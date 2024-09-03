import numpy as np
import torch
import pdb
from torch import masked_select, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import networks.pe_relu
import pdb

from networks.triplane import MultiResBiplane, Triplane
# from networks.triplane_tcnn import Triplane, MultiResTriplane


def logit(x):
    return torch.log(x / (1 - x))


class InterpGrad(torch.autograd.Function):

    @staticmethod
    @torch.jit.script
    def forw_compute(x, y, qx):
        grad_qx = torch.zeros_like(qx)
        grad_x = torch.zeros_like(x)
        grad_y = torch.zeros_like(y)
        out = torch.zeros_like(qx)
        all_xr = x[..., 1:]
        all_xl = x[..., :-1]
        all_yr = y[..., 1:]
        all_yl = y[..., :-1]
        qx = torch.clamp(qx, all_xl[..., 0:1]*0.99, all_xr[..., -1:]*0.99)
        in_range = (qx >= all_xl) & (qx < all_xr)
        xr = all_xr[in_range] # N
        xl = all_xl[in_range]
        yr = all_yr[in_range]
        yl = all_yl[in_range]
        x_range = xr - xl
        # try:
        dYdqx = (yr - yl) / x_range
        dYdyr = (qx[..., 0] - xl) / x_range
        dYdyl = (xr - qx[..., 0]) / x_range
        dYdxr = -dYdyr * (yr - yl) / x_range
        dYdxl = -dYdyl * (yr - yl) / x_range
        # except:
            # pdb.set_trace()
            # i = torch.where(torch.sum(in_range, dim=-1) == 0)
            # i = i[0][0]
            # print("i", i)
            # print("qx", qx[i])
            # print("xl", all_xl[i])
            # print("xr", all_xr[i])

        grad_qx[..., 0] += dYdqx
        grad_x[..., :-1][in_range] += dYdxl
        grad_x[..., 1:][in_range] += dYdxr
        grad_y[..., :-1][in_range] += dYdyl
        grad_y[..., 1:][in_range] += dYdyr
        out[..., 0] = dYdqx * (qx[..., 0] - xl) + yl
        return out, grad_x, grad_y, grad_qx

    @staticmethod
    def forward(ctx, x, y, qx):
        '''
        x: [N, n_samples], should be monotonic
        y: [N, n_samples]
        qx: [N, 1]
        '''
        N, n_samples = x.shape
        out, grad_x, grad_y, grad_qx = InterpGrad.forw_compute(x, y, qx)
        
        ctx.save_for_backward(grad_x, grad_y, grad_qx)

        return out

    @staticmethod
    @torch.jit.script
    def back_compute(grad_out, grad_x, grad_y, grad_qx):
        grad_x = grad_x * grad_out
        grad_y = grad_y * grad_out
        grad_qx = grad_qx * grad_out
        return grad_x, grad_y, grad_qx

    @staticmethod
    def backward(ctx, grad_out):
        grad_x, grad_y, grad_qx = ctx.saved_tensors
        # grad_x = grad_x * grad_out
        # grad_y = grad_y * grad_out
        # grad_qx = grad_qx * grad_out
        grad_x, grad_y, grad_qx = InterpGrad.back_compute(grad_out, grad_x, grad_y, grad_qx)
        return grad_x, grad_y, grad_qx
    
interp_grad = InterpGrad.apply
        


def interp(x, y, qx):
    '''
    x: [N, n_samples], should be monotonic
    y: [N, n_samples]
    qx: [N, m_samples]
    '''
    N, n_samples = x.shape
    qy = torch.zeros_like(qx)
    
    for i in range(n_samples-1):
        in_range = (qx >= x[:, i:i+1]) & (qx <= x[:, i+1:i+2])
        qy[in_range] = ((y[:, i+1:i+2][in_range] - y[:, i:i+1][in_range]) / (x[:, i+1:i+2][in_range] - x[:, i:i+1][in_range]) * (qx[in_range] - x[:, i:i+1][in_range]) + y[:, i:i+1][in_range])
    
    return qy

class CouplingLayer(nn.Module):
    def __init__(self, map_st, mask):
        super().__init__()
        self.map_st = map_st
        self.register_buffer('mask', mask) 
        # self.mask = mask # old version
        self.cnt = 0
        self.multiple = torch.tensor([-1, -1, 1, 1, -1, -1, 1, 1, 2, 2.], device=mask.device, requires_grad=True)
        self.plus = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1e-8, 1e-8], device=mask.device, requires_grad=True)
        self.out_range = torch.tensor([-1000., 1000.], device=mask.device, requires_grad=True)

    @staticmethod
    @torch.jit.script
    def get_xy_compute(dxl1, dxl2, dxr1, dxr2, dyl1, dyl2, dyr1, dyr2, kl, kr):
        kl = kl * 2.0# + 1e-8
        kr = kr * 2.0# + 1e-8

       
        xL1 = -dxl1
        xL2 = -dxl1 - dxl2
        yL1 = - dyl1
        yL2 = - dyl1 - dyl2

        xR1 = dxr1
        xR2 = dxr1 + dxr2
        yR1 = dyr1
        yR2 = dyr1 + dyr2


        xR3 = xR2 + 1e4
        xL3 = xL2 - 1e4
        yR3 = yR2 + kr*1e4
        yL3 = yL2 - kl*1e4

        all_x = torch.cat([xL3, xL2, xL1, xR1, xR2, xR3], dim=-1)
        all_y = torch.cat([yL3, yL2, yL1, yR1, yR2, yR3], dim=-1)
        return all_x, all_y


    def get_all_xy(self, dxdykk):
        '''
        dxdykk: [N, P, 1, 10], dxl1, dxl2, dxr1, dxr2, dy...., kl, kr, all > 0
        '''

        dxl2, dxl1, dxr1, dxr2, dyl2, dyl1, dyr1, dyr2, kl, kr = torch.split(
            dxdykk, split_size_or_sections=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dim=-1)

        # print("max_err = ", abs(all_x -t_all_x).max().item(), abs(all_y - t_all_y).max().item())

        return CouplingLayer.get_xy_compute(dxl1, dxl2, dxr1, dxr2, dyl1, dyl2, dyr1, dyr2, kl, kr)

    def forward(self, x, t_feat):
        if self.mask.ndim - x.ndim == 1:
            self.mask = self.mask.squeeze(0)

        
        y = x * self.mask

        dxdykk = self.map_st(torch.tanh(
            x[..., self.mask.squeeze().bool()]), t_feat)  # [0, 1]

        

        # l_out xL2  left xL1  middle xR1  right xR2  r_out
        

        if self.cnt % 50 == 0:
            # with torch.no_grad():
            #     x_modify = x[..., (1 - self.mask).bool()]
            #     right = (x_modify > xR1) & (x_modify < xR2)
            #     left = (x_modify < xL1) & (x_modify > xL2)
            #     r_out = x_modify >= xR2
            #     l_out = x_modify <= xL2
            #     middle = (x_modify <= xR1) & (x_modify >= xL1)
            #     print(f"middle rate: {middle.sum().item()/x_modify.numel():.2f}", f'l_out rate: {l_out.sum().item()/x_modify.numel():.2f}',
            #       f'r_out rate: {r_out.sum().item()/x_modify.numel():.2f}', "mask: ", self.mask)
                
                # self.k_map = torch.zeros_like(x_modify)
                # self.k_map[middle] = (yR1[middle] - yL1[middle]) / \
                #     (xR1[middle] - xL1[middle])
                # self.k_map[right] = (yR2[right] - yR1[right]) / \
                #     (xR2[right] - xR1[right])
                # self.k_map[left] = (yL2[left] - yL1[left]) / \
                #     (xL2[left] - xL1[left])
                # self.k_map[r_out] = kr[r_out]
                # self.k_map[l_out] = kl[l_out]
            self.cnt = 0
        self.cnt += 1

        
        
        all_x, all_y = self.get_all_xy(dxdykk)
        grad_y = interp_grad(all_x.reshape(-1, 6), all_y.reshape(-1, 6), x[..., (1 - self.mask).bool()].reshape(-1, 1)).reshape(*y.shape[:-1], 1)
        # self.new_y.retain_grad()
        # self.grad_y.retain_grad()
        y[..., (1-self.mask).bool()] = grad_y #self.new_y
        

        return y, None

    def inverse(self, y, t_feat):
        if self.mask.ndim - y.ndim == 1:
            self.mask = self.mask.squeeze(0)

        
        x = y * self.mask

        dxdykk = self.map_st(torch.tanh(
            y[..., self.mask.squeeze().bool()]), t_feat)  # [0, 1]
        
        all_x, all_y = self.get_all_xy(dxdykk)

        # new_x = interp1d(all_y.reshape(-1, 6), all_x.reshape(-1, 6), y[..., (1 - self.mask).bool()].reshape(-1, 1), None).reshape(*y.shape[:-1], 1)
        # self.new_x = interp(all_y.reshape(-1, 6), all_x.reshape(-1, 6), y[..., (1 - self.mask).bool()].reshape(-1, 1)).reshape(*x.shape[:-1], 1)

        grad_x = interp_grad(all_y.reshape(-1, 6), all_x.reshape(-1, 6), y[..., (1 - self.mask).bool()].reshape(-1, 1)).reshape(*x.shape[:-1], 1)
        # self.new_x.retain_grad()
        # self.grad_x.retain_grad()
        x[..., (1-self.mask).bool()] = grad_x

        # y_modify[middle] = (y_modify[middle] - yL1[middle]) * (xR1[middle] -
        #                                                        xL1[middle]) / (yR1[middle] - yL1[middle]) + xL1[middle]
        # y_modify[right] = (y_modify[right] - yR1[right]) * (xR2[right] -
        #                                                     xR1[right]) / (yR2[right] - yR1[right]) + xR1[right]
        # y_modify[left] = (y_modify[left] - yL1[left]) * (xL2[left] -
        #                                                  xL1[left]) / (yL2[left] - yL1[left]) + xL1[left]
        # y_modify[r_out] = (y_modify[r_out] - yR2[r_out]) / \
        #     kr[r_out] + xR2[r_out]
        # y_modify[l_out] = (y_modify[l_out] - yL2[l_out]) / \
        #     kl[l_out] + xL2[l_out]

        # x[..., (1-self.mask).bool()] = y_modify

        return x, None


def apply_homography_xy1(mat, xy1):
    """
    :param mat (*, 3, 3) (# * dims must match uv dims)
    :param xy1 (*, H, W, 3)
    :returns warped coordinates (*, H, W, 2)
    """
    out_h = torch.matmul(mat, xy1[..., None])
    return out_h[..., :2, 0] / (out_h[..., 2:, 0] + 1e-8)


def apply_homography(mat, uv):
    """
    :param mat (*, 3, 3) (# * dims must match uv dims)
    :param uv (*, H, W, 2)
    :returns warped coordinates (*, H, W, 2)
    """
    uv_h = torch.cat([uv, torch.ones_like(uv[..., :1])], dim=-1)  # (..., 3)
    return apply_homography_xy1(mat, uv_h)


class NVPnonlin(nn.Module):
    def __init__(
            self,
            n_layers,
            n_frames,
            feature_dim=32,
            t_dim=8,
            bound=torch.tensor([[-1, -1, -1], [1, 1, 1]]),
            multires=False,
            base_res=8,
            net_layer=2,
            pe_freq=4,
            normalization=True,
            affine=False,
            activation=nn.LeakyReLU,
            device='cuda',
    ):
        super().__init__()
        self._checkpoint = False
        self.affine = affine
        # self.bound = bound.to(device)
        self.register_buffer('bound', bound.to(device))
        self.feature_dim = feature_dim
        self.multires = multires
        self.base_res = base_res
        self.net_layer = net_layer
        self.pe_freq = pe_freq
        self.normalization = normalization
        self.activation = activation
        self.device = device
        # make layers
        input_dims = 3
        normalization = nn.BatchNorm1d if normalization else None
        self.base_res = base_res
        # self.frames = n_frames
        self.register_buffer('frames', torch.tensor(n_frames))
        t_baseres = n_frames // 20
        t_res = [t_baseres, t_baseres*5, t_baseres*13]
        # t_res = [t_baseres]
        self.t_embeddings = nn.ParameterList()
        for res in t_res:
            self.t_embeddings.append(nn.Parameter(
                torch.randn(1, t_dim, 1, res)*0.001))

        # self.featbank = HashGrid(grid_levels=8,
        #                          max_grid_size=2**16,
        #                          feat_dim=16,
        #                          coarse_res=16,
        #                          fine_res=1024,
        #                          device=device).to(device)

        self.layers1 = nn.ModuleList()

        self.layer_idx = [i for i in range(n_layers)]

        i = 0
        mask_selection = []
        while i < n_layers:
            mask_selection.append(torch.randperm(input_dims))
            i += input_dims
        mask_selection = torch.cat(mask_selection)
        self.mask_selection = mask_selection

        for i in self.layer_idx:
            # get mask
            mask2 = torch.zeros(input_dims, device=device)
            mask2[self.mask_selection[i]] = 1
            mask1 = 1 - mask2

            # get transformation
            if multires:
                # base_res = 64

                map_st = MultiResBiplane(feat_dim=feature_dim,
                                         device=device,
                                         # , base_res*17],
                                         x_res=[base_res, base_res*8],
                                         t_dim=t_dim,
                                         output_dim=10,
                                         net_layer=net_layer,
                                         # x_res=[64, 64*5, 64*13, 64*27],
                                         # t_res=[8, 8*5, 8*13, 8*27]
                                         )
            else:
                map_st = Triplane(feat_dim=feature_dim, device=device)

            self.layers1.append(CouplingLayer(map_st, mask1))

        if self.affine:
            # this mlp takes time and depth as input and produce an affine transformation for x and y
            self.affine_mlp = networks.pe_relu.MLP(input_dim=2,
                                                   hidden_size=256,
                                                   n_layers=2,
                                                   skip_layers=[],
                                                   use_pe=True,
                                                   pe_dims=[1],
                                                   pe_freq=pe_freq,
                                                   output_dim=5).to(device)
        self.forw_path = []
        self.inv_path = []

    def _expand_features(self, F, x):
        _, N, K, _ = x.shape
        return F[:, None, None, :].expand(-1, N, K, -1)

    def _call(self, func, *args, **kwargs):
        if self._checkpoint:
            return checkpoint(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    def invert_affine(self, a, b, c, d, tx, ty, zeros, ones):
        determinant = a * d - b * c

        inverse_determinant = 1.0 / determinant

        inverted_a = d * inverse_determinant
        inverted_b = -b * inverse_determinant
        inverted_c = -c * inverse_determinant
        inverted_d = a * inverse_determinant
        inverted_tx = (b * ty - d * tx) * inverse_determinant
        inverted_ty = (c * tx - a * ty) * inverse_determinant

        return torch.cat([inverted_a, inverted_b, inverted_tx,
                          inverted_c, inverted_d, inverted_ty,
                          zeros, zeros, ones], dim=-1).reshape(*a.shape[:-1], 3, 3)

    def get_affine(self, theta, inverse=False):
        """
        expands the 5 parameters into 3x3 affine transformation matrix
        :param theta (..., 5)
        :returns mat (..., 3, 3)
        """
        angle = theta[..., 0:1]
        scale1 = torch.exp(theta[..., 1:2])
        scale2 = torch.exp(theta[..., 3:4])
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        a = cos * scale1
        b = -sin * scale1
        c = sin * scale2
        d = cos * scale2
        tx = theta[..., 2:3]
        ty = theta[..., 4:5]
        zeros = torch.zeros_like(a)
        ones = torch.ones_like(a)
        if inverse:
            return self.invert_affine(a, b, c, d, tx, ty, zeros, ones)
        else:
            return torch.cat([a, b, tx, c, d, ty, zeros, zeros, ones], dim=-1).reshape(*theta.shape[:-1], 3, 3)

    def _affine_input(self, t, x, inverse=False):
        depth = x[..., -1]  # [n_imgs, n_pts, n_samples]
        net_in = torch.stack(
            [t[..., None].repeat(1, *x.shape[1:3]), depth], dim=-1)
        # [n_imgs, n_pts, n_samples, 3, 3]
        affine = self.get_affine(self.affine_mlp(net_in), inverse=inverse)
        xy = x[..., :2]
        xy = apply_homography(affine, xy)
        x = torch.cat([xy, depth[..., None]], dim=-1)
        return x

    def get_t_feature(self, t):
        '''
        t: [n_imgs, 1]
        '''
        t = t * 2 - 1.0
        N = t.shape[0]
        t = t.reshape(1, N, 1, 1)
        zeros = torch.zeros_like(t)
        t = torch.cat([t, zeros], dim=-1)

        t_feat = []
        for featlist in self.t_embeddings:
            t_feat.append(F.grid_sample(
                featlist, t, align_corners=True).squeeze(0).squeeze(-1).T)
        t_feat = torch.cat(t_feat, dim=-1)
        return t_feat

    def forward(self, t, f, x):
        # x_step = [x]
        # t = t * 2 - 1.0
        t_feat = self.get_t_feature(t)
        x = (x - (self.bound[1] + self.bound[0]) / 2) / \
            ((self.bound[1] - self.bound[0])/2)
        # x[..., 2] = torch.log(x[..., 2] + 1e-8)
        y = x  # .double()
        if self.affine:
            y = self._affine_input(t, y)
        self.forw_path = []
        # self.forw_path.append(y)
        for i in self.layer_idx:
            # feat_i = self.code_projectors[i](feat)
            # feat_i = self._expand_features(feat_i, y)
            l1 = self.layers1[i]
            y, _ = self._call(l1, y, t_feat)
            # self.forw_path.append(y)
            # x_step.append(y)
        return y  # , x_step

    def inverse(self, t, f, y, prev=None):
        # t = t * 2 - 1.0
        t_feat = self.get_t_feature(t)

        x = y  # .double()
        self.inv_path = []
        # self.inv_path.append(x)
        # x_step = [x]
        for i in reversed(self.layer_idx):
            # feat_i = self.code_projectors[i](feat)
            # feat_i = self._expand_features(feat_i, x)
            l1 = self.layers1[i]
            x, _ = self._call(l1.inverse, x, t_feat)
            # self.inv_path.append(x)
            # x_step.append(x)
        if self.affine:
            x = self._affine_input(t, x, inverse=True)

        # x[..., 2] = torch.exp(x[..., 2]) - 1e-8

        x = x * ((self.bound[1] - self.bound[0])/2) + \
            (self.bound[1] + self.bound[0]) / 2
        return x  # , x_step
