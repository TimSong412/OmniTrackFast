import numpy as np
import torch
from torch import masked_select, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import networks.pe_relu

from networks.triplane import MultiResBiplane, Triplane
# from networks.triplane_tcnn import Triplane, MultiResTriplane


class CouplingLayer(nn.Module):
    def __init__(self, map_st, mask):
        super().__init__()
        self.map_st = map_st
        self.mask = mask

    def forward(self, y, t_feat):
        if self.mask.ndim - y.ndim == 1:
            self.mask = self.mask.squeeze(0)
        y1 = y * self.mask
        # F_y1 = torch.cat([F, self.projection(y[..., self.mask.squeeze().bool()])], dim=-1)
        # st = self.map_st(F_y1)
        
        st = self.map_st(torch.tanh(y[..., self.mask.squeeze().bool()]), t_feat)  # [0, 1]

        s, t = torch.split(st, split_size_or_sections=1, dim=-1)
        # s = torch.clamp(s, min=-8, max=8)
        # s = torch.tanh(s)*8.0

        # best: 5.0*s+5e-3, 10.0*t-5.0
        s = 5.0*s+5e-3  # [0, 5]
        t = t*6- 3.0  # [-3, 3]

        # x = y1 + (1 - self.mask) * ((y - t) * torch.exp(-s))
        x = y1 + (1 - self.mask) * ((y - t) / (s))
        ldj = (-s).sum(-1)

        # x = torch.clamp(x, min=0, max=1)
        # x = torch.sigmoid(x)

        return x, ldj

    def inverse(self, x, t_feat):
        if self.mask.ndim - x.ndim == 1:
            self.mask = self.mask.squeeze(0)

        # x = torch.log(x / (1 - x))

        x1 = x * self.mask

        # F_x1 = torch.cat([F, self.projection(x[..., self.mask.squeeze().bool()])], dim=-1)
        # st = self.map_st(F_x1)

        st = self.map_st(torch.tanh(x[..., self.mask.squeeze().bool()]), t_feat)

        s, t = torch.split(st, split_size_or_sections=1, dim=-1)
        # s = torch.clamp(s, min=-8, max=8)
        # s = torch.tanh(s)*8.0
        s = 5.0*s+5e-3  # [0, 10]
        t = t*6 - 3.0  # [-5, 5]

        y = x1 + (1 - self.mask) * (x * (s) + t)
        ldj = s.sum(-1)

        # y = torch.clamp(y, min=0, max=1)
        # y = torch.sigmoid(y)

        return y, ldj


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


class NVPbiplane(nn.Module):
    def __init__(
            self,
            n_layers,
            n_frames,
            feature_dim=32,
            t_dim = 8,
            bound=torch.tensor([[-1, -1, -1], [1, 1, 1]]),
            multires=False,
            base_res = 8,            
            pe_freq=4,
            normalization=True,
            affine=False,
            net_layer=2,
            activation=nn.LeakyReLU,
            device='cuda',
    ):
        super().__init__()
        self._checkpoint = False
        self.affine = affine
        self.bound = bound.to(device)
        # make layers
        input_dims = 3
        normalization = nn.BatchNorm1d if normalization else None
        self.base_res = base_res
        self.frames = n_frames
        t_baseres = n_frames // 20
        t_res = [t_baseres, t_baseres*5, t_baseres*13]
        self.t_embeddings = nn.ParameterList()
        for res in t_res:
            self.t_embeddings.append(nn.Parameter(torch.randn(1, t_dim, 1, res)*0.001))

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
            mask2[mask_selection[i]] = 1
            mask1 = 1 - mask2

            # get transformation
            if multires:
                # base_res = 64
  
                map_st = MultiResBiplane(feat_dim=feature_dim,
                                          device=device,
                                          x_res=[base_res, base_res*8], #, base_res*13], 
                                          t_dim=t_dim,
                                          net_layer=net_layer
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
            t_feat.append(F.grid_sample(featlist, t, align_corners=True).squeeze(0).squeeze(-1).T)
        t_feat = torch.cat(t_feat, dim=-1)
        return t_feat


    def forward(self, t, f, x):
        # x_step = [x]
        # t = t * 2 - 1.0
        t_feat = self.get_t_feature(t)
        x = (x - (self.bound[1] + self.bound[0]) / 2) / \
            ((self.bound[1] - self.bound[0])/2)
        y = x  # .double()
        if self.affine:
            y = self._affine_input(t, y)
        for i in self.layer_idx:
            # feat_i = self.code_projectors[i](feat)
            # feat_i = self._expand_features(feat_i, y)
            l1 = self.layers1[i]
            y, _ = self._call(l1, y, t_feat)
            # x_step.append(y)
        return y  # , x_step

    def inverse(self, t, f, y, prev=None):
        # t = t * 2 - 1.0
        t_feat = self.get_t_feature(t)

        x = y  # .double()
        # x_step = [x]
        for i in reversed(self.layer_idx):
            # feat_i = self.code_projectors[i](feat)
            # feat_i = self._expand_features(feat_i, x)
            l1 = self.layers1[i]
            x, _ = self._call(l1.inverse, x, t_feat)
            # x_step.append(x)
        if self.affine:
            x = self._affine_input(t, x, inverse=True)

        x = x * ((self.bound[1] - self.bound[0])/2) + \
            (self.bound[1] + self.bound[0]) / 2
        return x  # , x_step
