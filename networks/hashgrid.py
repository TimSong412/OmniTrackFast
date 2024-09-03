import torch
import torch.nn as nn
from networks.hashmem import HashMemory


def positional_encoding(x, num_frequencies=6, incl_input=True):
    """
    Apply positional encoding to the input.
    x: N, 3

    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """

    results = []
    if incl_input:
        results.append(x)
    for i in range(num_frequencies):
        results.append(torch.sin(x * 2 ** i*torch.pi))
        results.append(torch.cos(x * 2 ** i*torch.pi))

    return torch.cat(results, dim=-1)


class HashGrid(nn.Module):
    def __init__(self,
                 grid_levels=16,
                 max_grid_size=2**16,
                 feat_dim=16,
                 coarse_res=16,
                 fine_res=512,
                 device='cuda'):
        super(HashGrid, self).__init__()
        '''
        max_grid_size: the maximum number of grids T
        '''
        self.device = device
        self.grid_levels = grid_levels
        self.max_grid_size = max_grid_size
        self.hash_factors = torch.Tensor(
            [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]).long().to(device)
        self.feat_dim = feat_dim
        self.coarse_res = coarse_res
        self.fine_res = fine_res
        self.b = 2**(torch.log2(torch.tensor(fine_res /
                     float(coarse_res)))/(grid_levels-1))

        self.table_bank = nn.ParameterList([
            nn.Parameter(torch.randn((max_grid_size, feat_dim),
                         dtype=torch.float32).to(device), requires_grad=True),
            nn.Parameter(torch.randn((max_grid_size, feat_dim),
                         dtype=torch.float32).to(device), requires_grad=True),
            nn.Parameter(torch.randn((max_grid_size, feat_dim), dtype=torch.float32).to(device), requires_grad=True)])

    def hash(self, x: torch.Tensor):
        '''
        x: N, ..., 3
        '''
        idx = torch.zeros((*x.shape[0:-1], 1),
                          dtype=torch.int64, device=x.device)
        for i in range(x.shape[-1]):
            idx ^= (x[..., i] * self.hash_factors[i]).long().unsqueeze(-1)
        return idx % self.max_grid_size

    def query(self, x: torch.Tensor, tableidx: int):
        '''
        x: N, 3
        '''
        featlist = []
        for l in range(self.grid_levels):
            Nl = torch.floor(self.coarse_res*self.b**l)
            upper = torch.ceil(x*Nl).long()
            lower = torch.floor(x*Nl).long()
            corner_8pts = torch.stack([lower,
                                       torch.stack(
                                           [lower[..., 0], lower[..., 1], upper[..., 2]], dim=-1),
                                       torch.stack(
                                           [lower[..., 0], upper[..., 1], lower[..., 2]], dim=-1),
                                       torch.stack(
                                           [lower[..., 0], upper[..., 1], upper[..., 2]], dim=-1),
                                       torch.stack(
                                           [upper[..., 0], lower[..., 1], lower[..., 2]], dim=-1),
                                       torch.stack(
                                           [upper[..., 0], lower[..., 1], upper[..., 2]], dim=-1),
                                       torch.stack(
                                           [upper[..., 0], upper[..., 1], lower[..., 2]], dim=-1),
                                       upper], dim=-1).permute(0, 1, 2, 4, 3)
            corner_idx = self.hash(corner_8pts)

            weight = (x*Nl - lower)
            corner_feats = self.table_bank[tableidx][corner_idx.squeeze(-1)]
            inter_z = weight[..., 2, None, None] * corner_feats[:, :, :, [1, 3, 5, 7]] + (
                1-weight[..., 2, None, None]) * corner_feats[:, :, :, [0, 2, 4, 6]]
            inter_y = weight[..., 1, None, None] * inter_z[:, :, :, [1, 3]
                                                           ] + (1-weight[..., 1, None, None]) * inter_z[:, :, :, [0, 2]]
            inter_x = weight[..., 0, None, None] * inter_y[:, :, :, [1]
                                                           ] + (1-weight[..., 0, None, None]) * inter_y[:, :, :, [0]]
            featl = inter_x.squeeze(3)
            featlist.append(featl)
        return featlist

    def forward(self, x: torch.Tensor, t, mask: torch.Tensor):
        '''
        x: N, 3 (x, y, z)
        t: N, 1
        mask: N, 3 [0 or 1]
        '''
        x = x[..., mask.squeeze().bool()]
        # print("XMIN = ", x.min())
        idx = torch.where(mask.squeeze() == 0)[0]
        x_t = torch.cat(
            [x, t[:, None, None].repeat(1, *x.shape[1:3], 1)], dim=-1)
        x_encode = positional_encoding(x_t, num_frequencies=6, incl_input=True)
        featlist = self.query(x_t, tableidx=idx)
        featlist = torch.cat(featlist, dim=-1)
        # if len(featlist.shape) != len(x_encode.shape):
        #     print("featlist shape = ", featlist.shape)
        #     print("x_encode shape = ", x_encode.shape)
        #     featlist = featlist.unsqueeze(0)
        latent = torch.cat([featlist, x_encode], dim=-1)
        return latent





class HashGridTcnn(nn.Module):
    def __init__(self,
                 bbox=torch.tensor([[-1, -1, 0], [1, 1, 2.0]]),
                 grid_levels=16,
                 hash_dim=3,
                 bank_dim=2,
                 max_grid_size_exp=16,
                 feat_dim=16,
                 coarse_res=16,
                 fine_res=512,
                 num_couples=6,
                 couple_mask=[0, 1, 2, 0, 1, 2],
                 device='cuda'):
        super(HashGridTcnn, self).__init__()
        '''
        max_grid_size: the maximum number of grids T
        '''
        self.hash_dim = hash_dim
        self.device = device
        self.grid_levels = grid_levels
        self.max_grid_size = 2**max_grid_size_exp
        self.feat_dim = feat_dim
        self.coarse_res = coarse_res
        self.fine_res = fine_res
        self.n_couples = num_couples
        bboxlist = []
        # box: x, y, z, t
        for i in range(num_couples):

            couple = couple_mask[i]
            dimmask = torch.ones_like(bbox).bool()
            dimmask[:, couple] = False
            box = bbox[dimmask].clone().reshape(2, -1)
            if hash_dim == 3:
                box = torch.cat([box, torch.tensor([[0], [1.0]])], dim=-1)
            bboxlist.append(box)

        self.tablebank = nn.ModuleList([HashMemory(bbox=bboxlist[cid], levels=grid_levels,
                                       base_res=coarse_res, output_dim=feat_dim, bank_dim=bank_dim, input_dim=hash_dim) for cid in range(num_couples)])
        # self.outnorm = nn.GroupNorm(32, 128)

    def query(self, x: torch.Tensor, tableidx: int):
        '''
        x: N, ...,  3
        '''
        X_shape = x.shape
        x = x.reshape(-1, self.hash_dim)
        res = self.tablebank[tableidx](x)
        # res = self.outnorm(res)
        return res.reshape(*X_shape[:-1], -1)

    def forward(self, x: torch.Tensor, t, mask: torch.Tensor, layerid):
        '''
        x: N, 3 (x, y, z)
        t: N, 1
        mask: N, 3 [0 or 1]
        '''
        x = x[..., mask.squeeze().bool()]
        # print("XMIN = ", x.min())
        idx = layerid
        t_encode = None
        if self.hash_dim == 3:
            if x.ndim == 3:
                x_t = torch.cat(
                    [x, t[:, None].repeat(1, *x.shape[1:-1], 1)], dim=-1)
            else:
                x_t = torch.cat(
                [x, t[:, None, None].repeat(1, *x.shape[1:3], 1)], dim=-1)
        else:
            x_t = x
            if x.ndim == 3:
                t_encode = positional_encoding(t[:, None].repeat(1, *x.shape[1:-1], 1), num_frequencies=6, incl_input=True)
            else:
                t_encode = positional_encoding(t[:, None, None].repeat(1, *x.shape[1:-1], 1), num_frequencies=6, incl_input=True)

        feat = self.query(x_t, tableidx=idx)
        
        x_encode = positional_encoding(x_t, num_frequencies=6, incl_input=True)
        if t_encode is not None:
            x_encode = torch.cat([x_encode, t_encode], dim=-1)
        latent = torch.cat([feat, x_encode], dim=-1)
        return latent


if __name__ == "__main__":
    mem = HashMemory(bbox=torch.tensor([[-1, -1, -1], [1, 1, 1.0]]),
                     levels=8)
    a = torch.tensor([[0, 0, 0], [0, 0, 0.0]])
    res = mem(a)
    print(res.shape)
