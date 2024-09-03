import torch
import torch.nn as nn

import time


class FourierFeatureTransform(nn.Module):
    def __init__(self, num_input_channels, mapping_size, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn(
            (num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):
        B, N, C = x.shape
        x = (x.reshape(B*N, C) @ self._B).reshape(B, N, -1)
        x = 2 * torch.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class NerfTriplane(nn.Module):
    def __init__(self,
                 res=[256],
                 hidden_size=128,
                 feat_dim=16,
                 bound=torch.tensor([[-1, -1, -1.0], [1, 1, 1.0]]),
                 device='cuda') -> None:
        super().__init__()
        self.device = device
        self.bound = bound.to(device)
        self.res = res
        self.hidden_size = hidden_size
        # self.xy_embeddings = nn.ParameterList()
        # self.yz_embeddings = nn.ParameterList()
        # self.xz_embeddings = nn.ParameterList()
        self.xy_yz_xz_embeddings = nn.ParameterList()
        for r in res:
            self.xy_yz_xz_embeddings.append(nn.Parameter(
                torch.randn(3, feat_dim, int(r), int(r))*0.001))
            # self.yz_embeddings.append(nn.Parameter(torch.randn(1, feat_dim, int(r), int(r))*0.001))
            # self.xz_embeddings.append(nn.Parameter(torch.randn(1, feat_dim, int(r), int(r))*0.001))
        self.sample_time = 0
        self.forward_time = 0
        self.mlp = nn.Sequential(
            nn.Linear(len(res)*feat_dim, hidden_size),
            nn.LeakyReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
        )
        self.sigma_net = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size//2),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_size//2, hidden_size//2),
            # nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.color_net = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size//2, 3),
        )

    def sample_plane(self, coords2d, planelist: nn.ParameterList):
        assert len(coords2d.shape) == 3, coords2d.shape
        assert coords2d.min() >= -1 and coords2d.max() <= 1, (coords2d.min(), coords2d.max())
        all_feature = []
        for plane in planelist:
            sampled_features = torch.nn.functional.grid_sample(plane,
                                                               coords2d.reshape(
                                                                   coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                               mode='bilinear', padding_mode='zeros', align_corners=True)

            N, C, H, W = sampled_features.shape
            sampled_features = sampled_features.reshape(
                N, C, H*W).permute(0, 2, 1)
            all_feature.append(sampled_features)

        return torch.cat(all_feature, dim=-1)

    def sample_plane_xyyzxz(self, coords3d, plane):
        '''
        coords3d: N, 3 in [-1, 1]
        '''

        assert len(coords3d.shape) == 2, coords3d.shape
        assert coords3d.min() >= -1 and coords3d.max() <= 1, (coords3d.min(), coords3d.max())
        N, D = coords3d.shape
        xy = coords3d[:, 0:2]
        yz = coords3d[:, 1:3]
        xz = coords3d[:, [0, 2]]
        xy_yz_xz = torch.stack([xy, yz, xz], dim=0)
        xy_yz_xz = xy_yz_xz.reshape(3, N, 1, 2)
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           xy_yz_xz,
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled_features = torch.sum(sampled_features, dim=0).squeeze()
        sampled_features = sampled_features.permute(1, 0)
        return sampled_features

    def forward(self, coordinates, *kwargs):
        '''
        coordinates: xyz
        '''

        in_shape = coordinates.shape
        coordinates = coordinates.reshape(-1, 3)
        coordinates_normed = (
            coordinates - self.bound[0]) / (self.bound[1] - self.bound[0]) * 2 - 1
        coordinates_normed = torch.tanh(1.0*coordinates_normed)
        st = time.time()
        for i, r in enumerate(self.res):
            if i == 0:
                features = self.sample_plane_xyyzxz(
                    coordinates_normed, self.xy_yz_xz_embeddings[i])
            else:
                features = torch.cat([features, self.sample_plane_xyyzxz(
                    coordinates_normed, self.xy_yz_xz_embeddings[i])], dim=-1)
        self.sample_time = time.time() - st
        latent = self.mlp(features)
        sigma = self.sigma_net(latent).reshape(*in_shape[:-1], 1)
        color = self.color_net(latent).reshape(*in_shape[:-1], 3)
        self.forward_time = time.time() - st 
        # color = torch.sigmoid(color)
        # sigma = torch.exp(sigma)
        res = torch.cat([color, sigma], dim=-1)
        return res


class Triplane(nn.Module):
    def __init__(self, input_dim=3, output_dim=2, noise_val=None,  x_res=256, t_res=32, feat_dim=32, device='cuda'):
        super().__init__()
        print("Triplane Params:")
        print("x_res: ", x_res)
        print("t_res: ", t_res)
        print("feat_dim: ", feat_dim)

        self.device = device
        # self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, 32, 128, 128)*0.001) for _ in range(3*num_objs)])
        self.xt_embedding = nn.Parameter(
            torch.randn(1, feat_dim, t_res, x_res)*0.001)
        self.yt_embedding = nn.Parameter(
            torch.randn(1, feat_dim, t_res, x_res)*0.001)
        self.xy_embedding = nn.Parameter(
            torch.randn(1, feat_dim, x_res, x_res)*0.001)

        self.noise_val = noise_val
        # Use this if you want a PE
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid(),
        )
        # self.net = tcnn.Network(
        #     n_input_dims= feat_dim,
        #     n_output_dims= output_dim,
        #     network_config={
        #         "otype": "CutlassMLP",
        #         "activation": "LeakyReLU",
        #         # "output_activation": "None",
        #         'output_activation': 'Sigmoid',
        #         "n_neurons": 256,
        #         "n_hidden_layers": 4,
        #     },
        # )

    def expand_plane(self, magnitude):
        xt_shape = self.xt_embedding.shape
        xy_shape = self.xy_embedding.shape
        xt_x = torch.linspace(-1, 1, xt_shape[-1]*magnitude)
        xt_t = torch.linspace(-1, 1, xt_shape[-2]*magnitude)
        y, x = torch.meshgrid(xt_t, xt_x)
        xt_grid = torch.stack([x, y], dim=-1).to(self.device)
        xt_grid = xt_grid.reshape(1, *xt_grid.shape)
        new_xt_embedding = torch.nn.functional.grid_sample(
            self.xt_embedding, xt_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        new_yt_embedding = torch.nn.functional.grid_sample(
            self.yt_embedding, xt_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        xy_x = torch.linspace(-1, 1, xy_shape[-1]*magnitude)
        xy_y = torch.linspace(-1, 1, xy_shape[-2]*magnitude)
        y, x = torch.meshgrid(xy_y, xy_x)
        xy_grid = torch.stack([x, y], dim=-1).to(self.device)
        xy_grid = xy_grid.reshape(1, *xy_grid.shape)
        new_xy_embedding = torch.nn.functional.grid_sample(
            self.xy_embedding, xy_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        del self.xt_embedding
        del self.yt_embedding
        del self.xy_embedding

        self.xt_embedding = nn.Parameter(new_xt_embedding)
        self.yt_embedding = nn.Parameter(new_yt_embedding)
        self.xy_embedding = nn.Parameter(new_xy_embedding)

    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        assert coords2d.min() >= -1 and coords2d.max() <= 1, (coords2d.min(), coords2d.max())
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.reshape(
                                                               coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)

        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features

    def forward(self, coordinates, debug=False):
        '''
        coordinates: xy_t in [-1, 1]
        return: [0, 1]
        '''
        in_shape = coordinates.shape
        coordinates = coordinates.reshape(-1, 3)
        coordinates = coordinates[None]

        xy_embed = self.sample_plane(coordinates[..., 0:2], self.xy_embedding)
        yt_embed = self.sample_plane(coordinates[..., 1:3], self.yt_embedding)
        xt_embed = self.sample_plane(coordinates[..., :3:2], self.xt_embedding)

        # if self.noise_val != None:
        #    xy_embed = xy_embed + self.noise_val*torch.empty(xy_embed.shape).normal_(mean = 0, std = 0.5).to(self.device)
        #    yz_embed = yz_embed + self.noise_val*torch.empty(yz_embed.shape).normal_(mean = 0, std = 0.5).to(self.device)
        #    xz_embed = xz_embed + self.noise_val*torch.empty(xz_embed.shape).normal_(mean = 0, std = 0.5).to(self.device)

        features = torch.sum(torch.stack(
            [xy_embed, yt_embed, xt_embed]), dim=0).squeeze()
        if self.noise_val != None and self.training:
            features = features + self.noise_val * \
                torch.empty(features.shape).normal_(
                    mean=0, std=0.5).to(self.device)
        out = self.net(features).reshape(*in_shape[:-1], -1)
        # res = (out + 1) / 2.0
        return out


class MultiResBiplane(nn.Module):
    def __init__(self, output_dim=2, 
                 noise_val=None,  
                 x_res=[256], 
                 feat_dim=16, 
                 t_dim=8, 
                 net_layer = 2,
                 act=nn.Sigmoid(),
                 device='cuda') -> None:
        super().__init__()
        self.xy_embeddings = nn.ParameterList()
        for r in x_res:
            self.xy_embeddings.append(nn.Parameter(
                torch.randn(1, feat_dim, r, r)*0.0001))
        print("Biplane Params:")
        print("x_res: ", x_res)
        print("feat_dim: ", feat_dim)
        print("net_layer: ", net_layer)
        print("t_dim: ", t_dim)
        print("act: ", act)

        self.x_res = x_res
        width = 32
        input_dim = feat_dim*len(x_res)+t_dim*3
        if net_layer == 2:
            self.net = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.LeakyReLU(),
                # nn.Linear(input_dim // 2, width),
                # nn.LeakyReLU(),
                # nn.Linear(width, width),
                # nn.LeakyReLU(),
                # nn.Linear(width, width),
                # nn.LeakyReLU(),
                # nn.Linear(width, width),
                # nn.LeakyReLU(),
                nn.Linear(input_dim//2, output_dim),
                act,
            )
        elif net_layer ==3:
            self.net = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.LeakyReLU(),
                nn.Linear(input_dim // 2, input_dim // 4),
                nn.LeakyReLU(),
                nn.Linear(input_dim//4, output_dim),
                act,
            )
    
    def forward(self, coordinates, t_feat):
        '''
        coordinates: xy in [-1, 1]
        return: [0, 1]
        '''
        in_shape = coordinates.shape
        coordinates = coordinates.reshape(-1, 2)

        self.coord_record = coordinates
        
        coordinates = coordinates[None, :, None]

        

        xy_features = []

        for emb in self.xy_embeddings:
            xy_features.append(torch.nn.functional.grid_sample(emb,
                                                               coordinates,
                                                               mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(0).squeeze(-1).T)
        features = torch.cat(xy_features, dim=-1)
        # features = features.reshape(*in_shape[0:2], -1)
        if in_shape[2] > 1:
            t_feat = t_feat[:, None, None].expand(-1, in_shape[1], in_shape[2], -1).reshape(-1, t_feat.shape[-1])
        else:
            t_feat = t_feat[:, None].expand(-1, in_shape[1], -1).reshape(-1, t_feat.shape[-1])
        features = torch.cat([features, t_feat], dim=-1)

        out = self.net(features).reshape(*in_shape[:-1], -1)
        # res = (out + 1) / 2.0
        return out

class MultiResTriplane(nn.Module):
    def __init__(self, input_dim=3, output_dim=2, noise_val=None,  x_res=[256], t_res=[32], feat_dim=16, device='cuda'):
        super().__init__()
        print("Triplane Params:")
        print("x_res: ", x_res)
        print("t_res: ", t_res)
        print("feat_dim: ", feat_dim)
        assert len(x_res) == len(
            t_res), f"x_res and t_res must have the same length, but got x:{len(x_res)} and t:{len(t_res)}"
        self.x_res = x_res
        self.t_res = t_res
        self.device = device
        # self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, 32, 128, 128)*0.001) for _ in range(3*num_objs)])
        self.xt_yt_embeddings = nn.ParameterList()
        self.xy_embeddings = nn.ParameterList()
        for rx, rt in zip(x_res, t_res):
            self.xt_yt_embeddings.append(nn.Parameter(
                torch.randn(2, feat_dim, rt, rx)*0.001))
            self.xy_embeddings.append(nn.Parameter(
                torch.randn(1, feat_dim, rx, rx)*0.001))

        self.sample_time = 0
        self.forward_time = 0

        self.noise_val = noise_val
        width = 32
        # Use this if you want a PE
        self.net = nn.Sequential(
            nn.Linear(feat_dim*len(x_res), width),
            nn.LeakyReLU(),
            # nn.Linear(width, width),
            # nn.LeakyReLU(),
            # nn.Linear(width, width),
            # nn.LeakyReLU(),
            # nn.Linear(width, width),
            # nn.LeakyReLU(),
            nn.Linear(width, output_dim),
            nn.Sigmoid(),
        )
        # self.net = tcnn.Network(
        #     n_input_dims= feat_dim,
        #     n_output_dims= output_dim,
        #     network_config={
        #         "otype": "CutlassMLP",
        #         "activation": "LeakyReLU",
        #         # "output_activation": "None",
        #         'output_activation': 'Sigmoid',
        #         "n_neurons": 256,
        #         "n_hidden_layers": 4,
        #     },
        # )

    def sample_plane_xtytxy(self, coords3d, plane_xtyt, plane_xy):
        '''
        coords3d: N, 3 in [-1, 1]
        '''
        assert len(coords3d.shape) == 2, coords3d.shape
        assert coords3d.min() >= -1 and coords3d.max() <= 1, (coords3d.min(), coords3d.max())
        N, D = coords3d.shape
        xy = coords3d[:, 0:2]
        yt = coords3d[:, 1:3]
        xt = coords3d[:, [0, 2]]
        xt_yt = torch.stack([xt, yt], dim=0)

        feat_xtyt = torch.nn.functional.grid_sample(plane_xtyt,
                                                    xt_yt.reshape(2, N, 1, 2),
                                                    mode='bilinear', padding_mode='zeros', align_corners=True)
        feat_xy = torch.nn.functional.grid_sample(plane_xy,
                                                  xy.reshape(1, N, 1, 2),
                                                  mode='bilinear', padding_mode='zeros', align_corners=True)

        sampled_features = torch.cat([feat_xtyt, feat_xy], dim=0)

        sampled_features = torch.sum(sampled_features, dim=0).squeeze()
        sampled_features = sampled_features.permute(1, 0)
        return sampled_features

    def forward(self, coordinates, debug=False):
        '''
        coordinates: xy_t in [-1, 1]
        return: [0, 1]
        '''
        in_shape = coordinates.shape
        coordinates = coordinates.reshape(-1, 3)
        
        st = time.time()
        for i in range(len(self.x_res)):
            if i == 0:
                features = self.sample_plane_xtytxy(
                    coordinates, self.xt_yt_embeddings[i], self.xy_embeddings[i])
            else:
                features = torch.cat([features, self.sample_plane_xtytxy(
                    coordinates, self.xt_yt_embeddings[i], self.xy_embeddings[i])], dim=-1)
        self.sample_time = time.time() - st
        
        out = self.net(features).reshape(*in_shape[:-1], -1)
        self.forward_time = time.time() - st
        # res = (out + 1) / 2.0
        return out


if __name__ == "__main__":

    p = Triplane().to('cuda')

    opt = torch.optim.Adam(p.parameters(), lr=1e-3)

    coord = torch.rand(100, 3).to('cuda')
    res = p(coord)

    label = torch.rand(100, 2).to('cuda')
    loss = abs(label - res).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()

    p.expand_plane(2)

    coord = torch.rand(100, 3).to('cuda')
    res = p(coord)
    loss = abs(label - res).mean()
    loss.backward()
    opt.step()
