import glob
import os
import pdb
import time

import cv2
import imageio.v2 as imageio
import trimesh
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import random
import csv
import kornia
import tqdm
import matplotlib.pyplot as plt

import util
from criterion import masked_mse_loss, masked_l1_loss, compute_depth_range_loss, lossfun_distortion
from networks.mfn import GaborNet
from networks.nvp_simplified import NVPSimplified
from kornia import morphology as morph
from torchsummary import summary

from pathlib import Path

# torch.autograd.set_detect_anomaly(True) 




def init_weights(m):
    # Initializes weights according to the DCGAN paper
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def de_parallel(model):
    return model.module if hasattr(model, 'module') else model

class DepthMem(nn.Module):
    def __init__(self, args, depthmaps, device='cuda'):
        super(DepthMem, self).__init__()
        self.args = args
        self.device = device
        if args.opt_depth:
            self.depthmaps = nn.parameter.Parameter(depthmaps.clone())
        else:
            self.depthmaps = depthmaps.clone().to(device)
        

class TriplaneDepTrainer():
    def __init__(self, args, images=None, device='cuda'):
        self.args = args
        self.device = device
        self.out_dir = args.out_dir

        self.backtime = 0

        self.read_data(images)
        self.depthmem = DepthMem(args, self.depthmaps, device=device)

        self.feature_mlp = None
        if args.deformer == 'triplane':
            from networks.nvp_triplane import NVPtriplane
            box_err = 1.
            feat_dim = args.feat_dim
            bound = torch.tensor([[-1.0, -1.0, 0.1], [1.0, 1.0, 2.1]])*box_err
            self.deform_mlp = NVPtriplane(n_layers=6,
                                          n_frames=self.images.shape[0],
                                          feature_dim=feat_dim,
                                          multires=args.multires,
                                          base_res=args.triplane_res,
                                          bound=bound,
                                          device=device).to(device)
        
        elif args.deformer == 'biplane':
            from networks.nvp_decom import NVPbiplane
            box_err = 1.
            feat_dim = args.feat_dim            
            # bound = torch.tensor([[-1.0, -1.0, 0.1], [1.0, 1.0, 2.1]])*box_err
            # bound = torch.tensor([[-self.w / 2 / self.f * 1., -self.h / 2 / self.f * 1., 0.1], [self.w / 2 / self.f * 1, self.h / 2 / self.f * 1, 2.1]])*box_err
            bound = self.bound
            self.deform_mlp = NVPbiplane(n_layers=6,
                                         n_frames=self.images.shape[0],
                                          feature_dim=feat_dim,
                                          t_dim = args.t_dim,                                          
                                          multires=args.multires,
                                          base_res=args.triplane_res,
                                          bound=bound,
                                          net_layer=args.net_layer,
                                          device=device).to(device)
        elif args.deformer == 'nonlin':
            from networks.nvp_nonlin import NVPnonlin
            box_err = 1.
            feat_dim = args.feat_dim
            # bound = torch.tensor([[-self.w / 2 / self.f * 1., -self.h / 2 / self.f * 1., 0.1], [self.w / 2 / self.f * 1, self.h / 2 / self.f * 1, 2.1]])*box_err
            bound = self.bound
            self.deform_mlp = NVPnonlin(n_layers=6,
                                         n_frames=self.images.shape[0],
                                          feature_dim=feat_dim,
                                          t_dim = 16,
                                          multires=args.multires,
                                          base_res=args.triplane_res,
                                          net_layer=args.net_layer,
                                          bound=bound,
                                          device=device).to(device)
            
        else:
            self.deform_mlp = NVPSimplified(n_layers=6,
                                            feature_dims=128,
                                            hidden_size=[256, 256, 256],
                                            proj_dims=256,
                                            code_proj_hidden_size=[],
                                            proj_type='fixed_positional_encoding',
                                            pe_freq=args.pe_freq,
                                            normalization=False,
                                            affine=args.use_affine,
                                            device=device).to(device)
    
        self.optimizer = torch.optim.Adam([
            # {'params': self.featbank.parameters() if args.hashdeform else self.feature_mlp.parameters() , 'lr': args.lr_feature},
            # {'params': self.feature_mlp.parameters(), 'lr': args.lr_feature},
            {'params': self.deform_mlp.parameters(), 'lr': args.lr_deform},
            
            {'params': self.depthmem.parameters(), 'lr': args.lr_depth},
        ])

        def check_gradient_with_name(name):
            def check_gradient(grad):
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    # print(name)
                    # print(grad)
                    # pdb.set_trace()
                    # print("Nan in grad")
                    pass
            return check_gradient
     
            
        

        self.learnable_params = list(self.deform_mlp.parameters()) + \
                                list(self.depthmem.parameters())
      

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=args.lrate_decay_steps,
                                                         gamma=args.lrate_decay_factor)


        seq_name = os.path.basename(args.data_dir.rstrip('/'))
        # self.out_dir = os.path.join(args.save_dir, '{}_{}_{}'.format(now, args.expname, seq_name))

        self.step = self.load_from_ckpt(self.out_dir if self.args.load_dir == '' else self.args.load_dir,
                                        load_opt=self.args.load_opt,
                                        load_scheduler=self.args.load_scheduler)
        self.time_steps = torch.linspace(
            1, self.num_imgs, self.num_imgs, device=self.device)[:, None] / self.num_imgs

        if args.distributed:
            self.feature_mlp = torch.nn.parallel.DistributedDataParallel(
                self.feature_mlp,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )
            self.deform_mlp = torch.nn.parallel.DistributedDataParallel(
                self.deform_mlp,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )
           
    def read_annotation(self, seq_name=None):
        if "RGB" in self.args.data_dir:
            annotation_dir = "dataset/tapvid_rgb_stacking/annotations"
            annotation_file = "{}/{}.npy".format(annotation_dir, seq_name)
            dataset_name = 'rgb_stacking'
            inputs = np.load(annotation_file, allow_pickle=True).item()

        elif "kinetics" in self.args.data_dir:
            annotation_dir = "dataset/tapvid_kinetics_256_100/annotations"
            # annotation_file = '{}/{}.pkl'.format(annotation_dir, seq_name.split('_')[0])
            annotation_file = list(Path(annotation_dir).glob(seq_name + "*"))[0]
            dataset_name = 'kinetics'
            inputs = np.load(annotation_file, allow_pickle=True)

        else:
            annotation_dir = "dataset/tapvid_davis_256/annotations"
            annotation_file = '{}/{}.pkl'.format(annotation_dir, seq_name)
            dataset_name = 'davis'
            # inputs = np.load(annotation_file, allow_pickle=True)
        if not os.path.exists(annotation_file):
            print("Annotation file not found")
            self.eval = False
        else:
            self.eval = True
            
            inputs = np.load(annotation_file, allow_pickle=True)
            # Load tapvid data
            

            self.query_points = inputs[dataset_name]['query_points']
            self.target_points = inputs[dataset_name]['target_points']
            self.gt_occluded = inputs[dataset_name]['occluded']

            one_hot_eye = np.eye(self.target_points.shape[2])
            query_frame = self.query_points[..., 0]
            query_frame = np.round(query_frame).astype(np.int32)
            self.evaluation_points = one_hot_eye[query_frame] == 0

            self.query_points *= (1, self.h / 256, self.w / 256)
            self.eval_ids1 = self.query_points[0, :, 0].astype(int)
            self.eval_px1s = torch.from_numpy(self.query_points[:, :, [2, 1]]).transpose(0, 1).float().to(self.device)


    def read_data(self, images=None):
        self.seq_dir = self.args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')
        print("read sequence: ", self.seq_name)

        self.eval_w = 256.0
        self.eval_h = 256.0

        img_files = sorted(glob.glob(os.path.join(self.img_dir, '*')))
        self.num_imgs = min(self.args.num_imgs, len(img_files))
        if images == None:
            self.img_files = img_files[:self.num_imgs]
            images = np.array([imageio.imread(img_file) / 255. for img_file in self.img_files])
            self.images = torch.from_numpy(images).float() # [n_imgs, h, w, 3]
        else:
            self.images = images
            assert self.images.shape[0] == self.num_imgs
            
        self.h, self.w = self.images.shape[1:3]

        mask_files = [img_file.replace('color', 'mask').replace('.jpg', '.png') for img_file in self.img_files]
        if os.path.exists(mask_files[0]):
            masks = np.array([imageio.imread(mask_file)[..., :3].sum(axis=-1) / 255.
                              if imageio.imread(mask_file).ndim == 3 else
                              imageio.imread(mask_file) / 255.
                              for mask_file in mask_files])
            self.masks = torch.from_numpy(masks).to(self.device) > 0.  # [n_imgs, h, w]
            self.with_mask = True
        else:
            self.masks = torch.ones(self.images.shape[:-1]).to(self.device) > 0.
            self.with_mask = False
        self.grid = util.gen_grid(self.h, self.w, device=self.device, normalize=False, homogeneous=True).float()
        
        if self.args.eval:
            self.read_annotation(self.seq_name)
        
        if self.args.depth_res != 1.0:
            print(f"Resize depthmaps to {self.args.depth_res}x")
        self.depthmaps = torch.zeros((self.num_imgs, int(self.h*self.args.depth_res), int(self.w*self.args.depth_res)), device=self.device)
        # self.depthdir = os.path.join(self.seq_dir, 'depth', 'mix', 'gray')
        if self.args.perspective:
            print("reading depth: ", self.args.depth)
            self.depthdir = os.path.join(self.seq_dir, self.args.depth, 'depth')
        else:
            self.depthdir = os.path.join(self.seq_dir, 'zoe_depth', 'depth')
        depth_files = sorted(glob.glob(os.path.join(self.depthdir, '*.npz')))
        depth_files = depth_files[:self.num_imgs]
        assert len(depth_files) == self.num_imgs
        for i, depth_file in enumerate(depth_files):
            # depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
            # depth = depth.astype(np.float32) / 65535. * 2
            # depth = 2.0 - depth
            depth = np.load(depth_file)['depth']
            depth = depth.astype(np.float32)
            depth = cv2.resize(depth, (int(self.w*self.args.depth_res), int(self.h*self.args.depth_res)), interpolation=cv2.INTER_NEAREST)
            self.depthmaps[i] = torch.from_numpy(depth).to(self.device)
        self.fov = torch.tensor(40).to(self.device)# deg
        self.mean_depth = self.depthmaps.mean()
        # self.depthmaps /= self.mean_depth
        # self.depthmaps *= 1.5
        self.f = self.w / (2 * torch.tan((self.fov )/ 2 / 180 * torch.pi))
        all_pts = []
        for i in tqdm.trange(self.depthmaps.shape[0]):
            depthi = self.get_init_depth_maps([i])
            all_pts.append(self.unproject(self.grid[..., 0:2].reshape(-1, 2), depthi.reshape(-1, 1)).reshape(-1, 3))
        all_pts = torch.cat(all_pts, dim=0)
        self.bound = torch.zeros(2, 3)

        x_mid = torch.median(all_pts[:, 0])
        y_mid = torch.median(all_pts[:, 1])
        z_mid = torch.median(all_pts[:, 2])

        x_range = min(x_mid - all_pts[:, 0].min(), all_pts[:, 0].max() - x_mid) *0.8
        y_range = min(y_mid - all_pts[:, 1].min(), all_pts[:, 1].max() - y_mid) *0.8
        z_range = min(z_mid - all_pts[:, 2].min(), all_pts[:, 2].max() - z_mid) *0.8

        assert x_range > 0 and y_range > 0 and z_range > 0

        self.bound[0, 0] = x_mid - x_range
        self.bound[1, 0] = x_mid + x_range
        self.bound[0, 1] = y_mid - y_range
        self.bound[1, 1] = y_mid + y_range
        self.bound[0, 2] = z_mid - z_range
        self.bound[1, 2] = z_mid + z_range

        self.bound = self.bound.to(self.device)
        print("bound: ", self.bound)
        
        fig = plt.figure()
        plt.hist(all_pts[:, 2].cpu().numpy(), bins=100)
        plt.savefig(os.path.join(self.out_dir, "hist_z.png"))
        plt.close(fig)
        fig = plt.figure()
        plt.hist(all_pts[:, 0].cpu().numpy(), bins=100)
        plt.savefig(os.path.join(self.out_dir, "hist_x.png"))
        plt.close(fig)
        fig = plt.figure()
        plt.hist(all_pts[:, 1].cpu().numpy(), bins=100)
        plt.savefig(os.path.join(self.out_dir, "hist_y.png"))
        plt.close(fig)

        all_pts = (all_pts - (self.bound[1] + self.bound[0]) / 2) / \
            ((self.bound[1] - self.bound[0])/2)
        fig = plt.figure()
        plt.hist(all_pts[:, 2].cpu().numpy(), bins=100)
        plt.savefig(os.path.join(self.out_dir, "hist_z_norm.png"))
        plt.close(fig)
        fig = plt.figure()
        plt.hist(all_pts[:, 0].cpu().numpy(), bins=100)
        plt.savefig(os.path.join(self.out_dir, "hist_x_norm.png"))
        plt.close(fig)
        fig = plt.figure()
        plt.hist(all_pts[:, 1].cpu().numpy(), bins=100)
        plt.savefig(os.path.join(self.out_dir, "hist_y_norm.png"))
        plt.close(fig)


        
    
    def get_init_depth_maps(self, ids):
        grid = self.grid[..., :2].clone()
        normed_grid = util.normalize_coords(grid, self.h, self.w)
        init_maps = self.depthmaps[ids][:, None]
        sampled_maps = F.grid_sample(init_maps, normed_grid[None], align_corners=True, mode='nearest')
        sampled_maps = sampled_maps.squeeze(1)
        return sampled_maps

    # @staticmethod
    # @torch.jit.script
    # def project_compute(x, w, h, f):
    #     depth = x[..., -1:]
    #     x = x[..., :2] / depth
    #     x = x * f
    #     x[..., 0] += w / 2.0
    #     x[..., 1] += h / 2.0
    #     return x, depth

    def project(self, x, return_depth=False):
        '''
        orthographic projection
        :param x: [..., 3]
        :param return_depth: if returning depth
        :return: pixel_coords in image space [..., 2], depth [..., 1]
        '''
        if self.args.perspective:
            # f = self.w / (2 * np.tan(self.fov / 2 / 180 * np.pi))

            depth = x[..., -1:]
            x = x[..., :2] / depth
            x = x * self.f
            x[..., 0] += self.w / 2.0
            x[..., 1] += self.h / 2.0
            if return_depth:
                return x, depth
            else:
                return x

        else:
            pixel_coords, depth = torch.split(x, dim=-1, split_size_or_sections=[2, 1])
            pixel_coords = util.denormalize_coords(pixel_coords, self.h, self.w)
            if return_depth:
                return pixel_coords, depth
            else:
                return pixel_coords
            
    # @staticmethod
    # @torch.jit.script
    # def unproject_compute(pixels, depths, w, h, f):
    #     pixels[..., 0] -= w / 2.0
    #     pixels[..., 1] -= h / 2.0
    #     pixels = pixels / f
    #     pixels = pixels * depths
    #     xyz = torch.cat([pixels, depths], dim=-1)
    #     return xyz

    def unproject(self, pixels, depths):
        '''
        orthographic unprojection
        :param pixels: [..., 2] pixel coordinates (unnormalized), in -w, w, -h, h / 2
        :param depths: [..., 1]
        :return: 3d locations in normalized space [..., 3]
        '''
        assert pixels.shape[-1] in [2, 3]
        assert pixels.ndim == depths.ndim
        px = pixels.clone()
        if self.args.perspective:
            
            px[..., 0] -= self.w / 2.0
            px[..., 1] -= self.h / 2.0
            px = px / self.f
            px = px * depths
            return torch.cat([px, depths], dim=-1)

        else:
            pixels = util.normalize_coords(pixels[..., :2].clone(), self.h, self.w)
            return torch.cat([pixels, depths], dim=-1)
       
    def get_in_range_mask(self, x, max_padding=0):
        mask = (x[..., 0] >= -max_padding) * \
               (x[..., 0] <= self.w - 1 + max_padding) * \
               (x[..., 1] >= -max_padding) * \
               (x[..., 1] <= self.h - 1 + max_padding)
        return mask

    def sample_3d_pts_for_pixels(self, pixels, return_depth=False, fids=None, max_batch=16, original=False):
        '''
        stratified sampling
        sample points on ray for each pixel
        :param pixels: [n_imgs, n_pts, 2]
        :param return_depth: True or False
        :param det: if deterministic
        :param near_depth: nearest depth
        :param far_depth: farthest depth
        :return: sampled 3d locations [n_imgs, n_pts, n_samples, 3]
        '''
        assert fids is not None

        if original:
            sample_frames = self.depthmaps[fids][..., None].permute(0, 3, 1, 2).clone()
        else:
            sample_frames = self.depthmem.depthmaps[fids][..., None].permute(0, 3, 1, 2).clone()

        sample_grid = util.normalize_coords(pixels, self.h, self.w)
        if sample_grid.ndim == 3:
            sample_grid = sample_grid[:, :, None]
        
        if self.args.blur_sample:
            kernel_sz = sample_frames.shape[2]//64
            if kernel_sz % 2 == 0:
                kernel_sz += 1
            mean_depth = sample_frames.mean()
            sample_frames /= 2*mean_depth
            if sample_frames.shape[0] > max_batch:
                frame_batches = torch.split(sample_frames, max_batch, dim=0)
                sample_frames = torch.cat([kornia.filters.bilateral_blur(frame_batch, (kernel_sz, kernel_sz), 0.1, (2, 2)) for frame_batch in frame_batches], dim=0)
            else:
                sample_frames = kornia.filters.bilateral_blur(sample_frames, (kernel_sz, kernel_sz), 0.1, (2, 2))
            sample_frames *= 2*mean_depth

        depths = F.grid_sample(sample_frames, sample_grid, align_corners=True, mode='nearest').permute(0, 2, 3, 1)

        '''
        if isinstance(fids, list):
            fids = torch.tensor(fids, device=self.device)
        else:
            fid = torch.from_numpy(fids)[:, None, None, None].expand(-1, pixels.shape[1], 1, 1)
        
        x = pixels[..., 0] / self.w * self.depthmem.depthmaps.shape[2]
        y = pixels[..., 1] / self.h * self.depthmem.depthmaps.shape[1]
        x = torch.round(x).long()
        y = torch.round(y).long()
        x = x[:, :, None, None].expand(-1, -1, 1, 1)
        y = y[:, :, None, None].expand(-1, -1, 1, 1)
        x = torch.clamp(x, 0, self.depthmem.depthmaps.shape[2]-1)
        y = torch.clamp(y, 0, self.depthmem.depthmaps.shape[1]-1)
        depths = self.depthmem.depthmaps[fid, y, x]
        '''

        # depths = depths[..., None]
        pixels_expand = pixels[:, :, None, :].expand(-1, -1, 1, -1)

        x = self.unproject(pixels_expand, depths)  # [n_imgs, n_pts, n_samples, 3]
        if return_depth:
            return x, depths
        else:
            return x
        

    def get_prediction_one_way(self, x, id, inverse=False):
        '''
        mapping 3d points from local to canonical or from canonical to local (inverse=True)
        :param x: [n_imgs, n_pts, n_samples, 3]
        :param id: [n_imgs, ]
        :param inverse: True or False
        :return: [n_imgs, n_pts, n_samples, 3]
        '''
        t = self.time_steps[id]  # [n_imgs, 1]
        st = time.time()
        # feature = self.feature_mlp(t)  # [n_imgs, n_feat]
        # print("feature_mlp time: ", time.time() - st)

        if inverse:
            if self.args.distributed:
                out = self.deform_mlp.module.inverse(t, None, x)
            else:
                out = self.deform_mlp.inverse(t, None, x)
        else:
            st = time.time()
            out = self.deform_mlp.forward(t, None, x)
            # print("deform_mlp time: ", time.time() - st)

        return out  # [n_imgs, n_pts, n_samples, 3]

    def get_predictions(self, x1, id1, id2, return_canonical=False):
        '''
        mapping 3d points from one frame to another frame
        :param x1: [n_imgs, n_pts, n_samples, 3]
        :param id1: [n_imgs,]
        :param id2: [n_imgs,]
        :return: [n_imgs, n_pts, n_samples, 3]
        '''
        x1_canonical = self.get_prediction_one_way(x1, id1)
        # x1c_ana = x1_canonical.reshape(-1, 3).detach().cpu().numpy()
        # print(f"max: {np.max(x1c_ana, axis=0).round(3)} min: {np.min(x1c_ana, axis=0).round(3)} mean: {np.mean(x1c_ana, axis=0).round(3)} std: {np.std(x1c_ana, axis=0).round(3)}")
        # x1_c = self.get_prediction_one_way(x1_canonical, id1, inverse=True)
        # print("mean err = ", abs(x1_c - x1).mean().item())
        x2_pred = self.get_prediction_one_way(x1_canonical, id2, inverse=True)
        if return_canonical:
            return x2_pred, x1_canonical
        else:
            return x2_pred  # [n_imgs, n_pts, n_samples, 3]

    def get_canonical_color_and_density(self, x_canonical, apply_contraction=True):
        def contraction(x):
            x_norm = x.norm(dim=-1)
            x_out = torch.zeros_like(x)
            x_out[x_norm <= 1] = x[x_norm <= 1]
            x_out[x_norm > 1] = ((2. - 1. / x_norm[..., None])
                                 * (x / x_norm[..., None]))[x_norm > 1]
            return x_out

        if apply_contraction:
            x_canonical = contraction(x_canonical)
        st = time.time()
        out_canonical = self.color_mlp(x_canonical)
        # print("color_mlp time: ", time.time() - st)
        # [n_imgs, n_pts, n_samples, 3]
        color = torch.sigmoid(out_canonical[..., :3])
        density = F.softplus(out_canonical[..., -1] - 1.)
        return color, density

    def get_blending_weights(self, x_canonical):
        '''
        query the nerf network to color, density and blending weights
        :param x_canonical: input canonical 3D locations
        :return: dict containing colors, weights, alphas and rendered rgbs
        '''
        color, density = self.get_canonical_color_and_density(x_canonical)

        alpha = util.sigma2alpha(density)  # [n_imgs, n_pts, n_samples]

        # mask out the nearest 20% of samples. This trick may be helpful to avoid one local minimum solution
        # where surfaces that are not nearest to the camera are initialized at nearest depth planes
        if self.args.mask_near and self.step < 4000:
            mask = torch.ones_like(alpha)
            mask[..., :int(self.args.num_samples_ray * 0.2)] = 0
            alpha *= mask
        # [n_imgs, n_pts, n_samples-1]
        T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[..., :-1]
        # [n_imgs, n_pts, n_samples]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T), dim=-1)

        weights = alpha * T  # [n_imgs, n_pts, n_samples]

        rendered_rgbs = torch.sum(
            weights.unsqueeze(-1) * color, dim=-2)  # [n_imgs, n_pts, 3]

        out = {'colors': color,
               'weights': weights,
               'alphas': alpha,
               'rendered_rgbs': rendered_rgbs,
               }
        return out

    def get_pred_rgbs_for_pixels(self, ids, pixels, return_weights=False):
        xs_samples, pxs_depths_samples = self.sample_3d_pts_for_pixels(
            pixels, return_depth=True)
        xs_canonical_samples = self.get_prediction_one_way(xs_samples, ids)
        out = self.get_blending_weights(xs_canonical_samples)
        blending_weights = out['weights']
        rendered_rgbs = out['rendered_rgbs']
        if return_weights:
            # [n_imgs, n_pts, 3], [n_imgs, n_pts, n_samples]
            return rendered_rgbs, blending_weights
        else:
            return rendered_rgbs

    def get_pred_depths_for_pixels(self, ids, pixels):
        '''
        :param ids: list [n_imgs,]
        :param pixels: [n_imgs, n_pts, 2]
        :return: pred_depths: [n_imgs, n_pts, 1]
        '''
        xs_samples, pxs_depths_samples = self.sample_3d_pts_for_pixels(pixels, return_depth=True, fids=ids)
        # xs_canonical_samples = self.get_prediction_one_way(xs_samples, ids)
        # out = self.get_blending_weights(xs_canonical_samples)
        # pred_depths = torch.sum(out['weights'].unsqueeze(-1) * pxs_depths_samples, dim=-2)
        # x2_pred = self.get_prediction_one_way(x1_canonical, id2, inverse=True)
        pred_depths = xs_samples[..., -1]
        return pred_depths  # [n_imgs, n_pts, 1]

    def get_pred_colors_and_depths_for_pixels(self, ids, pixels):
        '''
        :param ids: list [n_imgs,]
        :param pixels: [n_imgs, n_pts, 2]
        :return: pred_depths: [n_imgs, n_pts, 1]
        '''
        xs_samples, pxs_depths_samples = self.sample_3d_pts_for_pixels(
            pixels, return_depth=True)
        xs_canonical_samples = self.get_prediction_one_way(xs_samples, ids)
        out = self.get_blending_weights(xs_canonical_samples)
        pred_depths = torch.sum(
            out['weights'].unsqueeze(-1) * pxs_depths_samples, dim=-2)
        rendered_rgbs = out['rendered_rgbs']
        return rendered_rgbs, pred_depths  # [n_imgs, n_pts, 1]

    def compute_depth_consistency_loss(self, proj_depths, pred_depths, visibilities, normalize=True):
        '''
        :param proj_depths: [n_imgs, n_pts, 1]
        :param pred_depths: [n_imgs, n_pts, 1]
        :param visibilities: [n_imgs, n_pts, 1]
        :return: depth loss
        '''
        if normalize:
            mse_error = torch.mean((proj_depths - pred_depths) **
                                   2 * visibilities) / (torch.mean(visibilities) + 1e-6)
        else:
            mse_error = torch.mean(
                (proj_depths - pred_depths) ** 2 * visibilities)
        return mse_error

    def get_correspondences_for_pixels(self, ids1, px1s, ids2,
                                       return_depth=False,
                                       use_max_loc=False):
        '''
        get correspondences for pixels in one frame to another frame
        :param ids1: [num_imgs]
        :param px1s: [num_imgs, num_pts, 2]
        :param ids2: [num_imgs]
        :param return_depth: if returning the depth of the mapped point in the target frame
        :param use_max_loc: if using only the sample with the maximum blending weight to
                            compute the corresponding location rather than doing over composition.
                            set to True leads to better results on occlusion boundaries,
                            by default it is set to True for inference.

        :return: px2s_pred: [num_imgs, num_pts, 2], and optionally depth: [num_imgs, num_pts, 1]
        '''
        # [n_pair, n_pts, n_samples, 3]
        x1s_samples = self.sample_3d_pts_for_pixels(px1s, fids=ids1)
        x2s_proj_samples, xs_canonical_samples = self.get_predictions(x1s_samples, ids1, ids2, return_canonical=True)
        # out = self.get_blending_weights(xs_canonical_samples)  # [n_imgs, n_pts, n_samples]
        # if use_max_loc:
        #     blending_weights = out['weights']
        #     indices = torch.max(blending_weights, dim=-1, keepdim=True)[1]
        #     x2s_pred = torch.gather(x2s_proj_samples, 2, indices[..., None].repeat(1, 1, 1, 3)).squeeze(-2)
        #     return self.project(x2s_pred, return_depth=return_depth)
        # else:
        #     x2s_pred = torch.sum(out['weights'].unsqueeze(-1) * x2s_proj_samples, dim=-2)
        #     return self.project(x2s_pred, return_depth=return_depth)
        x2s_pred = x2s_proj_samples.squeeze(-2)
        return self.project(x2s_pred, return_depth=return_depth)

    def get_correspondences_and_occlusion_masks_for_pixels(self, ids1, px1s, ids2,
                                                           return_depth=False,
                                                           use_max_loc=False,
                                                           depth_err = 0.1):
        px2s_pred, depth_proj = self.get_correspondences_for_pixels(ids1, px1s, ids2,
                                                                    return_depth=True,
                                                                    use_max_loc=use_max_loc)
        
        px2s_pred_samples, px2s_pred_depths_samples = self.sample_3d_pts_for_pixels(px2s_pred, return_depth=True, fids=ids2)
        # xs_canonical_samples = self.get_prediction_one_way(px2s_pred_samples, ids2)
        # out = self.get_blending_weights(xs_canonical_samples)
        # weights = out['weights']
        # eps = 1.1 * (self.args.max_depth - self.args.min_depth) / self.args.num_samples_ray
        # mask_zero = px2s_pred_depths_samples.squeeze(-1) >= (depth_proj.expand(-1, -1, self.args.num_samples_ray)) - eps)
        # weights[mask_zero] = 0.
        # occlusion_score = weights.sum(dim=-1, keepdim=True)

        # zero = no occulusion
        occlusion_score = px2s_pred_depths_samples.squeeze(-1) <= (depth_proj - depth_err)
        occlusion_score = occlusion_score.float()
        if return_depth:
            return px2s_pred, occlusion_score, depth_proj
        else:
            return px2s_pred, occlusion_score

    def compute_scene_flow_smoothness_loss(self, ids, xs):
        mask_valid = (ids >= 1) * (ids < self.num_imgs - 1)
        ids = ids[mask_valid]
        if len(ids) == 0:
            return torch.tensor(0.)
        xs = xs[mask_valid]
        ids_prev = ids - 1
        ids_after = ids + 1
        xs_prev_after = self.get_predictions(torch.cat([xs, xs], dim=0),
                                             np.concatenate([ids, ids]),
                                             np.concatenate([ids_prev, ids_after]))
        xs_prev, xs_after = torch.split(
            xs_prev_after, split_size_or_sections=len(xs), dim=0)
        scene_flow_prev = xs - xs_prev
        scene_flow_after = xs_after - xs
        loss = masked_l1_loss(scene_flow_prev, scene_flow_after)
        return loss

    def canonical_sphere_loss(self, xs_canonical_samples, radius=1.):
        ''' encourage mapped locations to be within a (unit) sphere '''
        xs_canonical_norm = (xs_canonical_samples ** 2).sum(dim=-1)
        if (xs_canonical_norm >= 1.).any():
            canonical_unit_sphere_loss = (
                (xs_canonical_norm[xs_canonical_norm >= radius] - 1) ** 2).mean()
        else:
            canonical_unit_sphere_loss = torch.tensor(0.)
        return canonical_unit_sphere_loss

    def gradient_loss(self, pred, gt, weight=None):
        '''
        coordinate
        :param pred: [n_imgs, n_pts, 2] or [n_pts, 2]
        :param gt:
        :return:
        '''
        pred_grad = pred[..., 1:, :] - pred[..., :-1, :]
        gt_grad = gt[..., 1:, :] - gt[..., :-1, :]
        if weight is not None:
            weight_grad = weight[..., 1:, :] * weight[..., :-1, :]
        else:
            weight_grad = None
        loss = masked_l1_loss(pred_grad, gt_grad, weight_grad)
        return loss

    def compute_all_losses(self,
                           batch,
                           w_rgb=10,
                           w_depth=100,
                           w_canonical = 0,  # original 100
                           w_depth_range=10,
                           w_distortion=1.,
                           w_scene_flow_smooth=20., # default 10, original 20
                           w_canonical_unit_sphere=0.,
                           w_flow_grad=0.01,
                           write_logs=True,
                           return_data=False,
                           log_prefix='loss',
                           ):
        # st = time.time()
        depth_min_th = self.args.min_depth
        depth_max_th = self.args.max_depth
        max_padding = self.args.max_padding

        ids1 = batch['ids1'].numpy()
        ids2 = batch['ids2'].numpy()
        px1s = batch['pts1'].to(self.device)
        px2s = batch['pts2'].to(self.device)
        # print("max interval: ", (ids2 - ids1).max())
        # depth1 = batch['depth1'].to(self.device)
        # depth2 = batch['depth2'].to(self.device)
        # gt_rgb1 = batch['gt_rgb1'].to(self.device)
        # gt_rgb2 = batch['gt_rgb2'].to(self.device)
        weights = batch['weights'].to(self.device)
        # self.data_time = time.time() - st
        num_pts = px1s.shape[1]
        st = time.time()
        # [n_pair, n_pts, n_samples, 3]
        # x1s_samples, px1s_depths_samples = self.sample_3d_pts_for_pixels(px1s, return_depth=True, det=False, fids=ids1)
        # x1s_samples = util.normalize_coords(px1s, self.h, self.w)
        # x2s_samples = util.normalize_coords(px2s, self.h, self.w)

        # px1s: [n_pairs, npts, 2]
        x1s_samples = self.sample_3d_pts_for_pixels(px1s, return_depth=False, fids=ids1)
        x2s_samples, depth2 = self.sample_3d_pts_for_pixels(px2s, return_depth=True, fids=ids2)
        origin_x2s, origin_depth2 = self.sample_3d_pts_for_pixels(px2s, return_depth=True, fids=ids2, original=True)
        # depth1 =depth1.squeeze()
        depth2 =depth2.squeeze()
        origin_depth2 = origin_depth2.squeeze()
        self.sample_time = time.time() - st
        # x1s_samples = torch.cat([x1s_samples, depth1[..., None]], dim=-1)
        #  = self.unproject(px1s, depth1[..., None])
        # = self.unproject(px2s, depth2[..., None])

        # x1s_samples = x1s_samples[:, :, None]
        # px1s_depths_samples = depth1[..., None, None]

        # x2s_samples = torch.cat([x2s_samples, depth2[..., None]], dim=-1)
        # x2s_samples = x2s_samples[:, :, None]
        

        st = time.time()
        x2s_proj_samples, x1s_canonical_samples = self.get_predictions(x1s_samples, ids1, ids2, return_canonical=True)
        # out = self.get_blending_weights(x1s_canonical_samples)
        # blending_weights1 = out['weights']
        # alphas1 = out['alphas']
        # pred_rgb1 = out['rendered_rgbs']
        

        # mask = (x2s_proj_samples[..., -1] >= depth_min_th) * (x2s_proj_samples[..., -1] <= depth_max_th)
        # blending_weights1 = blending_weights1 * mask.float()
        # x2s_pred = torch.sum(blending_weights1.unsqueeze(-1) * x2s_proj_samples, dim=-2)
        x2s_pred = x2s_proj_samples.squeeze(-2)

        # [n_imgs, n_pts, n_samples, 2]
        # px2s_proj_samples, px2s_proj_depth_samples = self.project(x2s_proj_samples, return_depth=True)
        px2s_proj, px2s_proj_depths = self.project(x2s_pred, return_depth=True)


        self.pred_time = time.time() - st


        mask = self.get_in_range_mask(px2s_proj, max_padding)
        # rgb_mask = self.get_in_range_mask(px1s)

        if mask.sum() > 0:
            # loss_rgb = F.mse_loss(pred_rgb1[rgb_mask], gt_rgb1[rgb_mask])
            # loss_rgb_grad = self.gradient_loss(pred_rgb1[rgb_mask], gt_rgb1[rgb_mask])

            optical_flow_loss = masked_l1_loss(px2s_proj[mask], px2s[mask], weights[mask], normalize=False)
            # optical_flow_loss = abs(px2s_proj - px2s).mean()
            optical_flow_grad_loss = self.gradient_loss(px2s_proj[mask], px2s[mask], weights[mask])
        else:
            loss_rgb = loss_rgb_grad = optical_flow_loss = optical_flow_grad_loss = torch.tensor(0.)

        # mapped depth should be within the predefined range
        # depth_range_loss = compute_depth_range_loss(px2s_proj_depth_samples, depth_min_th, depth_max_th)

        # distortion loss to remove floaters
        # t = torch.cat([px1s_depths_samples[..., 0], px1s_depths_samples[..., 0][..., -1:]], dim=-1)
        # distortion_loss = lossfun_distortion(t, blending_weights1)

        # scene flow smoothness
        # only apply to 25% of points to reduce cost
        # scene_flow_smoothness_loss = self.compute_scene_flow_smoothness_loss(ids1, x1s_samples[:, :int(num_pts / 4)])

        # loss for mapped points to stay within canonical sphere
        canonical_unit_sphere_loss = self.canonical_sphere_loss(x1s_canonical_samples)

        depth_pred_loss = torch.mean(abs(px2s_proj_depths.squeeze() - depth2))
        # depth_pred_loss = torch.clamp(depth_pred_loss, min=0, max=0.1)

        # depth_pred_loss = abs(torch.max(px2s_proj_depths.squeeze(), depth2) / (torch.min(px2s_proj_depths.squeeze(), depth2) - 1)).mean()

        # time_disp = torch.tensor(abs(ids1 - ids2)>2).reshape(-1, 1).to(self.device)
        depth_consist_loss = torch.mean(abs(px2s_proj_depths.squeeze() - origin_depth2))

        

        if w_canonical > 0:
            x2s_canonical = self.get_prediction_one_way(x2s_samples, ids2)

            canonical_dist_loss = torch.mean(abs(x2s_canonical.squeeze() - x1s_canonical_samples.squeeze()))
        else:
            canonical_dist_loss = torch.tensor(0.).to(self.device)
        # optical_flow_loss = torch.clamp(optical_flow_loss, min=0, max=30)

        # w_scene_flow_smooth * scene_flow_smoothness_loss + \
        loss = optical_flow_loss + \
               w_flow_grad * optical_flow_grad_loss + \
               w_canonical_unit_sphere * canonical_unit_sphere_loss + \
               w_depth * depth_pred_loss + \
               w_depth * 0.1 * depth_consist_loss
        
        # loss = torch.clamp(loss, min=0, max=10)
        

        if write_logs:
            self.scalars_to_log['{}/Loss'.format(log_prefix)] = loss.item()
            self.scalars_to_log['{}/loss_depth'.format(log_prefix)] = depth_pred_loss.item()
            self.scalars_to_log['{}/loss_canonical'.format(log_prefix)] = canonical_dist_loss.item()
            self.scalars_to_log['{}/loss_depth_consist'.format(log_prefix)] = depth_consist_loss.item()
            self.scalars_to_log['{}/min_depth'.format(log_prefix)] = self.depthmem.depthmaps.min().item()
            self.scalars_to_log['{}/loss_flow'.format(log_prefix)] = optical_flow_loss.item()
            # self.scalars_to_log['{}/loss_rgb'.format(log_prefix)] = loss_rgb.item()
            # self.scalars_to_log['{}/loss_depth_range'.format(log_prefix)] = depth_range_loss.item()
            # self.scalars_to_log['{}/loss_distortion'.format(log_prefix)] = distortion_loss.item()
            # self.scalars_to_log['{}/loss_scene_flow_smoothness'.format(log_prefix)] = scene_flow_smoothness_loss.item()
            self.scalars_to_log['{}/loss_canonical_unit_sphere'.format(log_prefix)] = canonical_unit_sphere_loss.item()
            self.scalars_to_log['{}/loss_flow_gradient'.format(log_prefix)] = optical_flow_grad_loss.item()
            
            

        data = {'ids1': ids1,
                'ids2': ids2,
                'x1s': x1s_samples,
                'x2s_pred': x2s_pred,
                'xs_canonical': x1s_canonical_samples,
                'mask': mask,
                'px2s_proj': px2s_proj,
                'px2s_proj_depths': px2s_proj_depths,
                # 'blending_weights': blending_weights1,
                # 'alphas': alphas1,
                # 't': t
                }
        if return_data:
            return loss, data
        else:
            return loss

    def weight_scheduler(self, step, start_step, w, min_weight, max_weight):
        if step <= start_step:
            weight = 0.0
        else:
            weight = w * (step - start_step)
        weight = np.clip(weight, a_min=min_weight, a_max=max_weight)
        return weight


    def train_one_step(self, step, batch):
        # self.feature_mlp.train()
        self.deform_mlp.train()
        self.step = step
        start = time.time()
        self.scalars_to_log = {}

        self.optimizer.zero_grad()
        w_rgb = self.weight_scheduler(step, 0, 1./5000, 0, 10)
        w_flow_grad = self.weight_scheduler(step, 0, 1./500000, 0, 0.1)
        w_distortion = self.weight_scheduler(step, 40000, 1./2000, 0, 10)
        w_scene_flow_smooth = 20.

        # scaler = GradScaler()
        
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        loss, flow_data = self.compute_all_losses(batch,
                                                w_rgb=w_rgb,
                                                # w_depth=10,
                                                # w_scene_flow_smooth=w_scene_flow_smooth,
                                                w_distortion=w_distortion,
                                                w_flow_grad=w_flow_grad,
                                                return_data=True)
        is_break = False
        for p in self.deform_mlp.parameters():
            if torch.isnan(p.data).any() or (p.grad is not None and torch.isnan(p.grad).any()):
                is_break = True
                pdb.set_trace()
        if torch.isnan(loss):
            # pass
            pdb.set_trace()
        self.forward_time = time.time() - start
        back_st = time.time()

        # scaler.scale(loss).backward()
        loss.backward()
        self.backtime = time.time() - back_st

        if self.args.grad_clip > 0 and self.depthmem.depthmaps.grad.norm() > self.args.grad_clip:
            print(f"Warning! Clip gradient from {self.depthmem.depthmaps.grad.norm()} to {self.args.grad_clip} \n")
            torch.nn.utils.clip_grad_norm_(
                self.depthmem.parameters(), self.args.grad_clip)
        if self.depthmem.depthmaps.grad.norm() > 1:
            print("depth grad norm: ", self.depthmem.depthmaps.grad.norm().item())
        


        
        if is_break:
            pass
            # if self.args.deformer == 'triplane':
            #     print("all grad = ", all_triplane_grad)
            #     print("all mlp grad = ", all_mlp_grad)
            #     print("all grad sum = ", sum(all_triplane_grad) + sum(all_mlp_grad))
            # pdb.set_trace()

        # scaler.step(self.optimizer)
        
        self.optimizer.step()
        self.scheduler.step()
        # scaler.update()
        # print(prof.key_averages().table(sort_by="cuda_time_total"))
        

        # print("time= ", time.time() - start)
        # print("backtime= ", self.backtime)

        # metrics = self.metrics(flow_data, batch)
        # for k, v in metrics.items():
        #     self.scalars_to_log['metric/{}'.format(k)] = v

        self.scalars_to_log['lr'] = self.optimizer.param_groups[0]['lr']

        self.scalars_to_log['time'] = time.time() - start
        
        self.ids1 = flow_data['ids1']
        self.ids2 = flow_data['ids2']

    def sample_pts_within_mask(self, mask, num_pts, return_normed=False, seed=None,
                               use_mask=False, reverse_mask=False, regular=False, interval=10):
        rng = np.random.RandomState(seed) if seed is not None else np.random
        if use_mask:
            if reverse_mask:
                mask = ~mask
            kernel = torch.ones(7, 7, device=self.device)
            mask = morph.erosion(mask.float()[None, None], kernel).bool().squeeze()  # Erosion
        else:
            mask = torch.ones_like(self.grid[..., 0], dtype=torch.bool)

        if regular:
            coords = self.grid[::interval, ::interval, :2][mask[::interval, ::interval]].clone()
        else:
            coords_valid = self.grid[mask][..., :2].clone()
            rand_inds = rng.choice(len(coords_valid), num_pts, replace=(num_pts > len(coords_valid)))
            coords = coords_valid[rand_inds]

        coords_normed = util.normalize_coords(coords, self.h, self.w)
        if return_normed:
            return coords, coords_normed
        else:
            return coords  # [num_pts, 2]

    def generate_uniform_3d_samples(self, num_pts, radius=2):
        num_pts = int(num_pts)
        pts = 2. * torch.rand(num_pts * 2, 3, device=self.device) - 1.  # [-1, 1]^3
        pts_norm = torch.norm(pts, dim=-1)
        pts = pts[pts_norm < 1.]
        rand_ids = np.random.choice(len(pts), num_pts, replace=len(pts) < num_pts)
        pts = pts[rand_ids]
        pts *= radius
        return pts

    def get_canonical_uvw_from_frames(self, num_pts_per_frame=10000):
        uvws = []
        for i in range(self.num_imgs):
            pixels_normed = 2 * torch.rand(num_pts_per_frame, 2, device=self.device) - 1.
            pixels = util.denormalize_coords(pixels_normed, self.h, self.w)[None]
            pixel_samples = self.sample_3d_pts_for_pixels(pixels, det=False, fids=[i])
            with torch.no_grad():
                uvw = self.get_prediction_one_way(pixel_samples, [i])[0]
            uvws.append(uvw.reshape(-1, 3))
        uvws = torch.cat(uvws, dim=0)
        return uvws

    def save_canonical_rgba_volume(self, num_pts, sample_points_from_frames=False):
        save_dir = os.path.join(self.out_dir, 'pcd')
        os.makedirs(save_dir, exist_ok=True)
        chunk_size = self.args.chunk_size
        if sample_points_from_frames:
            num_pts_per_frame = num_pts // (self.args.num_imgs * self.args.num_samples_ray)
            uvw = self.get_canonical_uvw_from_frames(num_pts_per_frame)
            suffix = '_frames'
            apply_contraction = True
        else:
            uvw = self.generate_uniform_3d_samples(num_pts, radius=1)
            suffix = ''
            apply_contraction = False
        uvw_np = uvw.cpu().numpy()
        rgbas = []
        for chunk in torch.split(uvw, chunk_size, dim=0):
            with torch.no_grad():
                color, density = self.get_canonical_color_and_density(chunk, apply_contraction=apply_contraction)
            alpha = util.sigma2alpha(density)
            rgba = torch.cat([color, alpha[..., None]], dim=-1)
            rgbas.append(rgba.cpu().numpy())
        rgbas = np.concatenate(rgbas, axis=0)
        out = np.ascontiguousarray(np.concatenate([uvw_np, rgbas], axis=-1))
        np.save(os.path.join(save_dir, '{:06d}{}.npy'.format(self.step, suffix)), out)

    def vis_pairwise_correspondences(self, ids=None, num_pts=200, use_mask=False, use_max_loc=True,
                                     reverse_mask=False, regular=True, interval=20):
        if ids is not None:
            id1, id2 = ids
        else:
            id1 = self.ids1[0]
            id2 = self.ids2[0]

        px1s = self.sample_pts_within_mask(self.masks[id1], num_pts, seed=1234,
                                           use_mask=use_mask, reverse_mask=reverse_mask,
                                           regular=regular, interval=interval)
        num_pts = len(px1s)

        with torch.no_grad():
            px2s_pred, occlusion_score = \
                self.get_correspondences_and_occlusion_masks_for_pixels([id1], px1s[None], [id2],
                                                                        use_max_loc=use_max_loc)
            px2s_pred = px2s_pred[0]
            mask = occlusion_score > self.args.occlusion_th

        kp1 = px1s.detach().cpu().numpy()
        kp2 = px2s_pred.detach().cpu().numpy()
        img1 = self.images[id1].cpu().numpy()
        img2 = self.images[id2].cpu().numpy()
        mask = mask[0].squeeze(-1).cpu().numpy()
        out = util.drawMatches(img1, img2, kp1, kp2, num_vis=num_pts, mask=mask)
        out = cv2.putText(out, str(id2 - id1), org=(30, 50), fontScale=1, color=(255, 255, 255),
                          fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=2)
        out = util.uint82float(out)
        return out

    def plot_correspondences_for_pixels(self, query_kpt, query_id, num_pts=100,
                                        vis_occlusion=False,
                                        occlusion_th=0.95,
                                        use_max_loc=False,
                                        radius=2,
                                        return_kpts=False):
        frames = []
        kpts = []
        with torch.no_grad():
            img_query = self.images[query_id].cpu().numpy()
            for id in range(0, self.num_imgs):
                if vis_occlusion:
                    if id == query_id:
                        kp_i = query_kpt
                        occlusion_score = torch.zeros_like(query_kpt[..., :1])
                    else:
                        kp_i, occlusion_score = \
                            self.get_correspondences_and_occlusion_masks_for_pixels([query_id], query_kpt[None], [id],
                                                                                    use_max_loc=use_max_loc)
                        kp_i = kp_i[0]
                        occlusion_score = occlusion_score[0]

                    mask = occlusion_score > occlusion_th

                    out_pts = kp_i.cpu().numpy()
                    out_vis = 1 - mask.cpu().numpy().astype(float)
                    all_out = np.concatenate([out_pts, out_vis], axis=-1)
                    # np.save(f"{self.args.out_dir}/kp_{id}.npy", kp_i.cpu().numpy())
                    # np.save(f"{self.args.out_dir}/kp_{id:03d}.npy", all_out)
                    
                    kp_i = torch.cat([kp_i, mask.float()], dim=-1)
                    mask = mask.squeeze(-1).cpu().numpy()
                else:
                    if id == query_id:
                        kp_i = query_kpt
                    else:
                        kp_i = self.get_correspondences_for_pixels([query_id], query_kpt[None], [id],
                                                                   use_max_loc=use_max_loc)[0]
                    mask = None
                img_i = self.images[id].cpu().numpy()
                out = util.drawMatches(img_query, img_i, query_kpt.cpu().numpy(), kp_i.cpu().numpy(),
                                       num_vis=num_pts, mask=mask, radius=radius)
                frames.append(out)
                kpts.append(kp_i)
        kpts = torch.stack(kpts, dim=0)
        if return_kpts:
            return frames, kpts
        return frames

    def eval_video_correspondences(self, query_id, pts=None, num_pts=100, seed=1234, use_mask=False,
                                   mask=None, reverse_mask=False, vis_occlusion=False, occlusion_th=0.99,
                                   use_max_loc=False, regular=True,
                                   interval=20, radius=2, return_kpts=False):
        with torch.no_grad():
            if mask is not None:
                mask = torch.from_numpy(mask).bool().to(self.device)
            else:
                mask = self.masks[query_id]

            if pts is None:
                x_0 = self.sample_pts_within_mask(mask, num_pts, seed=seed, use_mask=use_mask,
                                                  reverse_mask=reverse_mask, regular=regular, interval=interval)
                num_pts = 1e7 if regular else num_pts
            else:
                x_0 = torch.from_numpy(pts).float().to(self.device)
            return self.plot_correspondences_for_pixels(x_0, query_id, num_pts=num_pts,
                                                        vis_occlusion=vis_occlusion,
                                                        occlusion_th=occlusion_th,
                                                        use_max_loc=use_max_loc,
                                                        radius=radius, return_kpts=return_kpts)

    def get_pred_depth_maps(self, ids, chunk_size=40000):
        grid = self.grid[..., :2].reshape(-1, 2).clone()
        pred_depths = []
        for id in ids:
            depth_map = []
            for coords in torch.split(grid, split_size_or_sections=chunk_size, dim=0):
                depths_chunk = self.get_pred_depths_for_pixels([id], coords[None])[0]
                depths_chunk = torch.nan_to_num(depths_chunk)
                depth_map.append(depths_chunk)
            depth_map = torch.cat(depth_map, dim=0).reshape(self.h, self.w)
            pred_depths.append(depth_map)
        pred_depths = torch.stack(pred_depths, dim=0)
        return pred_depths  # [n, h, w]

    def get_pred_imgs(self, ids, chunk_size=40000, return_weights_stats=False):
        grid = self.grid[..., :2].reshape(-1, 2).clone()
        pred_rgbs = []
        weights_stats = []
        for id in ids:
            rgb = []
            weights_stat = []
            for coords in torch.split(grid, split_size_or_sections=chunk_size, dim=0):
                if return_weights_stats:
                    rgbs_chunk, weights_stats_chunk = self.get_pred_rgbs_for_pixels([id], coords[None],
                                                                                    return_weights=return_weights_stats)
                    weights_sum = weights_stats_chunk[0].sum(dim=-1)
                    weights_max = weights_stats_chunk[0].max(dim=-1)[0]
                    weights_stats_chunk = torch.stack([weights_sum, weights_max], dim=-1)
                    weights_stat.append(weights_stats_chunk)
                else:
                    rgbs_chunk = self.get_pred_rgbs_for_pixels([id], coords[None])
                rgb.append(rgbs_chunk[0])
            img = torch.cat(rgb, dim=0).reshape(self.h, self.w, 3)
            pred_rgbs.append(img)
            if return_weights_stats:
                weights_stats.append(torch.cat(weights_stat, dim=0).reshape(self.h, self.w, 2))

        pred_rgbs = torch.stack(pred_rgbs, dim=0)
        if return_weights_stats:
            weights_stats = torch.stack(weights_stats, dim=0)  # [n, h, w, 2]
            return pred_rgbs, weights_stats
        return pred_rgbs  # [n, h, w, 3]

    def get_pred_color_and_depth_maps(self, ids, chunk_size=40000):
        grid = self.grid[..., :2].reshape(-1, 2).clone()
        pred_rgbs = []
        pred_depths = []
        for id in ids:
            rgb = []
            depth_map = []
            for coords in torch.split(grid, split_size_or_sections=chunk_size, dim=0):
                rgbs_chunk = self.get_pred_rgbs_for_pixels([id], coords[None])
                rgb.append(rgbs_chunk[0])
                depths_chunk = self.get_pred_depths_for_pixels([id], coords[None])
                depths_chunk = torch.nan_to_num(depths_chunk)
                depth_map.append(depths_chunk[0])

            img = torch.cat(rgb, dim=0).reshape(self.h, self.w, 3)
            pred_rgbs.append(img)
            depth_map = torch.cat(depth_map, dim=0).reshape(self.h, self.w)
            pred_depths.append(depth_map)

        pred_rgbs = torch.stack(pred_rgbs, dim=0)
        pred_depths = torch.stack(pred_depths, dim=0)
        return pred_rgbs, pred_depths  # [n, h, w, 3/1]

    def get_pred_flows(self, ids1, ids2, chunk_size=40000, use_max_loc=False, return_original=False):
        grid = self.grid[..., :2].reshape(-1, 2).clone()
        flows = []
        for id1, id2 in zip(ids1, ids2):
            flow_map = []
            for coords in torch.split(grid, split_size_or_sections=chunk_size, dim=0):
                with torch.no_grad():
                    flows_chunk = self.get_correspondences_for_pixels([id1], coords[None], [id2],
                                                                      use_max_loc=use_max_loc)[0]
                    flow_map.append(flows_chunk)
            flow_map = torch.cat(flow_map, dim=0).reshape(self.h, self.w, 2)
            flow_map = (flow_map - self.grid[..., :2]).cpu().numpy()
            flows.append(flow_map)
        flows = np.stack(flows, axis=0)
        flow_imgs = util.flow_to_image(flows)
        if return_original:
            return flow_imgs, flows
        else:
            return flow_imgs  # [n, h, w, 3], numpy arra

    def get_pred_flows_and_occlusions(self, ids1, ids2, chunk_size=40000, return_original=False):
        grid = self.grid[..., :2].reshape(-1, 2).clone()
        flows = []
        for id1, id2 in zip(ids1, ids2):
            flow_map = []
            for coords in torch.split(grid, split_size_or_sections=chunk_size, dim=0):
                with torch.no_grad():
                    flows_chunk, occlusion_chunk = self.get_correspondences_and_occlusion_masks_for_pixels([id1],
                                                                                                           coords[None],
                                                                                                           [id2])
                    flows_chunk = torch.cat([flows_chunk[0], occlusion_chunk[0].float()], dim=-1)
                    flow_map.append(flows_chunk)
            flow_map = torch.cat(flow_map, dim=0).reshape(self.h, self.w, 3)
            flow_map[..., :2] -= self.grid[..., :2]
            flow_map = flow_map.cpu().numpy()
            flows.append(flow_map)
        flows = np.stack(flows, axis=0)
        flow_imgs = util.flow_to_image(flows[..., :2])
        if return_original:
            return flow_imgs, flows
        else:
            return flow_imgs  # [n, h, w, 3], numpy arra

    def render_color_and_depth_videos(self, start_id, end_id, chunk_size=40000, colorize=True):
        depths_np = []
        colors_np = []
        for id in range(start_id, end_id):
            with torch.no_grad():
                color, depth = self.get_pred_color_and_depth_maps([id], chunk_size=chunk_size)
                colors_np.append(color.cpu().numpy())
                depths_np.append(depth.cpu().numpy())

        colors_np = np.concatenate(colors_np, axis=0)
        depths_np = np.concatenate(depths_np, axis=0)
        depths_vis_min, depths_vis_max = depths_np.min(), depths_np.max()
        depths_vis = (depths_np - depths_vis_min) / (depths_vis_max - depths_vis_min)
        if colorize:
            depths_vis_color = []
            for depth_vis in depths_vis:
                depth_vis_color = util.colorize_np(depth_vis, range=(0, 1))
                depths_vis_color.append(depth_vis_color)
            depths_vis_color = np.stack(depths_vis_color, axis=0)
        else:
            depths_vis_color = depths_vis
        colors_np = (255 * colors_np).astype(np.uint8)
        depths_vis_color = (255 * depths_vis_color).astype(np.uint8)
        return colors_np, depths_vis_color

    def log(self, writer, step):
        if self.args.local_rank == 0:
            if step % self.args.i_print == 0:
                logstr = '{}_{} | step: {} |'.format(self.args.expname, self.seq_name, step)
                for k in self.scalars_to_log.keys():
                    logstr += ' {}: {:.6f}'.format(k, self.scalars_to_log[k])
                    if k != 'time':
                        writer.add_scalar(k, self.scalars_to_log[k], step)
                print(logstr)

            if step % self.args.i_img == 0:
                # flow
                flows = self.get_pred_flows(self.ids1[0:1], self.ids2[0:1], chunk_size=self.args.chunk_size)[0]
                writer.add_image('flow', flows, step, dataformats='HWC')

                # correspondences
                out_trained = self.vis_pairwise_correspondences()
                out_fix_10 = self.vis_pairwise_correspondences(ids=(0, min(self.num_imgs // 10, 10)))
                out_fix_half = self.vis_pairwise_correspondences(ids=(0, self.num_imgs // 2))
                out_fix_full = self.vis_pairwise_correspondences(ids=(0, self.num_imgs - 1))

                writer.add_image('correspondence/trained', out_trained, step, dataformats='HWC')
                writer.add_image('correspondence/fix_10', out_fix_10, step, dataformats='HWC')
                writer.add_image('correspondence/fix_half', out_fix_half, step, dataformats='HWC')
                writer.add_image('correspondence/fix_whole', out_fix_full, step, dataformats='HWC')


            if step % self.args.i_weight == 0 and step > 0:
                # save checkpoints
                os.makedirs(self.out_dir, exist_ok=True)
                print('Saving checkpoints at {} to {}...'.format(step, self.out_dir))
                fpath = os.path.join(self.out_dir, 'model_{:06d}.pth'.format(step))
                self.save_model(fpath)

                vis_dir = os.path.join(self.out_dir, 'vis')
                os.makedirs(vis_dir, exist_ok=True)
                print('saving visualizations to {}...'.format(vis_dir))
                if self.with_mask:
                    video_correspondences = self.eval_video_correspondences(0,
                                                                            use_mask=True,
                                                                            vis_occlusion=self.args.vis_occlusion,
                                                                            use_max_loc=self.args.use_max_loc,
                                                                            occlusion_th=self.args.occlusion_th)
                    imageio.mimwrite(os.path.join(vis_dir, '{}_corr_foreground_{:06d}.mp4'.format(self.seq_name, step)),
                                     video_correspondences,
                                     quality=8, fps=10)
                    video_correspondences = self.eval_video_correspondences(0,
                                                                            use_mask=True,
                                                                            reverse_mask=True,
                                                                            vis_occlusion=self.args.vis_occlusion,
                                                                            use_max_loc=self.args.use_max_loc,
                                                                            occlusion_th=self.args.occlusion_th)
                    imageio.mimwrite(os.path.join(vis_dir, '{}_corr_background_{:06d}.mp4'.format(self.seq_name, step)),
                                     video_correspondences,
                                     quality=8, fps=10)
                else:
                    video_correspondences = self.eval_video_correspondences(0,
                                                                            vis_occlusion=self.args.vis_occlusion,
                                                                            use_max_loc=self.args.use_max_loc,
                                                                            occlusion_th=self.args.occlusion_th)
                    imageio.mimwrite(os.path.join(vis_dir, '{}_corr_{:06d}.mp4'.format(self.seq_name, step)),
                                     video_correspondences,
                                     quality=8, fps=10)
                # color_frames, depth_frames = self.render_color_and_depth_videos(0, self.num_imgs,
                #                                                                 chunk_size=self.args.chunk_size)
                # imageio.mimwrite(os.path.join(vis_dir, '{}_depth_{:06d}.mp4'.format(self.seq_name, step)), depth_frames,
                #                  quality=8, fps=10)
                # imageio.mimwrite(os.path.join(vis_dir, '{}_color_{:06d}.mp4'.format(self.seq_name, step)), color_frames,
                #                  quality=8, fps=10)

                ids1 = np.arange(self.num_imgs)
                ids2 = ids1 + 1
                ids2[-1] -= 2
                # pred_optical_flows_vis, pred_optical_flows = self.get_pred_flows(ids1, ids2,
                #                                                                  use_max_loc=self.args.use_max_loc,
                #                                                                  chunk_size=self.args.chunk_size,
                #                                                                  return_original=True
                #                                                                  )
                # imageio.mimwrite(os.path.join(vis_dir, '{}_flow_{:06d}.mp4'.format(self.seq_name, step)),
                #                  pred_optical_flows_vis[:-1],
                #                  quality=8, fps=10)

            if self.args.use_error_map and (step % self.args.i_cache == 0) and (step > 0):
                flow_save_dir = os.path.join(self.out_dir, 'flow')
                os.makedirs(flow_save_dir, exist_ok=True)
                flow_errors = []
                for i, (id1, id2) in enumerate(zip(ids1, ids2)):
                    save_path = os.path.join(flow_save_dir, '{}_{}.npy'.format(os.path.basename(self.img_files[id1]),
                                                                               os.path.basename(self.img_files[id2])))
                    # np.save(save_path, pred_optical_flows[i])
                    # gt_flow = np.load(os.path.join(self.seq_dir, 'raft_exhaustive',
                    #                                '{}_{}.npy'.format(os.path.basename(self.img_files[id1]),
                    #                                                   os.path.basename(self.img_files[id2]))
                    #                                ))
                #     flow_error = np.linalg.norm(gt_flow - pred_optical_flows[i], axis=-1).mean()
                #     flow_errors.append(flow_error)

                # flow_errors = np.array(flow_errors)
                # np.savetxt(os.path.join(self.out_dir, 'flow_error.txt'), flow_errors)

    def save_model(self, filename):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(),
                   'deform_mlp': de_parallel(self.deform_mlp).state_dict(),
                #    'feature_mlp': de_parallel(self.feature_mlp).state_dict(),
                #    'color_mlp': de_parallel(self.color_mlp).state_dict(),
                   'num_imgs': self.num_imgs
                   }
        if self.args.opt_depth:
            to_save['depth_mem'] = de_parallel(self.depthmem).state_dict()

        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.args.distributed:
            to_load = torch.load(filename, map_location='cuda:{}'.format(self.args.local_rank))
        else:
            to_load = torch.load(filename)

        if load_opt:
            self.optimizer.load_state_dict(to_load['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load['scheduler'])

        self.deform_mlp.load_state_dict(to_load['deform_mlp'])
        # self.feature_mlp.load_state_dict(to_load['feature_mlp'])
        # self.color_mlp.load_state_dict(to_load['color_mlp'])
        self.num_imgs = to_load['num_imgs']
        if self.args.opt_depth:
            self.depthmem.load_state_dict(to_load['depth_mem'])

    def load_from_ckpt(self, out_folder,
                       load_opt=True,
                       load_scheduler=True,
                       force_latest_ckpt=False):
        '''
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        '''

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f)
                     for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print('Reloading from {}, starting at step={}'.format(fpath, step))
        else:
            print('No ckpts found, from scratch...')
            step = 0

        return step