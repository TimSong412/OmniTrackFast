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
from loaders.animreader import anime_read
from torchsummary import summary

from torch.cuda.amp import GradScaler
from wis3d import Wis3D
import pdb
import open3d as o3d
from pathlib import Path
# torch.autograd.set_detect_anomaly(True) 

from networks.nvp_nonlin import NVPnonlin


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
            # w, h = depthmaps.shape[-2:]
            # grid = util.gen_grid(h//4, w//4, device=device, normalize=True, homogeneous=False).float()
            # grid = grid[None].expand(depthmaps.shape[0], -1, -1, -1)
            # self.coarse_depthmaps = F.grid_sample(depthmaps[:, None], grid, align_corners=True, mode='nearest').squeeze(1)   
            
            # self.depthmaps = nn.parameter.Parameter(self.coarse_depthmaps)
            # self.delta_depthmaps = nn.parameter.Parameter(torch.zeros_like(depthmaps))
        else:
            self.depthmaps = depthmaps.clone().to(device)
        

class ComboTrainer():
    def __init__(self, args, images=None, device='cuda'):
        self.args = args
        self.device = device
        self.out_dir = args.out_dir

        print("decay steps: ", args.lrate_decay_steps)

        self.backtime = 0

        self.read_data(images)
        self.depthmem = DepthMem(args, self.depthmaps, device=device)
        self.wis3d = Wis3D(self.out_dir+"/viserr", f"vis_err", "xyz")

        self.feature_mlp = None


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
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.num_iters, eta_min=3e-5, verbose=True)

      
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
        else:
            annotation_dir = "dataset/tapvid_davis_256/annotations"
            annotation_file = '{}/{}.pkl'.format(annotation_dir, seq_name)
            dataset_name = 'davis'
            inputs = np.load(annotation_file, allow_pickle=True)
        # Load tapvid data
        if not os.path.exists(annotation_file):
            print("Annotation file not found")
            self.eval = False
        else:
            self.eval = True
            
            # inputs = np.load(annotation_file, allow_pickle=True)

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
        self.read_done = False
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
        
        if self.images.shape[0] == 0:
            self.h, self.w = 480, 854
        else:
            self.h, self.w = self.images.shape[1:3]

        mask_files = [img_file.replace('color', 'mask').replace('.jpg', '.png') for img_file in self.img_files]
        if len(mask_files) > 0 and os.path.exists(mask_files[0]):
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
        self.gradmaps = torch.zeros((*self.depthmaps.shape, 2), device=self.device)
        self.coarse_grid = util.gen_grid(self.depthmaps.shape[-2], self.depthmaps.shape[-1], device=self.device, normalize=False, homogeneous=False)
        self.median_grad_per_frame = torch.zeros(self.num_imgs, device=self.device)
        for fid in range(self.num_imgs):
            self.gradmaps[fid] = self.get_pixel_depth_gradient(self.coarse_grid.reshape(1, -1, 2), torch.tensor([fid]).to(self.device), original=True, scale=False).reshape(self.depthmaps.shape[-2], self.depthmaps.shape[-1], 2)
            self.median_grad_per_frame[fid] = torch.median(torch.norm(self.gradmaps[fid], dim=-1))

        self.fov = torch.tensor(40).to(self.device)# deg
        self.mean_depth = self.depthmaps.mean()
        # self.depthmaps /= self.mean_depth
        # self.depthmaps *= 1.5
        self.f = self.w / (2 * torch.tan((self.fov )/ 2 / 180 * torch.pi))
        all_pts = []
        for i in tqdm.trange(self.depthmaps.shape[0]):
            depthi = self.get_init_depth_maps([i])
            all_pts.append(self.unproject(self.grid[..., 0:2].reshape(-1, 2), depthi.reshape(-1, 1)).reshape(-1, 3))
        if len(all_pts) > 0:
            all_pts = torch.cat(all_pts, dim=0)
        else:
            all_pts = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5],[1, 1, 1.0]], device=self.device)
            
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
        if os.path.exists(self.out_dir):
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
        self.read_done = True


        
    
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
    def vis_grad_diff(self, step):
        grad_map_dir = os.path.join(self.out_dir, "grad_maps")
        if not os.path.exists(grad_map_dir):
            os.makedirs(grad_map_dir)
        for fids in range(self.num_imgs):
            
            grid = self.grid[..., :2].clone().reshape(-1, 2)[None]
            
            init_grad_map = self.get_pixel_depth_gradient(grid, [fids], original=True).reshape(self.h, self.w, 2)
            current_grad_map = self.get_pixel_depth_gradient(grid, [fids], original=False).reshape(self.h, self.w, 2)
            diff_grad_map = current_grad_map - init_grad_map
            diff_grad_map = diff_grad_map.norm(dim=-1)
            max = diff_grad_map.max()
            mean = diff_grad_map.mean()
            if max == 0:
                continue
            diff_grad_map = diff_grad_map / (diff_grad_map.max()+1e-6)
            diff_grad_map = diff_grad_map.detach().cpu().numpy()
            diff_grad_map = (diff_grad_map * 255).astype(np.uint8)
            cv2.putText(diff_grad_map, f"max: {max:.3f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(diff_grad_map, f"mean {mean:.3f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(grad_map_dir, f"grad_diff_step{step:06d}_{fids:03d}.png"), diff_grad_map)

        
        

    def get_pixel_depth_gradient(self, pixels, fids, original=False, scale=True):
        if original:
            if self.read_done:
                sample_frames = self.gradmaps[fids].permute(0, 3, 1, 2)#.clone()
                normed_px = util.normalize_coords(pixels, self.h, self.w)[:, None]
                return F.grid_sample(sample_frames, normed_px, align_corners=True, mode='nearest').permute(0, 2, 3, 1)
            sample_frames = self.depthmaps[fids][..., None].permute(0, 3, 1, 2)#.clone()                

        else:
            sample_frames = self.depthmem.depthmaps[fids][..., None].permute(0, 3, 1, 2)#.clone()
            # delta_frames = self.depthmem.delta_depthmaps[fids][..., None].permute(0, 3, 1, 2).clone()
        
        if scale:
            scaled_pixels = pixels * torch.tensor([self.depthmaps.shape[-1]/float(self.w), self.depthmaps.shape[-2]/float(self.h)], device=self.device)
        else:
            scaled_pixels = pixels
        
        pix_l = scaled_pixels.clone()
        pix_l[..., 0] = torch.clamp(pix_l[..., 0]-1, 0, self.depthmaps.shape[-1]-1)
        # pix_r = scaled_pixels.clone()
        # pix_r[..., 0] = torch.clamp(pix_r[..., 0]+1, 0, self.depthmaps.shape[-1]-1)
        pix_u = scaled_pixels.clone()
        pix_u[..., 1] = torch.clamp(pix_u[..., 1]-1, 0, self.depthmaps.shape[-2]-1)
        # pix_d = scaled_pixels.clone()
        # pix_d[..., 1] = torch.clamp(pix_d[..., 1]+1, 0, self.depthmaps.shape[-2]-1)
        
        pix_c = util.normalize_coords(scaled_pixels, self.depthmaps.shape[-2], self.depthmaps.shape[-1])[:, None]
        pix_l = util.normalize_coords(pix_l, self.depthmaps.shape[-2], self.depthmaps.shape[-1])[:, None]
        # pix_r = util.normalize_coords(pix_r, self.depthmaps.shape[-2], self.depthmaps.shape[-1])[:, None]
        pix_u = util.normalize_coords(pix_u, self.depthmaps.shape[-2], self.depthmaps.shape[-1])[:, None]
        # pix_d = util.normalize_coords(pix_d, self.depthmaps.shape[-2], self.depthmaps.shape[-1])[:, None]

        depth_c = F.grid_sample(sample_frames, pix_c, align_corners=True, mode='nearest').permute(0, 2, 3, 1)
        depth_l = F.grid_sample(sample_frames, pix_l, align_corners=True, mode='nearest').permute(0, 2, 3, 1)
        # depth_r = F.grid_sample(sample_frames, pix_r, align_corners=True, mode='nearest').permute(0, 2, 3, 1)
        depth_u = F.grid_sample(sample_frames, pix_u, align_corners=True, mode='nearest').permute(0, 2, 3, 1)
        # depth_d = F.grid_sample(sample_frames, pix_d, align_corners=True, mode='nearest').permute(0, 2, 3, 1)

        x_grad = (depth_l - depth_c)
        y_grad = (depth_u - depth_c)
        return torch.cat([x_grad, y_grad], dim=-1)
        

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
        sample_grid = torch.clamp(sample_grid, -1, 1)
        depths = F.grid_sample(sample_frames, sample_grid, align_corners=True, mode='nearest').permute(0, 2, 3, 1)
        # if not original:
        #     d_depths = F.grid_sample(delta_frames, sample_grid, align_corners=True, mode='nearest').permute(0, 2, 3, 1)
        #     # depths = depths + d_depths

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
                                                           depth_err = 0.02):
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
    


    def compute_match_losses(self,
                           batch,
                           step,
                           w_depth=100,
                           w_smooth=10,
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
        
        max_padding = self.args.max_padding

        ids1 = batch['ids1'].numpy()
        ids2 = batch['ids2'].numpy()
        px1s = batch['pts1'].to(self.device)
        px2s = batch['pts2'].to(self.device)
        
        weights = batch['weights'].to(self.device)
        # self.data_time = time.time() - st
        num_pts = px1s.shape[1]
        st = time.time()
        

        # px1s: [n_pairs, npts, 2]
        x1s_samples = self.sample_3d_pts_for_pixels(px1s, return_depth=False, fids=ids1)
        x2s_samples, depth2 = self.sample_3d_pts_for_pixels(px2s, return_depth=True, fids=ids2)
        origin_x2s, origin_depth2 = self.sample_3d_pts_for_pixels(px2s, return_depth=True, fids=ids2, original=True)
        local_grad1 = self.get_pixel_depth_gradient(px1s, ids1, original=True, scale=True)
        opt_grad1 = self.get_pixel_depth_gradient(px1s, ids1, original=False, scale=True)
        local_grad2 = self.get_pixel_depth_gradient(px2s, ids2, original=True, scale=True)
        opt_grad2 = self.get_pixel_depth_gradient(px2s, ids2, original=False, scale=True)
        median_grad2 = self.median_grad_per_frame[ids2]
        # depth1 =depth1.squeeze()
        depth2 =depth2.squeeze()
        origin_depth2 = origin_depth2.squeeze()
        self.sample_time = time.time() - st

        st = time.time()
        try:
            x2s_proj_samples, x1s_canonical_samples = self.get_predictions(x1s_samples, ids1, ids2, return_canonical=True)
        except Exception as e:
            print(e)
            pdb.set_trace()
            print("mindepth= ", self.depthmem.depthmaps.min().item())
        
        x2s_pred = x2s_proj_samples.squeeze(-2)

        # [n_imgs, n_pts, n_samples, 2]
        # px2s_proj_samples, px2s_proj_depth_samples = self.project(x2s_proj_samples, return_depth=True)
        px2s_proj, px2s_proj_depths = self.project(x2s_pred, return_depth=True)


        self.pred_time = time.time() - st


        mask = self.get_in_range_mask(px2s_proj, max_padding)
        

        if mask.sum() > 0:
            # loss_rgb = F.mse_loss(pred_rgb1[rgb_mask], gt_rgb1[rgb_mask])
            # loss_rgb_grad = self.gradient_loss(pred_rgb1[rgb_mask], gt_rgb1[rgb_mask])
            
            flow_diff = abs(px2s_proj - px2s)

            # print("median match flow diff: ", torch.median(abs(flow_diff)).item())
            # print("max match flow diff: ", abs(flow_diff).max().item())

            if step > 2000:
                inliers = abs(flow_diff) < 2*torch.median(abs(flow_diff))
                inliers = inliers.all(-1)
                flow_diff = flow_diff[inliers]
            
            # optical_flow_loss = flow_diff[flow_loss_mask].abs().mean()
            # optical_flow_loss = torch.mean(flow_diff.sum(-1)/2. / torch.clamp(torch.norm(local_grad2.squeeze(), dim=-1), median_grad2[:, None]) * median_grad2[:, None])
            optical_flow_loss = torch.mean(flow_diff)
            # optical_flow_grad_loss = self.gradient_loss(px2s_proj[mask], px2s[mask], weights[mask])
        else:
            loss_rgb = loss_rgb_grad = optical_flow_loss = optical_flow_grad_loss = torch.tensor(0.)

        canonical_unit_sphere_loss = self.canonical_sphere_loss(x1s_canonical_samples)
        
        
        depth_diff = (px2s_proj_depths.squeeze() - depth2).abs()

        # print("max depth diff: ", depth_diff.abs().max().item())
        # print("mean depth diff: ", depth_diff.abs().mean().item())
        # print("median depth diff: ", torch.median(depth_diff.abs()).item())

        # depth_loss_mask = depth_diff > 2.

        
        if step > 2000:
            depth_diff = depth_diff[inliers]
        depth_pred_loss = torch.mean(depth_diff)
        # depth_pred_loss = torch.mean(depth_diff * (median_grad2[:, None]/torch.clamp(torch.norm(local_grad2.squeeze(), dim=-1), median_grad2[:, None])))

        # depth_pred_loss = masked_l1_loss(px2s_proj_depths,, depth2, mask)
            # depth_pred_loss = torch.mean(abs(px2s_proj_depths.squeeze() - depth2))
        # flow_loss_mask = torch.sum(flow_loss_mask, dim=-1).bool()
        # mask_iou = torch.sum(depth_loss_mask * flow_loss_mask) / torch.sum(depth_loss_mask + flow_loss_mask)

        # print(f"outlier iou: {mask_iou.item():.3f}")
    
        depth_bias = torch.mean(depth2 - origin_depth2)
        # depth_consist_loss = torch.mean(abs(depth2 - origin_depth2))
        depth_consist_loss = torch.mean(abs(px2s_proj_depths.squeeze() - origin_depth2))
        

        depth_grad_loss = torch.mean(torch.norm(local_grad2 - opt_grad2, dim=-1)) + torch.mean(torch.norm(local_grad1 - opt_grad1, dim=-1))
        self.local_grad2 = local_grad2
        self.opt_grad2 = opt_grad2
        

        if w_canonical > 0:
            x2s_canonical = self.get_prediction_one_way(x2s_samples, ids2)

            canonical_dist_loss = torch.mean(abs(x2s_canonical.squeeze() - x1s_canonical_samples.squeeze()))
        else:
            canonical_dist_loss = torch.tensor(0.).to(self.device)
        # optical_flow_loss = torch.clamp(optical_flow_loss, min=0, max=30)
        
        # w_scene_flow_smooth * scene_flow_smoothness_loss + \
        loss = optical_flow_loss + \
               w_depth * depth_pred_loss + \
               w_smooth*0.1 * depth_consist_loss + \
               w_smooth * depth_grad_loss + \
               w_canonical * canonical_dist_loss
        
        # loss = torch.clamp(loss, min=0, max=10)
        
        if torch.isnan(loss) or abs(px2s_proj_depths[mask]).min() < 1e-3: #or (loss > 100 and step > 1000)
            pdb.set_trace()
            # loss*= 0.1
        if write_logs:
            self.scalars_to_log['{}/Loss_match'.format(log_prefix)] = loss.item()
            self.scalars_to_log['{}/loss_depth_match'.format(log_prefix)] = depth_pred_loss.item()
            self.scalars_to_log['{}/loss_depth_consist_match'.format(log_prefix)] = depth_consist_loss.item()
            self.scalars_to_log['{}/loss_depth_grad_match'.format(log_prefix)] = depth_grad_loss.item()
            self.scalars_to_log['{}/loss_flow_match'.format(log_prefix)] = optical_flow_loss.item()
            self.scalars_to_log['{}/min_depth_match'.format(log_prefix)] = abs(px2s_proj_depths).min().item()
            self.scalars_to_log['{}/loss_depth_bias_match'.format(log_prefix)] = depth_bias.item()
            
            self.scalars_to_log['{}/loss_canonical_match'.format(log_prefix)] = canonical_dist_loss.item()
            self.scalars_to_log['{}/loss_canonical_unit_sphere'.format(log_prefix)] = canonical_unit_sphere_loss.item()
            # self.scalars_to_log['{}/loss_flow_gradient'.format(log_prefix)] = optical_flow_grad_loss.item()
            
            

        data = {'ids1': ids1,
                'ids2': ids2,
                'x1s': x1s_samples,
                'x2s_pred': x2s_pred,
                'xs_canonical': x1s_canonical_samples,
                'mask': mask,
                'px2s_proj': px2s_proj,
                'px2s_proj_depths': px2s_proj_depths,
                }
        if return_data:
            return loss, data
        else:
            return loss
        


    def compute_flow_losses(self,
                           batch,
                           step,
                        #    w_rgb=10,
                           w_depth=100,
                           w_smooth=10,
                           w_canonical = 0,  # original 100
                        #    w_depth_range=10,
                        #    w_distortion=1.,
                        #    w_scene_flow_smooth=20., # default 10, original 20
                        #    w_canonical_unit_sphere=0.,
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
       
        weights = batch['weights'].to(self.device)
        # self.data_time = time.time() - st
        num_pts = px1s.shape[1]
        st = time.time()


        # px1s: [n_pairs, npts, 2]
        x1s_samples = self.sample_3d_pts_for_pixels(px1s, return_depth=False, fids=ids1)
        x2s_samples, depth2 = self.sample_3d_pts_for_pixels(px2s, return_depth=True, fids=ids2)
        origin_x2s, origin_depth2 = self.sample_3d_pts_for_pixels(px2s, return_depth=True, fids=ids2, original=True)

        local_grad1 = self.get_pixel_depth_gradient(px1s, ids1, original=True, scale=True).squeeze()
        opt_grad1 = self.get_pixel_depth_gradient(px1s, ids1, original=False, scale=True).squeeze()
        local_grad2 = self.get_pixel_depth_gradient(px2s, ids2, original=True, scale=True).squeeze()
        opt_grad2 = self.get_pixel_depth_gradient(px2s, ids2, original=False, scale=True).squeeze()
        median_grad2 = self.median_grad_per_frame[ids2]
        # depth1 =depth1.squeeze()
        depth2 = depth2.squeeze()
        origin_depth2 = origin_depth2.squeeze()
        self.sample_time = time.time() - st

        st = time.time()
        x2s_proj_samples, x1s_canonical_samples = self.get_predictions(x1s_samples, ids1, ids2, return_canonical=True)

        x2s_pred = x2s_proj_samples.squeeze(-2)

        px2s_proj, px2s_proj_depths = self.project(x2s_pred, return_depth=True)


        self.pred_time = time.time() - st


        mask = self.get_in_range_mask(px2s_proj, max_padding)
        label_inlier = self.get_in_range_mask(px2s, max_padding)
        # print("label outliers: ", (~label_inlier).sum().item())

        zero_depth = (px2s_proj_depths.abs()<1e-2).squeeze()
        mask = mask * ~zero_depth
        # rgb_mask = self.get_in_range_mask(px1s)

        if mask.sum() > 0:


            optical_flow_loss = masked_l1_loss(px2s_proj[mask], px2s[mask], weights[mask], normalize=False)
            # optical_flow_loss = masked_l1_loss(px2s_proj, px2s, weights, normalize=False)
            outlier_flow_loss = (px2s_proj[(~mask)*(~zero_depth)]-px2s[(~mask)*(~zero_depth)]).abs().mean()
            # if ~mask.sum() > 0:
            #     optical_flow_loss += 0.1 * masked_l1_loss(px2s_proj[~mask], px2s[~mask], weights[~mask], normalize=False)
            # optical_flow_loss = abs(px2s_proj - px2s).mean()
            optical_flow_grad_loss = self.gradient_loss(px2s_proj[mask], px2s[mask], weights[mask])
        else:
            loss_rgb = loss_rgb_grad = optical_flow_loss = optical_flow_grad_loss = torch.tensor(0.)


        canonical_unit_sphere_loss = self.canonical_sphere_loss(x1s_canonical_samples)

        depth_diff = (px2s_proj_depths.squeeze() - depth2).abs()
        # depth_diff[~mask] *= 0.5
        
        inlier_depth_loss = torch.mean(depth_diff[mask])
        outlier_depth_loss = torch.mean(depth_diff[~mask])

        depth_diff = depth_diff[~zero_depth]
        depth_pred_loss = torch.mean(depth_diff.abs())


        grad_diff_1 = (local_grad1 - opt_grad1).abs()
        grad_diff_2 = (local_grad2 - opt_grad2).abs()
        grad_diff_1[~mask] *= 0.1
        grad_diff_2[~mask] *= 0.1

        depth_grad_loss = torch.mean(torch.norm(grad_diff_1, dim=-1)) + torch.mean(torch.norm(grad_diff_2, dim=-1))

        # time_disp = torch.tensor(abs(ids1 - ids2)>2).reshape(-1, 1).to(self.device)
        depth_consist_loss = torch.mean(abs(px2s_proj_depths.squeeze() - origin_depth2))

        if w_canonical > 0:
            x2s_canonical = self.get_prediction_one_way(x2s_samples, ids2)

            canonical_dist_loss = torch.mean(abs(x2s_canonical.squeeze() - x1s_canonical_samples.squeeze()))
        else:
            canonical_dist_loss = torch.tensor(0.).to(self.device)

        loss = optical_flow_loss + \
               w_flow_grad * optical_flow_grad_loss + \
               w_smooth * depth_grad_loss + \
               w_smooth * depth_consist_loss + \
               w_depth * depth_pred_loss
        outlier_ratio = (~mask).sum().item()/(mask.numel())
        # print(f"outlier ratio: {outlier_ratio:.3f}", f"inlier depth: {inlier_depth_loss.item():.3f}", f"outlier depth: { outlier_depth_loss.item():.3f}", f"inlier flow:{optical_flow_loss.item():.3f}", f"outlier flow:{outlier_flow_loss.item():.3f}", f"max_interval: {abs(ids1 - ids2).max().item()}", f" min_zerodepth: {px2s_proj_depths.abs().min().item():.3f}")
        
        if torch.isnan(loss) or (loss > 100 and step > 2000) or abs(px2s_proj_depths[mask]).min() < 1e-3 or (outlier_ratio > 0.1 and step > 2000):
            print("out")
            outdir = Path("outlier")
            outdir.mkdir(exist_ok=True)
            for i in range(0, len(ids1), 10):
                id1 = ids1[i]
                id2 = ids2[i]
                image1 = self.images[id1].detach().cpu().numpy().copy()*255
                image2 = self.images[id2].detach().cpu().numpy().copy()*255
                image1 = image1.astype(np.uint8)
                image2 = image2.astype(np.uint8)
                px1s_ = px1s[i].detach().cpu().numpy()
                px2s_ = px2s[i].detach().cpu().numpy()
                pred_px2s_ = px2s_proj[i].detach().cpu().numpy()
                colors = plt.cm.hsv(np.linspace(0, 1, len(px2s_)))[..., :3]*255
                colors = colors.astype(np.uint8)
                for px, color in zip(px1s_, colors):
                    cv2.circle(image1, (int(px[0]), int(px[1])), 3, color.tolist(), -1)
                for px, color in zip(px2s_, colors):
                    cv2.circle(image2, (int(px[0]), int(px[1])), 3, color.tolist(), -1)
                inliers = mask[i].detach().cpu().numpy()
                for px, pred_px, color in zip(px2s_[inliers], pred_px2s_[inliers], colors[inliers]):
                    cv2.line(image2, (int(px[0]), int(px[1])), (int(pred_px[0]), int(pred_px[1])), color.tolist(), 1)
                for px, colro in zip(px2s_[~inliers], colors[~inliers]):
                    cv2.circle(image2, (int(px[0]), int(px[1])), 5, (0, 0, 255), 1)
                    
                # cv2.imwrite(f"outlier/outlier_{step}_{i}_1_{id1}.png", image1)
                # cv2.imwrite(f"outlier/outlier_{step}_{i}_2_{id2}.png", image2)
            # print("done")
            pdb.set_trace()
            

            # pdb.set_trace()
        

        if write_logs:
            self.scalars_to_log['{}/Loss_dense'.format(log_prefix)] = loss.item()
            self.scalars_to_log['{}/loss_depth_dense'.format(log_prefix)] = depth_pred_loss.item()
            self.scalars_to_log['{}/loss_depth_grad_dense'.format(log_prefix)] = depth_grad_loss.item()
            self.scalars_to_log['{}/loss_canonical_dense'.format(log_prefix)] = canonical_dist_loss.item()
            self.scalars_to_log['{}/loss_depth_consist_dense'.format(log_prefix)] = depth_consist_loss.item()
            self.scalars_to_log['{}/min_depth_dense'.format(log_prefix)] = abs(px2s_proj_depths).min().item()
            self.scalars_to_log['{}/loss_flow_dense'.format(log_prefix)] = optical_flow_loss.item()
            self.scalars_to_log['{}/outlier_ratio'.format(log_prefix)] = outlier_ratio
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

        longterm_end = 24000

        # scaler = GradScaler()
        
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        if len(batch) != 2:
            longterm_end = 0
            batch = {'gm': batch}
        if "simple" in batch.keys():
            batch["gm"] = batch["simple"]
        loss_flow, data_flow = self.compute_flow_losses(batch['gm'],
                                                step,
                                                w_smooth=1.,
                                                w_depth=100,
                                                # w_scene_flow_smooth=w_scene_flow_smooth,
                                                # w_distortion=w_distortion,
                                                w_flow_grad=w_flow_grad,
                                                return_data=True)
        if step < longterm_end and ('long' in batch.keys()) and batch['long']['ids1'][0] >= 0:
            loss_match, data_match = self.compute_match_losses(batch['long'],
                                                step,
                                                w_smooth=self.args.smooth_weight,
                                                w_depth=100,
                                                # w_depth=180,
                                                # w_scene_flow_smooth=w_scene_flow_smooth,
                                                w_distortion=w_distortion,
                                                w_flow_grad=w_flow_grad,
                                                return_data=True)
        else:
            loss_match = 0
        
        
        match_overfit_iters = 1000
        if step < match_overfit_iters:
            loss = loss_flow + loss_match

        elif step < match_overfit_iters*2:
            loss = loss_flow + loss_match*0.1

            
        else:
            # loss = loss_flow
            loss = loss_flow + loss_match*0.01

        # zero_loss = 1.0/abs(self.depthmem.depthmaps[self.depthmem.depthmaps<1e-3]).mean()

        self.scalars_to_log['loss/Loss'] = loss.item()

        is_break = False
        # for p in self.deform_mlp.parameters():
        #     if torch.isnan(p.data).any() or (p.grad is not None and torch.isnan(p.grad).any()):
        #         is_break = True
        #         pdb.set_trace()
        if torch.isnan(loss):
            # pass
            pdb.set_trace()
        self.forward_time = time.time() - start
        back_st = time.time()

        mindep = self.depthmem.depthmaps.min()
        mindep_pos = torch.where(self.depthmem.depthmaps == mindep)


        # scaler.scale(loss).backward()
        loss.backward()
        self.backtime = time.time() - back_st

        

        if self.args.grad_clip > 0 and self.depthmem.depthmaps.grad.norm() > self.args.grad_clip:
            print(f"Warning! Clip gradient from {self.depthmem.depthmaps.grad.norm()} to {self.args.grad_clip} \n")
            torch.nn.utils.clip_grad_norm_(
                self.depthmem.parameters(), self.args.grad_clip)
            print("depth grad norm: ", self.depthmem.depthmaps.grad.norm().item())
            
            # for param in self.learnable_params:
            #     grad_norm = torch.nn.utils.clip_grad_norm_(
            #         param, self.args.grad_clip)
            #     if grad_norm > self.args.grad_clip:
                    # print("Warning! Clip gradient from {} to {}".format(
                    #     grad_norm, self.args.grad_clip))

        
        if is_break:
            pass

        
        self.optimizer.step()
        self.scheduler.step()
        # scaler.update()


        self.scalars_to_log['loss/min_depth'] = self.depthmem.depthmaps.min().item()
        self.scalars_to_log['loss/max_depth'] = self.depthmem.depthmaps.max().item()
        self.scalars_to_log['loss/depth<0_ratio'] = (self.depthmem.depthmaps < 0).float().mean().item() 
        try:
            self.scalars_to_log['loss/min_depth_pos'] = torch.where(self.depthmem.depthmaps == self.depthmem.depthmaps.min())[0][0].item()
        except:
            pdb.set_trace()
        self.scalars_to_log['lr'] = self.optimizer.param_groups[0]['lr']

        self.scalars_to_log['time'] = time.time() - start
        
        self.ids1 = data_flow['ids1']
        self.ids2 = data_flow['ids2']
        if (step % 1000 == 1 or (step < 10 and step % 2 == 0))and self.args.dbg:
            for fid in range(0, self.depthmem.depthmaps.shape[0], 8):
                grid0 = self.grid[..., 0:2].clone()
                depth = self.get_pred_depth_maps([fid])[0].detach()
                xyz = self.unproject(grid0, depth[..., None])
                color = self.images[fid].reshape(-1, 3).cpu().numpy()
                xyz = xyz.reshape(-1, 3).cpu().numpy()



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

    def plot_correspondences_for_pixels(self, query_kpt, query_id, num_pts=200,
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

    def eval_video_correspondences(self, query_id, pts=None, num_pts=200, seed=1234, use_mask=False,
                                   mask=None, reverse_mask=False, vis_occlusion=False, occlusion_th=0.99,
                                   use_max_loc=False, regular=True,
                                   interval=10, radius=2, return_kpts=False):
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
                

            if self.args.use_error_map and (step % self.args.i_cache == 0) and (step > 0):
                flow_save_dir = os.path.join(self.out_dir, 'flow')
                os.makedirs(flow_save_dir, exist_ok=True)
                flow_errors = []
                for i, (id1, id2) in enumerate(zip(ids1, ids2)):
                    save_path = os.path.join(flow_save_dir, '{}_{}.npy'.format(os.path.basename(self.img_files[id1]),
                                                                               os.path.basename(self.img_files[id2])))
               
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
        num_imgs = to_load['num_imgs']
        self.num_imgs = num_imgs
        self.deform_mlp = NVPnonlin(n_layers=6,
                                        n_frames=num_imgs,
                                        feature_dim=self.deform_mlp.feature_dim,
                                        t_dim = 16,
                                        multires=self.deform_mlp.multires,
                                        base_res=self.deform_mlp.base_res,
                                        net_layer=self.deform_mlp.net_layer,
                                        bound=self.deform_mlp.bound,
                                        device=self.device).to(self.device)

        self.deform_mlp.load_state_dict(to_load['deform_mlp'])
        # self.feature_mlp.load_state_dict(to_load['feature_mlp'])
        # self.color_mlp.load_state_dict(to_load['color_mlp'])
        
        if self.args.opt_depth:
            self.depthmaps = torch.zeros((num_imgs, self.depthmaps.shape[1], self.depthmaps.shape[2]), device=self.device)
            self.depthmem = DepthMem( self.args, self.depthmaps, self.device).to(self.device)
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
            fpath = ckpts[self.args.ckpt]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print('Reloading from {}, starting at step={}'.format(fpath, step))
        else:
            print('No ckpts found, from scratch...')
            step = 0

        return step
