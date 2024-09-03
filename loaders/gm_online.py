import os
import glob
import json
import imageio.v2 as imageio
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import multiprocessing as mp
import tqdm
from util import normalize_coords, gen_grid_np, gen_grid


class GMDepthDataset(Dataset):
    def __init__(self, args, max_interval=None):
        self.args = args
        self.seq_dir = args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')
        # self.depth_dir = os.path.join(self.seq_dir, 'depth', "npz")
        # self.depthmask_dir = os.path.join(self.seq_dir, 'depth', "mask")
        self.depth_dir = os.path.join(self.seq_dir, 'raw_depth', "depth")
        self.depthmask_dir = os.path.join(self.seq_dir, 'raw_depth', "mask")

        self.flow_dir = os.path.join(self.seq_dir, 'gmflow')
        img_names = sorted(os.listdir(self.img_dir))
        self.num_imgs = min(self.args.num_imgs, len(img_names))

        # if self.num_imgs < 50:
        #     args.lrate_decay_steps = 2500 
        print("decay steps: ", args.lrate_decay_steps)

        self.img_names = img_names[:self.num_imgs]
        self.inc_step = args.inc_step
        if self.inc_step > 0:
            self.img_range = mp.Value('i', self.inc_step)
        self.step = mp.Value('i', 0)

        h, w, _ = imageio.imread(os.path.join(self.img_dir, img_names[0])).shape
        self.h, self.w = h, w
        max_interval = self.num_imgs - 1 if not max_interval else max_interval
        # self.max_interval = mp.Value('i', max_interval)
        self.max_interval = max_interval
        self.num_pts = self.args.num_pts
        self.grid = gen_grid_np(self.h, self.w).reshape(-1, 2)

        if self.args.norm_neighbor > 0:
            normal = self.args.norm_neighbor
        else:
            normal = 10
        normal = torch.distributions.Normal(0, normal)
        print("noraml_neighbor: ", normal)
        self.dist_weights = normal.log_prob(torch.linspace(0, self.num_imgs-1, self.num_imgs)).exp()
        self.dist_weights[0] = 0
        self.dist_weights[1:5] = self.dist_weights[4]
        self.dist_weights /= self.dist_weights.sum()

        # self.depthmaps = np.zeros((self.num_imgs, self.h, self.w), dtype=np.float32)
        # self.depthmasks = np.zeros((self.num_imgs, self.h, self.w), dtype=np.float32)
        # for i in range(self.num_imgs):
            # depthmap = np.load(os.path.join(self.depth_dir, self.img_names[i].replace('.jpg', ".npz")))["depth"]
            # depthmask = cv2.imread(os.path.join(self.depthmask_dir, self.img_names[i]), cv2.IMREAD_ANYDEPTH) / 255.
            # self.depthmaps[i] = depthmap
            # self.depthmasks[i] = depthmask
        
        # self.depthmasks = np.round(self.depthmasks)/
        self.flow = np.zeros((self.num_imgs, max_interval*2, self.h, self.w, 2), dtype=np.float32)
        self.masks = np.zeros((self.num_imgs, max_interval*2, self.h, self.w), dtype=np.float32)
        self.candidate_pair_range = np.zeros((self.num_imgs, 2), dtype=np.int32)
        print("max interval: ", max_interval)
        print("loading flow and mask")
        for i in tqdm.trange(self.num_imgs):
            for j in range(-max_interval, max_interval+1):
                if j == 0:
                    continue
                if i + j < 0 or i + j >= self.num_imgs:
                    continue
                
                flow_file = os.path.join(self.flow_dir, '{}_{}.npy'.format(self.img_names[i], self.img_names[i+j]))
                if not os.path.exists(flow_file):
                    flow_file = os.path.join(self.flow_dir, '{}_{}.npz'.format(self.img_names[i], self.img_names[i+j]))
                    flow = np.load(flow_file)['flow']
                else:
                    flow = np.load(flow_file)
                mask_file = flow_file.replace('gmflow', 'full_mask').replace('.npy', '.png')
                masks = imageio.imread(mask_file) / 255.
                if masks.sum() < 4096:
                    continue
                self.candidate_pair_range[i, 0] = min(self.candidate_pair_range[i, 0], j)
                self.candidate_pair_range[i, 1] = max(self.candidate_pair_range[i, 1], j)
                if j > 0:
                    self.flow[i, j+max_interval-1] = flow
                    self.masks[i, j+max_interval-1] = masks
                    
                else:
                    self.flow[i, j+max_interval] = flow
                    self.masks[i, j+max_interval] = masks
                
        countmaps = np.zeros((self.num_imgs, self.h, self.w), dtype=np.float32)
        for i in tqdm.trange(self.num_imgs):
            countmap = imageio.imread(os.path.join(self.seq_dir, 'gm_count_maps', self.img_names[i].replace('.jpg', '.png')))
            countmaps[i] = countmap
        
        self.masks = np.round(self.masks)
        self.masks = self.masks * countmaps[:, None]
        # self.masks = np.clip(self.masks, 0, 10)
        self.masks[self.masks>0] = 1. / np.sqrt(self.masks[self.masks>0] + 1.)
        for i in range(self.masks.shape[0]):
            for j in range(self.masks.shape[1]):
                if np.sum(self.masks[i, j]) > 0:
                    self.masks[i, j] /= np.sum(self.masks[i, j])

        
        
        # outdir = os.path.join("viscntmap")
        # os.makedirs(outdir, exist_ok=True)
        # maxmask = self.masks.max()
        # for i in tqdm.trange(self.masks.shape[0]):
        #     for j in range(self.masks.shape[1]):
        #         cv2.imwrite(os.path.join(outdir, "{}_{}.png".format(i, i+j-16)), (self.masks[i, j]/maxmask*255.0).astype(np.uint8))
        print("loading GMflow done")

    def __len__(self):
        return self.num_imgs**2*1000
    
    def set_step(self, step):
        self.step.value = step
        print("set step to ", self.step.value)

    def increase_range(self):
        current_range = self.img_range.value
        self.img_range.value = min(self.inc_step + current_range, self.num_imgs)
        print("increasing range to ", self.img_range)

    def set_max_interval(self, max_interval):
        self.max_interval.value = min(max_interval, self.num_imgs - 1)

    def increase_max_interval_by(self, increment):
        curr_max_interval = self.max_interval.value
        self.max_interval.value = min(curr_max_interval + increment, self.num_imgs - 1)

    def __getitem__(self, idx):
        if self.inc_step > 0:
            id1 = idx % self.img_range.value - self.img_range.value // 2 + self.num_imgs // 2
            # if id1 == 9:
            #     print("range= ", self.img_range, "idx= ", idx)
        else:   
            id1 = idx % self.num_imgs
            # id1 = idx % self.center_range + self.num_imgs // 2 - self.center_range // 2

        # img_name1 = self.img_names[id1]
        max_interval = min(self.max_interval, self.num_imgs - 1)
        # img2_candidates = sorted(list(self.sample_weights[img_name1].keys()))
        # img2_candidates = img2_candidates[max(id1 - max_interval, 0):min(id1 + max_interval, self.num_imgs - 1)]
        id2s = np.arange(max(id1 + self.candidate_pair_range[id1, 0], 0, id1-max_interval), min(id1 + self.candidate_pair_range[id1, 1], self.num_imgs - 1, id1+max_interval))

        # sample noramally from i-10 and i+10
        # id2s = np.array([self.img_names.index(n) for n in img2_candidates])
        # sample_weights = np.array([self.sample_weights[img_name1][i] for i in img2_candidates])
        

        id2s[id2s>=id1]+=1


        
        # normal_weights = self.dist_weights[abs(id2s - id1)]
        normal_weights = np.ones_like(id2s) / len(id2s)
        normal_weights = torch.from_numpy(normal_weights).float()

        if self.args.norm_neighbor >= 1:
            sample_weights = normal_weights.numpy()
        else:
            sample_weights = np.ones_like(id2s) / len(id2s)
        
        longterm = False
        # if self.step.value < 1000 and np.random.rand() < 0.5:
        #     longterm = True
        #     sample_weights[np.abs(id2s - id1) < 30] = 0
        # sample_weights[np.abs(id2s - id1) > 16] = 0
        # sample_weights[np.abs(id2s - id1) < 2] = 0
            # sample_weights /= np.sum(sample_weights)
        #     sample_weights[np.abs(id2s - id1) <= 1] = 0.5
        #     sample_weights /= np.sum(sample_weights)
        # sample_weights[np.abs(id2s - id1) > 5] = 0
        sample_weights /= np.sum(sample_weights)
        

        id2 = np.random.choice(id2s, p=sample_weights)
        # id2 = self.img_names.index(img_name2)
        frame_interval = abs(id1 - id2)


        if id2 - id1 < 0:
            flow = self.flow[id1, id2 - id1 + self.max_interval].reshape(-1, 2)
            mask = self.masks[id1, id2 - id1 + self.max_interval].reshape(-1)
        else:
            flow = self.flow[id1, id2 - id1 + self.max_interval -1].reshape(-1, 2)
            mask = self.masks[id1, id2 - id1 + self.max_interval -1].reshape(-1)

        #     mask = cycle_consistency_mask | occlusion_mask & depthmask1# & depthmask2

        if mask.sum() == 0:
            invalid = True
            mask = np.ones_like(mask)
            mask /= mask.sum()
        else:
            invalid = False
        # flow_mag = np.linalg.norm(flow, axis=1)
        # if longterm:
        #     mask[flow_mag > flow_mag.mean()] *= 10
        # else:
        # mask[flow_mag > flow_mag.mean()] *= 3
        
        # mask /= mask.sum()

        select_mask = np.random.choice(mask.shape[0], self.num_pts, replace=((mask>0).sum() < self.num_pts), p=mask)

        # if self.args.use_count_map:
        #     # count_map = imageio.imread(os.path.join(self.seq_dir, 'count_maps', img_name1.replace('.jpg', '.png')))
        #     count_map = self.countmaps[id1]
        #     pixel_sample_weight = 1 / np.sqrt(count_map + 1.)
        #     pixel_sample_weight = pixel_sample_weight[mask]
        #     pixel_sample_weight /= pixel_sample_weight.sum()
        #     select_ids = np.random.choice(mask.sum(), self.num_pts, replace=(mask.sum() < self.num_pts),
        #                                     p=pixel_sample_weight)
        # else:
        #     select_ids = np.random.choice(mask.sum(), self.num_pts, replace=(mask.sum() < self.num_pts))

        pair_weight = np.cos((frame_interval - 1.) / max_interval * np.pi / 3)  
        pair_weight = np.ones_like(pair_weight)

        coord1 = self.grid[select_mask]
        coord2 = coord1 + flow[select_mask]

        pts1 = torch.from_numpy(coord1).float()
        pts2 = torch.from_numpy(coord2).float()
        # pts2_normed = normalize_coords(pts2, self.h, self.w)[None, None]


        weights = torch.ones((*pts1.shape[0:1], 1)) * pair_weight
        # weights = np.linalg.norm(flow[select_mask], axis=1, keepdims=True)
        # weights = np.clip((weights+1)*0.5, 0, 4)
        # weights = torch.from_numpy(weights).float()


        if invalid:
            weights = torch.zeros_like(weights)

        if np.random.choice([0, 1]):
            id1, id2, pts1, pts2= id2, id1, pts2, pts1
            # weights[mask == 0.] = 0

        data = {'ids1': id1,
                'ids2': id2,
                'pts1': pts1,  # [n_pts, 2]
                'pts2': pts2,  # [n_pts, 2]
                # 'depth1': pts_depth1,
                # 'depth2': pts_depth2,
                # 'gt_rgb1': gt_rgb1,  # [n_pts, 3]
                # 'gt_rgb2': gt_rgb2,
                'weights': weights,  # [n_pts, 1]
                # 'covisible_mask': covisible_mask,  # [n_pts, 1]
                # 'occlusion_mask': torch.from_numpy(occlusion_mask[mask][select_ids]),
                }
        return data




class GMExhaustiveDataset(Dataset):
    def __init__(self, args, max_interval=None):
        self.args = args
        self.seq_dir = args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')
        self.flow_dir = os.path.join(self.seq_dir, 'gmflow')
        img_names = sorted(os.listdir(self.img_dir))
        self.num_imgs = min(self.args.num_imgs, len(img_names))
        self.img_names = img_names[:self.num_imgs]

        h, w, _ = imageio.imread(os.path.join(self.img_dir, img_names[0])).shape
        self.h, self.w = h, w
        max_interval = self.num_imgs - 1 if not max_interval else max_interval
        # self.max_interval = mp.Value('i', max_interval)
        self.max_interval = max_interval
        self.num_pts = self.args.num_pts
        self.grid = gen_grid_np(self.h, self.w)
                    
        normal = self.args.norm_neighbor
        normal = torch.distributions.Normal(0, normal)
        print("noraml_neighbor: ", normal)
        self.dist_weights = normal.log_prob(torch.linspace(0, self.num_imgs-1, self.num_imgs)).exp()
        self.dist_weights[0] = 0
        self.dist_weights[1:5] = self.dist_weights[4]

        self.dist_weights /= self.dist_weights.sum()

        self.inc_step = args.inc_step
        if self.inc_step > 0:
            self.img_range = mp.Value('i', self.inc_step)
        max_interval = 32
        self.max_interval = max_interval

        self.flow = np.zeros((self.num_imgs, 2*max_interval, self.h, self.w, 2), dtype=np.float32)
        self.masks = np.zeros((self.num_imgs, 2*max_interval, self.h, self.w, 3), dtype=np.float32)
        print("loading flow and mask")
        for i in tqdm.trange(self.num_imgs):
            for j in range(-max_interval, max_interval+1):
                if j == 0:
                    continue
                if i + j < 0 or i + j >= self.num_imgs:
                    continue
                flow_file = os.path.join(self.flow_dir, '{}_{}.npy'.format(self.img_names[i], self.img_names[i+j]))
                flow = np.load(flow_file)
                mask_file = flow_file.replace('gmflow', 'gm_masks').replace('.npy', '.png')
                masks = imageio.imread(mask_file) / 255.
                if j > 0:
                    self.flow[i, j+15] = flow
                    self.masks[i, j+15] = masks
                else:
                    self.flow[i, j+16] = flow
                    self.masks[i, j+16] = masks
        
        # self.countmaps = np.zeros((self.num_imgs, self.h, self.w), dtype=np.float32)
        # for i in range(self.num_imgs):
        #     countmap = imageio.imread(os.path.join(self.seq_dir, 'count_maps', self.img_names[i].replace('.jpg', '.png')))
        #     self.countmaps[i] = countmap
        
        self.images = np.zeros((self.num_imgs, self.h, self.w, 3), dtype=np.float32)
        for i in range(self.num_imgs):
            self.images[i] = imageio.imread(os.path.join(self.img_dir, self.img_names[i]))

    def __len__(self):
        return self.num_imgs * 100000

    def set_max_interval(self, max_interval):
        self.max_interval.value = min(max_interval, self.num_imgs - 1)

    def increase_max_interval_by(self, increment):
        curr_max_interval = self.max_interval.value
        self.max_interval.value = min(curr_max_interval + increment, self.num_imgs - 1)

    def increase_range(self):
        current_range = self.img_range.value
        self.img_range.value = min(self.inc_step + current_range, self.num_imgs)
        print("increasing range to ", self.img_range)

    def set_step(self, step):
        pass

    def __getitem__(self, idx):
        cached_flow_pred_dir = os.path.join('out', '{}_{}'.format(self.args.expname, self.seq_name), 'flow')
        cached_flow_pred_files = sorted(glob.glob(os.path.join(cached_flow_pred_dir, '*')))
        # flow_error_file = os.path.join(os.path.dirname(cached_flow_pred_dir), 'flow_error.txt')
        # if os.path.exists(flow_error_file):
        #     flow_error = np.loadtxt(flow_error_file)
        #     id1_sample_weights = flow_error / np.sum(flow_error)
        #     id1 = np.random.choice(self.num_imgs, p=id1_sample_weights)
        # else:
        if self.inc_step > 0:
            id1 = idx % self.img_range.value - self.img_range.value // 2 + self.num_imgs // 2
            # if id1 == 9:
            #     print("range= ", self.img_range, "idx= ", idx)
        else:   
            id1 = idx % self.num_imgs

        # img_name2 = np.random.choice(img2_candidates, p=sample_weights)
        # id2 = self.img_names.index(img_name2)
        max_interval = min(self.max_interval, self.num_imgs - 1)
        # img2_candidates = sorted(list(self.sample_weights[img_name1].keys()))
        # img2_candidates = img2_candidates[max(id1 - max_interval, 0):min(id1 + max_interval, self.num_imgs - 1)]
        id2s = np.arange(max(id1 - max_interval, 0), min(id1 + max_interval, self.num_imgs - 1))

        # sample noramally from i-10 and i+10
        # id2s = np.array([self.img_names.index(n) for n in img2_candidates])
        # sample_weights = np.array([self.sample_weights[img_name1][i] for i in img2_candidates])
        
        normal_weights = self.dist_weights[abs(id2s - id1)]

        sample_weights = np.ones_like(id2s) / len(id2s)

        id2s[id2s>=id1]+=1

        if self.args.norm_neighbor >= 1:
            sample_weights *= normal_weights.numpy()
        else:
            sample_weights /= np.sum(sample_weights)
            sample_weights[np.abs(id2s - id1) <= 1] = 0.5
            sample_weights /= np.sum(sample_weights)
        # sample_weights[np.abs(id2s - id1) > 16] = 0
        sample_weights /= np.sum(sample_weights)
        

        id2 = np.random.choice(id2s, p=sample_weights)

        frame_interval = abs(id1 - id2)

        # read image, flow and confidence
        img1 = self.images[id1] / 255.
        img2 = self.images[id2] / 255.

        # flow_file = os.path.join(self.flow_dir, '{}_{}.npy'.format(img_name1, img_name2))
        # flow = np.load(flow_file)
        # mask_file = flow_file.replace('raft_exhaustive', 'raft_masks').replace('.npy', '.png')
        # masks = imageio.imread(mask_file) / 255.

        if id2 - id1 < 0:
            flow = self.flow[id1, id2 - id1 + 16]
            masks = self.masks[id1, id2 - id1 + 16]
        else:
            flow = self.flow[id1, id2 - id1 + 15]
            masks = self.masks[id1, id2 - id1 + 15]

        coord1 = self.grid
        coord2 = self.grid + flow

        cycle_consistency_mask = masks[..., 0] > 0
        occlusion_mask = masks[..., 1] > 0

        if frame_interval == 1:
            mask = np.ones_like(cycle_consistency_mask)
        else:
            mask = cycle_consistency_mask | occlusion_mask

        if mask.sum() == 0:
            invalid = True
            mask = np.ones_like(cycle_consistency_mask)
        else:
            invalid = False

        if len(cached_flow_pred_files) > 0 and self.args.use_error_map:
            cached_flow_pred_file = cached_flow_pred_files[id1]
            # assert img_name1 + '_' in cached_flow_pred_file
            sup_flow_file = os.path.join(self.flow_dir, os.path.basename(cached_flow_pred_file))
            pred_flow = np.load(cached_flow_pred_file)
            sup_flow = np.load(sup_flow_file)
            error_map = np.linalg.norm(pred_flow - sup_flow, axis=-1)
            error_map = cv2.GaussianBlur(error_map, (5, 5), 0)
            error_selected = error_map[mask]
            prob = error_selected / np.sum(error_selected)
            select_ids_error = np.random.choice(mask.sum(), self.num_pts, replace=(mask.sum() < self.num_pts), p=prob)
            select_ids_random = np.random.choice(mask.sum(), self.num_pts, replace=(mask.sum() < self.num_pts))
            select_ids = np.random.choice(np.concatenate([select_ids_error, select_ids_random]), self.num_pts,
                                          replace=False)
        else:
            if self.args.use_count_map:
                # count_map = imageio.imread(os.path.join(self.seq_dir, 'count_maps', img_name1.replace('.jpg', '.png')))
                # count_map = self.countmaps[id1]
                # pixel_sample_weight = 1 / np.sqrt(count_map + 1.)
                pixel_sample_weight = np.ones_like(cycle_consistency_mask).astype(np.float32)
                pixel_sample_weight = pixel_sample_weight[mask]
                pixel_sample_weight /= pixel_sample_weight.sum()
                select_ids = np.random.choice(mask.sum(), self.num_pts, replace=(mask.sum() < self.num_pts),
                                              p=pixel_sample_weight)
            else:
                select_ids = np.random.choice(mask.sum(), self.num_pts, replace=(mask.sum() < self.num_pts))

        pair_weight = np.cos((frame_interval - 1.) / max_interval * np.pi / 2)

        pts1 = torch.from_numpy(coord1[mask][select_ids]).float()
        pts2 = torch.from_numpy(coord2[mask][select_ids]).float()
        pts2_normed = normalize_coords(pts2, self.h, self.w)[None, None]

        covisible_mask = torch.from_numpy(cycle_consistency_mask[mask][select_ids]).float()[..., None]
        weights = torch.ones_like(covisible_mask) * pair_weight

        # pt1coords = pts1.cpu().long().numpy()
        # pt2coords = pts2.cpu().long().numpy()
        # pt1img = (img1.copy())
        # for i in range(pt1coords.shape[0]):
        #     pt1img = cv2.circle(pt1img, (pt1coords[i, 0], pt1coords[i, 1]), 3, (0, 0, 1.0), -1)
        # pt2img = img2.copy()
        # for i in range(pt2coords.shape[0]):
        #     pt2img = cv2.circle(pt2img, (pt2coords[i, 0], pt2coords[i, 1]), 3, (0, 0, 1.0), -1)
        

        gt_rgb1 = torch.from_numpy(img1[mask][select_ids]).float()
        gt_rgb2 = F.grid_sample(torch.from_numpy(img2).float().permute(2, 0, 1)[None], pts2_normed,
                                align_corners=True).squeeze().T

        if invalid:
            weights = torch.zeros_like(weights)

        if np.random.choice([0, 1]):
            id1, id2, pts1, pts2, gt_rgb1, gt_rgb2 = id2, id1, pts2, pts1, gt_rgb2, gt_rgb1
            weights[covisible_mask == 0.] = 0

        data = {'ids1': id1,
                'ids2': id2,
                'pts1': pts1,  # [n_pts, 2]
                'pts2': pts2,  # [n_pts, 2]
                'gt_rgb1': gt_rgb1,  # [n_pts, 3]
                'gt_rgb2': gt_rgb2,
                'weights': weights,  # [n_pts, 1]
                'covisible_mask': covisible_mask,  # [n_pts, 1]
                'occlusion_mask': torch.from_numpy(occlusion_mask[mask][select_ids]),
                }
        return data

