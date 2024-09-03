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


def get_sample_weights(flow_stats):
    sample_weights = {}
    for k in flow_stats.keys():
        sample_weights[k] = {}
        total_num = np.array(list(flow_stats[k].values())).sum()
        for j in flow_stats[k].keys():
            sample_weights[k][j] = 1. * flow_stats[k][j] / total_num
    return sample_weights


class OfflineDepthDataset(Dataset):
    def __init__(self, args, max_interval=None):
        self.args = args
        self.seq_dir = args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')
        # self.depth_dir = os.path.join(self.seq_dir, 'depth', "npz")
        # self.depthmask_dir = os.path.join(self.seq_dir, 'depth', "mask")
        self.depth_dir = os.path.join(self.seq_dir, 'raw_depth', "depth")
        self.depthmask_dir = os.path.join(self.seq_dir, 'raw_depth', "mask")

        self.flow_dir = os.path.join(self.seq_dir, 'raft_exhaustive')
        img_names = sorted(os.listdir(self.img_dir))
        self.num_imgs = min(self.args.num_imgs, len(img_names))

        # if self.num_imgs < 50:
        #     args.lrate_decay_steps = 2500 
        print("decay steps: ", args.lrate_decay_steps)

        self.img_names = img_names[:self.num_imgs]
        self.inc_step = args.inc_step
        if self.inc_step > 0:
            self.img_range = mp.Value('i', self.inc_step)

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
        self.flow = np.zeros((self.num_imgs, 32, self.h, self.w, 2), dtype=np.float32)
        self.masks = np.zeros((self.num_imgs, 32, self.h, self.w), dtype=np.float32)
        self.candidate_pair_range = np.zeros((self.num_imgs, 2), dtype=np.int32)
        
        print("loading mask only")
        for i in tqdm.trange(self.num_imgs):
            for j in range(-16, 17):
                if j == 0:
                    continue
                if i + j < 0 or i + j >= self.num_imgs:
                    continue
                flow_file = os.path.join(self.flow_dir, '{}_{}.npy'.format(self.img_names[i], self.img_names[i+j]))
                # flow = np.load(flow_file)
                mask_file = flow_file.replace('raft_exhaustive', 'full_mask').replace('.npy', '.png')
                masks = imageio.imread(mask_file) / 255.
                if masks.sum() < 4096:
                    continue
                self.candidate_pair_range[i, 0] = min(self.candidate_pair_range[i, 0], j)
                self.candidate_pair_range[i, 1] = max(self.candidate_pair_range[i, 1], j)
                if j > 0:
                    # self.flow[i, j+15] = flow
                    self.masks[i, j+15] = masks
                    
                else:
                    # self.flow[i, j+16] = flow
                    self.masks[i, j+16] = masks
                
        countmaps = np.zeros((self.num_imgs, self.h, self.w), dtype=np.float32)
        for i in tqdm.trange(self.num_imgs):
            countmap = imageio.imread(os.path.join(self.seq_dir, 'count_maps', self.img_names[i].replace('.jpg', '.png')))
            countmaps[i] = countmap
        
        # self.masks = np.round(self.masks)
        # self.masks = self.masks * countmaps[:, None]
        # self.masks = np.clip(self.masks, 0, 10)
        # self.masks[self.masks>0] = 1. / np.sqrt(self.masks[self.masks>0] + 1.)
        print("normalizing masks")
        for i in tqdm.trange(self.masks.shape[0]):
            for j in range(self.masks.shape[1]):
                if np.sum(self.masks[i, j]) > 0:
                    # self.masks[i, j] /= np.sum(self.masks[i, j])
                    self.masks[i, j] *= countmaps[i]
                    self.masks[i, j] /= np.sum(self.masks[i, j])

        print("loading done")

    def __len__(self):
        return self.num_imgs**2*1000
    
    def set_step(self, step):
        pass

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


        
        normal_weights = self.dist_weights[abs(id2s - id1)]

        if self.args.norm_neighbor >= 1:
            sample_weights = normal_weights.numpy()
        else:
            sample_weights = np.ones_like(id2s) / len(id2s)
        sample_weights[np.abs(id2s - id1) > 16] = 0
            # sample_weights /= np.sum(sample_weights)
        #     sample_weights[np.abs(id2s - id1) <= 1] = 0.5
        #     sample_weights /= np.sum(sample_weights)
        # sample_weights[np.abs(id2s - id1) > 5] = 0
        sample_weights /= np.sum(sample_weights)
        

        id2 = np.random.choice(id2s, p=sample_weights)
        # id2 = self.img_names.index(img_name2)
        frame_interval = abs(id1 - id2)

        flow_file = os.path.join(self.flow_dir, '{}_{}.npy'.format(self.img_names[id1], self.img_names[id2]))
        flow = np.load(flow_file).reshape(-1, 2)
        if id2 - id1 < 0:
            mask = self.masks[id1, id2 - id1 + 16].reshape(-1)
        else:
            mask = self.masks[id1, id2 - id1 + 15].reshape(-1)

        #     mask = cycle_consistency_mask | occlusion_mask & depthmask1# & depthmask2

        if mask.sum() == 0:
            invalid = True
            mask = np.ones_like(mask)
            mask /= mask.sum()
        else:
            invalid = False

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

        coord1 = self.grid[select_mask]
        coord2 = coord1 + flow[select_mask]

        pts1 = torch.from_numpy(coord1).float()
        pts2 = torch.from_numpy(coord2).float()
        # pts2_normed = normalize_coords(pts2, self.h, self.w)[None, None]


        weights = torch.ones((*pts1.shape[0:1], 1)) * pair_weight


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

