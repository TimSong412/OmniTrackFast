import os
import glob
import json
import imageio
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import multiprocessing as mp
from util import normalize_coords, gen_grid_np


def get_sample_weights(flow_stats):
    sample_weights = {}
    for k in flow_stats.keys():
        sample_weights[k] = {}
        total_num = np.array(list(flow_stats[k].values())).sum()
        for j in flow_stats[k].keys():
            sample_weights[k][j] = 1. * flow_stats[k][j] / total_num
    return sample_weights

class RAFTDepthDataset(Dataset):
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
        self.img_names = img_names[:self.num_imgs]

        h, w, _ = imageio.imread(os.path.join(self.img_dir, img_names[0])).shape
        self.h, self.w = h, w
        max_interval = self.num_imgs - 1 if not max_interval else max_interval
        self.max_interval = mp.Value('i', max_interval)
        self.num_pts = self.args.num_pts
        self.grid = gen_grid_np(self.h, self.w)
        flow_stats = json.load(open(os.path.join(self.seq_dir, 'flow_stats.json')))
        self.sample_weights = get_sample_weights(flow_stats)
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
        # self.center_range = 10 # +- 5 frames initially

    def __len__(self):
        return self.num_imgs * 100000

    def increase_center_range_by(self, increment):
        self.center_range = min(self.center_range + increment, self.num_imgs)

    def set_max_interval(self, max_interval):
        self.max_interval.value = min(max_interval, self.num_imgs - 1)

    def increase_max_interval_by(self, increment):
        curr_max_interval = self.max_interval.value
        self.max_interval.value = min(curr_max_interval + increment, self.num_imgs - 1)

    def __getitem__(self, idx):
        cached_flow_pred_dir = os.path.join('out', '{}_{}'.format(self.args.expname, self.seq_name), 'flow')
        cached_flow_pred_files = sorted(glob.glob(os.path.join(cached_flow_pred_dir, '*')))
        flow_error_file = os.path.join(os.path.dirname(cached_flow_pred_dir), 'flow_error.txt')
        if os.path.exists(flow_error_file):
            flow_error = np.loadtxt(flow_error_file)
            id1_sample_weights = flow_error / np.sum(flow_error)
            id1 = np.random.choice(self.num_imgs, p=id1_sample_weights)
        else:
            id1 = idx % self.num_imgs
            # id1 = idx % self.center_range + self.num_imgs // 2 - self.center_range // 2

        img_name1 = self.img_names[id1]
        max_interval = min(self.max_interval.value, self.num_imgs - 1)
        img2_candidates = sorted(list(self.sample_weights[img_name1].keys()))
        img2_candidates = img2_candidates[max(id1 - max_interval, 0):min(id1 + max_interval, self.num_imgs - 1)]

        # sample noramally from i-10 and i+10
        id2s = np.array([self.img_names.index(n) for n in img2_candidates])
        sample_weights = np.array([self.sample_weights[img_name1][i] for i in img2_candidates])

        normal_weights = self.dist_weights[abs(id2s - id1)]
        if self.args.norm_neighbor >= 1:
            sample_weights *= normal_weights.numpy()
        else:
            sample_weights /= np.sum(sample_weights)
            sample_weights[np.abs(id2s - id1) <= 1] = 0.5
            sample_weights /= np.sum(sample_weights)

        sample_weights /= np.sum(sample_weights)

        # sample_weights /= np.sum(sample_weights)
        # sample_weights[np.abs(id2s - id1) <= 1] = 0.5
        # sample_weights /= np.sum(sample_weights)

        
        

        img_name2 = np.random.choice(img2_candidates, p=sample_weights)
        id2 = self.img_names.index(img_name2)
        frame_interval = abs(id1 - id2)

        # read image, flow and confidence
        # img1 = imageio.imread(os.path.join(self.img_dir, img_name1)) / 255.
        # img2 = imageio.imread(os.path.join(self.img_dir, img_name2)) / 255.

        # depth1 = cv2.imread(os.path.join(self.depth_dir, img_name1.replace('.jpg', '.png')), cv2.IMREAD_ANYDEPTH)
        depth1 = np.load(os.path.join(self.depth_dir, img_name1.replace('.jpg', ".npz")))["depth"]
        # depth1 = depth1.astype(np.float32) / 65535.0 * 2.0
        # depth1 = 2.0 - depth1
        # depth1 = np.clip(depth1, 1e-3, 65535)
        # depth1 = 0.05 / depth1
        

        depth2 = np.load(os.path.join(self.depth_dir, img_name2.replace('.jpg', ".npz")))["depth"]
        # depth2 = depth2.astype(np.float32) / 65535.0 * 2.0
        # depth2 = 2.0 - depth2
        # depth2 = np.clip(depth2, 1e-3, 65535)
        # depth2 = 0.05 / depth2

        depthmask1 = cv2.imread(os.path.join(self.depthmask_dir, img_name1), cv2.IMREAD_ANYDEPTH) / 255.
        depthmask2 = cv2.imread(os.path.join(self.depthmask_dir, img_name2), cv2.IMREAD_ANYDEPTH) / 255.

        flow_file = os.path.join(self.flow_dir, '{}_{}.npy'.format(img_name1, img_name2))
        flow = np.load(flow_file)
        mask_file = flow_file.replace('raft_exhaustive', 'raft_masks').replace('.npy', '.png')
        masks = imageio.imread(mask_file) / 255.

        coord1 = self.grid
        coord2 = self.grid + flow

        # warpped_depth2 = F.grid_sample(torch.from_numpy(depth2).float()[None, None], normalize_coords(torch.from_numpy(coord2), self.h, self.w).float()[None], align_corners=True).squeeze()
        warpped_mask2 = F.grid_sample(torch.from_numpy(depthmask2).float()[None, None], normalize_coords(torch.from_numpy(coord2), self.h, self.w).float()[None], align_corners=True).squeeze()

        depthmask1 = depthmask1 > 0
        depthmask2 = (warpped_mask2 > 0.5).numpy()

        cycle_consistency_mask = masks[..., 0] > 0
        occlusion_mask = masks[..., 1] > 0

        if frame_interval == 1:
            mask = np.ones_like(cycle_consistency_mask) & depthmask1 & depthmask2
        else:
            mask = cycle_consistency_mask | occlusion_mask & depthmask1 & depthmask2

        if mask.sum() == 0:
            invalid = True
            mask = np.ones_like(cycle_consistency_mask)
        else:
            invalid = False

        if len(cached_flow_pred_files) > 0 and self.args.use_error_map:
            cached_flow_pred_file = cached_flow_pred_files[id1]
            assert img_name1 + '_' in cached_flow_pred_file
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
                count_map = imageio.imread(os.path.join(self.seq_dir, 'count_maps', img_name1.replace('.jpg', '.png')))
                pixel_sample_weight = 1 / np.sqrt(count_map + 1.)
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

        pts_depth1 = torch.from_numpy(depth1[mask][select_ids]).float()
        # _pts_depth2 = warpped_depth2[mask][select_ids].float()
        pts_depth2 = F.grid_sample(torch.from_numpy(depth2).float()[None, None], pts2_normed, align_corners=True).squeeze()

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
        

        # gt_rgb1 = torch.from_numpy(img1[mask][select_ids]).float()
        # gt_rgb2 = F.grid_sample(torch.from_numpy(img2).float().permute(2, 0, 1)[None], pts2_normed,
        #                         align_corners=True).squeeze().T

        if invalid:
            weights = torch.zeros_like(weights)

        if np.random.choice([0, 1]):
            id1, id2, pts1, pts2, pts_depth1, pts_depth2 = id2, id1, pts2, pts1, pts_depth2, pts_depth1
            weights[covisible_mask == 0.] = 0

        data = {'ids1': id1,
                'ids2': id2,
                'pts1': pts1,  # [n_pts, 2]
                'pts2': pts2,  # [n_pts, 2]
                'depth1': pts_depth1,
                'depth2': pts_depth2,
                # 'gt_rgb1': gt_rgb1,  # [n_pts, 3]
                # 'gt_rgb2': gt_rgb2,
                'weights': weights,  # [n_pts, 1]
                'covisible_mask': covisible_mask,  # [n_pts, 1]
                # 'occlusion_mask': torch.from_numpy(occlusion_mask[mask][select_ids]),
                }
        return data


class RAFTExhaustiveDataset(Dataset):
    def __init__(self, args, max_interval=None):
        self.args = args
        self.seq_dir = args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')
        self.flow_dir = os.path.join(self.seq_dir, 'raft_exhaustive')
        img_names = sorted(os.listdir(self.img_dir))
        self.num_imgs = min(self.args.num_imgs, len(img_names))
        self.img_names = img_names[:self.num_imgs]

        h, w, _ = imageio.imread(os.path.join(self.img_dir, img_names[0])).shape
        self.h, self.w = h, w
        max_interval = self.num_imgs - 1 if not max_interval else max_interval
        self.max_interval = mp.Value('i', max_interval)
        self.num_pts = self.args.num_pts
        self.grid = gen_grid_np(self.h, self.w)
        flow_stats = json.load(open(os.path.join(self.seq_dir, 'flow_stats.json')))
        self.sample_weights = get_sample_weights(flow_stats)
        
        normal = self.args.norm_neighbor
        normal = torch.distributions.Normal(0, normal)
        print("noraml_neighbor: ", normal)
        self.dist_weights = normal.log_prob(torch.linspace(0, self.num_imgs-1, self.num_imgs)).exp()
        self.dist_weights[0] = 0
        self.dist_weights[1:5] = self.dist_weights[4]
        self.dist_weights /= self.dist_weights.sum()

    def __len__(self):
        return self.num_imgs * 100000

    def set_max_interval(self, max_interval):
        self.max_interval.value = min(max_interval, self.num_imgs - 1)

    def increase_max_interval_by(self, increment):
        curr_max_interval = self.max_interval.value
        self.max_interval.value = min(curr_max_interval + increment, self.num_imgs - 1)

    def __getitem__(self, idx):
        cached_flow_pred_dir = os.path.join('out', '{}_{}'.format(self.args.expname, self.seq_name), 'flow')
        cached_flow_pred_files = sorted(glob.glob(os.path.join(cached_flow_pred_dir, '*')))
        flow_error_file = os.path.join(os.path.dirname(cached_flow_pred_dir), 'flow_error.txt')
        if os.path.exists(flow_error_file):
            flow_error = np.loadtxt(flow_error_file)
            id1_sample_weights = flow_error / np.sum(flow_error)
            id1 = np.random.choice(self.num_imgs, p=id1_sample_weights)
        else:
            id1 = idx % self.num_imgs

        img_name1 = self.img_names[id1]
        max_interval = min(self.max_interval.value, self.num_imgs - 1)
        img2_candidates = sorted(list(self.sample_weights[img_name1].keys()))
        img2_candidates = img2_candidates[max(id1 - max_interval, 0):min(id1 + max_interval, self.num_imgs - 1)]

        # sample more often from i-1 and i+1
        id2s = np.array([self.img_names.index(n) for n in img2_candidates])
        sample_weights = np.array([self.sample_weights[img_name1][i] for i in img2_candidates])
        sample_weights /= np.sum(sample_weights)
        # sample_weights[np.abs(id2s - id1) <= 1] = 0.5
        # sample_weights /= np.sum(sample_weights)
        normal_weights = self.dist_weights[abs(id2s - id1)]
        if self.args.norm_neighbor >= 1:
            sample_weights *= normal_weights.numpy()
        else:
            sample_weights /= np.sum(sample_weights)
            sample_weights[np.abs(id2s - id1) <= 1] = 0.5
            sample_weights /= np.sum(sample_weights)

        sample_weights /= np.sum(sample_weights)

        img_name2 = np.random.choice(img2_candidates, p=sample_weights)
        id2 = self.img_names.index(img_name2)
        frame_interval = abs(id1 - id2)

        # read image, flow and confidence
        img1 = imageio.imread(os.path.join(self.img_dir, img_name1)) / 255.
        img2 = imageio.imread(os.path.join(self.img_dir, img_name2)) / 255.

        flow_file = os.path.join(self.flow_dir, '{}_{}.npy'.format(img_name1, img_name2))
        flow = np.load(flow_file)
        mask_file = flow_file.replace('raft_exhaustive', 'raft_masks').replace('.npy', '.png')
        masks = imageio.imread(mask_file) / 255.

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
            assert img_name1 + '_' in cached_flow_pred_file
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
                count_map = imageio.imread(os.path.join(self.seq_dir, 'count_maps', img_name1.replace('.jpg', '.png')))
                pixel_sample_weight = 1 / np.sqrt(count_map + 1.)
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


class RAFTEvalDataset(Dataset):
    def __init__(self, args, num_pts=512, max_interval=None):
        self.args = args
        self.seq_dir = args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')
        self.flow_dir = os.path.join(self.seq_dir, 'raft_exhaustive')
        img_names = sorted(os.listdir(self.img_dir))
        self.num_imgs = min(self.args.num_imgs, len(img_names))
        self.img_names = img_names[:self.num_imgs]

        h, w, _ = imageio.imread(os.path.join(self.img_dir, img_names[0])).shape
        self.h, self.w = h, w
        max_interval = self.num_imgs - 1 if not max_interval else max_interval
        self.max_interval = mp.Value('i', max_interval)
        self.num_pts = num_pts
        self.grid = gen_grid_np(self.h, self.w)
        flow_stats = json.load(open(os.path.join(self.seq_dir, 'flow_stats.json')))
        self.sample_weights = get_sample_weights(flow_stats)

    def __len__(self):
        return self.num_imgs * 100000

    def set_max_interval(self, max_interval):
        self.max_interval.value = min(max_interval, self.num_imgs - 1)

    def increase_max_interval_by(self, increment):
        curr_max_interval = self.max_interval.value
        self.max_interval.value = min(curr_max_interval + increment, self.num_imgs - 1)

    def __getitem__(self, idx):
        cached_flow_pred_dir = os.path.join('out', '{}_{}'.format(self.args.expname, self.seq_name), 'flow')
        cached_flow_pred_files = sorted(glob.glob(os.path.join(cached_flow_pred_dir, '*')))
        flow_error_file = os.path.join(os.path.dirname(cached_flow_pred_dir), 'flow_error.txt')
        if os.path.exists(flow_error_file):
            flow_error = np.loadtxt(flow_error_file)
            id1_sample_weights = flow_error / np.sum(flow_error)
            id1 = np.random.choice(self.num_imgs, p=id1_sample_weights)
        else:
            id1 = idx % self.num_imgs

        img_name1 = self.img_names[id1]
        max_interval = min(self.max_interval.value, self.num_imgs - 1)
        img2_candidates = sorted(list(self.sample_weights[img_name1].keys()))
        img2_candidates = img2_candidates[max(id1 - max_interval, 0):min(id1 + max_interval, self.num_imgs - 1)]

        # sample more often from i-5 and i+5
        id2s = np.array([self.img_names.index(n) for n in img2_candidates])
        sample_weights = np.array([self.sample_weights[img_name1][i] for i in img2_candidates])
        sample_weights /= np.sum(sample_weights)
        sample_weights[np.abs(id2s - id1) <= 5] = 0.5
        sample_weights /= np.sum(sample_weights)

        img_name2 = np.random.choice(img2_candidates, p=sample_weights)
        id2 = self.img_names.index(img_name2)
        frame_interval = abs(id1 - id2)

        # read image, flow and confidence
        img1 = imageio.imread(os.path.join(self.img_dir, img_name1)) / 255.
        img2 = imageio.imread(os.path.join(self.img_dir, img_name2)) / 255.

        flow_file = os.path.join(self.flow_dir, '{}_{}.npy'.format(img_name1, img_name2))
        flow = np.load(flow_file)
        mask_file = flow_file.replace('raft_exhaustive', 'raft_masks').replace('.npy', '.png')
        masks = imageio.imread(mask_file) / 255.

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
            assert img_name1 + '_' in cached_flow_pred_file
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
                count_map = imageio.imread(os.path.join(self.seq_dir, 'count_maps', img_name1.replace('.jpg', '.png')))
                pixel_sample_weight = 1 / np.sqrt(count_map + 1.)
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