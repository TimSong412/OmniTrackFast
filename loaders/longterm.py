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
from pathlib import Path
from util import normalize_coords, gen_grid_np, gen_grid


def get_sample_weights(flow_stats):
    sample_weights = {}
    for k in flow_stats.keys():
        sample_weights[k] = {}
        total_num = np.array(list(flow_stats[k].values())).sum()
        for j in flow_stats[k].keys():
            sample_weights[k][j] = 1. * flow_stats[k][j] / total_num
    return sample_weights

class LongtermDataset(Dataset):
    def __init__(self, args, max_interval=8, minbatch=4):
        self.args = args
        self.seq_dir = args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')

        # self.depth_dir = os.path.join(self.seq_dir, 'depth', "npz")
        # self.depthmask_dir = os.path.join(self.seq_dir, 'depth', "mask")
        self.matchdir = os.path.join(self.seq_dir, 'match')
        if not os.path.exists(self.matchdir):
            self.longterm = False
            print("ERROR: longterm match not found")
            return
        all_matches = sorted(glob.glob(os.path.join(self.matchdir, '*.npz')))
        all_images = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        self.num_imgs = min(self.args.num_imgs, len(all_images))
        self.matches = None
        for matchfile in all_matches[:self.num_imgs]:
            matchi = np.load(matchfile)['match']
            if self.matches is None:
                self.matches = matchi
            else:
                self.matches = np.concatenate([self.matches, matchi], axis=0)
        self.inc_step = args.inc_step
        if self.inc_step > 0:
            self.img_range = mp.Value('i', self.inc_step)
        
        self.matchid = []
        self.candidates = []
        self.batchsize = 1e10
        valid_pairs = 0
        total_pairs = 0
        for i in range(self.num_imgs):
            if sum(self.matches[..., 0] == i) > 0:
                
                candidate_ids = self.matches[self.matches[..., 0] == i, 3]
                batchsize = 1e10
                candidate_mask = np.ones_like(np.unique(candidate_ids))
                for idid, id2 in enumerate(np.unique(candidate_ids)):
                    total_pairs += 1
                    if sum(candidate_ids == id2) < minbatch:
                        candidate_mask[idid] = 0
                        continue
                    batchsize = min(batchsize, sum(candidate_ids == id2))
                
                self.batchsize = min(self.batchsize, batchsize)
                if sum(candidate_mask) == 0:
                    continue
                self.matchid.append(i)
                self.candidates.append(np.unique(candidate_ids)[candidate_mask.astype(bool)])
                valid_pairs += sum(candidate_mask)
        print("valid pairs: ", valid_pairs, "total pairs: ", total_pairs)
        # outdir = os.path.join("viscntmap")
        # os.makedirs(outdir, exist_ok=True)
        # maxmask = self.masks.max()
        # for i in tqdm.trange(self.masks.shape[0]):
        #     for j in range(self.masks.shape[1]):
        #         cv2.imwrite(os.path.join(outdir, "{}_{}.png".format(i, i+j-16)), (self.masks[i, j]/maxmask*255.0).astype(np.uint8))
        print("batchsize: ", self.batchsize)
        
        if valid_pairs < self.num_imgs*2:
            print("longterm dataset no enough pairs")
            self.longterm = False
        else:
            self.longterm = True

    def __len__(self):
        if not self.longterm:
            return 100
        return len(self.matchid)**2*self.batchsize**2
    
    def increase_range(self):
        
        pass

    def __getitem__(self, idx):
        # if self.inc_step > 0:
        #     print("here")
        #     idx = idx % self.img_range.value - self.img_range.value // 2 + len(self.matchid) // 2
        # else:
        if not self.longterm:
            return {
                'ids1': -1,
                'ids2': -1,
                'pts1': torch.zeros(0, 2),
                'pts2': torch.zeros(0, 2),
                'weights': torch.zeros(0, 1),
            }

        idx = idx % len(self.matchid)
        id1 = self.matchid[idx]
        id2 = self.candidates[idx][np.random.choice(len(self.candidates[idx]))]
        
        candidates = self.matches[(self.matches[..., 0] == id1) & (self.matches[..., 3] == id2)]
        if len(candidates) < self.batchsize:
            print("err")
        select_id = np.random.choice(len(candidates), self.batchsize, replace=False)

        select_pts = torch.from_numpy(candidates[select_id])
        pts1 = select_pts[..., 1:3]
        pts2 = select_pts[..., 4:6]
        weights = select_pts[..., 6:7]

        swap = torch.rand(1) > 0.5

        if swap:
            id1, id2 = id2, id1
            pts1, pts2 = pts2, pts1

        data = {
            'ids1': id1,
            'ids2': id2,
            'pts1': pts1,
            'pts2': pts2,
            'weights': weights,
        }
        return data


class CoTrackerSingle(Dataset):

    def __init__(self, args, max_interval=10, batchsize=8) -> None:
        super().__init__()
        self.args = args
        self.seq_dir = args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')
        img0 = imageio.imread(os.path.join(self.img_dir, "00000.jpg"))
        self.img_shape = img0.shape[:2]


        self.step = 0

        self.match_dir = Path("dataset/cotracker_eval") / self.seq_name
        trackfile = self.match_dir / "track.npy"
        visifile = self.match_dir / "visibility.npy"
        query_file = self.match_dir / f"{self.seq_name}.npy"


        self.all_tracks = np.load(trackfile, allow_pickle=True)[0]
        self.all_visibility = np.load(visifile, allow_pickle=True)[0] > 0.99

        self.all_tracks *= (self.img_shape[1] / 256., self.img_shape[0] / 256.)
        # query = np.load(query_file, allow_pickle=True).item()
        self.num_imgs = len(self.all_tracks)
        self.batchsize = batchsize

        # valid_ids = self.all_visibility.sum(axis=1) > self.batchsize

        id2_candidates = []

        for i in range(self.num_imgs):
            candidates =[]
            visible_i = self.all_visibility[i]
            for j in range(self.num_imgs):
                if abs(i-j) < 10:
                    continue    
                visible_j = self.all_visibility[j]
                visible = np.logical_and(visible_i, visible_j)
                if np.sum(visible) < self.batchsize:
                    continue
                candidates.append(j)
            id2_candidates.append(candidates)
        
        self.id2_candidates = id2_candidates
        self.id1_candidates = []
        for i in range(self.num_imgs):
            if len(id2_candidates[i]) > 0:
                self.id1_candidates.append(i)
        self.id1_candidates = np.array(self.id1_candidates)
    
    def __len__(self):
        return self.num_imgs**2*self.all_tracks.shape[1]
    
    def set_step(self, step):
        self.step = step

    def increase_range(self):
        # self.range = range
        pass
    
    def __getitem__(self, idx):
        id1 = idx % len(self.id1_candidates)
        id1 = self.id1_candidates[id1]
        id2 = self.id2_candidates[id1][np.random.choice(len(self.id2_candidates[id1]))]

        visible_1 = self.all_visibility[id1]
        visible_2 = self.all_visibility[id2]
        visible = np.logical_and(visible_1, visible_2)
        pts1_candidate = self.all_tracks[id1][visible]
        pts2_candidate = self.all_tracks[id2][visible]
        if len(pts1_candidate) < self.batchsize:
            print("err")
        select_id = np.random.choice(len(pts1_candidate), self.batchsize, replace=False)
        pts1 = torch.from_numpy(pts1_candidate[select_id])
        pts2 = torch.from_numpy(pts2_candidate[select_id])
        weights = torch.tensor([1.])
        return {
            'ids1': id1,
            'ids2': id2,
            'pts1': pts1,
            'pts2': pts2,
            'weights': weights,
        }




class CoTrackerDense(Dataset):

    def __init__(self, args, max_interval=10, batchsize=256) -> None:
        super().__init__()
        self.args = args
        self.seq_dir = args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')
        img0 = imageio.imread(os.path.join(self.img_dir, "00000.jpg"))
        self.img_shape = img0.shape[:2]


        self.step = 0

        self.match_dir = Path("dataset/cotracker_mask") / self.seq_name
        if args.cotracker == "mask":
            trackfile = self.match_dir / "traj_strided.npz"
            visifile = self.match_dir / "vis_strided.npz"
        elif args.cotracker == "filter":
            trackfile = self.match_dir / "traj_filtered.npz"
            visifile = self.match_dir / "vis_filtered.npz"
        elif args.cotracker == "single":
            trackfile = self.match_dir / "traj_single.npz"
            visifile = self.match_dir / "vis_single.npz"
        


        self.all_tracks = np.load(trackfile, allow_pickle=True)["traj"][0]
        self.all_visibility = np.load(visifile, allow_pickle=True)["vis"][0] > 0.9

        
        # query = np.load(query_file, allow_pickle=True).item()
        self.num_imgs = len(self.all_tracks)
        self.batchsize = batchsize

        self.inc_step = args.inc_step
        if self.inc_step > 0:
            self.img_range = mp.Value('i', self.inc_step)

        # valid_ids = self.all_visibility.sum(axis=1) > self.batchsize

        id2_candidates = []
        min_vis = 1e10

        for i in range(self.num_imgs):
            candidates =[]
            visible_i = self.all_visibility[i]
            for j in range(self.num_imgs):
                if abs(i-j) < 5:
                    continue    
                visible_j = self.all_visibility[j]
                visible = np.logical_and(visible_i, visible_j)
                if np.sum(visible) < self.batchsize:
                    continue
                if np.sum(visible) < min_vis:
                    min_vis = np.sum(visible)
                candidates.append(j)
            id2_candidates.append(candidates)
        self.batchsize = min(min_vis, 1024)
        

        print("batchsize = ", self.batchsize)

        self.id2_candidates = id2_candidates
        self.id1_candidates = []
        for i in range(self.num_imgs):
            if len(id2_candidates[i]) > 0:
                self.id1_candidates.append(i)
        self.id1_candidates = np.array(self.id1_candidates)
    
    def __len__(self):
        return self.num_imgs**2*min(self.all_tracks.shape[1], 500)
    
    def set_step(self, step):
        self.step = step

    def increase_range(self):
        current_range = self.img_range.value
        self.img_range.value = min(self.inc_step + current_range, len(self.id1_candidates))
        print("increasing range to ", self.img_range)
    
    def __getitem__(self, idx):
        if self.inc_step > 0:
            id1 = idx % self.img_range.value - self.img_range.value // 2 + len(self.id1_candidates) // 2
        else:
            id1 = idx % len(self.id1_candidates)
        id1 = self.id1_candidates[id1]
        id2 = self.id2_candidates[id1][np.random.choice(len(self.id2_candidates[id1]))]

        visible_1 = self.all_visibility[id1]
        visible_2 = self.all_visibility[id2]
        visible = np.logical_and(visible_1, visible_2)
        pts1_candidate = self.all_tracks[id1][visible]
        pts2_candidate = self.all_tracks[id2][visible]
        if len(pts1_candidate) < self.batchsize:
            print("err")
        select_id = np.random.choice(len(pts1_candidate), self.batchsize, replace=False)
        pts1 = torch.from_numpy(pts1_candidate[select_id])
        pts2 = torch.from_numpy(pts2_candidate[select_id])

        vis = False
        if vis:
            img1 = imageio.imread(os.path.join(self.img_dir, f"{id1:05d}.jpg"))
            img2 = imageio.imread(os.path.join(self.img_dir, f"{id2:05d}.jpg"))
            for i in range(len(pts1)):
                cv2.circle(img1, (int(pts1[i, 0]), int(pts1[i, 1])), 3, (0, 255, 0), -1)
                cv2.circle(img2, (int(pts2[i, 0]), int(pts2[i, 1])), 3, (0, 255, 0), -1)
            imageio.imwrite("img1.png", img1)
            imageio.imwrite("img2.png", img2)
            print("img1 and img2 saved")
        weights = torch.tensor([1.])
        return {
            'ids1': id1,
            'ids2': id2,
            'pts1': pts1,
            'pts2': pts2,
            'weights': weights,
        }






class CotrackerDataset(Dataset):
    def __init__(self, args, max_interval=8, minbatch=8):
        self.args = args
        self.seq_dir = args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')

        # self.depth_dir = os.path.join(self.seq_dir, 'depth', "npz")
        # self.depthmask_dir = os.path.join(self.seq_dir, 'depth', "mask")
        self.matchdir = Path(self.seq_dir) / 'cotracker'
        all_matches = sorted(self.matchdir.glob('*.npz'))
        all_images = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        self.num_imgs = min(self.args.num_imgs, len(all_images))
        self.matches = None
        self.ids = []
        self.candidates = []
        self.batchsize = 1024
        total_pairs = 0
        valid_pairs = 0
        for matchfile in all_matches:
            framei = int(matchfile.stem.split('_')[0])
            framej = int(matchfile.stem.split('_')[1])
            total_pairs += 1

            matchij = np.load(matchfile)['match']
            if len(matchij) < minbatch:
                continue
            valid_pairs += 1
            if framei not in self.ids:
                self.ids.append(framei)
                self.candidates.append([])
            self.candidates[-1].append(framej)
            
            if self.matches is None:
                self.matches = matchij
            else:
                self.matches = np.concatenate([self.matches, matchij], axis=0)

            if len(matchij) < self.batchsize:
                self.batchsize = len(matchij)
                print("framei = ", framei, "framej = ", framej, "batchsize = ", self.batchsize)
        self.inc_step = args.inc_step
        if self.inc_step > 0:
            self.img_range = mp.Value('i', self.inc_step)
        print("valid pairs: ", valid_pairs, "total pairs: ", total_pairs)
        print("batchsize: ", self.batchsize)


    def __len__(self):
        return len(self.ids)**2*self.batchsize**2
    
    def increase_range(self):
        
        pass

    def __getitem__(self, idx):
        # if self.inc_step > 0:
        #     print("here")
        #     idx = idx % self.img_range.value - self.img_range.value // 2 + len(self.matchid) // 2
        # else:
        idx = idx % len(self.ids)
        id1 = self.ids[idx]
        id2 = self.candidates[idx][np.random.choice(len(self.candidates[idx]))]
        # if not self.longterm:
        #     return {
        #         'ids1': -1,
        #         'ids2': -1,
        #         'pts1': torch.zeros(0, 2),
        #         'pts2': torch.zeros(0, 2),
        #         'weights': torch.zeros(0, 1),
        #     }

        candidates = self.matches[(self.matches[..., 0] == id1) & (self.matches[..., 3] == id2)]
        if len(candidates) < self.batchsize:
            print("err")
        select_id = np.random.choice(len(candidates), self.batchsize, replace=False)

        select_pts = torch.from_numpy(candidates[select_id])
        pts1 = select_pts[..., 1:3]
        pts2 = select_pts[..., 4:6]
        
        weights = torch.tensor([1.])

        swap = torch.rand(1) > 0.5

        if swap:
            id1, id2 = id2, id1
            pts1, pts2 = pts2, pts1

        data = {
            'ids1': id1,
            'ids2': id2,
            'pts1': pts1,
            'pts2': pts2,
            'weights': weights,
        }
        return data


