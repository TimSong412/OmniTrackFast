import os
import subprocess
import random
import datetime
import shutil
import numpy as np
import torch
import torch.utils.data
import torch.distributed as dist
from config import config_parser
from tensorboardX import SummaryWriter
from loaders.create_training_dataset import get_training_dataset
import time
# from trainer_tcnn import TcnnTrainer

import matplotlib
import imageio


from wis3d import Wis3D

from pathlib import Path
import tqdm

import cv2

import matplotlib.pyplot as plt

from setup_trainer import setup_trainer


def warp_frame_pts(from_id, to_id, trainer, return_canonical=True):
    grid0 = trainer.grid[..., 0:2].clone()
    depth0 = trainer.get_pred_depth_maps([from_id])[0]
    depth0 = depth0[..., None]
    pts0 = trainer.unproject(grid0, depth0)
    pts0 = pts0.reshape(-1, 3)[None, :, None]
    pts1_ptscano = trainer.get_predictions(pts0, [from_id], [to_id], return_canonical=return_canonical)
    if return_canonical:
        pts1, pts0_canonical = pts1_ptscano
        return pts1.reshape(-1, 3), pts0_canonical.reshape(-1, 3)
    else:
        pts1 = pts1_ptscano
        return pts1.reshape(-1, 3)


def vis(args, gap=5):
    seq_name = os.path.basename(args.data_dir.rstrip('/'))
    now = time.strftime("%y%m%d-%H%M", time.localtime())
    out_dir = os.path.join(args.save_dir, '{}_exhaust{}_{}'.format(now, args.expname, seq_name))
    os.makedirs(out_dir, exist_ok=True)
    # if 'depth' in args.trainer:
    #     seq_name += "_depth"
    print('visualize for {}...\n output is saved in {}'.format(seq_name, out_dir))
    
    args.out_dir = out_dir

    wis3d = Wis3D("viscanonical", 'exhaust_'+args.expname+seq_name, "xyz")
    print("wis3d dir: ", 'exhaust_'+args.expname+seq_name)

    print("vis name", args.expname+'_'+seq_name)

    print('=> will save pts to {}'.format(args.expname+'_'+seq_name))

    # save the args and config files
    f = os.path.join(out_dir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            if not arg.startswith('_'):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))

    if args.config:
        f = os.path.join(out_dir, 'config.txt')
        if not os.path.isfile(f):
            shutil.copy(args.config, f)
   
    per_frame = True
    # get trainer
    trainer = setup_trainer(args, eval=True)

    N_frames = len(trainer.images)
   
    with torch.no_grad():
        for st in tqdm.trange(0, N_frames, gap):
            
            color = trainer.images[st].reshape(-1, 3).cpu().numpy()
            mask = trainer.masks[st].reshape(-1).cpu().numpy()  
            mask_color = plt.cm.hsv(np.linspace(0, 1, mask.sum()))[:, :3]
            local_depth = trainer.get_init_depth_maps([st])[0]
            local_pts = trainer.unproject(trainer.grid[..., 0:2], local_depth[..., None])
            wis3d.set_scene_id(st)
            wis3d.add_point_cloud(local_pts.reshape(-1, 3), color, name=f"local_pts")
            
            for to in tqdm.trange(0, N_frames, gap):
                try:
                    wis3d.set_scene_id(to)
                                
                    pts = warp_frame_pts(st, to, trainer, return_canonical=False)
                    wis3d.add_point_cloud(pts, color, name=f"pts{st:03d}")
                    mask_pts = pts[mask]
                    mask_pts[..., 2] -= 0.001
                    wis3d.add_point_cloud(mask_pts, mask_color, name=f"mask{st:03d}")
                except:
                    print(f"error in {st} to {to}")
                    
            wis3d.set_scene_id(N_frames)
            _, pts0_canonical = warp_frame_pts(st, 0, trainer, return_canonical=True)
            wis3d.add_point_cloud(pts0_canonical, color, name=f"pts{st:03d}_canonical")
            mask_pts = pts0_canonical[mask]
            mask_pts[..., 2] -= 0.001
            wis3d.add_point_cloud(mask_pts, mask_color, name=f"mask{st:03d}_canonical")


def vis_canonical(args, gap=10):
    seq_name = os.path.basename(args.data_dir.rstrip('/'))
    now = time.strftime("%y%m%d-%H%M", time.localtime())
    out_dir = os.path.join(args.save_dir, '{}_exhaust{}_{}'.format(now, args.expname, seq_name))
    os.makedirs(out_dir, exist_ok=True)
    # if 'depth' in args.trainer:
    #     seq_name += "_depth"
    print('visualize for {}...\n output is saved in {}'.format(seq_name, out_dir))
    
    args.out_dir = out_dir

    wis3d = Wis3D("viscanonical", 'exhaust_'+args.expname+seq_name, "xyz")
    print("wis3d dir: ", 'exhaust_'+args.expname+seq_name)

    print("vis name", args.expname+'_'+seq_name)

    print('=> will save pts to {}'.format(args.expname+'_'+seq_name))

    # save the args and config files
    f = os.path.join(out_dir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            if not arg.startswith('_'):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))

    if args.config:
        f = os.path.join(out_dir, 'config.txt')
        if not os.path.isfile(f):
            shutil.copy(args.config, f)
   
    per_frame = True
    # get trainer
    trainer = setup_trainer(args, eval=True)

    N_frames = len(trainer.images)

    with torch.no_grad():
        for st in tqdm.trange(0, N_frames, gap):
            
            color = trainer.images[st].reshape(-1, 3).cpu().numpy()
            mask = trainer.masks[st].reshape(-1).cpu().numpy()  
            mask_color = plt.cm.hsv(np.linspace(0, 1, mask.sum()))[:, :3]
            local_depth = trainer.get_init_depth_maps([st])[0]
            local_pts = trainer.unproject(trainer.grid[..., 0:2], local_depth[..., None])
            
            _, pts0_canonical = warp_frame_pts(st, 0, trainer, return_canonical=True)
            pts0_canonical[..., 2] *= 0.1
            pts0_canonical[..., 0] *= 10
            wis3d.add_point_cloud(pts0_canonical, color, name=f"pts{st:03d}_canonical")
            
   


        

if __name__ == '__main__':
    args = config_parser()

    
    # vis_gtdepth(Path("dataset/soapbox"))
    # vis_flow(args)
    args.save_dir = "vis_out"
    # vis(args, gap=10)
    vis_canonical(args, gap=5)

    # vis_eval(args)
    # visRT(args)
    # vis(args)
    # vis_couple(args, frameid = 30)
