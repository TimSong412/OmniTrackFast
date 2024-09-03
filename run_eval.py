
import os
import numpy as np
import csv
import cv2
from config import config_parser
import torch
import glob
import pandas as pd
import imageio
import time
from pathlib import Path
import shutil
from setup_trainer import setup_trainer
from eval import eval_one_step



if __name__ == '__main__':
    
    args = config_parser("configs/default.txt")
    args.out_dir = "eval_out"
    data_dir = Path("dataset")

    ckptpath = Path("eval_ckpts")
    ckpt_dirs = sorted(ckptpath.glob("*"))

    AJ = []
    Delta = []
    OA = []
    TC = []

    for ckpt in ckpt_dirs:
        dataname = ckpt.stem.split("_")[-1]
        args.data_dir = (data_dir / dataname).__str__()
        args.load_dir = ckpt.__str__()

        trainer = setup_trainer(args)
        trainer.scalars_to_log = {}
        res = eval_one_step(trainer, 0)

        AJ.append(res['average_jaccard'].item())
        Delta.append(res['average_pts_within_thresh'].item())
        OA.append(res['occlusion_accuracy'].item())
        TC.append(res['temporal_coherence'])
    
    print("AJ: ", sum(AJ)/len(AJ), "Delta: ", sum(Delta)/len(Delta), "OA: ", sum(OA)/len(OA), "TC: ", sum(TC)/len(TC))


    