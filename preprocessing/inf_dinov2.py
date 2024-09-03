import torch
import cv2
import numpy as np
from pathlib import Path
import torchvision.transforms as T
import random
import tqdm
from typing import List
import glob
import torch.nn.functional as f
from typing import Sequence
from sklearn.decomposition import PCA
import argparse


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> T.Normalize:
    return T.Normalize(mean=mean, std=std)

class PCAvis():
    def __init__(self, samples) -> None:
        self.samples = samples
        self.pca = PCA(n_components=3)
        self.pca.fit(samples)
    def transform(self, samples):
        return self.pca.transform(samples)
        


def get_featmap(basedir):
    dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
    dinov2_vitl14.eval()  
    # specify the image size (h, w) here
    h = 480
    w = 854
    patch_h = h//14
    patch_w = w//14
    transform = T.Compose([
            T.GaussianBlur(9, sigma=(0.1, 2.0)),
            T.Resize((patch_h * 14, patch_w * 14)),
            # T.CenterCrop((patch_h*14, patch_w*14)),
            make_normalize_transform(),
        ])
    
    color_dir = basedir / "color"
    # print("color dir: ", color_dir)
    color_list = sorted(list(color_dir.glob("*.jpg")))
    feat_dir = basedir / "dinov2" / "featmap"
    vis_dir = basedir / "dinov2" / "vis"
    feat_dir.mkdir(exist_ok=True, parents=True)
    vis_dir.mkdir(exist_ok=True, parents=True)
    all_feat = []
    for imgfile in tqdm.tqdm(color_list):
        img = cv2.imread(imgfile.__str__())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
        img = transform(img)
        with torch.no_grad():
            features_dict0 = dinov2_vitl14.forward_features(img)
        feat = features_dict0["x_norm_patchtokens"]
        feat = feat.reshape((1, patch_h, patch_w, -1))
        feat = feat.cpu().numpy()
        feat = feat.squeeze()
        np.savez_compressed(feat_dir / (imgfile.stem + ".npz"), featmap = feat)
        all_feat.append(feat)
    samples = np.concatenate(all_feat[:10]).reshape(-1, all_feat[0].shape[-1])
    
    vis = PCAvis(samples)
    sample_trasformed = vis.transform(samples)
    sample_min = sample_trasformed.min(axis=0)
    sample_max = sample_trasformed.max(axis=0)

    for imgfile in tqdm.tqdm(color_list):
        feat = np.load(feat_dir / (imgfile.stem + ".npz"))["featmap"]
        feat = vis.transform(feat.reshape(-1, feat.shape[-1])).reshape(*feat.shape[:-1], 3)
        feat = (feat - sample_min) / (sample_max - sample_min)
        feat = np.clip(feat, 0, 1)
        feat = (feat * 255).astype(np.uint8)
        feat = cv2.resize(feat, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(vis_dir / (imgfile.stem + ".png")), feat)
        

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='dataset dir')
    args = parser.parse_args()

    basedirs = [Path(args.data_dir)]
    for basedir in sorted(basedirs):
        print("inference dinov2 for", basedir.stem)
        get_featmap(basedir)