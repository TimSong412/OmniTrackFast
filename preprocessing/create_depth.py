import torch
from wis3d import Wis3D
# Local file
from PIL import Image
import numpy as np
import sys
from pathlib import Path
import open3d as o3d
import cv2
import tqdm
import subprocess
import imageio
import argparse


def genmask_simple(basedir:Path):
    raft_mask_dir = basedir / 'raft_masks'
    depth_mask_dir = basedir / 'raw_depth' / 'mask'
    outdir = basedir / 'full_mask'
    outdir.mkdir(exist_ok=True)
    raftmasks = sorted(raft_mask_dir.glob('*.png'))

    for raftmaskfile in tqdm.tqdm(raftmasks):
        raftmask = imageio.imread(raftmaskfile) / 255.
        
        
        imgname0 = raftmaskfile.name[:-4].split('_')[0]
        imgname1 = raftmaskfile.name[:-4].split('_')[1]
        # if abs(int(imgname0[:-4]) - int(imgname1[:-4])) > 20:
        #     continue
        depthmask0 = imageio.imread(depth_mask_dir / (imgname0)) / 255.
        depthmask1 = imageio.imread(depth_mask_dir / (imgname1)) / 255.
        depthmask0 = np.round(depthmask0)
        depthmask1 = np.round(depthmask1)

        cycle_consistency_mask = raftmask[..., 0] > 0
        occlusion_mask = raftmask[..., 1] > 0
        depth_mask = (depthmask0 > 0)# & (warpped_mask2 > 0)
        final_mask = (cycle_consistency_mask | occlusion_mask) & depth_mask
        final_mask = final_mask.astype(np.uint8) * 255
        imageio.imwrite(outdir / raftmaskfile.name, final_mask)




def gen_depth_mask(basedir:Path):
    repo = "isl-org/ZoeDepth"
    model_zoe_n = torch.hub.load(repo, "ZoeD_K", pretrained=True)
    color_dir = basedir / "color"
    depth_dir = basedir / "raw_depth"
    depth_save_dir = depth_dir / "depth"
    mask_save_dir = depth_dir / "mask"

    depth_save_dir.mkdir(parents=True, exist_ok=True)
    mask_save_dir.mkdir(parents=True, exist_ok=True)

    rgbfiles = sorted(list(color_dir.glob("*.jpg")))

    all_rgbs = []
    all_depths = []
    all_masks = []

    seqname = basedir.stem
    
    # Zoe_N

    wis3d = Wis3D("./visdavis", f"vis_zoe-{seqname}", "xyz")

    

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_n.to(DEVICE)

    image = Image.open(rgbfiles[0]).convert("RGB")
    w, h = image.size
    print("image size: ", w, h)

    # w = 256
    # h = 256
    grid = np.meshgrid(np.arange(w), np.arange(h))
    grid = np.stack(grid, axis=-1).astype(np.float32)
    # grid[..., 0] /= float(w)
    # grid[..., 1] /= float(h)
    grid[..., 0] -= w / 2.0
    grid[..., 1] -= h / 2.0

    zoe.eval()
    fov = 40 # deg
    f = w / (2 * np.tan(fov / 2 / 180 * np.pi))
    # f = 1000
    print("focal length: ", f)
        
    print("Start inference")
    for rgbfile in tqdm.tqdm(rgbfiles):
        
        image = Image.open(rgbfile).convert("RGB")  # load
        # if image.size != (w, h):
        #     print("resize from ", image.size, " to ", (w, h))
        #     image = image.resize((w, h))
        #     image.save(rgbfile)
        w, h = image.size
        depth = zoe.infer_pil(image)  # as numpy
        all_rgbs.append(image)
        all_depths.append(depth)

        xyz = np.concatenate([grid, depth[..., None]], axis=-1)
        
        xyz = xyz.reshape(-1, 3)

        xyz[..., 0:2] /= f
        xyz[..., 0:2] *= xyz[..., 2:3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=2.)
        mask = np.zeros_like(depth)
        mask = mask.reshape(-1)
        mask[ind] = 255
        mask = mask.reshape(depth.shape)
        mask = mask.astype(np.uint8)
        all_masks.append(mask)
        cv2.imwrite((mask_save_dir / rgbfile.name).__str__(), mask)
        
    print("Start saving")
    all_depths = np.stack(all_depths, axis=0)
    print("max depth: ", all_depths.max())
    print("min depth: ", all_depths.min())
    print("mean depth: ", all_depths.mean())
    # sacle = 2.0 / all_depths.max()
    # all_depths *= sacle
    for i, img in  enumerate(tqdm.tqdm(all_rgbs)):
        wis3d.set_scene_id(i)
        depth = all_depths[i]

        np.savez_compressed((depth_save_dir / (rgbfiles[i].stem)).__str__(), depth=depth)


    
def gen_configfile(basefile, datalist):
    with open(basefile, "r") as f:
        lines = f.readlines()
    basename = lines[1].split("/")[-1]
    basename = basename.split()[0]

    for dataname in datalist:
        line_out = lines.copy()
        outfile = basefile.split("_")[0:-1] + [dataname+".txt"]
        outfile = "_".join(outfile)
        outfile = Path(outfile)
        outfile = Path("DGX_configs/RGB_stack/run_4549") / outfile.name
        line_out[1] = line_out[1].replace(basename, dataname)
        with open(outfile, "w") as f:
            f.writelines(line_out)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='dataset dir')
    args = parser.parse_args()
    tasklist = [Path(args.data_dir)]
    
    for datadir in tasklist:
        print(f"processing {datadir}")
        gen_depth_mask(datadir)
        genmask_simple(datadir)