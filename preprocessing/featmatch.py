from pathlib import Path
import numpy as np
import torch    
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm
import imageio
import pdb
import argparse

def match_feat(query_feat, all_feat):
    '''
    query_feat: np.ndarray, (d)
    all_feat: np.ndarray, (H, W, d)
    '''
    query_feat = query_feat / torch.norm(query_feat)
    all_feat = all_feat / torch.norm(all_feat, dim=2)[..., None]
    sim = torch.sum(query_feat * all_feat, dim=2)
    return torch.where(sim == sim.max()), sim

def count_sim_area(query_feat, all_feat, threshold=0.5):
    '''
    query_feat: np.ndarray, (d)
    all_feat: np.ndarray, (H, W, d)
    '''
    query_feat = query_feat / torch.norm(query_feat)
    all_feat = all_feat / torch.norm(all_feat, dim=2)[..., None]
    sim = torch.sum(query_feat * all_feat, dim=2)
    return (sim > threshold).sum()

def count_local_sim(x, y, featmap):
    '''
    x: int, 
    y: int, 
    featmap: np.ndarray, (H, W, d)
    '''
    query_feat = featmap[y, x]
    l = max(0, x-5)
    r = min(featmap.shape[1]-1, x+5)
    t = max(0, y-5)
    b = min(featmap.shape[0]-1, y+5)
    local_feat = featmap[t:b, l:r]
    query_feat = query_feat / torch.norm(query_feat)
    local_feat = local_feat / torch.norm(local_feat, dim=2)[..., None]
    sim = torch.sum(query_feat * local_feat)
    return sim

def sparse_filter(matchid_0_xy, matchid_1_xy, simvalue, mindist=4):
    if len(matchid_0_xy) < 4 or len(matchid_1_xy) < 4:
        return torch.empty((0, 2)), torch.empty((0, 2)), torch.empty((0))
    mindist0 = torch.zeros(len(matchid_0_xy), len(matchid_0_xy))
    mindist1 = torch.zeros(len(matchid_1_xy), len(matchid_0_xy))
    for i in range(len(matchid_0_xy)):
        mindist0[i] = ((matchid_0_xy - matchid_0_xy[i])**2).sum(dim=1).sqrt()
        mindist1[i] = ((matchid_1_xy - matchid_1_xy[i])**2).sum(dim=1).sqrt()
    mindist0[mindist0 == 0] = 1e10
    mindist1[mindist1 == 0] = 1e10
    # mindist0 = mindist0.min(dim=1).values
    # mindist1 = mindist1.min(dim=1).values
    
    try:
        mindist0 = torch.topk(mindist0, 2, dim=1, largest=False).values[:, :2]
        mindist1 = torch.topk(mindist1, 2, dim=1, largest=False).values[:, :2]
    except:
        pdb.set_trace()

    # print(mindist0.max())
    # print(mindist1.max())

    mindist_mask = ((mindist0 < mindist).sum(dim=1)==2) & ((mindist1 < mindist).sum(dim=1)==2)
    matchid_0_xy = matchid_0_xy[mindist_mask]
    matchid_1_xy = matchid_1_xy[mindist_mask]
    simvalue = simvalue[mindist_mask]
    return matchid_0_xy, matchid_1_xy, simvalue

# bmx-trees: area thresh = 100, self sim 0.5
def matchpair(featmap0, featmap1, match_sim_th=0.75, self_sim_th=0.55, area_threshold=100, local_min=30, local_max=100, mindist=4):
    '''
    feat0: np.ndarray, (H, W, d)
    feat1: np.ndarray, (H, W, d)
    '''
    h, w, d = featmap0.shape
    featmap0 = torch.tensor(featmap0).cuda()
    featmap1 = torch.tensor(featmap1).cuda()
    feat0 = featmap0.reshape(-1, d)
    feat1 = featmap1.reshape(-1, d)
    feat0 = feat0 / torch.norm(feat0, dim=1)[..., None]
    feat1 = feat1 / torch.norm(feat1, dim=1)[..., None]

    sim_matrix = feat0 @ (feat1.T)
    maxid_0 = sim_matrix.argmax(dim=1)
    sim_1 = sim_matrix.max(dim=0).values
    maxid_1 = sim_matrix.argmax(dim=0)
    matchedmask1 = (maxid_0[maxid_1] == torch.arange(len(maxid_1)).cuda())
    matchid_0 = maxid_1[matchedmask1]
    matchid_1 = torch.arange(len(maxid_1)).cuda()[matchedmask1]

    simvalue = sim_1[matchedmask1]
    simmask = simvalue > match_sim_th
    matchid_0 = matchid_0[simmask]
    matchid_1 = matchid_1[simmask]
    simvalue = simvalue[simmask]

    matched_feat0 = feat0[matchid_0]
    matched_feat1 = feat1[matchid_1]

    selfsim_feat0 = ((matched_feat0 @ (feat0.T)) > self_sim_th).sum(dim=1)
    selfsim_feat1 = ((matched_feat1 @ (feat1.T)) > self_sim_th).sum(dim=1)

    area_mask = (selfsim_feat0 < area_threshold) & (selfsim_feat1 < area_threshold)
    matchid_0 = matchid_0[area_mask]
    matchid_1 = matchid_1[area_mask]
    simvalue = simvalue[area_mask]

    matchid_0_xy = torch.stack([matchid_0 % w, matchid_0 // w], dim=1)
    matchid_1_xy = torch.stack([matchid_1 % w, matchid_1 // w], dim=1)

    local_sim0 = torch.tensor([count_local_sim(x, y, featmap0) for x, y in matchid_0_xy])
    local_sim1 = torch.tensor([count_local_sim(x, y, featmap1) for x, y in matchid_1_xy])
    local_mask = (local_sim0 > local_min) & (local_sim0 < local_max) & (local_sim1 > local_min) & (local_sim1 < local_max)

    matchid_0_xy = matchid_0_xy[local_mask]
    matchid_1_xy = matchid_1_xy[local_mask]
    simvalue = simvalue[local_mask]

   
    
    matchid_0_xy, matchid_1_xy, simvalue = sparse_filter(matchid_0_xy, matchid_1_xy, simvalue, mindist=mindist)

    matchid_0_xy, matchid_1_xy, simvalue = sparse_filter(matchid_0_xy, matchid_1_xy, simvalue, mindist=mindist)

    return matchid_0_xy, matchid_1_xy, simvalue

    
def matchframe(basedir:Path, f0id, f1id, out_dir = Path("matchmaps")):
    color_dir = basedir / "color"
    feat_dir = basedir / "dinov2" / "featmap"
    depthmap_dir = basedir / "raw_depth" / "mask"
    
    # out_dir.mkdir(exist_ok=True, parents=True)

    color0 = cv2.imread((color_dir / f"{f0id:05d}.jpg").__str__())
    color1 = cv2.imread((color_dir / f"{f1id:05d}.jpg").__str__())
    feat0 = np.load(feat_dir / f"{f0id:05d}.npz")["featmap"]
    feat1 = np.load(feat_dir / f"{f1id:05d}.npz")["featmap"]

    h, w = color0.shape[:2]
    ph, pw = feat0.shape[:2]

    mask0 = cv2.imread((depthmap_dir / f"{f0id:05d}.jpg").__str__(), cv2.IMREAD_GRAYSCALE)
    mask1 = cv2.imread((depthmap_dir / f"{f1id:05d}.jpg").__str__(), cv2.IMREAD_GRAYSCALE)
    coarse_mask0 = cv2.GaussianBlur(mask0, (3, 3), 0)
    coarse_mask1 = cv2.GaussianBlur(mask1, (3, 3), 0)
    coarse_mask0[coarse_mask0<255] = 0
    coarse_mask1[coarse_mask1<255] = 0  

    color0 = color0 * coarse_mask0[..., None].astype(np.float32) / 255
    color1 = color1 * coarse_mask1[..., None].astype(np.float32) / 255

    color0 = color0.astype(np.uint8)
    color1 = color1.astype(np.uint8)
    
    coarse_mask0 = coarse_mask0 > 0
    coarse_mask1 = coarse_mask1 > 0



    grid = torch.meshgrid(torch.arange(0, ph, 1), torch.arange(0, pw, 1))
    grid = torch.stack(grid, 2).reshape(-1, 2)
    native_grid = grid.clone()
    native_grid = native_grid.flip(1)
    grid = torch.tensor(grid).float()
    grid = grid + 0.5
    grid = grid.flip(1)
    grid = (grid * torch.tensor([w/pw, h/ph])).int()

    matchxy_0, matchxy_1, simvalue = matchpair(feat0, feat1)
    matchxy_0 = matchxy_0.cpu()
    matchxy_1 = matchxy_1.cpu()
    simvalue = simvalue.cpu()

    matchxy_0 = matchxy_0 + 0.5
    matchxy_1 = matchxy_1 + 0.5
    matchxy_0 = matchxy_0 / torch.tensor([pw, ph]).float() * torch.tensor([w, h]).float()
    matchxy_1 = matchxy_1 / torch.tensor([pw, ph]).float() * torch.tensor([w, h]).float()

    coarse_mask0 = torch.from_numpy(coarse_mask0)
    coarse_mask1 = torch.from_numpy(coarse_mask1)

    validpts_0 = coarse_mask0[torch.clamp(matchxy_0[:, 1].int(), 0, h-1), torch.clamp(matchxy_0[:, 0].int(), 0, w-1)]
    validpts_1 = coarse_mask1[torch.clamp(matchxy_1[:, 1].int(), 0, h-1), torch.clamp(matchxy_1[:, 0].int(), 0, w-1)]
    validpts = validpts_0 & validpts_1
    matchxy_0 = matchxy_0[validpts]
    matchxy_1 = matchxy_1[validpts]
    simvalue = simvalue[validpts]

    matchxy_0, matchxy_1, simvalue = sparse_filter(matchxy_0, matchxy_1, simvalue, mindist=32)

    id0s = torch.tensor([f0id]*len(matchxy_0))
    id1s = torch.tensor([f1id]*len(matchxy_1))
    id0xy0id1xy1sim = torch.cat([id0s[:, None], matchxy_0, id1s[:, None], matchxy_1, simvalue[:, None]], dim=1)

    colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(matchxy_0)))[..., :3]

    for i in range(len(matchxy_0)):
        xy0 = matchxy_0[i]
        xy1 = matchxy_1[i]
        xy0 = xy0.int()
        xy1 = xy1.int()
        cv2.circle(color0, (xy0[0].item(), xy0[1].item()), 5, (255*colors[i]).astype(np.uint8).tolist(), -1)
        cv2.circle(color1, (xy1[0].item(), xy1[1].item()), 5, (255*colors[i]).astype(np.uint8).tolist(), -1)
        cv2.putText(color1, f"{simvalue[i]:.2f}", (xy1[0].item(), xy1[1].item()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255*colors[i]).astype(np.uint8).tolist(), 1, cv2.LINE_AA)

    full_image = np.concatenate([color0, color1], axis=0)
    # cv2.imwrite((out_dir/f"match_{f0id:05d}_{f1id:05d}.png").__str__(), full_image)
    return full_image, id0xy0id1xy1sim


def match(basedir:Path):
    color_dir = basedir / "color"
    feat_dir = basedir / "dinov2" / "featmap"
    out_dir = Path("matchmaps")
    out_dir.mkdir(exist_ok=True, parents=True)



    color0 = cv2.imread((color_dir / "00010.jpg").__str__())
    color1 = cv2.imread((color_dir / "00025.jpg").__str__())
    feat0 = np.load(feat_dir / "00010.npz")["featmap"]
    feat1 = np.load(feat_dir / "00025.npz")["featmap"]

    h, w = color0.shape[:2]
    ph, pw = feat0.shape[:2]

    grid = torch.meshgrid(torch.arange(0, ph, 1), torch.arange(0, pw, 1))
    grid = torch.stack(grid, 2).reshape(-1, 2)
    native_grid = grid.clone()
    native_grid = native_grid.flip(1)
    grid = torch.tensor(grid).float()
    grid = grid + 0.5
    grid = grid.flip(1)
    grid = (grid * torch.tensor([w/pw, h/ph])).int()

    feat0 = torch.tensor(feat0)
    feat1 = torch.tensor(feat1)
    query_feat = feat0.reshape(-1, feat0.shape[2])
   

    
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(query_feat)))[..., :3]
    
    origin_color1 = color1.copy()
    for i in tqdm.trange(len(query_feat)):

        local_sim_0 = count_local_sim(native_grid[i, 0], native_grid[i, 1], feat0)
        if local_sim_0 < 30 or local_sim_0 > 100:
            print(f"local_sim_0: {local_sim_0}")
            continue
        
        sim_area00 = count_sim_area(query_feat[i], feat0)
        # sim_area01 = count_sim_area(query_feat[i], feat1)
        
        if sim_area00 > 100:
            print(f"sim_area00: {sim_area00}")
            continue
        
        (y, x), matchmap = match_feat(query_feat[i], feat1)

        target_feat = feat1[y, x]

        (back_y, back_x), _ = match_feat(target_feat, feat0)

        if back_x != native_grid[i, 0] or back_y != native_grid[i, 1]:
            print(f"back_x: {back_x}, back_y: {back_y}")
            continue

        sim_area11 = count_sim_area(target_feat, feat1)
        # sim_area10 = count_sim_area(target_feat, feat0)
        if sim_area11 > 100:
            print(f"sim_area11: {sim_area11}")
            continue
        
        local_sim_1 = count_local_sim(x, y, feat1)

        if local_sim_1 < 30 or local_sim_1 > 100:
            print(f"local_sim_0: {local_sim_0}")
            continue

        

        max_sim = matchmap.max().item()
        if max_sim < 0.8:
            continue

        cv2.circle(color0, (grid[i, 0].item(), grid[i, 1].item()), 5, (255*colors[i]).astype(np.uint8).tolist(), -1)

        y = y[0].item()
        x = x[0].item()
        y = y / feat1.shape[0] * h
        x = x / feat1.shape[1] * w
        
        cv2.circle(color1, (int(x), int(y)), 5, (255*colors[i]).astype(np.uint8).tolist(), -1)
        cv2.putText(color1, f"{local_sim_1:.2f}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255*colors[i]).astype(np.uint8).tolist(), 1, cv2.LINE_AA)
        mergemap = cv2.circle(origin_color1.copy(), (int(x), int(y)), 5, (255*colors[i]).astype(np.uint8).tolist(), -1)
        matchmap = torch.clamp(matchmap, 0, 1).cpu().numpy()
        matchmap = np.ones((*matchmap.shape, 3))*255*colors[i]*matchmap[..., None]
        matchmap = matchmap.astype(np.uint8)
        matchmap = cv2.resize(matchmap, (w, h), interpolation=cv2.INTER_LINEAR)
        matchmap = 0.5*matchmap + 0.5*mergemap
        matchmap = matchmap.astype(np.uint8)
        cv2.imwrite((out_dir/f"match_{grid[i, 1]:03d}_{grid[i, 0]:03d}.png").__str__(), matchmap)

    cv2.imwrite("color0.png", color0)
    cv2.imwrite("color1.png", color1)

def eval_feat(px0, featmap0, featmap1):
    '''
    px0: torch.tensor, (N, 2), in[-1, 1]
    featmap0: torch.tensor, (H, W, d)
    featmap1: torch.tensor, (H, W, d)
    '''
    result = torch.zeros_like(px0)
    H, W, d = featmap0.shape
    ph, pw = featmap0.shape[:2]
    grid = torch.meshgrid(torch.arange(0, ph, 1), torch.arange(0, pw, 1))
    grid = torch.stack(grid, 2).reshape(-1, 2)
    native_grid = grid.clone()
    native_grid = native_grid.flip(1)
    grid = torch.tensor(grid).float()
    grid = grid + 0.5
    grid = grid.flip(1)

    pxfeat0 = F.grid_sample(featmap0[None].permute(0, 3, 1, 2), px0[None, None], align_corners=True).squeeze().permute(1, 0)
    pxfeat0 /= torch.norm(pxfeat0, dim=1)[..., None]

    feat0 = featmap0.reshape(-1, d)
    feat1 = featmap1.reshape(-1, d)
    feat0 = feat0 / torch.norm(feat0, dim=1)[..., None]
    feat1 = feat1 / torch.norm(feat1, dim=1)[..., None]

    sim_matrix = feat0 @ (feat1.T)

    pxsim_matrix = pxfeat0 @ (feat1.T)

    maxid_0 = sim_matrix.argmax(dim=1)
    sim_0 = sim_matrix.max(dim=1).values
    maxid_1 = sim_matrix.argmax(dim=0)
    sim_1 = sim_matrix.max(dim=0).values

    sim_px_1 = pxsim_matrix.max(dim=1).values
    sim_px_1_id = pxsim_matrix.argmax(dim=1)

    sim_1_0 = sim_1[sim_px_1_id]

    valid_px = sim_1_0 < sim_px_1

    result[~valid_px] = np.zeros(2)
    
    sim_px_1_id = sim_px_1_id[valid_px]
    valid_res = torch.zeros(len(sim_px_1_id), 2)
    for i, id in enumerate(sim_px_1_id):
        feat_px_1 = feat1[id]
        self_sim_1 = sum((feat_px_1 @ feat1.T) > 0.5)
        if self_sim_1 > 100:
            continue
        x = id % W
        y = id // W
        local_sim_1 = count_local_sim(x, y, featmap1)
        if local_sim_1 < 30 or local_sim_1 > 100:
            continue
        x = x / W
        y = y / H
        valid_res[i] = (x, y)

    result[valid_px] = valid_res
    
    return result

def eval_match(basedir:Path):
    color_dir = basedir / "color"
    feat_dir = basedir / "dinov2" / "featmap"
    out_dir = Path("matchmaps")
    out_dir.mkdir(exist_ok=True, parents=True)
    dataset_name = 'davis'
    annotation_file = Path("dataset/tapvid_davis_256/annotations") / f"{ basedir.stem}.pkl"

    h = 480
    w = 854

    inputs = np.load(annotation_file, allow_pickle=True)

    query_points = inputs[dataset_name]['query_points']
    target_points = inputs[dataset_name]['target_points']
    gt_occluded = inputs[dataset_name]['occluded']

    one_hot_eye = np.eye(target_points.shape[2])
    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = one_hot_eye[query_frame] == 0

    query_points *= (1, h / 256., w / 256.)
    ids1 = query_points[0, :, 0].astype(int)
    px1s = torch.from_numpy(query_points[:, :, [2, 1]]).transpose(0, 1).float().cuda()

    results = np.zeros(target_points.shape)
    for id1 in range(target_points.shape[2]):
        print(id1)
        
        color1 = cv2.imread((color_dir / f"{id1:05d}.jpg").__str__())
        feat1 = np.load(feat_dir / f"{id1:05d}.npz")["featmap"]
        feat1 = torch.tensor(feat1).cuda()
        all_eval = []
        for id0 in range(target_points.shape[2]):
            feat0 = np.load(feat_dir / f"{id0:05d}.npz")["featmap"]
            feat0 = torch.tensor(feat0).cuda()
            px = px1s[ids1 == id0].squeeze()
            normed_px = px / torch.tensor([w, h]).float().cuda() * 2 - 1.
            matched_pts = eval_feat(normed_px, feat0, feat1)
            all_eval.append(matched_pts)
        all_eval = torch.cat(all_eval, dim=0)
        results[:, :, id1] = all_eval.cpu().numpy()
        

   


        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='dataset dir')
    args = parser.parse_args()

    vid_outdir = Path("matchvids")
    vid_outdir.mkdir(exist_ok=True, parents=True)
    basedirs = [Path(args.data_dir)]

    for basedir in sorted(basedirs):
       
        print("inference", basedir.stem)
        matchdir = basedir / "match"
        colordir = basedir / "color"
        N_frames = len(list(colordir.glob("*.jpg")))
        matchdir.mkdir(exist_ok=True, parents=True)
        
        for st in tqdm.trange(N_frames):
            all_frames = []
            matchs = []
            for i in tqdm.trange(st, N_frames):   
                
                if abs(i-st) < 10:
                    continue
                frame, matchi = matchframe(basedir, st, i)
                all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                matchs.append(matchi)
            if len(matchs) == 0:
                continue
            matchi = torch.cat(matchs, dim=0)
            np.savez_compressed(matchdir / f"match{st:05d}.npz", match=matchi.cpu().numpy())
            if st % 10 == 0:
                imageio.mimsave(vid_outdir/f"{basedir.stem}_match{st:05d}.mp4", all_frames, fps=10, quality=5)