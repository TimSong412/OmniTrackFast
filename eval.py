
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


def eval_temporal_coherence(pred_tracks, gt_tracks, gt_occluded, pred_occluded=None):
    '''
    :param pred_tracks: [1, n_pts, n_imgs, 2]
    :param gt_tracks: [1, n_pts, n_imgs, 2]
    :param gt_occluded: [1, n_pts, n_imgs] bool
    :return:
    '''
    pred_flow_01 = pred_tracks[..., 1:-1, :] - pred_tracks[..., :-2, :]
    pred_flow_12 = pred_tracks[..., 2:, :] - pred_tracks[..., 1:-1, :]
    gt_occluded_3 = gt_occluded[..., :-2] | gt_occluded[..., 1:-1] | gt_occluded[..., 2:]
    gt_visible_3 = ~gt_occluded_3
    if pred_occluded is not None:
        pred_occluded_3 = pred_occluded[..., :-2] | pred_occluded[..., 1:-1] | pred_occluded[..., 2:]
        pred_visible = ~pred_occluded_3
        gt_visible_3 = gt_visible_3 & pred_visible

    # difference in acceleration
    gt_flow_01 = gt_tracks[..., 1:-1, :] - gt_tracks[..., :-2, :]
    gt_flow_12 = gt_tracks[..., 2:, :] - gt_tracks[..., 1:-1, :]
    flow_diff = np.linalg.norm(pred_flow_12 - pred_flow_01 - (gt_flow_12 - gt_flow_01), axis=-1)
    if np.sum(gt_visible_3) == 0:
        return 0
    error = flow_diff[gt_visible_3].sum() / (gt_visible_3).sum()
    return error


def eval_one_step(trainer, step, occlusion_th=0.99, depth_err=0.02):
    print(f"eval_step: {step}")

    out_dir = os.path.join(trainer.out_dir, 'eval')
    seq_name = trainer.seq_name
    # need to keep the same seed as training otherwise will lead to issues when doing things like for-loops 
    # model = BaseTrainer(args)

    os.makedirs(out_dir, exist_ok=True)
    # annotation_file = '{}/{}.pkl'.format(annotation_dir, seq_name)
    print('evaluating {}...'.format(seq_name))

    use_max_loc = True

    # Load tapvid data
    # inputs = np.load(annotation_file, allow_pickle=True)

    # query_points = inputs[dataset_name]['query_points']
    # target_points = inputs[dataset_name]['target_points']
    # gt_occluded = inputs[dataset_name]['occluded']

    # one_hot_eye = np.eye(target_points.shape[2])
    # query_frame = query_points[..., 0]
    # query_frame = np.round(query_frame).astype(np.int32)
    # evaluation_points = one_hot_eye[query_frame] == 0

    # query_points *= (1, model.h / 256, model.w / 256)
    # ids1 = query_points[0, :, 0].astype(int)
    # px1s = torch.from_numpy(query_points[:, :, [2, 1]]).transpose(0, 1).float().cuda()
    query_points = trainer.query_points.copy()
    target_points = trainer.target_points.copy()
    gt_occluded = trainer.gt_occluded.copy()
    evaluation_points = trainer.evaluation_points.copy()
    ids1 = trainer.eval_ids1.copy()
    px1s = trainer.eval_px1s.clone()

    results = np.zeros(target_points.shape)
    occlusions = np.zeros(gt_occluded.shape)
    with torch.no_grad():
        for i in range(gt_occluded.shape[-1]):
            results_, occlusions_ = trainer.get_correspondences_and_occlusion_masks_for_pixels(
                ids1=ids1, 
                px1s=px1s, ids2=[i for _ in range(px1s.shape[0])],
                use_max_loc=use_max_loc,
                depth_err=depth_err)
            results[:, :, i, :] = results_.transpose(0, 1).cpu().numpy()
            occlusions[:, :, i] = occlusions_.squeeze().cpu().numpy()
    target_points *= (trainer.w / 256, trainer.h / 256)

    results[..., 0] *= trainer.eval_w / trainer.w
    results[..., 1] *= trainer.eval_h / trainer.h

    target_points[..., 0] *= trainer.eval_w / trainer.w
    target_points[..., 1] *= trainer.eval_h / trainer.h


    metrics = {}

    occlusion_mask = occlusions > occlusion_th
    out_of_boundary_mask = (results[..., 0] < 0) | (results[..., 0] > trainer.w - 1) | \
                            (results[..., 1] < 0) | (results[..., 1] > trainer.h - 1)
    occlusion_mask = occlusion_mask | out_of_boundary_mask
    occ_acc = np.sum(np.equal(occlusion_mask, gt_occluded) & evaluation_points, axis=(1, 2)) / np.sum(evaluation_points)
    metrics['occlusion_accuracy'] = occ_acc

    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(occlusion_mask)
    all_frac_within = []
    all_jaccard = []

    for thresh in [1, 2, 4, 8, 16]:
        within_dist = np.sum(
            np.square(results - target_points),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points = np.sum(
            visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics['pts_within_' + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=(1, 2))
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics['jaccard_' + str(thresh)] = jaccard
        all_jaccard.append(jaccard)

    metrics['average_pts_within_thresh'] = np.mean(np.stack(all_frac_within, axis=1), axis=1)
    metrics['average_jaccard'] = np.mean(np.stack(all_jaccard, axis=1), axis=1)
    metrics['temporal_coherence'] = eval_temporal_coherence(results, target_points, gt_occluded)#, pred_occluded=occlusion_mask)

    metrics = dict(sorted(metrics.items()))

    for k, v in metrics.items():
        trainer.scalars_to_log['metric/{}'.format(k)] = v.item()
        
    print("eval finished")
    return metrics


def eval_one_step_occ(trainer, step, occlusion_th=0.99, depth_err=0.02):
    print(f"eval_step: {step}")

    out_dir = os.path.join(trainer.out_dir, 'eval')
    seq_name = trainer.seq_name
    # need to keep the same seed as training otherwise will lead to issues when doing things like for-loops 
    # model = BaseTrainer(args)

    os.makedirs(out_dir, exist_ok=True)
    # annotation_file = '{}/{}.pkl'.format(annotation_dir, seq_name)
    print('evaluating {}...'.format(seq_name))

    use_max_loc = True

    # Load tapvid data
    # inputs = np.load(annotation_file, allow_pickle=True)

    # query_points = inputs[dataset_name]['query_points']
    # target_points = inputs[dataset_name]['target_points']
    # gt_occluded = inputs[dataset_name]['occluded']

    # one_hot_eye = np.eye(target_points.shape[2])
    # query_frame = query_points[..., 0]
    # query_frame = np.round(query_frame).astype(np.int32)
    # evaluation_points = one_hot_eye[query_frame] == 0

    # query_points *= (1, model.h / 256, model.w / 256)
    # ids1 = query_points[0, :, 0].astype(int)
    # px1s = torch.from_numpy(query_points[:, :, [2, 1]]).transpose(0, 1).float().cuda()
    query_points = trainer.query_points.copy()
    target_points = trainer.target_points.copy()
    gt_occluded = trainer.gt_occluded.copy()
    evaluation_points = trainer.evaluation_points.copy()
    ids1 = trainer.eval_ids1.copy()
    px1s = trainer.eval_px1s.clone()

    results = np.zeros(target_points.shape)
    occlusions = np.zeros(gt_occluded.shape)
    feats = np.zeros((*occlusions.shape[0:3], 384))
    with torch.no_grad():
        for i in range(gt_occluded.shape[-1]):
            results_, occlusions_, feats_ = trainer.get_correspondences_and_occlusion_masks_for_pixels(
                ids1=ids1, 
                px1s=px1s, ids2=[i for _ in range(px1s.shape[0])],
                use_max_loc=use_max_loc,
                depth_err=depth_err)
            results[:, :, i, :] = results_.transpose(0, 1).cpu().numpy()
            occlusions[:, :, i] = occlusions_.squeeze().cpu().numpy()
            feats[:, :, i, :] = feats_.permute(1, 0, 2).cpu().numpy()
    target_points *= (trainer.w / 256, trainer.h / 256)

    for pt in range(feats.shape[1]):
        for f in range(feats.shape[2]):
            if occlusions[0, pt, f] > 0:
                continue
            most_recent_visible = f
            far_visible = f
            
            for vis_f in range(f-1, -1, -1):
                if occlusions[0, pt, vis_f] == 0 and most_recent_visible == f:
                    most_recent_visible = vis_f
                if occlusions[0, pt, vis_f] == 1 and most_recent_visible < f:
                    far_visible = vis_f+1
                    break
                if vis_f == 0 and most_recent_visible < f:
                    far_visible = 0

            if most_recent_visible == f:
                continue
            visible_feats = feats[0, pt, far_visible:most_recent_visible+1]
            local_feats = feats[0, pt, f]
            cos_sim = np.dot(visible_feats, local_feats) / (np.linalg.norm(visible_feats, axis=1) * np.linalg.norm(local_feats))
            mean_cos_sim = np.mean(cos_sim)
            if mean_cos_sim < 0.8:
                occlusions[0, pt, f] = 1

    results[..., 0] *= trainer.eval_w / trainer.w
    results[..., 1] *= trainer.eval_h / trainer.h

    target_points[..., 0] *= trainer.eval_w / trainer.w
    target_points[..., 1] *= trainer.eval_h / trainer.h


    metrics = {}

    occlusion_mask = occlusions > occlusion_th
    out_of_boundary_mask = (results[..., 0] < 0) | (results[..., 0] > trainer.w - 1) | \
                            (results[..., 1] < 0) | (results[..., 1] > trainer.h - 1)
    occlusion_mask = occlusion_mask | out_of_boundary_mask
    occ_acc = np.sum(np.equal(occlusion_mask, gt_occluded) & evaluation_points, axis=(1, 2)) / np.sum(evaluation_points)
    metrics['occlusion_accuracy'] = occ_acc

    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(occlusion_mask)
    all_frac_within = []
    all_jaccard = []

    for thresh in [1, 2, 4, 8, 16]:
        within_dist = np.sum(
            np.square(results - target_points),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points = np.sum(
            visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics['pts_within_' + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=(1, 2))
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics['jaccard_' + str(thresh)] = jaccard
        all_jaccard.append(jaccard)

    metrics['average_pts_within_thresh'] = np.mean(np.stack(all_frac_within, axis=1), axis=1)
    metrics['average_jaccard'] = np.mean(np.stack(all_jaccard, axis=1), axis=1)
    metrics['temporal_coherence'] = eval_temporal_coherence(results, target_points, gt_occluded)

    metrics = dict(sorted(metrics.items()))
    trainer.scalars_to_log = {}
    for k, v in metrics.items():
        trainer.scalars_to_log['metric/{}'.format(k)] = v.item()
        
    print("eval finished")
    return metrics


def eval_one_sequence(args, annotation_dir, dataset_name, seq_name, out_dir, occlusion_th=0.99):
    torch.manual_seed(1234)  # need to keep the same seed as training otherwise will lead to issues when doing things like for-loops 
    model = BaseTrainer(args)

    os.makedirs(out_dir, exist_ok=True)
    annotation_file = '{}/{}.pkl'.format(annotation_dir, seq_name)
    print('evaluating {}...'.format(seq_name))

    use_max_loc = True

    # Load tapvid data
    inputs = np.load(annotation_file, allow_pickle=True)

    query_points = inputs[dataset_name]['query_points']
    target_points = inputs[dataset_name]['target_points']
    gt_occluded = inputs[dataset_name]['occluded']

    one_hot_eye = np.eye(target_points.shape[2])
    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = one_hot_eye[query_frame] == 0

    query_points *= (1, model.h / 256, model.w / 256)
    ids1 = query_points[0, :, 0].astype(int)
    px1s = torch.from_numpy(query_points[:, :, [2, 1]]).transpose(0, 1).float().cuda()

    results = np.zeros(target_points.shape)
    occlusions = np.zeros(gt_occluded.shape)
    with torch.no_grad():
        for i in range(gt_occluded.shape[-1]):
            results_, occlusions_ = model.get_correspondences_and_occlusion_masks_for_pixels(
                ids1=ids1, 
                px1s=px1s, ids2=[i for _ in range(px1s.shape[0])],
                use_max_loc=use_max_loc)
            results[:, :, i, :] = results_.transpose(0, 1).cpu().numpy()
            occlusions[:, :, i] = occlusions_.squeeze().cpu().numpy()
    target_points *= (model.w / 256, model.h / 256)

    metrics = {}

    occlusion_mask = occlusions > occlusion_th
    out_of_boundary_mask = (results[..., 0] < 0) | (results[..., 0] > model.w - 1) | \
                            (results[..., 1] < 0) | (results[..., 1] > model.h - 1)
    occlusion_mask = occlusion_mask | out_of_boundary_mask
    occ_acc = np.sum(np.equal(occlusion_mask, gt_occluded) & evaluation_points, axis=(1, 2)) / np.sum(evaluation_points)
    metrics['occlusion_accuracy'] = occ_acc

    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(occlusion_mask)
    all_frac_within = []
    all_jaccard = []

    for thresh in [1, 2, 4, 8, 16]:
        within_dist = np.sum(
            np.square(results - target_points),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points = np.sum(
            visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics['pts_within_' + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=(1, 2))
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics['jaccard_' + str(thresh)] = jaccard
        all_jaccard.append(jaccard)

    metrics['average_pts_within_thresh'] = np.mean(np.stack(all_frac_within, axis=1), axis=1)
    metrics['average_jaccard'] = np.mean(np.stack(all_jaccard, axis=1), axis=1)
    metrics['temporal_coherence'] = eval_temporal_coherence(results, target_points, gt_occluded)

    metrics = dict(sorted(metrics.items()))
    print(metrics)
    with open(os.path.join(out_dir, '{}.csv'.format(seq_name)),
                'w', newline='') as csvfile:
        fieldnames = ['video_name',
                        'average_jaccard',
                        'average_pts_within_thresh',
                        'occlusion_accuracy',
                        'temporal_coherence',
                        'jaccard_1',
                        'jaccard_2',
                        'jaccard_4',
                        'jaccard_8',
                        'jaccard_16',
                        'pts_within_1',
                        'pts_within_2',
                        'pts_within_4',
                        'pts_within_8',
                        'pts_within_16',
                        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            'video_name': seq_name,
            'average_jaccard': metrics['average_jaccard'].item(),
            'average_pts_within_thresh': metrics['average_pts_within_thresh'].item(),
            'occlusion_accuracy': metrics['occlusion_accuracy'].item(),
            'temporal_coherence': metrics['temporal_coherence'].item(),
            'jaccard_1': metrics['jaccard_1'].item(),
            'jaccard_2': metrics['jaccard_2'].item(),
            'jaccard_4': metrics['jaccard_4'].item(),
            'jaccard_8': metrics['jaccard_8'].item(),
            'jaccard_16': metrics['jaccard_16'].item(),
            'pts_within_1': metrics['pts_within_1'].item(),
            'pts_within_2': metrics['pts_within_2'].item(),
            'pts_within_4': metrics['pts_within_4'].item(),
            'pts_within_8': metrics['pts_within_8'].item(),
            'pts_within_16': metrics['pts_within_16'].item()
        })
        # del model
        # torch.cuda.empty_cache()


def summarize(out_dir, dataset_name):
    result_files = sorted(glob.glob(os.path.join(out_dir, '*.csv')))
    sum_file = os.path.join('{}.csv'.format(dataset_name))
    flag = True
    num_seqs = 0
    with open(sum_file, 'w', newline='') as outfile:
        for i, result_file in enumerate(result_files):
            with open(result_file, 'r', newline='') as infile:
                reader = csv.DictReader(infile)
                if flag:
                    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    flag = False
                for row in reader:
                    writer.writerow(row)
                num_seqs += 1

    df = pd.read_csv(sum_file)
    average = {'video_name': 'average'}
    for k in df.keys()[1:]:
        average[k] = np.round(df[k].mean(), 5)
        df[k] = np.round(df[k], 5)

    df.loc[-1] = average
    df.index = df.index + 1
    df.sort_index(inplace=True)
    df.to_csv(sum_file, index=False)
    print('{} | average_jaccard: {:.5f} | average_pts_within_thresh: {:.5f} '
            '| occlusion_acc: {:.5f} | temporal_coherence: {:.5f}'.
            format(num_seqs,
                   average['average_jaccard'],
                   average['average_pts_within_thresh'],
                   average['occlusion_accuracy'],
                   average['temporal_coherence']
                   ))

def render_vid(args):
    model = BaseTrainer(args)
    print("out_dir: ", out_dir)
    if model.with_mask:
        video_correspondences = model.eval_video_correspondences(0,
                                                                use_mask=True,
                                                                vis_occlusion=model.args.vis_occlusion,
                                                                use_max_loc=model.args.use_max_loc,
                                                                occlusion_th=model.args.occlusion_th)
        imageio.mimwrite(os.path.join(out_dir, '{}_corr_foreground_{:06d}.mp4'.format(model.seq_name, 200000)),
                            video_correspondences,
                            quality=8, fps=10)
        video_correspondences = model.eval_video_correspondences(0,
                                                                use_mask=True,
                                                                reverse_mask=True,
                                                                vis_occlusion=model.args.vis_occlusion,
                                                                use_max_loc=model.args.use_max_loc,
                                                                occlusion_th=model.args.occlusion_th)
        imageio.mimwrite(os.path.join(out_dir, '{}_corr_background_{:06d}.mp4'.format(model.seq_name, 200000)),
                            video_correspondences,
                            quality=8, fps=10)
    else:
        video_correspondences = model.eval_video_correspondences(0,
                                                                vis_occlusion=model.args.vis_occlusion,
                                                                use_max_loc=model.args.use_max_loc,
                                                                occlusion_th=model.args.occlusion_th)
        imageio.mimwrite(os.path.join(out_dir, '{}_corr_{:06d}.mp4'.format(model.seq_name, 20000)),
                            video_correspondences,
                            quality=8, fps=10)


def eval_sequences(seqlist):
    all_res= []
    for seqdir in seqlist:
        args = config_parser(dafault_config=seqdir+"/config.txt")
        args.out_dir = "eval_out"
        args.load_dir = seqdir
        print("seqdir: ", seqdir)
        print("-------------------")
        res = eval_ckpt(args)
        
        print("occ_acc: ", res['occlusion_accuracy'])
        print("-------------------")
        all_res.append(res)
        torch.cuda.empty_cache()
    print("-------------------")
    return all_res


def count_motion_magnitude(dataname):
    anno = Path("dataset/tapvid_davis_256/annotations") / f"{dataname}.pkl"
    data = np.load(anno, allow_pickle=True)
    query_points = data['davis']['query_points']
    target_points = data['davis']['target_points']
    gt_occluded = data['davis']['occluded']
    all_mag = 0
    all_pairs = 0
    for pt in range(target_points.shape[1]):
        motion_mag = 0
        pairs = 0
        traj = target_points[0, pt]
        for f in range(len(traj)-1):
            if gt_occluded[0, pt, f] or gt_occluded[0, pt, f+1]:
                continue
            motion_mag += np.linalg.norm(traj[f+1] - traj[f])
            pairs += 1
        all_mag += motion_mag
        all_pairs += pairs
        if pairs == 0:
            continue
        print(f"pt: {pt} motion magnitude: {motion_mag/pairs}")
    print(f"scene: {dataname} mean motion magnitude: {all_mag/all_pairs}")
    return all_mag/all_pairs

def eval_opw(depthmap0, depthmap1, flow, mask):
    grid = np.stack(np.meshgrid(np.arange(0, 854), np.arange(0, 480)), axis=-1)
    grid = grid.astype(np.float32)
    grid += flow
    depth0 = depthmap0[mask]
    grid = grid[mask]
    depth1 = torch.functional.F.grid_sample(depthmap1[None, None], torch.from_numpy(grid[None]).permute(0, 3, 1, 2), mode='bilinear', align_corners=True).permute(0, 2, 3, 1).squeeze().numpy()
    diff = np.abs(depth0 - depth1)
    return diff.mean()

def eval_scene_opw(basedir:Path):
    depthdir = basedir / "raw_depth" / "depth"
    flow_dir = basedir / "raft_exhaustive"
    maskdir = basedir / "full_mask"
    depthmaps = sorted(list(depthdir.glob("*.npz")))
    for i in range(len(depthmaps)-1):
        depthmap0 = np.load(depthmaps[i])['depth']
        depthmap1 = np.load(depthmaps[i+1])['depth']
        flow = np.load(flow_dir / f"{i:05d}.jpg_{i+1:05d}.jpg.npy")
        mask = cv2.imread(maskdir / f"{i:05d}.jpg_{i+1:05d}.jpg.png", cv2.IMREAD_GRAYSCALE) > 0
        print(f"scene {i} opw error: {eval_opw(depthmap0, depthmap1, flow, mask)}")
        



if __name__ == '__main__':
    
    args = config_parser("cotracker_out/240311-0324_depth_cotrackerFilter-l3335d5R12b4C10_dog/config.txt")
    args.out_dir = "eval_out"
    trainer = setup_trainer(args)
    res = eval_one_step(trainer, 0)
    print(res)

    