import numpy as np
import cv2
from matplotlib import pyplot as plt
from pathlib import Path

def draw_flow(img, flow, grid_size = 20):
    for i in range(0, img.shape[0], grid_size):
        for j in range(0, img.shape[1], grid_size):
            x = int(flow[i, j, 0])
            y = int(flow[i, j, 1])
            cv2.line(img, (j, i), (j + x, i + y), (0, 255, 0), 1)
            cv2.circle(img, (j, i), 2, (0, 0, 255), -1)
            cv2.circle(img, (j + x, i + y), 2, (255, 0, 0), -1)
    return img

def draw_corr(img1, img2, flow, grid_size = 20):
    num_pts = (img1.shape[0] // grid_size + 1) * (img1.shape[1] // grid_size + 1)
    colors = plt.cm.hsv(np.linspace(0, 1, num_pts+1))
    colors = np.random.permutation(colors)
    colors = colors * 255
    colors = colors.astype(int)
    pid = 0
    for i in range(0, img1.shape[0], grid_size):
        for j in range(0, img1.shape[1], grid_size):
            
            x = int(flow[i, j, 0])
            y = int(flow[i, j, 1])
            color = colors[pid].tolist()
            # cv2.line(img1, (j, i), (j + x, i + y), (0, 255, 0), 1)
            cv2.circle(img1, (j, i), 2, color, -1)
            cv2.circle(img2, (j + x, i + y), 2, color, -1)
            pid += 1
    return img1, img2

def wrap_pixel(img1, img2, flow):
    h, w = img1.shape[:2]
    grid = np.indices((h, w)).astype(np.float32)
    grid[0] += flow[:,:,1]
    grid[1] += flow[:,:,0]
    return cv2.remap(img2, grid[1], grid[0], cv2.INTER_LINEAR)
    

def vis_sparse_flow(flowfile, rgbdir, outdir):
    imgnames = flowfile.stem.split("_")
    img1 = cv2.imread((rgbdir / (imgnames[0])).__str__())
    img2 = cv2.imread((rgbdir / (imgnames[1])).__str__())
    flow = np.load(flowfile)
    w, h = img1.shape[:2]
    grid = np.meshgrid(np.arange(h), np.arange(w), indexing='xy')
    grid = np.stack(grid, axis=-1).astype(np.float32)
    dst = grid + flow
    dst = dst.reshape(-1, 2)
    grid = grid.reshape(-1, 2)
    dst = dst[::211]
    grid = grid[::211]
    colors = plt.cm.hsv(np.linspace(0, 1, dst.shape[0]+1))[..., :3] * 0.7 * 255
    colors = colors.astype(int)
    for i in range(dst.shape[0]):
        cv2.arrowedLine(img1, (int(grid[i, 0]), int(grid[i, 1])), (int(dst[i, 0]), int(dst[i, 1])), colors[i].tolist(), 1)
        cv2.arrowedLine(img2, (int(grid[i, 0]), int(grid[i, 1])), (int(dst[i, 0]), int(dst[i, 1])), colors[i].tolist(), 1)
    cv2.imwrite((outdir / (flowfile.stem + "_1.png")).__str__(), img1)
    cv2.imwrite((outdir / (flowfile.stem + "_2.png")).__str__(), img2)



if __name__ == '__main__':
    basedir = Path("dataset/DAVIS_tapvid/gold-fish")
    rgbdir = basedir / "color"
    flowdir = basedir / "raft_exhaustive"
    outdir = basedir / "vis_raft"
    outdir.mkdir(exist_ok=True, parents=True)
    flowfiles = sorted(flowdir.glob("*.npy"))
    for flowfile in flowfiles:
        vis_sparse_flow(flowfile, rgbdir, outdir)

