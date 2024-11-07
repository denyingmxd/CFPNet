import os
import numpy as np
import cv2
import tqdm
import h5py
import imageio

def normalize(value, vmin=0.0, vmax=4.0):
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    value = (value*255.0).astype(np.uint8)
    return value

image_list = []
h5_folder = '../data/demo/room/'
output_folder = 'tmp_vis'
os.system(f'mkdir -p {output_folder}')
files = sorted(os.listdir(h5_folder))
for f in tqdm.tqdm(files):
    fname = os.path.basename(f)
    h5_fname = f'{h5_folder}/{f}'
    output_fname = f'{output_folder}/{f.replace("h5", "png")}'

    h5_file = h5py.File(h5_fname, 'r')
    fr = h5_file['fr'][:]
    hist_data = h5_file['hist_data'][:]
    mask = h5_file['mask'][:]

    pad_size = int(5)
    vis_img = np.zeros([480, 320+640*1+pad_size*2, 3], dtype=np.uint8)

    L5_depth = np.zeros([480, 640])
    for i in range(mask.shape[0]):
        if not mask[i]: continue
        if fr[i,2] < 0 or fr[i,3] < 0: continue
        fr[i, 0] = np.clip(fr[i, 0], 0, 10000)
        fr[i, 1] = np.clip(fr[i, 1], 0, 10000)
        sy, sx, ey, ex = fr[i]
        L5_depth[sy:ey, sx:ex] = hist_data[i, 0]
    L5_depth = cv2.applyColorMap(normalize(L5_depth), cv2.COLORMAP_MAGMA)
    L5_depth = cv2.resize(L5_depth, (320, 240-pad_size), interpolation = cv2.INTER_AREA)

    realsense_depth = h5_file['depth'][:]
    realsense_depth = cv2.applyColorMap(normalize(realsense_depth), cv2.COLORMAP_MAGMA)
    rgb = h5_file['rgb'][:]
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    rgb = cv2.resize(rgb, (320, 240-pad_size), interpolation = cv2.INTER_AREA)

    vis_img[0:240-pad_size, 0:320, :] = rgb
    vis_img[-(240-pad_size)-1:-1, 0:320, :] = L5_depth
    vis_img[:, 320+pad_size:320+pad_size+640, :] = realsense_depth

    vis_img = cv2.resize(vis_img, (640, 300), interpolation = cv2.INTER_AREA)
    cv2.imwrite(f'{output_fname}', vis_img)

