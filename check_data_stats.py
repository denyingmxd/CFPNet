import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from src.utils.model_io import load_weights
from src.dataloader.zjuL5 import ZJUL5
from src.dataloader.nyu import NYUV2
from src.models.deltar import Deltar
from src.models.deltar_rgb import Deltar_rgb
from src.utils.utils import RunningAverageDict, colorize
from src.config import args
import matplotlib.pyplot as plt



if __name__ == '__main__':

    device = torch.device('cuda:0')
    if 'ZJU' in args.filenames_file_eval:
        test_loader = ZJUL5(args, 'online_eval').data
    elif 'nyu' in args.filenames_file_eval:
        test_loader = NYUV2(args, 'online_eval').data
    gts = []
    masks = []
    for index, batch in enumerate(tqdm(test_loader)):
        gt = batch['depth']
        img = batch['image']
        input_data = {'rgb': img}

        hist_data = batch['additional']['hist_data']
        rect_data = batch['additional']['rect_data']
        mask= batch['additional']['mask']
        patch_info = batch['additional']['patch_info']

        # gts.append(gt[0][0])
        gts.append(hist_data[0])
        masks.append(mask[0])
        # break
    x = np.stack(gts,0)
    masks = np.stack(masks,0)
    # x = x[x>0]
    x = x[masks]
    x = x.flatten()
    x = x[x>0]
    _ = plt.hist(x, bins=100,range=(0,10))
    plt.show()







