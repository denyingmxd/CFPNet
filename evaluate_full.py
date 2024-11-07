import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from src.utils.model_io import load_weights
from src.utils.utils import RunningAverageDict, colorize,make_model
from src.config import args
from src.dataloader.zjuL5 import ZJUL5
from src.dataloader.nyu import NYUV2
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from torch.distributions import categorical
import matplotlib.pyplot as plt
def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse,
                log_10=log_10, rmse_log=rmse_log, silog=silog, sq_rel=sq_rel)

def predict_tta(model, input_data, args):

    pred_refine,pred, prob, prob_residual = model(input_data)
    pred = np.clip(pred.cpu().numpy(), args.min_depth, args.max_depth)
    pred = nn.functional.interpolate(torch.from_numpy(pred), input_data['rgb'].shape[-2:], mode='bilinear', align_corners=True)
    # prob = nn.functional.interpolate(prob, input_data['rgb'].shape[-2:], mode='bilinear', align_corners=True)

    return None, pred,  prob, None

def eval(model, test_loader, args, device):
    if args.save_dir is not None and not os.path.exists(f'{args.save_dir}'):
        os.system(f'mkdir -p {args.save_dir}')

    metrics = RunningAverageDict()
    full = np.zeros((args.input_height, args.input_width),dtype=np.float32)
    with torch.no_grad():
        model.eval()
        for index, batch in enumerate(tqdm(test_loader)):
            gt = batch['depth'].to(device)
            img = batch['image'].to(device)
            input_data = {'rgb': img}
            additional_data = {}
            additional_data['hist_data'] = batch['additional']['hist_data'].to(device)
            additional_data['rect_data'] = batch['additional']['rect_data'].to(device)
            additional_data['mask'] = batch['additional']['mask'].to(device)
            additional_data['patch_info'] = batch['additional']['patch_info']
            input_data.update({
                'additional': additional_data
            })

            final_refine, final, prob, prob_residual = predict_tta(model, input_data, args)
            final = final.squeeze().cpu().numpy()

            gt_cc = gt.squeeze().cpu().numpy()
            eee = (gt_cc - final) ** 2
            valid_mask = np.logical_and(gt_cc > args.min_depth, gt_cc < args.max_depth)
            error_local = eee*valid_mask
            full+=error_local
            # if index==100:
            #     break
    full = full/(index+1)
    plt.imshow(full, vmin=0, vmax=2,cmap='jet')
    plt.colorbar()
    plt.show()
    plt.hist(full)
    plt.show()



if __name__ == '__main__':

    device = torch.device('cuda:0')
    if 'nyu' in args.test_dataset:
        pass
    elif 'zjuL5' in args.test_dataset:
        args.data_path_eval = 'data/ZJUL5'
        args.filenames_file_eval = 'data/ZJUL5/data.json'
        args.input_height = 480
        args.input_width = 640
        args.max_depth = 10
        args.min_depth = 1e-3
        args.n_bins = 256
        args.min_depth_eval = 1e-3
        args.max_depth_eval = 10
        args.zone_sample_num = 16
    else:
        raise NotImplementedError

    if 'ZJU' in args.filenames_file_eval:
        test_loader = ZJUL5(args, 'online_eval').data
    elif 'nyu' in args.filenames_file_eval:
        test_loader = NYUV2(args, 'online_eval').data

    model = make_model(args)
    model = model.to(device)
    model = load_weights(model, args.weight_path)
    model = model.eval()

    eval(model, test_loader, args, device)

