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
from thop import profile
from thop import clever_format
import time
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

def predict_tta(model, input_data, args,profiled=False):

    pred_refine,pred, prob, prob_residual = model(input_data)
    if profiled:
        macs, params = profile(model, inputs=(input_data,))
        macs, params = clever_format([macs, params], "%.3f")
        print(f"macs: {macs}, params: {params}")

    return None, pred,  prob, None

def eval(model, test_loader, args, device):
    if args.save_dir is not None and not os.path.exists(f'{args.save_dir}'):
        os.system(f'mkdir -p {args.save_dir}')

    metrics = RunningAverageDict()
    with torch.no_grad():
        model.eval()
        for index, batch in enumerate(tqdm(test_loader)):
            if index == 100:
                break
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

            final_refine, final, prob, prob_residual = predict_tta(model, input_data, args,profiled=False)

        diff = []
        niters = 500
        for i in tqdm(range(niters)):
            torch.cuda.synchronize()
            t = time.perf_counter()
            final_refine, final, prob, prob_residual = predict_tta(model, input_data, args)
            torch.cuda.synchronize()
            diff.append((time.perf_counter() - t) * 1000)
        diff = sum(sorted(diff)[1:-2]) / (niters - 3)
        print(f"{diff:.3f} ms")
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

