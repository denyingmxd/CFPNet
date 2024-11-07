import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from src.utils.model_io import load_weights
from src.dataloader.zjuL5 import ZJUL5
from src.models.deltar import Deltar
from src.models.deltar_rgb import Deltar_rgb
from src.utils.utils import RunningAverageDict, colorize
from src.config import args


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
    _, pred = model(input_data)
    pred = np.clip(pred.cpu().numpy(), args.min_depth, args.max_depth)
    pred = nn.functional.interpolate(torch.Tensor(pred), input_data['rgb'].shape[-2:], mode='bilinear', align_corners=True)

    return torch.Tensor(pred)

def eval(model1, model2, test_loader, args, device):
    if args.save_dir is not None and not os.path.exists(f'{args.save_dir}'):
        os.system(f'mkdir -p {args.save_dir}')

    metrics = RunningAverageDict()
    with torch.no_grad():
        model1.eval()
        model2.eval()
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

            final1 = predict_tta(model1, input_data, args)
            final2 = predict_tta(model2, input_data, args)
            final1 = final1.squeeze().cpu().numpy()
            final2 = final2.squeeze().cpu().numpy()



            gt = gt.squeeze().cpu().numpy()

            valid_mask = np.logical_and(gt > args.min_depth, gt < args.max_depth)
            # valid_mask = np.logical_and(valid_mask, gt > 4.)
            final1[final2>3.] = final2[final2>3.]

            if valid_mask.sum()>0:
                metrics.update(compute_errors(gt[valid_mask], final1[valid_mask]))

    metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
    print(f"Metrics: {metrics}")


if __name__ == '__main__':

    device = torch.device('cuda:0')
    test_loader = ZJUL5(args, 'online_eval').data

    model1 = Deltar(n_bins=args.n_bins, min_val=args.min_depth,
                    max_val=args.max_depth, norm=args.norm)

    model2 = Deltar_rgb(n_bins=args.n_bins, min_val=args.min_depth,
                   max_val=args.max_depth, norm=args.norm)
    model1 = model1.to(device)
    model1 = load_weights(model1, "/data/laiyan/codes/deltar/weights/train_deltar_27-Nov_14-26-nodebs16-tep25-lr0.0003-wd0.1-a476389e-f60f-4b66-8d6b-433f947ea2ee_best.pt")
    model1 = model1.eval()

    model2 = model2.to(device)
    model2 = load_weights(model2, "/data/laiyan/codes/deltar/weights/train_only_rgb_28-Nov_16-18-nodebs16-tep25-lr0.0003-wd0.1-1d1921ab-f16c-43aa-a706-f0838964386a_best.pt")
    model2 = model2.eval()

    eval(model1, model2, test_loader, args, device)
