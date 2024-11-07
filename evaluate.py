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
            # final_refine = final_refine.squeeze().cpu().numpy()

            if 'Deltar_rgb' == model._get_name():
                final *= np.median(np.median(batch['additional']['hist_data'][0].mean(-1)[batch['additional']['mask'][0]>0]))/np.median(final)


            impath = f"{batch['image_path'][0].replace('/', '__').replace('.jpg', '')}"
            im_subfolder = batch['image_folder'][0]
            vis_folder = f'{im_subfolder}'
            if not os.path.exists(f'{args.save_dir}/{vis_folder}'):
                os.system(f'mkdir -p {args.save_dir}/{vis_folder}')

            gt_cc = gt.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt_cc > args.min_depth, gt_cc < args.max_depth)

            if args.save_pred:
                pred_path = os.path.join(args.save_dir, vis_folder, f"{impath}_pred.jpg")
                pred = colorize(torch.from_numpy(final).unsqueeze(0), vmin=0.0, vmax=3.0, cmap='magma')
                Image.fromarray(pred).save(pred_path)

            if args.save_refined_pred:
                pred_path = os.path.join(args.save_dir, vis_folder, f"{impath}_pred_refine.jpg")
                pred = colorize(torch.from_numpy(final).unsqueeze(0), vmin=0.0, vmax=3.0, cmap='magma')
                Image.fromarray(pred).save(pred_path)

            if args.save_residual:
                pred_path = os.path.join(args.save_dir, vis_folder, f"{impath}_residual.jpg")
                pred = colorize(torch.from_numpy(final_refine - final).unsqueeze(0), vmin=0, vmax=0.2, cmap='magma')
                Image.fromarray(pred).save(pred_path)

            if args.save_entropy:
                # prob = prob.detach().cpu()
                dist = categorical.Categorical(probs=prob.permute(0,2,3,1))
                entropy = dist.entropy().cpu()
                viz = colorize(entropy,vmin=0,vmax=np.log(prob.shape[1]), cmap='jet')
                Image.fromarray(viz).save(os.path.join(args.save_dir, vis_folder, f"{impath}_entropy.jpg"))

            if args.save_residual_entropy:
                # prob_residual = prob_residual.detach().cpu()
                dist = categorical.Categorical(probs=prob_residual.permute(0, 2, 3, 1))
                entropy = dist.entropy().cpu()
                viz = colorize(entropy, vmin=0, vmax=np.log(prob_residual.shape[1]), cmap='jet')
                Image.fromarray(viz).save(os.path.join(args.save_dir, vis_folder, f"{impath}_residual_entropy.jpg"))

            if args.save_error_map:
                error_map = np.abs(gt.squeeze().cpu().numpy() - final) * valid_mask
                viz = colorize(torch.from_numpy(error_map).unsqueeze(0), vmin=0, vmax=1.2, cmap='jet')
                Image.fromarray(viz).save(os.path.join(args.save_dir, vis_folder, f"{impath}_error.jpg"))

            if args.save_refined_error_map:
                error_map = np.abs(gt.squeeze().cpu().numpy() - final_refine) * valid_mask
                viz = colorize(torch.from_numpy(error_map).unsqueeze(0), vmin=0, vmax=1.2, cmap='jet')
                Image.fromarray(viz).save(os.path.join(args.save_dir, vis_folder, f"{impath}_refined_error.jpg"))


            if args.save_rgb:
                img = (img-torch.min(img))/(torch.max(img)-torch.min(img))
                img = img.squeeze().permute([1,2,0]).cpu().numpy()
                img = (img*255).astype(np.uint8)
                # import ipdb; ipdb.set_trace()
                rgb = img.copy()
                rgb = Image.fromarray(rgb)
                draw = ImageDraw.Draw(rgb)
                for i, xxx in enumerate(batch['additional']['rect_data'][batch['additional']['mask'] == 1]):
                    a, b, c, d = xxx
                    draw.rectangle((b, a, d, c),outline=(255, 0, 0))

                rgb.save(os.path.join(args.save_dir, vis_folder, f"{impath}_rgb.jpg"))

            if args.save_for_demo:
                pred = (final * 1000).astype('uint16')
                pred_path = os.path.join(args.save_dir, vis_folder, f"{impath}_demo.jpg")
                Image.fromarray(pred).save(pred_path)

            gt = gt.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt > args.min_depth, gt < args.max_depth)
            if args.save_gt:
                pred_path = os.path.join(args.save_dir, vis_folder, f"{impath}_gt.jpg")
                pred = colorize(torch.from_numpy(gt).unsqueeze(0), vmin=0.0, vmax=3.0, cmap='magma')
                Image.fromarray(pred).save(pred_path)

            if valid_mask.sum()>0:
                if args.zone_area_only:
                    here_mask = (valid_mask * batch['additional']['my_mask'][0][0].numpy()).astype(bool)
                elif args.outside_zone_area_only:
                    here_mask = valid_mask * ~batch['additional']['my_mask'][0][0].numpy().astype(bool)
                else:
                    here_mask = valid_mask
                metrics.update(compute_errors(gt[here_mask], final[here_mask]))
            # break
    metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
    print(f"Metrics: {metrics}")
    print(','.join([str(i) for i in metrics.values()]))


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

