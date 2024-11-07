import os
import uuid
from datetime import datetime as dt
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
import json
import random
from src.utils.model_io import load_checkpoint, save_checkpoint, save_weights
from src.utils.utils import RunningAverage, RunningAverageDict, make_model
from src.utils.metrics import compute_errors


from src.dataloader.nyu import NYUV2
from src.loss import *

import os
os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa

PROJECT = "deltar"
logging = True

def main_worker(args):
    model = make_model(args)
    if args.resume != '':
        model, optimizer_state_dict, epoch = load_checkpoint(args.resume, model)
        if epoch is None:
            args.epoch =0
            args.last_epoch = -1
            optimizer_state_dict = None
        else:
            epoch = args.epoch + 1
            args.last_epoch = epoch
    else:
        args.epoch = 0
        args.last_epoch = -1
        optimizer_state_dict = None

    model = model.cuda()
    model = torch.nn.DataParallel(model)

    train(model, args, epochs=args.epochs, lr=args.lr, experiment_name=args.name,
          optimizer_state_dict=optimizer_state_dict)


def train(model, args, epochs=10, experiment_name="DeepLab", lr=0.0001,
          optimizer_state_dict=None):
    global PROJECT
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f"Training {experiment_name}")

    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{args.bs}-tep{epochs}-lr{lr}-wd{args.wd}-{uuid.uuid4()}"
    name = f"{experiment_name}_{run_id}"
    if logging:
        tags = args.tags.split(',') if args.tags != '' else None
        if args.dataset != 'nyu':
            PROJECT = PROJECT + f"-{args.dataset}"
        # wandb.init(project=PROJECT, name=name, config=args, dir='./', tags=tags, notes=args.notes)
        # with open(f'{wandb.run.dir}/run_args.json', 'w') as f:
        #     json.dump(args.__dict__, f, indent=2)

    if args.dataset == 'nyu':
        train_loader = NYUV2(args, 'train').data

    if args.dataset_eval == 'nyu':
        test_loader = NYUV2(args, 'online_eval').data

    criterion_ueff = SILogLoss()
    if args.use_my_loss:
        if args.my_loss_name == 'loss1':
            my_loss = MyLoss1()
        if args.my_loss_name == 'loss2':
            my_loss = MyLoss2()
        if args.my_loss_name == 'loss3':
            my_loss = MyLoss3()
    model.train()

    m = model.module
    params = [{"params": m.get_1x_lr_params(), "lr": lr / 10},
            {"params": m.get_10x_lr_params(), "lr": lr}]

    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    # if optimizer_state_dict is not None:
    #     optimizer.load_state_dict(optimizer_state_dict)

    iters = len(train_loader)
    step = args.epoch * iters
    best_loss = np.inf

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader),
                                            cycle_momentum=True,
                                            base_momentum=0.85, max_momentum=0.95, last_epoch=args.last_epoch,
                                            div_factor=args.div_factor,
                                            final_div_factor=args.final_div_factor)

    for epoch in range(args.epoch, epochs):

        # if logging: wandb.log({"Epoch": epoch}, step=step)
        # import ipdb; ipdb.set_trace()
        for i, batch in tqdm(enumerate(train_loader), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train",
                             total=len(train_loader)):
            optimizer.zero_grad()

            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            input_data = {'rgb': img}
            additional_data = {}
            additional_data['hist_data'] = batch['additional']['hist_data'].to(device)
            additional_data['rect_data'] = batch['additional']['rect_data'].to(device)
            additional_data['mask'] = batch['additional']['mask'].to(device)
            additional_data['my_mask'] = batch['additional']['my_mask'].to(device)
            additional_data['patch_info'] = batch['additional']['patch_info']
            input_data.update({
                'additional': additional_data
            })
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            pred_refine, pred = model(input_data)

            mask = depth > args.min_depth
            pred = torch.clip(pred,args.min_depth)
            loss = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True)
            # xxx = loss.detach().clone()
            # yyy= 0
            if args.use_my_loss:
                yyy = my_loss(pred, depth,additional_data['my_mask'].to(torch.bool),
                              sample_num = args.my_loss_num,
                              mask=mask.to(torch.bool), interpolate=True) * args.my_loss_weight
                loss += yyy
            # print(xxx,yyy)
            loss.backward()
            if args.disable_clip_grad:
                pass
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
            optimizer.step()
            # if logging and step % 5 == 0:
            #     wandb.log({f"Train/{criterion_ueff.name}": loss.detach().item()}, step=step)

            # if logging and step % 100 == 0:
            #     xxx = wandb.Image(batch['image'][0], caption='step: {}'.format(step))
            #     wandb.log({"Train/rgb": xxx})
            #     xxx = wandb.Image(batch['depth'][0][0], caption='step: {}'.format(step))
            #     wandb.log({"Train/depth": xxx})
            #     xxx = wandb.Image(pred[0][0], caption='step: {}'.format(step))
            #     wandb.log({"Train/pred": xxx})
            #     xxx = wandb.Image(additional_data['mask'][0].reshape(args.train_zone_num,args.train_zone_num).float(), caption='step: {}'.format(step))
            #     wandb.log({"Train/mask": xxx})
            # break

            step += 1
            scheduler.step()

        if step % args.validate_every == 0:
        # if logging and step % args.validate_every == 0:

            model.eval()

            criterion_loss = {}
            criterion_loss['ueff'] = criterion_ueff
            metrics, val_loss = validate(args, model, test_loader, criterion_loss, epoch, epochs, device)

            # print("Validated: {}".format(metrics))
            if logging:
                # wandb.log({f"Test/{criterion_ueff.name}": val_loss['val_si'].get_value()}, step=step)
                # wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=step)
                save_checkpoint(model, optimizer, epoch, fpath=f"checkpoints/{experiment_name}/{epoch}_{metrics['rmse']:3f}.pt")
                save_weights(model.module, fpath=f"weights/{experiment_name}/{epoch}_{metrics['rmse']:.3f}.pt")
                if metrics['rmse'] < best_loss:
                    save_checkpoint(model, optimizer, epoch, fpath=f"checkpoints/{experiment_name}/best.pt")
                    save_weights(model.module, fpath=f"weights/{experiment_name}/best.pt")
                    best_loss = metrics['rmse']
            model.train()
    if logging:
        wandb.finish()
    return model


def validate(args, model, test_loader, criterion_loss, epoch, epochs, device='cpu'):
    with torch.no_grad():
        criterion_ueff = criterion_loss['ueff']
        val_si = RunningAverage()
        metrics = RunningAverageDict()
        for i, batch in tqdm(enumerate(test_loader), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation", total=len(test_loader)):
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            input_data = {'rgb': img}
            additional_data = {}
            additional_data['hist_data'] = batch['additional']['hist_data'].to(device)
            additional_data['rect_data'] = batch['additional']['rect_data'].to(device)
            additional_data['mask'] = batch['additional']['mask'].to(device)
            additional_data['patch_info'] = batch['additional']['patch_info']
            input_data.update({
                'additional': additional_data
            })
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            pred_refine, pred, prob, _ = model(input_data)

            mask = depth > args.min_depth
            loss = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True)
            val_si.append(loss.detach().item())
            pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)

            # if logging and i == 0:
            #     wandb_rgb = wandb.Image(img[0], caption=f"epoch:{epoch}")
            #     wandb_pred =  wandb.Image(pred[0,0], caption=f"epoch:{epoch}")
            #     wandb_depth =  wandb.Image(depth[0,0], caption=f"epoch:{epoch}")
            #
            #     wandb.log({"Test/rgb":wandb_rgb})
            #     wandb.log({"Test/depth":wandb_depth})
            #     wandb.log({"Test/pred": wandb_pred})


            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth_eval] = args.min_depth_eval
            pred[pred > args.max_depth_eval] = args.max_depth_eval
            pred[np.isinf(pred)] = args.max_depth_eval
            pred[np.isnan(pred)] = args.min_depth_eval

            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
            metrics.update(compute_errors(gt_depth[valid_mask], pred[valid_mask]))

        return metrics.get_value(), {'val_si': val_si}



def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':

    from src.config import args

    if args.no_logging:
        globals()['logging'] = False

    set_seeds(117010053)
    main_worker(args)