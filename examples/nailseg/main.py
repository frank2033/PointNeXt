"""
Training script for NailSeg fingernail point cloud segmentation.

Supports training PointNet, PointNet++, and PointNeXt models.

Usage:
    # Train PointNet
    python examples/nailseg/main.py --cfg cfgs/nailseg/pointnet.yaml

    # Train PointNet++
    python examples/nailseg/main.py --cfg cfgs/nailseg/pointnet++.yaml

    # Train PointNeXt-S
    python examples/nailseg/main.py --cfg cfgs/nailseg/pointnext-s.yaml

    # Resume training
    python examples/nailseg/main.py --cfg cfgs/nailseg/pointnext-s.yaml mode=resume pretrained_path=<path_to_ckpt>

    # Test only
    python examples/nailseg/main.py --cfg cfgs/nailseg/pointnext-s.yaml mode=test pretrained_path=<path_to_ckpt>
"""
import __init__
import argparse
import yaml
import os
import sys
import json
import logging
import numpy as np
import csv
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import warnings

torch.backends.cudnn.benchmark = False
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import openpoints modules
from openpoints.models import build_model_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys, get_class_weights
from openpoints.transforms import build_transforms_from_cfg
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.utils import (set_random_seed, save_checkpoint, load_checkpoint,
                               resume_checkpoint, setup_logger_dist, cal_model_parm_nums,
                               Wandb, generate_exp_directory, resume_exp_directory,
                               EasyConfig, dist_utils, find_free_port)

# Import and register the NailSeg dataset
from nailseg_dataset import NailSeg


def write_to_csv(oa, macc, miou, ious, best_epoch, cfg, write_header=True):
    """Write evaluation results to CSV file."""
    ious_table = [f'{item:.2f}' for item in ious]
    header = ['method', 'OA', 'mACC', 'mIoU'] + cfg.classes + ['best_epoch', 'log_path']
    data = [cfg.cfg_basename, f'{oa:.2f}', f'{macc:.2f}',
            f'{miou:.2f}'] + ious_table + [str(best_epoch), cfg.run_dir]
    with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)


def save_metrics(metrics_history, save_path):
    """Save training metrics history to JSON for plotting."""
    with open(save_path, 'w') as f:
        json.dump(metrics_history, f, indent=2)


def main(gpu, cfg):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()

    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0:
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = SummaryWriter(log_dir=cfg.run_dir) if cfg.is_training else None
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed
                                           )
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    num_classes = val_loader.dataset.num_classes if hasattr(
        val_loader.dataset, 'num_classes') else None
    if num_classes is not None:
        assert cfg.num_classes == num_classes
    logging.info(f"number of classes of the dataset: {num_classes}")
    cfg.classes = val_loader.dataset.classes if hasattr(
        val_loader.dataset, 'classes') else [str(i) for i in range(num_classes)]

    # optionally resume from a checkpoint
    model_module = model.module if hasattr(model, 'module') else model
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler,
                              pretrained_path=cfg.pretrained_path)
        elif cfg.mode in ['val', 'test']:
            best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
            val_miou, val_macc, val_oa, val_ious, val_accs = validate(
                model, val_loader, cfg)
            with np.printoptions(precision=2, suppress=True):
                logging.info(
                    f'Best ckpt @E{best_epoch}, val_oa {val_oa:.2f}, '
                    f'val_macc {val_macc:.2f}, val_miou {val_miou:.2f}, '
                    f'\niou per cls: {val_ious}')
            return val_miou
        elif cfg.mode == 'finetune':
            logging.info(f'Finetuning from {cfg.pretrained_path}')
            load_checkpoint(model, pretrained_path=cfg.pretrained_path)
        elif cfg.mode == 'finetune_encoder':
            logging.info(f'Load encoder only, finetuning from {cfg.pretrained_path}')
            load_checkpoint(model_module.encoder, pretrained_path=cfg.pretrained_path)
    else:
        logging.info('Training from scratch')

    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

    cfg.criterion_args.weight = None
    if cfg.get('cls_weighed_loss', False):
        if hasattr(train_loader.dataset, 'num_per_class'):
            cfg.criterion_args.weight = get_class_weights(
                train_loader.dataset.num_per_class, normalize=True)
        else:
            logging.info('`num_per_class` attribute is not found in dataset')
    criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()

    # metrics history for plotting
    metrics_history = {
        'train_loss': [], 'train_miou': [], 'train_macc': [], 'train_oa': [],
        'val_miou': [], 'val_macc': [], 'val_oa': [], 'lr': [], 'epochs': []
    }
    metrics_path = os.path.join(cfg.run_dir, 'metrics.json')

    # ===> start training
    val_miou, val_macc, val_oa, val_ious, val_accs = 0., 0., 0., [], []
    best_val, macc_when_best, oa_when_best, ious_when_best, best_epoch = 0., 0., 0., [], 0
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch - 1
        cfg.epoch = epoch

        train_loss, train_miou, train_macc, train_oa, _, _ = \
            train_one_epoch(model, train_loader, criterion,
                            optimizer, scheduler, epoch, cfg)

        is_best = False
        if epoch % cfg.val_freq == 0:
            val_miou, val_macc, val_oa, val_ious, val_accs = validate(
                model, val_loader, cfg, epoch=epoch)
            if val_miou > best_val:
                is_best = True
                best_val = val_miou
                macc_when_best = val_macc
                oa_when_best = val_oa
                ious_when_best = val_ious
                best_epoch = epoch
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Find a better ckpt @E{epoch}, '
                        f'val_miou {val_miou:.2f} val_macc {macc_when_best:.2f} '
                        f'val_oa {oa_when_best:.2f}'
                        f'\nmious: {val_ious}')

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_loss {train_loss:.4f} train_miou {train_miou:.2f} '
                     f'val_miou {val_miou:.2f} best_val_miou {best_val:.2f}')

        # Save metrics for plotting
        if cfg.rank == 0:
            metrics_history['epochs'].append(epoch)
            metrics_history['train_loss'].append(train_loss)
            metrics_history['train_miou'].append(float(train_miou))
            metrics_history['train_macc'].append(float(train_macc))
            metrics_history['train_oa'].append(float(train_oa))
            metrics_history['val_miou'].append(float(val_miou))
            metrics_history['val_macc'].append(float(val_macc))
            metrics_history['val_oa'].append(float(val_oa))
            metrics_history['lr'].append(lr)
            save_metrics(metrics_history, metrics_path)

        if writer is not None:
            writer.add_scalar('best_val', best_val, epoch)
            writer.add_scalar('val_miou', val_miou, epoch)
            writer.add_scalar('val_macc', val_macc, epoch)
            writer.add_scalar('val_oa', val_oa, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_miou', train_miou, epoch)
            writer.add_scalar('train_macc', train_macc, epoch)
            writer.add_scalar('train_oa', train_oa, epoch)
            writer.add_scalar('lr', lr, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)

        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best)

    # Final summary
    with np.printoptions(precision=2, suppress=True):
        logging.info(
            f'Best ckpt @E{best_epoch}, val_oa {oa_when_best:.2f}, '
            f'val_macc {macc_when_best:.2f}, val_miou {best_val:.2f}, '
            f'\niou per cls: {ious_when_best}')

    # Test with best checkpoint
    if cfg.rank == 0:
        load_checkpoint(model, pretrained_path=os.path.join(
            cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
        cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + '.csv')
        test_miou, test_macc, test_oa, test_ious, test_accs = validate(
            model, val_loader, cfg, epoch=cfg.epochs)
        with np.printoptions(precision=2, suppress=True):
            logging.info(
                f'Test with best ckpt @E{best_epoch}, '
                f'test_oa {test_oa:.2f}, test_macc {test_macc:.2f}, '
                f'test_miou {test_miou:.2f}, '
                f'\niou per cls: {test_ious}')
        write_to_csv(test_oa, test_macc, test_miou, test_ious,
                     best_epoch, cfg, write_header=True)

        # Save final metrics
        metrics_history['best_epoch'] = best_epoch
        metrics_history['best_val_miou'] = float(best_val)
        metrics_history['test_miou'] = float(test_miou)
        metrics_history['test_macc'] = float(test_macc)
        metrics_history['test_oa'] = float(test_oa)
        save_metrics(metrics_history, metrics_path)
        logging.info(f'Metrics saved to {metrics_path}')
        logging.info(f'Run: python examples/nailseg/plot_curves.py --metrics_path {metrics_path}')

    if writer is not None:
        writer.close()
    if cfg.distributed:
        dist.destroy_process_group()
    wandb.finish(exit_code=True)


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, cfg):
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    model.train()
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1
        target = data['y'].squeeze(-1)
        data['x'] = get_features_by_keys(data, cfg.feature_keys)

        logits = model(data)
        loss = criterion(logits, target)

        loss.backward()

        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            optimizer.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        cm.update(logits.argmax(dim=1), target)
        loss_meter.update(loss.item())

        if idx % cfg.print_freq:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")

    miou, macc, oa, ious, accs = cm.all_metrics()
    return loss_meter.avg, miou, macc, oa, ious, accs


@torch.no_grad()
def validate(model, val_loader, cfg, num_votes=1, data_transform=None, epoch=-1):
    model.eval()
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Val')
    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y'].squeeze(-1)
        data['x'] = get_features_by_keys(data, cfg.feature_keys)

        logits = model(data)
        cm.update(logits.argmax(dim=1), target)

    tp, union, count = cm.tp, cm.union, cm.count
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
    miou, macc, oa, ious, accs = get_mious(tp, union, count)
    with np.printoptions(precision=2, suppress=True):
        logging.info(f'Val Epoch [{epoch}/{cfg.epochs}], '
                     f'val_oa {oa:.2f}, val_macc {macc:.2f}, val_miou {miou:.2f}, '
                     f'\niou per cls: {ious}')
    return miou, macc, oa, ious, accs


if __name__ == "__main__":
    parser = argparse.ArgumentParser('NailSeg fingernail segmentation training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]
    tags = [
        cfg.task_name,
        cfg.mode,
        cfg.cfg_basename,
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    opt_list = []
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt \
                and 'pretrain' not in opt and 'path' not in opt \
                and 'wandb' not in opt and '/' not in opt:
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = '-'.join(opt_list)

    cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']

    if cfg.mode in ['resume', 'test', 'val']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    else:
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # wandb config
    cfg.wandb.name = cfg.run_name

    # multi processing
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg,))
    else:
        main(0, cfg)
