from __future__ import print_function, division
import sys
sys.path.append('core')

import os
import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from core.liteanystereo import LiteAnyStereo
import core.stereo_datasets as datasets
from core.utils.utils import InputPadder
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def validate_eth3d(model):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        flow_pr = model(image1, image2, args.max_disp, test_mode=True)
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()

        occ_mask = Image.open(GT_file.replace('disp0GT.pfm', 'mask0nocc.png'))
        occ_mask = np.ascontiguousarray(occ_mask).flatten()

        val = (valid_gt.flatten() >= 0.5) & (occ_mask == 255)

        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()

        logging.info(f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} Bad1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation ETH3D: EPE %f, Bad1 %f" % (epe, d1))
    return {'eth3d-epe': epe, 'eth3d-d1': d1}


@torch.no_grad()
def validate_kitti(model, year=2015):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set='training', year=year)
    torch.backends.cudnn.benchmark = (device.type == 'cuda')

    d1_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        imageL_file, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        start = time.perf_counter()
        flow_pr = model(image1, image2, max_disp=args.max_disp, test_mode=True)
        end = time.perf_counter()

        if val_id > 50:
            elapsed_list.append(end-start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()
        epe_flattened = epe.flatten()
        flow_gt_flat = flow_gt.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        # D1: both > 3px and > 5% relative error
        rel_error = epe_flattened / torch.clamp(flow_gt_flat.abs(), min=1e-5)
        d1_mask = (epe_flattened > 3.0) & (rel_error > 0.05)

        # Apply only to valid pixels
        epe_list.append(epe_flattened[val].mean().item())
        d1_list.append(d1_mask[val].cpu().numpy())

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(np.concatenate(d1_list))
    avg_runtime = np.mean(elapsed_list)
    print(f"Validation KITTI {year}: EPE {epe}, D1 {d1}"
          f" {format(1 / avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'kitti-epe': epe, 'kitti-d1': d1}


@torch.no_grad()
def validate_middlebury(model, split='MiddEval3', resolution='F'):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split, resolution=resolution)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        flow_pr = model(image1, image2, args.max_disp, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        occ_mask = Image.open(imageL_file.replace('im0.png', 'mask0nocc.png')).convert('L')
        occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32).flatten()

        val = (valid_gt.reshape(-1) >= 0.5) & (flow_gt[0].reshape(-1) < 192) & (occ_mask==255)

        out = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} Bad2 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print(f"Validation Middlebury{split}_{resolution}_192: EPE {epe}, Bad2 {d1}")
    return {f'middlebury{split}_{resolution}-epe': epe, f'middlebury{split}-d1': d1}


@torch.no_grad()
def validate_driving(model, split='cloudy'):
    """ Peform validation using the DrivingStereo (weather) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.DrivingStereoWeather(aug_params, image_set=split)
    torch.backends.cudnn.benchmark = (device.type == 'cuda')

    d1_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        start = time.time()
        flow_pr = model(image1, image2, test_mode=True)
        end = time.time()

        if val_id > 50:
            elapsed_list.append(end - start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()
        epe_flattened = epe.flatten()
        flow_gt_flat = flow_gt.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        # D1: both > 3px and > 5% relative error
        rel_error = epe_flattened / torch.clamp(flow_gt_flat.abs(), min=1e-5)
        d1_mask = (epe_flattened > 3.0) & (rel_error > 0.05)

        # Apply only to valid pixels
        epe_list.append(epe_flattened[val].mean().item())
        d1_list.append(d1_mask[val].cpu().numpy())

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(np.concatenate(d1_list))
    avg_runtime = np.mean(elapsed_list)
    print(f"Validation DrivingStereo {split}: EPE {epe}, D1 {d1}"
          f" {format(1 / avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'drivingstereo-epe': epe, 'drivingstereo-d1': d1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/sceneflow.pth')
    parser.add_argument('--dataset', help="dataset for evaluation", default='sceneflow', choices=["eth3d", "kitti", "sceneflow", "drivingstereo"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])

    # Architecure choices
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    args = parser.parse_args()

    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    if device.type == 'cpu':
        torch.set_num_threads(os.cpu_count())
        torch.set_num_interop_threads(1)

    model = LiteAnyStereo()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt, map_location=device)

        target_model = model.module if hasattr(model, 'module') else model
        target_model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.to(device)
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

    if args.dataset == 'eth3d':
        validate_eth3d(model)

    elif args.dataset == 'kitti':
        validate_kitti(model, year=2012)
        validate_kitti(model, year=2015)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, resolution=args.dataset[-1])

    elif args.dataset == 'sceneflow':
        validate_sceneflow(model)

    elif args.dataset == 'drivingstereo':
        validate_driving(model, split='cloudy')
        validate_driving(model, split='foggy')
        validate_driving(model, split='rainy')
        validate_driving(model, split='sunny')