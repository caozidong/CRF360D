from __future__ import absolute_import, division, print_function
import os
import argparse
import tqdm
import yaml
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from networks import *
import datasets
from metrics import Evaluator
from networks.models import *
from saver import Saver


def main(config):
    model_path = os.path.join(config["load_weights_dir"], "model.pth")
    model_dict = torch.load(model_path)

    # data
    datasets_dict = {"stanford2d3d": datasets.Stanford2D3D,
                     "matterport3d": datasets.Matterport3D}
    cf_test = config['test_dataset']
    dataset = datasets_dict[cf_test['name']]

    test_dataset = dataset(cf_test['root_path'], 
                            cf_test['list_path'],
                            cf_test['args']['height'],
                            cf_test['args']['width'], 
                            cf_test['args']['augment_color'],
                            cf_test['args']['augment_flip'],
                            cf_test['args']['augment_rotation'],
                            cf_test['args']['repeat'],
                            is_training=False)
    test_loader = DataLoader(test_dataset, 
                            cf_test['batch_size'], 
                            False,
                            num_workers=cf_test['num_workers'], 
                            pin_memory=True, 
                            drop_last=False)
    num_test_samples = len(test_dataset)
    num_steps = num_test_samples // cf_test['batch_size']
    print("Num. of test samples:", num_test_samples, "Num. of steps:", num_steps, "\n")

    # network
    model = make(config['model'])
    # model = nn.parallel.DataParallel(model)
    model.cuda()

    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict})
    model.eval()

    evaluator = Evaluator(config['median_align'])
    evaluator.reset_eval_metrics()
    saver = Saver(config["load_weights_dir"])
    pbar = tqdm.tqdm(test_loader)
    pbar.set_description("Testing")

    with torch.no_grad():
        for batch_idx, inputs in enumerate(pbar):

            equi_inputs = inputs["normalized_rgb"].cuda()

            cube_inputs = inputs["normalized_cube_rgb"].cuda()

            # from thop import profile
            # from thop import clever_format
            # macs, params = profile(model, inputs=(equi_inputs, cube_inputs))
            # print(macs, params) 
            # macs, params = clever_format([macs, params], "%.3f")
            # print(macs, params) 
            # assert False

            outputs = model(equi_inputs, cube_inputs)

            pred_depth = outputs["pred_depth"].detach().cpu()

            gt_depth = inputs["gt_depth"]
            mask = inputs["val_mask"]
            for i in range(gt_depth.shape[0]):
                evaluator.compute_eval_metrics(gt_depth[i:i + 1], pred_depth[i:i + 1], mask[i:i + 1])

            # if settings.save_samples:
            #     saver.save_samples(inputs["rgb"], gt_depth, pred_depth, mask)
            # saver.save_samples(inputs["rgb"], gt_depth, pred_depth, mask)

    # evaluator.print(config["load_weights_dir"])
    evaluator.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    main(config)