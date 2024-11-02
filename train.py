from __future__ import absolute_import, division, print_function

import argparse
import os
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train_matterport3d/segnet_v1.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')
        
    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    save_path = os.path.join('./tmp', save_name)

    from trainer import Trainer
    trainer = Trainer(config, save_path)
    trainer.train()
