import argparse
import os
import pathlib
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import utils
from eval import Eval as Evaluator
from losses import LossFn
from test import test
import sys
import numpy as np
from os import path
import socket
import random
from torch.backends import cudnn
import torch
import logging
from train import train
from torch import nn
from dataset import Loader3D
from model import Competitor


def save_weights(epoch, model, optim, score, path):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict(),
        'metric': score
    }
    torch.save(state, path)


def main(experiment_name, args):

    assert torch.cuda.is_available()
    logging.info(f"This model will run on {torch.cuda.get_device_name(torch.cuda.current_device())}")

    ## DETERMINISTIC SET-UP
    seed = config.get('seed', 47)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # END OF DETERMINISTIC SET-UP

    loader_config = config.get('data-loader', None)
    train_config = config.get('trainer', None)

    model = Competitor(n_classes=1)
    logging.info('using data parallel')
    model = nn.DataParallel(model).cuda()

    train_params = model.parameters()

    optim_config = config.get('optimizer')
    optim_name = optim_config.get('name', None)
    if not optim_name or optim_name == 'Adam':
        optimizer = torch.optim.Adam(params=train_params, lr=optim_config['learning_rate'])
    elif optim_name == 'SGD':
        optimizer = torch.optim.SGD(params=train_params, lr=optim_config['learning_rate'])
    else:
        raise Exception("optimizer not recognized")

    sched_config = config.get('lr_scheduler')
    scheduler_name = sched_config.get('name', None)
    if scheduler_name == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=sched_config['milestones'],
            gamma=sched_config.get('factor', 0.1),
        )
    elif scheduler_name == 'Plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=7)
    else:
        scheduler = None

    evaluator = Evaluator(loader_config, project_dir, skip_dump=args.skip_dump)

    loss = LossFn(config.get('loss'), loader_config, weights=None)  # TODO: fix this, weights are disabled now

    start_epoch = 0
    if train_config['checkpoint_path'] is not None:
        try:
            checkpoint = torch.load(train_config['checkpoint_path'])
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info(f"Checkpoint loaded successfully at epoch {start_epoch}, score:{checkpoint.get('metric', 'unavailable')})")
        except OSError as e:
            logging.info("No checkpoint exists from '{}'. Skipping...".format(train_config['checkpoint_path']))

    # DATA LOADING
    data_utils = Loader3D(loader_config, train_config.get("do_train", True), args.additional_dataset, args.competitor, args.skip_primary)
    train_d, test_d, val_d = data_utils.split_dataset()
    test_loader = [(test_p, data.DataLoader(test_p, loader_config['batch_size'], num_workers=loader_config['num_workers'])) for test_p in test_d]
    val_loader = [(val_p, data.DataLoader(val_p, loader_config['batch_size'], num_workers=loader_config['num_workers'])) for val_p in val_d]

    if train_config['do_train']:

        train_loader = data.DataLoader(train_d, loader_config['batch_size'], num_workers=loader_config['num_workers'])

        # creating training writer (purge on)
        writer = SummaryWriter(log_dir=os.path.join(config['tb_dir'], experiment_name), purge_step=start_epoch)

        best_val = 0
        best_test = 0

        for epoch in range(start_epoch, train_config['epochs']):

            train(model, train_loader, loss, optimizer, epoch, writer, evaluator, phase="Train")

            val_model = model.module
            val_iou, val_dice, val_haus = test(val_model, val_loader, epoch, writer, evaluator, phase="Validation")

            if val_iou < 1e-05 and epoch > 15:
                logging.info('WARNING: drop in performances detected.')

            if scheduler is not None:
                if optim_name == 'SGD' and scheduler_name == 'Plateau':
                    scheduler.step(val_iou)
                else:
                    scheduler.step(epoch)

            save_weights(epoch, model, optimizer, val_iou, os.path.join(project_dir, 'checkpoints', 'last.pth'))

            if val_iou > best_val:
                best_val = val_iou
                save_weights(epoch, model, optimizer, best_val, os.path.join(project_dir, 'best.pth'))

            if epoch % 5 == 0 and epoch != 0:
                test_iou, _, _ = test(val_model, test_loader, epoch, writer, evaluator, phase="Test")
                best_test = best_test if best_test > test_iou else test_iou

        logging.info('BEST TEST METRIC IS {}'.format(best_test))

    val_model = model.module
    final_iou, final_dice, _ = test(val_model, test_loader, epoch="Final", writer=None, evaluator=evaluator, phase="Final")


if __name__ == '__main__':

    RESULTS_DIR = r'/localpath/results'
    BASE_YAML_PATH = os.path.join('configs', 'config.yaml')

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--base_config', default="config.yaml", help='path to the yaml config file')
    arg_parser.add_argument('--verbose', action='store_true', help="if true sdout is not redirected, default: false")
    arg_parser.add_argument('--skip_dump', action='store_true', help="dump test data, default: false")
    arg_parser.add_argument('--test', action='store_true', help="set up test params, default: false")
    arg_parser.add_argument('--competitor', action='store_true', help='competitor trains on sparse, default: false')
    arg_parser.add_argument('--additional_dataset', action='store_true', help='add also the syntetic dataset, default: false')
    arg_parser.add_argument('--reload', action='store_true', help='reload experiment?, default: false')
    arg_parser.add_argument('--skip_primary', action='store_true', help='do not load primary train data, default: false')

    args = arg_parser.parse_args()
    yaml_path = args.base_config

    if path.exists(yaml_path):
        print(f"loading config file in {yaml_path}")
        config = utils.load_config_yaml(yaml_path)
        experiment_name = config.get('title')
        project_dir = os.path.join(RESULTS_DIR, experiment_name)
    else:
        config = utils.load_config_yaml(BASE_YAML_PATH)  # load base config (remote or local)
        experiment_name = config.get('title', 'test')
        print('this experiment is on debug. no folders are going to be created.')
        project_dir = os.path.join(RESULTS_DIR, 'test')

    log_dir = pathlib.Path(os.path.join(project_dir, 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)

    if not args.verbose:
        # redirect streams to project dir
        sys.stdout = open(os.path.join(log_dir, 'std.log'), 'a+')
        sys.stderr = sys.stdout
        utils.set_logger(os.path.join(log_dir, 'logging.log'))
    else:
        # not create folder here, just log to console
        utils.set_logger()

    if args.test:
        config['trainer']['do_train'] = False
        config['data-loader']['num_workers'] = 0
        config['trainer']['checkpoint_path'] = os.path.join(project_dir, 'checkpoints', 'last.pth')

    if args.reload:
        logging.info("RELOAD! setting checkpoint path to last.pth")
        config['trainer']['checkpoint_path'] = os.path.join(project_dir, 'checkpoints', 'last.pth')

    main(experiment_name, args)
