from config import cfg, get_student_cfg, mod_cfg_yaml
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch

from utils.utils import create_logger
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import setup_logger

from core.new_trainer import do_train
from core.loss import MultiLossFactory
from models.pose_higher_hrnet import PoseHigherResolutionNet
from models.hhrnet import HHRNet

from tensorboardX import SummaryWriter

from dataset import make_dataloader
import time
import copy
import yaml
import os

import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Run Linear Classifer.')
    parser.add_argument('--NUM_CHANNELS', default=32)
    parser.add_argument('--NO_STAGE', default=4)
    parser.add_argument('--TYPE', default='C')
    parser.add_argument('--NUM_MODULES', default=[4, 1, 4, 3])
    parser.add_argument('--NUM_BLOCKS', default=[4, 4, 4, 4, 4])
    parser.add_argument('--LOG_DIR', default='/content/drive/My Drive/AI_Colab/HigherHRNet/log')
    parser.add_argument('--OUTPUT_DIR', default='/content/drive/My Drive/AI_Colab/HigherHRNet/output')
    parser.add_argument('--YAML_DIR', default='/content/experiments')
    parser.add_argument('--WITH_HEATMAPS_TS_LOSS', default=[True, True])
    parser.add_argument('--WITH_TAGMAPS_TS_LOSS', default=[True, False])
    parser.add_argument('--TEACHER_WEIGHT', default=0.9)
    args = parser.parse_args()
    return args


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = get_args()
    # Modified these information
    NUM_CHANNELS = int(args.NUM_CHANNELS)
    NO_STAGE = int(args.NO_STAGE)
    TYPE = args.TYPE
    NUM_MODULES = eval(args.NUM_MODULES)
    NUM_BLOCKS = eval(args.NUM_BLOCKS)
    LOG_DIR =  args.LOG_DIR
    OUTPUT_DIR = args.OUTPUT_DIR
    YAML_DIR = args.YAML_DIR
    WITH_HEATMAPS_TS_LOSS = eval(args.WITH_HEATMAPS_TS_LOSS)
    WITH_TAGMAPS_TS_LOSS = eval(args.WITH_TAGMAPS_TS_LOSS)
    TEACHER_WEIGHT = float(args.TEACHER_WEIGHT)
    # create teacher
    model_path = './pose_higher_hrnet_w32_512_2.pth'
    pre_train_model = PoseHigherResolutionNet(cfg)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load pretrain
    pre_train_model.load_state_dict(torch.load(model_path,torch.device(dev)))

    # freeze teacher
    for param in pre_train_model.parameters():
        param.requires_grad = False

    # student = PoseHigherResolutionNet(new_cfg)
    student_cfg = mod_cfg_yaml(cfg, NUM_CHANNELS, TYPE, NO_STAGE, NUM_MODULES, NUM_BLOCKS,
                           LOG_DIR, OUTPUT_DIR, YAML_DIR,
                           WITH_HEATMAPS_TS_LOSS, WITH_TAGMAPS_TS_LOSS, TEACHER_WEIGHT)
    student = HHRNet(student_cfg)
    student = torch.nn.DataParallel(student)

    # Set up logger
    logger, final_output_dir, tb_log_dir = create_logger(
            student_cfg, 'simple_model', 'train'
        )

    final_output_dir = student_cfg.LOG_DIR

    if torch.cuda.is_available():
        # cudnn related setting
        cudnn.benchmark = student_cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = student_cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = student_cfg.CUDNN.ENABLED


    train_loader = make_dataloader(student_cfg,True,False)
    # iteration = 1

    loss_factory = MultiLossFactory(student_cfg).cuda()

    logger.info(train_loader.dataset)

    writer_dict = {
            'writer': SummaryWriter(log_dir=tb_log_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }

    best_perf = -1
    best_model = False
    last_epoch = -1

    optimizer = optim.Adam(
                student.parameters(),
                lr=student_cfg.TRAIN.LR
            )
    begin_epoch = student_cfg.TRAIN.BEGIN_EPOCH

    end_epoch = student_cfg.TRAIN.END_EPOCH

    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth.tar')

    if student_cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        student.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))
        
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, student_cfg.TRAIN.LR_STEP, student_cfg.TRAIN.LR_FACTOR,
                last_epoch=last_epoch
            )

    pre_train_model.to(dev)
    student.to(dev)
    for epoch in range(begin_epoch, end_epoch):
        start = time.time()
        do_train(cfg,student,train_loader,loss_factory,optimizer,epoch,final_output_dir,writer_dict, pre_train_model,dev)
        print('epoch',epoch,':',round((time.time() - start)/60,2),'minutes')
        # In PyTorch 1.1.0 and later, you should call `lr_scheduler.step()` after `optimizer.step()`.
        lr_scheduler.step()

        perf_indicator = epoch
        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': student.state_dict(),
            'best_state_dict': student.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)
        final_model_state_file = os.path.join(
            final_output_dir, 'final_state{}.pth.tar'.format(torch.cuda.get_device_name())
        )
        logger.info('saving final model state to {}'.format(
            final_model_state_file))
        torch.save(student.module.state_dict(), final_model_state_file)
        writer_dict['writer'].close()



if __name__ == "__main__":
    main()