# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time

from utils.utils import AverageMeter
from utils.vis import save_debug_images


def do_train(cfg, model, data_loader, loss_factory, optimizer, epoch, output_dir, writer_dict, teacher,device):
    logger = logging.getLogger("Training")

    batch_time = AverageMeter()
    data_time = AverageMeter()

    heatmaps_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    push_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    pull_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    teacher_heatmap_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    teacher_tagmap_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, heatmaps, masks, joints) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute student output
        student_outputs = model(images.to(device))
        # compute teacher output
        teacher_outputs = teacher(images.to(device))

        heatmaps = list(map(lambda x: x.cuda(non_blocking=True), heatmaps))
        masks = list(map(lambda x: x.cuda(non_blocking=True), masks))
        joints = list(map(lambda x: x.cuda(non_blocking=True), joints))

        # student loss
        # loss = loss_factory(student_outputs, heatmaps, masks)
        student_heatmaps_losses, student_push_losses, student_pull_losses, student_teacher_heatmap_losses, student_teacher_tagmap_losses = \
            loss_factory(student_outputs, heatmaps, masks, joints, teacher_outputs)
        # teaccher_loss
        teacher_loss = 0


        student_loss = 0
        for idx in range(cfg.LOSS.NUM_STAGES):
            if student_heatmaps_losses[idx] is not None:
                heatmaps_loss = student_heatmaps_losses[idx].mean(dim=0)
                heatmaps_loss_meter[idx].update(
                    heatmaps_loss.item(), images.size(0)
                )
                student_loss = student_loss + heatmaps_loss
            if student_push_losses[idx] is not None:
                push_loss = student_push_losses[idx].mean(dim=0)
                push_loss_meter[idx].update(
                    push_loss.item(), images.size(0)
                )
                student_loss = student_loss + push_loss
            if student_pull_losses[idx] is not None:
                pull_loss = student_pull_losses[idx].mean(dim=0)
                pull_loss_meter[idx].update(
                    pull_loss.item(), images.size(0)
                )
                student_loss = student_loss + pull_loss

            if student_teacher_heatmap_losses[idx] is not None:
                student_teacher_heatmap_loss = student_teacher_heatmap_losses[idx].mean(dim=0)
                teacher_heatmap_loss_meter[idx].update(
                    student_teacher_heatmap_loss.item(), images.size(0)
                )
                teacher_loss = teacher_loss + student_teacher_heatmap_loss
            
            if student_teacher_tagmap_losses[idx] is not None:
                student_teacher_tagmap_loss = student_teacher_tagmap_losses[idx].sum(dim=0)
                teacher_tagmap_loss_meter[idx].update(
                    student_teacher_tagmap_loss.item(), images.size(0)
                )
                teacher_loss = teacher_loss + student_teacher_tagmap_loss


        alpha = cfg.TRAIN.TEACHER_WEIGHT
        loss = student_loss * alpha + (1-alpha) * teacher_loss
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # log everything
        if i % cfg.PRINT_FREQ == 0 and cfg.RANK == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  '{heatmaps_loss}{push_loss}{pull_loss}{teacher_heatmap_loss}{teacher_tagmap_loss}'.format(
                      epoch, i, len(data_loader),
                      batch_time=batch_time,
                      speed=images.size(0)/batch_time.val,
                      data_time=data_time,
                      heatmaps_loss=_get_loss_info(heatmaps_loss_meter, 'heatmaps'),
                      push_loss=_get_loss_info(push_loss_meter, 'push'),
                      pull_loss=_get_loss_info(pull_loss_meter, 'pull'),
                      teacher_heatmap_loss=_get_loss_info(teacher_heatmap_loss_meter,'teacher_heatmap'),
                      teacher_tagmap_loss=_get_loss_info(teacher_tagmap_loss_meter,'teacher_tagmap')
                  )
            logger.info(msg)
            logger.info("Total losses:" + str(loss))

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            for idx in range(cfg.LOSS.NUM_STAGES):
                writer.add_scalar(
                    'train_stage{}_heatmaps_loss'.format(i),
                    heatmaps_loss_meter[idx].val,
                    global_steps
                )
                writer.add_scalar(
                    'train_stage{}_push_loss'.format(idx),
                    push_loss_meter[idx].val,
                    global_steps
                )
                writer.add_scalar(
                    'train_stage{}_pull_loss'.format(idx),
                    pull_loss_meter[idx].val,
                    global_steps
                )
                writer.add_scalar(
                    'train_stage{}_teacher_heatmap_loss'.format(i),
                    teacher_heatmap_loss_meter[idx].val,
                    global_steps
                )
                writer.add_scalar(
                    'train_stage{}_teacher_tagmap_loss'.format(i),
                    teacher_tagmap_loss_meter[idx].val,
                    global_steps
                )
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            for scale_idx in range(len(student_outputs)):
                prefix_scale = prefix + '_output_{}'.format(
                    cfg.DATASET.OUTPUT_SIZE[scale_idx]
                )
                save_debug_images(
                    cfg, images, heatmaps[scale_idx], masks[scale_idx],
                    student_outputs[scale_idx],teacher_outputs[scale_idx], prefix_scale
                )



def _get_loss_info(loss_meters, loss_name):
    msg = ''
    for i, meter in enumerate(loss_meters):
        msg += 'Stage{i}-{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(
            i=i, name=loss_name, meter=meter
        )

    return msg
