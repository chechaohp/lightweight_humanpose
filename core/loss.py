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

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


def make_input(t, requires_grad=False, need_cuda=True):
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    inp = inp.sum()
    if need_cuda:
        inp = inp.cuda()
    return inp


class TeacherStudentTagLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, student_tagmap, teacher_tagmap):
        assert student_tagmap.size() ==  teacher_tagmap.size()
        loss = torch.pow(student_tagmap-teacher_tagmap,2)
        return loss.sum(dim=3).sum(dim=2).sum(dim=1)



class TeacherStudentHeatMapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, student_heatmap, teacher_heatmap, mask):
        assert student_heatmap.size() == teacher_heatmap.size()
        loss = ((student_heatmap - teacher_heatmap)**2) * mask[:, None, :, :].expand_as(teacher_heatmap)
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
        # loss = loss.mean(dim=3).mean(dim=2).sum(dim=1)
        return loss


class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):
        assert pred.size() == gt.size()
        loss = ((pred - gt)**2) * mask[:, None, :, :].expand_as(pred)
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
        # loss = loss.mean(dim=3).mean(dim=2).sum(dim=1)
        return loss


class AELoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type

    def singleTagLoss(self, pred_tag, joints):
        """
        associative embedding loss for one image
        """
        tags = []
        pull = 0
        for joints_per_person in joints:
            tmp = []
            for joint in joints_per_person:
                if joint[1] > 0:
                    tmp.append(pred_tag[joint[0]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)
            tags.append(torch.mean(tmp, dim=0))
            pull = pull + torch.mean((tmp - tags[-1].expand_as(tmp))**2)

        num_tags = len(tags)
        if num_tags == 0:
            return make_input(torch.zeros(1).float()), \
                make_input(torch.zeros(1).float())
        elif num_tags == 1:
            return make_input(torch.zeros(1).float()), \
                pull/(num_tags)

        tags = torch.stack(tags)

        size = (num_tags, num_tags)
        A = tags.expand(*size)
        B = A.permute(1, 0)

        diff = A - B

        if self.loss_type == 'exp':
            diff = torch.pow(diff, 2)
            push = torch.exp(-diff)
            push = torch.sum(push) - num_tags
        elif self.loss_type == 'max':
            diff = 1 - torch.abs(diff)
            push = torch.clamp(diff, min=0).sum() - num_tags
        else:
            raise ValueError('Unkown ae loss type')

        return push/((num_tags - 1) * num_tags) * 0.5, \
            pull/(num_tags)

    def forward(self, tags, joints):
        """
        accumulate the tag loss for each image in the batch
        """
        pushes, pulls = [], []
        joints = joints.cpu().data.numpy()
        batch_size = tags.size(0)
        for i in range(batch_size):
            push, pull = self.singleTagLoss(tags[i], joints[i])
            pushes.append(push)
            pulls.append(pull)
        return torch.stack(pushes), torch.stack(pulls)


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class LossFactory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.heatmaps_loss = None
        self.ae_loss = None
        self.student_teacher_loss = None
        self.student_teacher_tag_loss = None
        self.heatmaps_loss_factor = 1.0
        self.push_loss_factor = 1.0
        self.pull_loss_factor = 1.0

        if cfg.LOSS.WITH_HEATMAPS_LOSS:
            self.heatmaps_loss = HeatmapLoss()
            self.heatmaps_loss_factor = cfg.LOSS.HEATMAPS_LOSS_FACTOR
        if cfg.LOSS.WITH_AE_LOSS:
            self.ae_loss = AELoss(cfg.LOSS.AE_LOSS_TYPE)
            self.push_loss_factor = cfg.LOSS.PUSH_LOSS_FACTOR
            self.pull_loss_factor = cfg.LOSS.PULL_LOSS_FACTOR

        if cfg.LOSS.WITH_TAGMAPS_TS_LOSS:
            self.student_teacher_tag_loss = TeacherStudentTagLoss()
        
        if cfg.LOSS.WITH_HEATMAPS_TS_LOSS:
            self.student_teacher_loss = TeacherStudentHeatMapLoss()

        if not self.heatmaps_loss and not self.ae_loss:
            logger.error('At least enable one loss!')

    def forward(self, outputs, heatmaps, masks, joints, teacher_outputs):
        # TODO(bowen): outputs and heatmaps can be lists of same length
        heatmaps_pred = outputs[:, :self.num_joints]
        tags_pred = outputs[:, self.num_joints:]

        heatmap_teacher = teacher_outputs[:,:self.num_joints]
        tags_teacher = teacher_outputs[:, self.num_joints:]

        heatmaps_loss = None
        push_loss = None
        pull_loss = None
        student_teacher_loss = None
        student_teacher_tag_loss = None

        if self.heatmaps_loss is not None:
            heatmaps_loss = self.heatmaps_loss(heatmaps_pred, heatmaps, masks)
            heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor

        if self.ae_loss is not None:
            batch_size = tags_pred.size()[0]
            tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)

            push_loss, pull_loss = self.ae_loss(tags_pred, joints)
            push_loss = push_loss * self.push_loss_factor
            pull_loss = pull_loss * self.pull_loss_factor

        if self.student_teacher_loss is not None:
            student_teacher_loss = self.student_teacher_loss(heatmaps_pred,heatmap_teacher)

        if self.student_teacher_tag_loss is not None:
            student_teacher_loss = self.student_teacher_loss(tags_pred, tags_teacher)

        return [heatmaps_loss], [push_loss], [pull_loss], [student_teacher_loss], [student_teacher_tag_loss]


class MultiLossFactory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # init check
        self._init_check(cfg)

        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.num_stages = cfg.LOSS.NUM_STAGES

        self.heatmaps_loss = \
            nn.ModuleList(
                [
                    HeatmapLoss()
                    if with_heatmaps_loss else None
                    for with_heatmaps_loss in cfg.LOSS.WITH_HEATMAPS_LOSS
                ]
            )
        self.heatmaps_loss_factor = cfg.LOSS.HEATMAPS_LOSS_FACTOR

        self.ae_loss = \
            nn.ModuleList(
                [
                    AELoss(cfg.LOSS.AE_LOSS_TYPE) if with_ae_loss else None
                    for with_ae_loss in cfg.LOSS.WITH_AE_LOSS
                ]
            )

        self.student_teacher_heatmap_loss = \
            nn.ModuleList(
                [
                    TeacherStudentHeatMapLoss() if with_heatmaps_loss else None
                    for with_heatmaps_loss in cfg.LOSS.WITH_HEATMAPS_TS_LOSS
                ]
            )

        self.student_teacher_tag_loss = \
            nn.ModuleList(
                [
                    TeacherStudentTagLoss() if with_tagmaps_loss else None
                    for with_tagmaps_loss in cfg.LOSS.WITH_TAGMAPS_TS_LOSS
                ]
            )
            
        self.push_loss_factor = cfg.LOSS.PUSH_LOSS_FACTOR
        self.pull_loss_factor = cfg.LOSS.PULL_LOSS_FACTOR

    def forward(self, outputs, heatmaps, masks, joints, teacher_outputs):
        # forward check
        self._forward_check(outputs, heatmaps, masks, joints)

        heatmaps_losses = []
        push_losses = []
        pull_losses = []
        student_teacher_heatmap_losses = []
        student_teacher_tagmap_losses = []

        for idx in range(len(outputs)):
            offset_feat = 0
            if self.heatmaps_loss[idx]:
                heatmaps_pred = outputs[idx][:, :self.num_joints]

                heatmap_teacher = teacher_outputs[idx][:, :self.num_joints]

                offset_feat = self.num_joints

                heatmaps_loss = self.heatmaps_loss[idx](
                    heatmaps_pred, heatmaps[idx], masks[idx]
                )

                student_teacher_heatmap_loss = self.student_teacher_heatmap_loss[idx](
                    heatmaps_pred, heatmap_teacher, masks[idx]
                )

                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                heatmaps_losses.append(heatmaps_loss)
                student_teacher_heatmap_losses.append(student_teacher_heatmap_loss)
            else:
                heatmaps_losses.append(None)
                student_teacher_heatmap_losses.append(None)

            if self.ae_loss[idx]:
                tags_pred = outputs[idx][:, offset_feat:]
                batch_size = tags_pred.size()[0]
                tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)

                push_loss, pull_loss = self.ae_loss[idx](
                    tags_pred, joints[idx]
                )
                push_loss = push_loss * self.push_loss_factor[idx]
                pull_loss = pull_loss * self.pull_loss_factor[idx]

                push_losses.append(push_loss)
                pull_losses.append(pull_loss)
            else:
                push_losses.append(None)
                pull_losses.append(None)

            if self.student_teacher_tag_loss[idx]:
                tags_pred = outputs[idx][:, self.num_joints:]
                teacher_tag_pred = teacher_outputs[idx][:, self.num_joints:]
                student_teacher_tagmap_loss = self.student_teacher_tag_loss[idx](tags_pred,teacher_tag_pred)
                student_teacher_tagmap_losses.append(student_teacher_heatmap_loss)
            else:
                student_teacher_tagmap_losses.append(None)

        # print(len(student_teacher_heatmap_losses))
        return heatmaps_losses, push_losses, pull_losses, student_teacher_heatmap_losses, student_teacher_tagmap_losses

    def _init_check(self, cfg):
        assert isinstance(cfg.LOSS.WITH_HEATMAPS_LOSS, (list, tuple)), \
            'LOSS.WITH_HEATMAPS_LOSS should be a list or tuple'
        assert isinstance(cfg.LOSS.HEATMAPS_LOSS_FACTOR, (list, tuple)), \
            'LOSS.HEATMAPS_LOSS_FACTOR should be a list or tuple'
        assert isinstance(cfg.LOSS.WITH_AE_LOSS, (list, tuple)), \
            'LOSS.WITH_AE_LOSS should be a list or tuple'
        assert isinstance(cfg.LOSS.PUSH_LOSS_FACTOR, (list, tuple)), \
            'LOSS.PUSH_LOSS_FACTOR should be a list or tuple'
        assert isinstance(cfg.LOSS.PUSH_LOSS_FACTOR, (list, tuple)), \
            'LOSS.PUSH_LOSS_FACTOR should be a list or tuple'
        assert len(cfg.LOSS.WITH_HEATMAPS_LOSS) == cfg.LOSS.NUM_STAGES, \
            'LOSS.WITH_HEATMAPS_LOSS and LOSS.NUM_STAGE should have same length, got {} vs {}.'.\
                format(len(cfg.LOSS.WITH_HEATMAPS_LOSS), cfg.LOSS.NUM_STAGES)
        assert len(cfg.LOSS.WITH_HEATMAPS_LOSS) == len(cfg.LOSS.HEATMAPS_LOSS_FACTOR), \
            'LOSS.WITH_HEATMAPS_LOSS and LOSS.HEATMAPS_LOSS_FACTOR should have same length, got {} vs {}.'.\
                format(len(cfg.LOSS.WITH_HEATMAPS_LOSS), len(cfg.LOSS.HEATMAPS_LOSS_FACTOR))
        assert len(cfg.LOSS.WITH_AE_LOSS) == cfg.LOSS.NUM_STAGES, \
            'LOSS.WITH_AE_LOSS and LOSS.NUM_STAGE should have same length, got {} vs {}.'.\
                format(len(cfg.LOSS.WITH_AE_LOSS), cfg.LOSS.NUM_STAGES)
        assert len(cfg.LOSS.WITH_AE_LOSS) == len(cfg.LOSS.PUSH_LOSS_FACTOR), \
            'LOSS.WITH_AE_LOSS and LOSS.PUSH_LOSS_FACTOR should have same length, got {} vs {}.'. \
                format(len(cfg.LOSS.WITH_AE_LOSS), len(cfg.LOSS.PUSH_LOSS_FACTOR))
        assert len(cfg.LOSS.WITH_AE_LOSS) == len(cfg.LOSS.PULL_LOSS_FACTOR), \
            'LOSS.WITH_AE_LOSS and LOSS.PULL_LOSS_FACTOR should have same length, got {} vs {}.'. \
                format(len(cfg.LOSS.WITH_AE_LOSS), len(cfg.LOSS.PULL_LOSS_FACTOR))

    def _forward_check(self, outputs, heatmaps, masks, joints):
        assert isinstance(outputs, list), \
            'outputs should be a list, got {} instead.'.format(type(outputs))
        assert isinstance(heatmaps, list), \
            'heatmaps should be a list, got {} instead.'.format(type(heatmaps))
        assert isinstance(masks, list), \
            'masks should be a list, got {} instead.'.format(type(masks))
        assert isinstance(joints, list), \
            'joints should be a list, got {} instead.'.format(type(joints))
        assert len(outputs) == self.num_stages, \
            'len(outputs) and num_stages should been same, got {} vs {}.'.format(len(outputs), self.num_stages)
        assert len(outputs) == len(heatmaps), \
            'outputs and heatmaps should have same length, got {} vs {}.'.format(len(outputs), len(heatmaps))
        assert len(outputs) == len(masks), \
            'outputs and masks should have same length, got {} vs {}.'.format(len(outputs), len(masks))
        assert len(outputs) == len(joints), \
            'outputs and joints should have same length, got {} vs {}.'.format(len(outputs), len(joints))
        assert len(outputs) == len(self.heatmaps_loss), \
            'outputs and heatmaps_loss should have same length, got {} vs {}.'. \
                format(len(outputs), len(self.heatmaps_loss))
        assert len(outputs) == len(self.ae_loss), \
            'outputs and ae_loss should have same length, got {} vs {}.'. \
                format(len(outputs), len(self.ae_loss))


def test_ae_loss():
    import numpy as np
    t = torch.tensor(
        np.arange(0, 32).reshape(1, 2, 4, 4).astype(np.float)*0.1,
        requires_grad=True
    )
    t.register_hook(lambda x: print('t', x))

    ae_loss = AELoss(loss_type='exp')

    joints = np.zeros((2, 2, 2))
    joints[0, 0] = (3, 1)
    joints[1, 0] = (10, 1)
    joints[0, 1] = (22, 1)
    joints[1, 1] = (30, 1)
    joints = torch.LongTensor(joints)
    joints = joints.view(1, 2, 2, 2)

    t = t.contiguous().view(1, -1, 1)
    l = ae_loss(t, joints)

    print(l)


if __name__ == '__main__':
    test_ae_loss()
