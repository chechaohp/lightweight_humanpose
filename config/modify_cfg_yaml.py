from yacs.config import CfgNode as CN
import numpy as np
import copy
import yaml

def create_var(NUM_CHANNELS, NO_STAGE, TYPE, NUM_MODULES, NUM_BLOCK, VERSION, DATASET_ROOT, LOG_DIR, OUTPUT_DIR):
    new_var = CN()

    NUM_CHANNELS = int(NUM_CHANNELS)
    new_var.NUM_CHANNELS = NUM_CHANNELS

    new_var.TYPE = TYPE.upper()

    NO_STAGE = int(NO_STAGE)
    new_var.NO_STAGE = NO_STAGE

    NUM_MODULES = NUM_MODULES.replace(',', '').replace(' ', '')
    assert len(NUM_MODULES) == NO_STAGE, \
        'NUM_MODULES should has the length as NO_STAGE'
    new_var.NUM_MODULES = [int(x) for x in NUM_MODULES]

    NUM_BLOCK = NUM_BLOCK.replace(',', '').replace(' ', '')
    assert len(NUM_BLOCK) == NO_STAGE+1, \
        'NUM_BLOCK should has the length as NO_STAGE+1, the last is DECONV blocks'    
    new_var.NUM_BLOCK = [int(x) for x in NUM_BLOCK]

    new_var.VERSION = int(VERSION)

    new_var.NAME = 'hhrnet_{}{}{}m{}b{}'.format(NUM_CHANNELS, TYPE.lower(),
                                                NO_STAGE, NUM_MODULES, NUM_BLOCK)
    new_var.DATASET_ROOT = DATASET_ROOT
    new_var.LOG_DIR = LOG_DIR
    new_var.OUTPUT_DIR = OUTPUT_DIR
    return new_var
    


def mod_cfg(cfg, new_var):
    new_cfg = copy.deepcopy(cfg)
    extra = new_cfg.MODEL.EXTRA
    extra.STEM_INPLANES = new_var.NUM_CHANNELS * 2

    for i in range(new_var.NO_STAGE):
        extra['STAGE{}'.format(i+1)]['NUM_MODULES'] = new_var.NUM_MODULES[i]
        extra['STAGE{}'.format(i+1)]['NUM_BLOCKS'] = np.ones((i+1)).astype(int) * new_var.NUM_BLOCK[i]
        extra['STAGE{}'.format(i+1)]['NUM_CHANNELS'] = 2**np.linspace(0,i,i+1).astype(int) * new_var.NUM_CHANNELS

    extra.DECONV.NUM_CHANNELS = [new_var.NUM_CHANNELS]
    extra.DECONV.NUM_BASIC_BLOCKS = new_var.NUM_BLOCK[-1]

    new_cfg.MODEL.NO_STAGE = new_var.NO_STAGE
    new_cfg.MODEL.NAME = new_var.NAME
    new_cfg.MODEL.TYPE = new_var.TYPE
    new_cfg.MODEL.VERSION = new_var.VERSION

    new_cfg.DATASET.ROOT = new_var.DATASET_ROOT
    new_cfg.LOG_DIR =  new_var.LOG_DIR
    new_cfg.OUTPUT_DIR =  new_var.OUTPUT_DIR

    return new_cfg
