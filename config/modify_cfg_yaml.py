import numpy as np
import copy
import os
import yaml


def mod_cfg_yaml(cfg, NUM_CHANNELS, TYPE, NO_STAGE, NUM_MODULES, NUM_BLOCKS,
                DATASET_ROOT, LOG_DIR, OUTPUT_DIR, DATA_DIR, default_yaml, yaml_folder):

    assert type(NUM_CHANNELS) == int, 'Input for NUM_CHANNELS should be an integer'
    assert type(NO_STAGE) == int, 'Input for NO_STAGE should be an integer'
    assert type(TYPE) == str, 'Input for TYPE should be a string'
    TYPE = TYPE.upper()
    assert isinstance(NUM_MODULES, (list, tuple)), 'Input for NUM_MODULES should be a list'
    assert isinstance(NUM_BLOCKS, (list, tuple)), 'Input for NUM_BLOCKS should be a list'
    for NUM_MODULE in NUM_MODULES:
        assert type(NUM_MODULE) == int, 'Input for NUM_MODULES should be integers'
    for NUM_BLOCK in NUM_BLOCKS:
        assert type(NUM_BLOCK) == int, 'Input for NUM_BLOCKS should be integers'
    assert len(NUM_MODULES) == NO_STAGE, 'Length of NUM_MODULES should be {}'.format(NO_STAGE)
    assert len(NUM_BLOCKS) == NO_STAGE+1, 'Length of NUM_BLOCKS should be {}'.format(NO_STAGE+1)

    new_cfg = copy.deepcopy(cfg)
    extra = new_cfg.MODEL.EXTRA
    extra.NO_STAGE = NO_STAGE
    extra.TYPE = TYPE
    new_cfg.DATASET.ROOT = DATASET_ROOT
    new_cfg.LOG_DIR =  LOG_DIR
    new_cfg.OUTPUT_DIR =  OUTPUT_DIR

    VERSION = 1
    NAME = 'hhrnet_{}{}{}_ver{}'.format(NUM_CHANNELS, TYPE, NO_STAGE, VERSION)
    while os.path.exists(yaml_folder + '/' + NAME + '.yaml'):
        VERSION += 1
        NAME = 'hhrnet_{}{}{}_ver{}'.format(NUM_CHANNELS, TYPE, NO_STAGE, VERSION)
    new_cfg.MODEL.NAME = NAME    

    extra.STEM_INPLANES = NUM_CHANNELS * 2
    for i in range(NO_STAGE):
        extra['STAGE{}'.format(i+1)]['NUM_MODULES'] = NUM_MODULES[i]
        extra['STAGE{}'.format(i+1)]['NUM_BLOCKS'] = np.ones((i+1)).astype(int) * NUM_BLOCKS[i]
        extra['STAGE{}'.format(i+1)]['NUM_CHANNELS'] = 2**np.linspace(0,i,i+1).astype(int) * NUM_CHANNELS
    extra.DECONV.NUM_CHANNELS = [NUM_CHANNELS]
    extra.DECONV.NUM_BASIC_BLOCKS = NUM_BLOCKS[-1]        
    

    with open(default_yaml, 'r') as file:
        cfg_tree = yaml.load(file)
    
    cfg_extra = cfg_tree['MODEL']['EXTRA']
    cfg_tree['MODEL']['NAME'] = NAME
    cfg_tree['DATASET']['ROOT'] = DATASET_ROOT    
    cfg_tree['LOG_DIR'] = LOG_DIR
    cfg_tree['OUTPUT_DIR'] = OUTPUT_DIR
    cfg_tree['DATA_DIR'] = DATA_DIR
    
    cfg_extra['STEM_INPLANES'] = NUM_CHANNELS * 2    

    for i in range(NO_STAGE):
        cfg_extra['STAGE{}'.format(i+1)]['NUM_MODULES'] = NUM_MODULES[i]
        cfg_extra['STAGE{}'.format(i+1)]['NUM_BLOCKS'] = (np.ones((i+1)).astype(int) * NUM_BLOCKS[i]).tolist()
        cfg_extra['STAGE{}'.format(i+1)]['NUM_CHANNELS'] = (2**np.linspace(0,i,i+1).astype(int) * NUM_CHANNELS).tolist()

    cfg_extra['DECONV']['NUM_BASIC_BLOCKS'] = NUM_BLOCKS[-1]
    cfg_extra['DECONV']['NUM_CHANNELS'] = [NUM_CHANNELS]

    new_yaml = yaml_folder + '/' + NAME + '.yaml'

    with open(new_yaml, 'w') as file:
        documents = yaml.dump(cfg_tree, file)    

    return new_cfg
