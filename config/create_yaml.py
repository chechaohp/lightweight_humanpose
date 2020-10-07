import yaml

def create_yaml(default_yaml, new_yaml_folder, new_var):
    with open(default_yaml, 'r') as file:
        cfg_tree = yaml.load(file)
    
    extra = cfg_tree['MODEL']['EXTRA']

    cfg_tree['MODEL']['NAME'] = new_var.NAME
    cfg_tree['DATASET']['ROOT'] = new_var.DATASET_ROOT
    #cfg_tree['DATA_DIR'] = cfg.DATA_DIR
    cfg_tree['LOG_DIR'] = new_var.LOG_DIR
    cfg_tree['OUTPUT_DIR'] = new_var.OUTPUT_DIR
    
    extra['STEM_INPLANES'] = new_var.NUM_CHANNELS * 2    

    for i in range(new_var.NO_STAGE):
        extra['STAGE{}'.format(i+1)]['NUM_MODULES'] = new_var.NUM_MODULES[i]
        extra['STAGE{}'.format(i+1)]['NUM_BLOCKS'] = (np.ones((i+1)).astype(int) * new_var.NUM_BLOCK[i]).tolist()
        extra['STAGE{}'.format(i+1)]['NUM_CHANNELS'] = (2**np.linspace(0,i,i+1).astype(int) * new_var.NUM_CHANNELS).tolist()

    extra['DECONV']['NUM_BASIC_BLOCKS'] = new_var.NUM_BLOCK[-1]
    extra['DECONV']['NUM_CHANNELS'] = [new_var.NUM_CHANNELS]

    new_yaml = new_yaml_folder + '/' + new_var.NAME + '.yaml'

    with open(new_yaml, 'w') as file:
        documents = yaml.dump(cfg_tree, file)
