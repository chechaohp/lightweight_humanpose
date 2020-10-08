# test_repo


**1. Setting up:**
  - Clone the repository
    ```javascript
    !git clone https://github.com/chechaohp/test_repo.git
    !cp -r test_repo/* ./
    !rm -rf test_repo
    ```
  - Install requirement package
    ```javascript
    !pip install -r requirements.txt
    ```

**1. Creating and Saving Student Configuration:**
    1. Load the default configuration and function to change it
    ```javascript
    from config import cfg, mod_cfg_yaml
    ```
    1. Choosing new parameters for the student model
    ```javascript
    NUM_CHANNELS = 32
    NO_STAGE = 4
    TYPE = 'C'
    NUM_MODULES = [4, 1, 4, 3]
    if TYPE == 'B':
        NUM_MODULES = [int(x) for x in np.ones(NO_STAGE).tolist()]
        print('For Type B, each later stages has 1 HR Modules')
    NUM_BLOCKS = [4, 4, 4, 4, 4]

    DATASET_ROOT = '/content/coco'
    LOG_DIR =  '/content/drive/My Drive/AI_Colab/HigherHRNet/log'
    OUTPUT_DIR = '/content/drive/My Drive/AI_Colab/HigherHRNet/output'
    DATA_DIR = ''
    default_yaml = '/content/experiments/default.yaml'
    #yaml_folder = '/content/drive/My Drive/AI_Colab/HigherHRNet/experiments'
    yaml_folder = '/content/experiments'
    ```
    > There are 3 method of choosing exchange units in HRNet: A, B, C
    > Type A has only final exchange unit
    > Type B has final and between-stage exchange units
    > Type C has final, between-stage and within-stage exchange units
    > The more exchange units, the higher the accuracy, but with the cost of more calculation and parameters
    > The code is only compatible for type B and C.
**1. Creating Student Model:**
    1. Load model structure
    1. Check model parameters
