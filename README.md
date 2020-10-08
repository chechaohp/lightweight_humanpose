# test_repo



## **1. Setting up**
### 1.1. Clone the repository
```javascript
!git clone https://github.com/chechaohp/test_repo.git
!cp -r test_repo/* ./
!rm -rf test_repo
```
### 1.2. Install requirement package
```javascript
!pip install -r requirements.txt
```


## **2. Creating and Saving Student Configuration**
### 2.1. Load the default configuration and function to change it
```javascript
from config import cfg, mod_cfg_yaml
```
    
### 2.2. Choosing new parameters for the student model
(The default parameters of the teacher model)
```javascript
NUM_CHANNELS = 32
NO_STAGE = 4
TYPE = 'C'
NUM_MODULES = [4, 1, 4, 3]
if TYPE == 'B':
    NUM_MODULES = [int(x) for x in np.ones(NO_STAGE).tolist()]
    print('For Type B, each later stages has 1 HR Modules')
NUM_BLOCKS = [4, 4, 4, 4, 4]
```
#### Note:
There are 3 method of choosing exchange units in HRNet: A, B, C
> Type A has only final exchange unit

> Type B has final and between-stage exchange units

> Type C has final, between-stage and within-stage exchange units

The more exchange units, the higher the accuracy, but with the cost of more calculation and parameters
> The code is only compatible for type B and C.

> Type B will basically has only 1 HR Module in each branch in each stage.

Also change the path:
`DATASET_ROOT`, `DATA_DIR`, `default_yaml` can be ignored with default installation
Change `LOG_DIR`, `OUTPUT_DIR`, `yaml_folder` to change where you want to save your results.
```javascript
DATASET_ROOT = '/content/coco'
LOG_DIR =  '/content/drive/My Drive/AI_Colab/HigherHRNet/log'
OUTPUT_DIR = '/content/drive/My Drive/AI_Colab/HigherHRNet/output'
DATA_DIR = ''
default_yaml = '/content/experiments/default.yaml'
yaml_folder = '/content/experiments'
```

Then, to create new cfg and save it to `.yaml` file
```javascript
student_cfg = mod_cfg_yaml(cfg, NUM_CHANNELS, TYPE, NO_STAGE, NUM_MODULES, NUM_BLOCKS,
                           DATASET_ROOT, LOG_DIR, OUTPUT_DIR, DATA_DIR, default_yaml, yaml_folder)
```
    
## **3. Creating Student Model**
### 3.1. Load model structure
```javascript
from models import HHRNet
import torch
```
### 3.2. Create model
```javascript
student = HHRNet(student_cfg)
student = torch.nn.DataParallel(student)
```
### 3.3. Count model parameters
```javascript
def count_params(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{count_params(student):,d}", "train parameters")
```
For reference, the initial Higher HRNet has 28,645,331 trainable parameters.
