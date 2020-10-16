## 2. Creating and Saving Student Configuration
##### 1. Load the default configuration and function to change it
```python
from config import cfg, mod_cfg_yaml, get_student_cfg
import numpy as np
```
##### 2. Choosing new parameters for the student model
```python
NUM_CHANNELS = 32
NO_STAGE = 4
TYPE = 'C'
NUM_MODULES = [4, 1, 4, 3]
NUM_BLOCKS = [4, 4, 4, 4, 4]
```
(Above are also the default parameters of our teacher model)
##### Note:
> `NUM_MODULES` are the number of HR modules in each branch in each stage.

> `NUM_BLOCKS` are the number of `Basic` or `Bottleneck` blocks in each module of each stage.

> `NUM_BLOCKS` has 1 more extra value for the number of blocks in the DECONV layers.

There are 3 method of choosing exchange units in HRNet: A, B, C
> Type A has only final exchange unit

> Type B has final and between-stage exchange units

> Type C has final, between-stage and within-stage exchange units

The more exchange units, the higher the accuracy, but with the cost of more calculation and parameters
> The code is only compatible for type B and C.

> Type B will basically has only 1 HR Module in each branch in each stage.

##### 3. Change the path
Change `LOG_DIR`, `OUTPUT_DIR`, `YAML_DIR` to change where you want to save your results.
```python
LOG_DIR =  '/content/drive/My Drive/AI_Colab/HigherHRNet/log'
OUTPUT_DIR = '/content/drive/My Drive/AI_Colab/HigherHRNet/output'
YAML_DIR = '/content/experiments'
```
##### 4. Training mode
Change the components of the student's loss function.
```python
WITH_HEATMAPS_TS_LOSS = [True, True]
WITH_TAGMAPS_TS_LOSS = [True, False]
TEACHER_WEIGHT = 0.9
```
##### 5. Create and save
Then, to create new cfg and save it to `.yaml` file
```python
student_cfg = mod_cfg_yaml(cfg, NUM_CHANNELS, TYPE, NO_STAGE, NUM_MODULES, NUM_BLOCKS,
                           LOG_DIR, OUTPUT_DIR, YAML_DIR,
                           WITH_HEATMAPS_TS_LOSS, WITH_TAGMAPS_TS_LOSS, DISTILLATION_WEIGHT)
```
You can also load your previously-saved configuration from `.yaml` file:
```python
from yacs import config

with open(YAML_PATH) as file:
    cfg = config.load_cfg(file)
```

## 3. Creating Student Model
##### 1. Load model structure
```python
from models import HHRNet
import torch
```
##### 2. Create model
```python
student = HHRNet(student_cfg)
student = torch.nn.DataParallel(student)
```
##### 3. Count model parameters
```python
def count_params(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{count_params(student):,d}", "train parameters")
```
For reference, the initial Higher HRNet has 28,645,331 trainable parameters.
