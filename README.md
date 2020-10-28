# test_repo

Modified from the original HigherHRNet code

How to run on colab

## 1. Setting up
##### 1. Clone the repository
```python
!git clone https://github.com/chechaohp/test_repo.git
!cp -r test_repo/* ./
!rm -rf test_repo
```
##### 2. Install required packages
```python
!pip install -r requirements.txt
```
## 2. Creating and Saving Student Configuration

Create a file student config like format of test_student.yaml

## 3. Start training

```\python
!python train.py \
    --student_file experiments/student2_3M.yaml \
    --log drive/"My Drive"/"Colab Notebooks"/ENGN8501/half_teacher
```

## 4. Validation 
```python
!python valid.py --student_file experiments/student2_3M.yaml \
                --log drive/"My Drive"/"Colab Notebooks"/ENGN8501/half_teacher/valid \
                --model_file drive/"My Drive"/"Colab Notebooks"/ENGN8501/half_teacher/model_best.pth.tar
```

## 5. Testing

```python
!python valid.py --student_file experiments/student2_3M.yaml \
                --log drive/"My Drive"/"Colab Notebooks"/ENGN8501/half_teacher/valid \
                --model_file drive/"My Drive"/"Colab Notebooks"/ENGN8501/half_teacher/model_best.pth.tar
```