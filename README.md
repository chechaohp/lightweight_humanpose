# test_repo


**1. *Setting up:**
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

**2. Creating and Saving Student Configuration:**
    1. Load the default configuration and function to change it
    2. Choosing new parameters for the student model
    > There are 3 method of choosing exchange units in HRNet: A, B, C
    > Type A has only final exchange unit
    > Type B has final and between-stage exchange units
    > Type C has final, between-stage and within-stage exchange units
    > The more exchange units, the higher the accuracy, but with the cost of more calculation and parameters
    > The code is only compatible for type B and C.
**3. Creating Student Model:**
    1. Load model structure
    2. Check model parameters
