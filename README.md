# mind

# PYTHON
Version: 3.6.8

# DATA
~/mind/MIND/train

~/mind/MIND/dev

~/mind/MIND/test

# TF
Name: tensorflow
Version: 2.3.4

# Package
- transformers
Version: 4.11.3

- spacy
Version: 3.1.4

- joblib
Version: 1.1.0

# CUDA 10.1 (cuDNN 7.6)
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# Horovod
HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_TENSORFLOW=1 pip3 install --no-cache-dir horovod

# CHECK
horovodrun -n 3 python3 test.py --check-build

# RUN
- for training: horovodrun -n 2 python3 run.py --mode train

- for evaluation: horovodrun -n 2 python3 run.py --mode test --test_dir 'dev'

- for prediction: horovodrun -n 2 python3 run.py --mode test --test_dir 'test'

# Submit
upload the prediction_<20211031233724>.tsv in model folder as described in the folder structure to leaderboard(https://competitions.codalab.org/competitions/24122#participate)

# Folder Structure:
By default: training data in train folder, evalution data in dev folder and test data in test folder.
  - training and evaluation supports multiple gpus, to speed up, the behavior data can be splitted to multiple files for interleave
    - command: 
      wc -l behaviors.tsv
      split behaviors.tsv -l your_num --verbose
By default: trained model will be saved to model folder, prediction result in tsv format ready for submission to leaderborad system will be saved to model folder.

![image](https://user-images.githubusercontent.com/28990806/139605879-06eb35b8-5749-4cbf-9977-0ab10d977a54.png)


# Result:
  - Trivial baseline

![image](https://user-images.githubusercontent.com/28990806/140473157-e78c708d-8fb1-4c8b-a785-2f8dd4425d19.png)

  - Two tower model

![image](https://user-images.githubusercontent.com/28990806/140453040-73cd7079-b181-4e61-aad6-6a2d52524c01.png)




