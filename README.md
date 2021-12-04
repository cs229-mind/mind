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
- for training: horovodrun -n 6 python3 run.py --mode train --batch_size=64

- for evaluation: horovodrun -n 6 python3 run.py --mode test --test_dir 'dev' --batch_size=1024

- for prediction: horovodrun -n 6 python3 run.py --mode test --test_dir 'test' --batch_size=2048
or horovodrun -n 6 python3 run_parallel.py --mode test --test_dir 'test' --batch_size=17408

Debug: while running, command kill -SIGUSR1 your_pid would tell which line of code the thread is running

Monitor: while running, command tensorboard --logdir 'runs' would be able to monitor metrics: loss, accuracy, auc, mrr, ndcg

# Submit
upload the prediction_<20211031233724>.tsv in model folder as described in the folder structure to leaderboard(https://competitions.codalab.org/competitions/24122#participate)

# Folder Structure:
By default: training data in train folder, evalution data in dev folder and test data in test folder.
  - training and evaluation supports multiple gpus, to speed up, the behavior data can be splitted to multiple files for interleave
    - command: 
      wc -l behaviors.tsv
      split behaviors.tsv -l your_num --verbose
By default: trained model will be saved to model folder, prediction result in tsv format ready for submission to leaderborad system will be saved to model folder.

By Default: the pretrained model(bert-base-uncased) can be downloaded from https://huggingface.co/bert-base-uncased/ and put it in folder ~/mind/

![image](https://user-images.githubusercontent.com/28990806/139605879-06eb35b8-5749-4cbf-9977-0ab10d977a54.png)


# Result:
  - Trivial baseline

![image](https://user-images.githubusercontent.com/28990806/140473157-e78c708d-8fb1-4c8b-a785-2f8dd4425d19.png)

  - Two tower model

![image](https://user-images.githubusercontent.com/28990806/140453040-73cd7079-b181-4e61-aad6-6a2d52524c01.png)
  - NRMS (Attention (news encoder) + Attention(user encoder))
<img width="1318" alt="nrms_model" src="https://user-images.githubusercontent.com/21976032/140476695-365d9b17-b694-46b3-aa0d-9c94250c5c45.png">

  - Two tower model(tuned)

![image](https://user-images.githubusercontent.com/28990806/140866972-2d9e8890-a883-4894-a416-bbe598de9688.png)

  - Personalized Mixed Network
![image](https://user-images.githubusercontent.com/28990806/144518339-9b91b33e-3564-4161-8d05-0dcaccfb30ce.png)


