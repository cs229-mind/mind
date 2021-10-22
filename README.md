# mind

# DATA
/mind/MIND/test
/mind/MIND/train

# TF
Name: tensorflow
Version: 2.0.0b1

# CUDA 10.1
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# Horovod
HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_TENSORFLOW=1 pip3 install --no-cache-dir horovod

# CHECK
horovodrun -n 3 python3 test.py --check-build

# RUN
horovodrun -n 2 python3 run.py
