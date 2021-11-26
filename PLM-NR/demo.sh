#!/bin/bash
export JAVA_HOME=/export/apps/jdk/JDK-1_8_0_121
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$JAVA_HOME/lib/server
export HADOOP_HOME=/home/hawei/hadoop-2.10.1
export PATH=${PATH}:${HADOOP_HOME}/bin:${JAVA_HOME}/bin
export LIBRARY_PATH=${LIBRARY_PATH}:${HADOOP_HOME}/lib/native
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${JAVA_HOME}/jre/lib/amd64/server:${HADOOP_HOME}/lib/native:/usr/local/cuda/extras/CUPTI/lib64
source $HADOOP_HOME/libexec/hadoop-config.sh
export PYTHONPATH="/usr/local/lib/python3.6/site-packages"
export CLASSPATH=$($HADOOP_HOME/bin/hadoop classpath --glob)

hvd_size=2 # two GPUs
mode=$1 # ['train', 'test', 'train_test']

root_data_dir=../
train_dir="train_valid"
test_dir="test"
dataset="data"

epoch=2
num_attention_heads=20
news_attributes=$2


model_dir=$3 
batch_size=8

user_log_mask=False
padded_news_different_word_index=False
use_padded_news_embedding=False

save_steps=2000
lr=0.00001
max_steps_per_epoch=120000
filter_num_word=1
mask_uet_bing_rate=0.8
neg_ratio=4



if [ ${mode} == train ] 
then
    mpirun -np ${hvd_size} -H localhost:${hvd_size} \
    python run.py --root_data_dir ${root_data_dir} \
    --mode ${mode} --epoch ${epoch} --dataset ${dataset} \
    --model_dir ${model_dir}  --batch_size ${batch_size} \
    --news_attributes ${news_attributes} --lr ${lr} \
    --padded_news_different_word_index ${padded_news_different_word_index} \
    --user_log_mask ${user_log_mask} --use_padded_news_embedding ${use_padded_news_embedding} \
    --train_dir ${train_dir} --test_dir ${test_dir} --save_steps ${save_steps} \
    --filter_num_word ${filter_num_word} --max_steps_per_epoch ${max_steps_per_epoch} \
    --neg_ratio ${neg_ratio}  --num_attention_heads ${num_attention_heads}
elif [ ${mode} == test ]
then
    batch_size=32
    log_steps=100
    load_ckpt_test=${11}
    CUDA_LAUNCH_BLOCKING=1 python run.py --root_data_dir ${root_data_dir} \
    --mode ${mode} --epoch ${epoch} --dataset ${dataset} \
    --model_dir ${model_dir}  --batch_size ${batch_size} \
    --news_attributes ${news_attributes} --lr ${lr} \
    --padded_news_different_word_index ${padded_news_different_word_index} \
    --user_log_mask ${user_log_mask} --use_padded_news_embedding ${use_padded_news_embedding} \
    --train_dir ${train_dir} --test_dir ${test_dir} --save_steps ${save_steps} \
    --log_steps ${log_steps} --num_attention_heads ${num_attention_heads} \
    --load_ckpt_test ${load_ckpt_test}
fi