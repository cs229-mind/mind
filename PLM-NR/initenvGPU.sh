#!/bin/bash
export JAVA_HOME=/export/apps/jdk/JDK-1_8_0_121
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$JAVA_HOME/lib/server
export HADOOP_HOME=/home/hawei/hadoop-2.10.1
export PATH=${PATH}:${HADOOP_HOME}/bin:${JAVA_HOME}/bin
export LIBRARY_PATH=${LIBRARY_PATH}:${HADOOP_HOME}/lib/native
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${JAVA_HOME}/jre/lib/amd64/server:${HADOOP_HOME}/lib/native:/usr/local/cuda/extras/CUPTI/lib64
source $HADOOP_HOME/libexec/hadoop-config.sh
export PYTHONPATH="/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages"
export CLASSPATH=$($HADOOP_HOME/bin/hadoop classpath --glob)