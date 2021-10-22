import horovod.torch as hvd
hvd.init()
print('go', hvd.rank(), hvd.size(), 'go')