# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim

import os
import json
import time
import pickle
import shutil
import argparse
import numpy as np
from json import encoder

from train import run_simple, markov, generate_input
from model import MyNewTrajPreModel
import config 
from tensorboardX import SummaryWriter
from config import args

np.random.seed(1)
torch.manual_seed(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
shutil.rmtree('logs',True)

print('*' * 15 + 'start training' + '*' * 15)
print('users:{}'.format(args.uid_size))
print(args)
model = MyNewTrajPreModel(args).cuda(device=0)

weight_p, bias_p = [],[]
for name, p in model.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]
l1 = args.L1 #0.0001
criterion = nn.NLLLoss().cuda(device=0)
optimizer = optim.RMSprop([{'params': filter(lambda p: p.requires_grad, weight_p), 'weight_decay':args.L2},
                      {'params': filter(lambda p: p.requires_grad, bias_p), 'weight_decay':0}], 
                      lr=args.learning_rate, 
                      alpha=0.9
                      )

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.lr_step,
                                                    factor=args.lr_decay, threshold=1e-3)

lr = args.learning_rate
metrics = {'train_loss': [], 'valid_loss': [], 'accuracy': [], 'valid_acc': {},'acc5':[], 'acc10':[], 'apr':[]}

#------markov model-----------------
candidate = args.data_neural.keys()
markov1, users_acc_markov1 = markov(args, candidate, 1)

markov5, users_acc_markov5 = markov(args, candidate, 5)

markov10, users_acc_markov10 = markov(args, candidate, 10)
print('markov acc1:', markov1)
print('markov acc5:', markov5)
print('markov acc10:', markov10)
# metrics['markov_acc'] = users_acc_markov
#---------------------------------
data_train, train_idx = generate_input(args.data_neural, 'train', candidate=candidate)
data_test, test_idx = generate_input(args.data_neural, 'test', candidate=candidate)
# print('users:{} markov:{} train:{} test:{}'.format(len(candidate), avg_acc_markov,
#                                                        len([y for x in train_idx for y in train_idx[x]]),
#                                                        len([y for x in test_idx for y in test_idx[x]])))

SAVE_PATH = args.save_path
tmp_path = 'checkpoint/'
if not os.path.exists((SAVE_PATH + tmp_path)):
    os.mkdir(SAVE_PATH + tmp_path)

print('user num：', args.uid_size)
print('poi num：',args.loc_size)
writer = SummaryWriter(logdir='./logs')
for epoch in range(args.epoch_max):
    print('------------',epoch)
    st = time.time()   
    model, avg_loss = run_simple(data_train, train_idx, 'train', lr, args.clip, model, optimizer,
                                    criterion, l1)
    print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_loss, lr))
    metrics['train_loss'].append(avg_loss)
    train_loss = avg_loss
    avg_loss, tacc, users_acc = run_simple(data_test, test_idx, 'test', lr, args.clip, model,
                                                optimizer, criterion, l1)
    print('==>Test Acc:{:.4f},Acc5:{:.4f},Acc10:{:.4f} Loss:{:.4f}'.format(tacc[0], tacc[1],tacc[2], avg_loss))
    test_loss = avg_loss
    test_acc, acc5, acc10,apr = tacc
    
    writer.add_scalars('test_acc', {'acc1':test_acc,'acc5':acc5,'acc10':acc10}, epoch)
    writer.add_scalars('train', {'train':train_loss,'test':test_loss}, epoch)
    metrics['valid_loss'].append(avg_loss)
    metrics['accuracy'].append(test_acc)
    metrics['acc5'].append(acc5)
    metrics['acc10'].append(acc10)
    metrics['apr'].append(apr)
    metrics['valid_acc'][epoch] = users_acc

    save_name_tmp = 'ep_' + str(epoch) + '.m'
    torch.save(model.state_dict(), SAVE_PATH + tmp_path + save_name_tmp)

    # scheduler.step(test_acc)
    scheduler.step(test_loss)
    lr_last = lr
    lr = optimizer.param_groups[0]['lr']
    if lr_last > lr:
        load_epoch = np.argmax(metrics['accuracy'])
        load_name_tmp = 'ep_' + str(load_epoch) + '.m'
        model.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp))
        print('load epoch={} model state'.format(load_epoch))
    if epoch == 0:
        print('single epoch time cost:{}'.format(time.time() - st))
    if lr < 1e-8:
        break

writer.close()
savename = 'Gresults'


pickle.dump(metrics, open( savename + '.pk', 'wb'))
mid = np.argmax(metrics['accuracy'])
avg_acc = metrics['accuracy'][mid]
load_name_tmp = 'ep_' + str(mid) + '.m'
model.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp))
save_name = 'res'
json.dump({'metrics': metrics}, fp=open(SAVE_PATH + save_name + '.rs', 'w'), indent=4)
metrics_view = {'train_loss': [], 'valid_loss': [], 'accuracy': []}
for key in metrics_view:
    metrics_view[key] = metrics[key]
json.dump({'metrics': metrics_view}, fp=open(SAVE_PATH + save_name + '.txt', 'w'), indent=4)
torch.save(model.state_dict(), SAVE_PATH + save_name + '.m')

for rt, dirs, files in os.walk(SAVE_PATH + tmp_path):
    for name in files:
        remove_path = os.path.join(rt, name)
        os.remove(remove_path)
os.rmdir(SAVE_PATH + tmp_path)

ours_acc =  avg_acc
print(ours_acc)








