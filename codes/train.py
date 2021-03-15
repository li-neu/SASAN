# coding: utf-8
import torch
from torch.autograd import Variable
import numpy as np
from collections import deque, Counter

def generate_input(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['trajs']
        train_id = data_neural[u][mode]
        data_train[u] = {}

        trace = {}
        session = []
        for c, i in enumerate(train_id):
            session.extend(sessions[i])
        target = np.array([s[0] for s in session[1:]])

        loc_tim = []
        loc_tim.extend([(s[0], s[1]) for s in session[:-1]])
        loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
        tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
        trace['loc'] = Variable(torch.LongTensor(loc_np))
        trace['tim'] = Variable(torch.LongTensor(tim_np))
        trace['target'] = Variable(torch.LongTensor(target))
        trace['user'] = Variable(torch.LongTensor([u]))
        trace['index'] = Variable(torch.LongTensor([sum([len(sessions[les]) for les in range(train_id[0])])]))
        data_train[u][i] = trace
        # train_idx[u] = train_id
        if mode == 'train':
            train_idx[u] = [0, i]
        else:
            train_idx[u] = [i]
    return data_train, train_idx

def generate_queue(train_idx, mode, mode2):
    """return a deque. You must use it by train_queue.popleft()"""
    user = list(train_idx.keys())
    train_queue = deque()
    if mode == 'random':
        initial_queue = {}
        for u in user:
            if mode2 == 'train':
                initial_queue[u] = deque(train_idx[u][1:])
            else:
                initial_queue[u] = deque(train_idx[u])
        queue_left = 1
        while queue_left > 0:
            np.random.shuffle(user)
            for j, u in enumerate(user):
                if len(initial_queue[u]) > 0:
                    train_queue.append((u, initial_queue[u].popleft()))
                if j >= int(0.01 * len(user)):
                    break
            queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue


def get_acc(target, scores):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = np.zeros((3, 1))
    rank = np.zeros(len(target))
    rank2 = np.zeros(len(target))
    for i, p in enumerate(predx):
        t = target[i]
        tarscore = scores[i][t]
        ranklen = len(scores[i])
        rank[i] = (torch.sum(scores[i] > tarscore) + 1)
        rank2[i] = 1.0 / rank[i]
        rank[i] = (ranklen-rank[i]+1) / ranklen
        
        if t in p[:10] and t > 0:
            acc[0] += 1
        if t in p[:5] and t > 0:
            acc[1] += 1
        if t == p[0] and t > 0:
            acc[2] += 1
    #print('---------',np.mean(rank), np.mean(rank2))
    return acc, np.mean(rank)
'''

def get_acc(target, scores):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = np.zeros((3, 1))
    for i, p in enumerate(predx):
        t = target[i]
        if t in p[:10] and t > 0:
            acc[0] += 1
        if t in p[:5] and t > 0:
            acc[1] += 1
        if t == p[0] and t > 0:
            acc[2] += 1
    return acc
'''

def get_hint(target, scores, users_visited):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(1, 1)
    predx = idxx.cpu().numpy()
    hint = np.zeros((3,))
    count = np.zeros((3,))
    count[0] = len(target)
    for i, p in enumerate(predx):
        t = target[i]
        if t == p[0] and t > 0:
            hint[0] += 1
        if t in users_visited:
            count[1] += 1
            if t == p[0] and t > 0:
                hint[1] += 1
        else:
            count[2] += 1
            if t == p[0] and t > 0:
                hint[2] += 1
    return hint, count


def run_simple(data, run_idx, mode, lr, clip, model, optimizer, criterion, l1):
    """mode=train: return model, avg_loss
       mode=test: return avg_loss,avg_acc,users_rnn_acc"""
    run_queue = None
    if mode == 'train':
        model.train(True)
        run_queue = generate_queue(run_idx, 'random', 'train')
    elif mode == 'test':
        model.train(False)
        run_queue = generate_queue(run_idx, 'normal', 'test')
    total_loss = []
    queue_len = len(run_queue)

    users_acc = {}
    apr = {}
    for c in range(queue_len):
        optimizer.zero_grad()
        u, i = run_queue.popleft()
        if u not in users_acc:
            users_acc[u] = [0, 0, 0, 0]
        loc = data[u][i]['loc'].cuda()
        tim = data[u][i]['tim'].cuda()
        target = data[u][i]['target'].cuda()
        user = data[u][i]['user'].cuda()
        if mode == 'test':
            index = data[u][i]['index'].cuda()
        else:
            index = Variable(torch.LongTensor([-1])).cuda()
        scores = model(loc, tim, user, index)
        
        if scores.data.size()[0] > target.data.size()[0]:
            scores = scores[-target.data.size()[0]:]
        regularization_loss = 0
        for param in model.parameters():
            regularization_loss += torch.sum(abs(param))
        
        loss1 = criterion(scores, target)
        loss = loss1 + l1*regularization_loss
        
        if mode == 'train':
            
            loss.backward()
            # gradient clipping
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.add_(-lr, p.grad.data)
            except:
                pass
            optimizer.step()
        elif mode == 'test':
            users_acc[u][0] += len(target)
            acc, apr[u] = get_acc(target, scores)
            users_acc[u][1] += acc[2] # top 1
            users_acc[u][2] += acc[1] #top 5
            users_acc[u][3] += acc[0] #top 10
        # total_loss.append(loss.data.cpu().numpy()[0])
        total_loss.append(loss1.item())

    avg_loss = np.mean(total_loss, dtype=np.float64)
    if mode == 'train':
        return model, avg_loss
    elif mode == 'test':
        users_rnn_acc = {}
        users_rnn_acc5 = {}
        users_rnn_acc10 = {}
        for u in users_acc:
            tmp_acc = users_acc[u][1] / users_acc[u][0]
            users_rnn_acc[u] = tmp_acc.tolist()[0]
            tmp5 = users_acc[u][2] / users_acc[u][0]
            tmp10 = users_acc[u][3] / users_acc[u][0]
            users_rnn_acc5[u] = tmp5.tolist()[0]
            users_rnn_acc10[u] = tmp10.tolist()[0]
        avg_acc = np.mean([users_rnn_acc[x] for x in users_rnn_acc])
        avg5 = np.mean([users_rnn_acc5[x] for x in users_rnn_acc5])
        avg10 = np.mean([users_rnn_acc10[x] for x in users_rnn_acc10])
        avgapr = np.mean([apr[x] for x in apr])
        avg_acc = (avg_acc, avg5, avg10, avgapr)
        print('apr:', avgapr)
        return avg_loss, avg_acc, users_rnn_acc

'''
def markov(parameters, candidate):
    validation = {}
    for u in candidate:
        traces = parameters.data_neural[u]['trajs']
        train_id = parameters.data_neural[u]['train']
        test_id = parameters.data_neural[u]['test']
        trace_train = []
        for tr in train_id:
            trace_train.append([t[0] for t in traces[tr]])
        locations_train = []
        for t in trace_train:
            locations_train.extend(t)
        trace_test = []
        for tr in test_id:
            trace_test.append([t[0] for t in traces[tr]])
        locations_test = []
        for t in trace_test:
            locations_test.extend(t)
        validation[u] = [locations_train, locations_test]
    acc = 0
    count = 0
    user_acc = {}
    for u in validation.keys():
        topk = list(set(validation[u][0]))
        transfer = np.zeros((len(topk), len(topk)))

        # train
        sessions = parameters.data_neural[u]['trajs']
        train_id = parameters.data_neural[u]['train']
        for i in train_id:
            for j, s in enumerate(sessions[i][:-1]):
                loc = s[0]
                target = sessions[i][j + 1][0]
                if loc in topk and target in topk:
                    r = topk.index(loc)
                    c = topk.index(target)
                    transfer[r, c] += 1
        for i in range(len(topk)):
            tmp_sum = np.sum(transfer[i, :])
            if tmp_sum > 0:
                transfer[i, :] = transfer[i, :] / tmp_sum

        # validation
        user_count = 0
        user_acc[u] = 0
        test_id = parameters.data_neural[u]['test']
        for i in test_id:
            for j, s in enumerate(sessions[i][:-1]):
                loc = s[0]
                target = sessions[i][j + 1][0]
                count += 1
                user_count += 1
                if loc in topk:
                    pred = np.argmax(transfer[topk.index(loc), :])
                    if pred >= len(topk) - 1:
                        pred = np.random.randint(len(topk))

                    pred2 = topk[pred]
                    if pred2 == target:
                        acc += 1
                        user_acc[u] += 1
        user_acc[u] = user_acc[u] / user_count
    avg_acc = np.mean([user_acc[u] for u in user_acc])
    return avg_acc, user_acc
'''
def markov(parameters, candidate, K=1):
    validation = {}
    for u in candidate:
        traces = parameters.data_neural[u]['trajs']
        train_id = parameters.data_neural[u]['train']
        test_id = parameters.data_neural[u]['test']
        trace_train = []
        for tr in train_id:
            trace_train.append([t[0] for t in traces[tr]])
        locations_train = []#all train POI
        for t in trace_train:
            locations_train.extend(t)
        trace_test = []
        for tr in test_id:
            trace_test.append([t[0] for t in traces[tr]])
        locations_test = [] #all test poi
        for t in trace_test:
            locations_test.extend(t)
        validation[u] = [locations_train, locations_test]
    acc = 0
    count = 0
    user_acc = {}
    # apr = {}
    for u in validation.keys():
        topk = list(set(validation[u][0]))
        transfer = np.zeros((len(topk), len(topk)))

        # train
        sessions = parameters.data_neural[u]['trajs']
        train_id = parameters.data_neural[u]['train']
        for i in train_id:
            for j, s in enumerate(sessions[i][:-1]):
                loc = s[0]
                target = sessions[i][j + 1][0]
                if loc in topk and target in topk:
                    r = topk.index(loc)
                    c = topk.index(target)
                    transfer[r, c] += 1
        for i in range(len(topk)):
            tmp_sum = np.sum(transfer[i, :])
            if tmp_sum > 0:
                transfer[i, :] = transfer[i, :] / tmp_sum

        # validation
        user_count = 0
        user_acc[u] = 0
        # K = 1
        # temp = 0.0
        # temp_count = 0
        test_id = parameters.data_neural[u]['test']
        for i in test_id:
            for j, s in enumerate(sessions[i][:-1]):
                loc = s[0]
                target = sessions[i][j + 1][0]
                count += 1
                user_count += 1
                if loc in topk:
                    pred = np.argmax(transfer[topk.index(loc), :])
                    if pred >= len(topk) - 1:
                        pred = np.random.randint(len(topk))
                    pset = np.argsort(transfer[topk.index(loc), :]) 
                    # tlist =  list(pset)
                    # rank = len(topk)
                    # for ind, val in enumerate(tlist):
                    #     if val == target:
                    #         rank = ind
                    #         break
                    
                    # ranklen = len(topk)
                    # temp += ranklen+1-rank / ranklen
                    # temp_count += 1
                    pset = pset[-K:]
                    
                    pre = [topk[k] for k in pset]
                    if target in pre:
                        acc += 1
                        user_acc[u] += 1
                
        user_acc[u] = user_acc[u] / user_count
        # apr[u] = temp / temp_count 
    avg_acc = np.mean([user_acc[u] for u in user_acc])
    
    
    return avg_acc, user_acc  