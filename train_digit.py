import numpy as np
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from net import Net
import matplotlib.pyplot as plt
import os
import pandas as pd

def load_dataset():
    train_file = pd.read_csv('train.csv')
    train_file = np.array(train_file.loc[:,:])
    #train_data = train_file[:,1:]
    #label_data = train_file[:,0]
    train_set_x_orig = np.array(train_file[:,1:])
    train_set_y_orig = np.array(train_file[:,0],dtype='int')
    print (train_set_x_orig.shape)
    print (train_set_y_orig.shape)
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))

    print (train_set_y_orig.shape)
    return train_set_x_orig, train_set_y_orig

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[1] #numbers of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1 : Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    Shuffle_X = X[:, permutation]
    Shuffle_Y = Y[:, permutation]

    # Step 2 : Partition
    num_complete_minibataches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibataches):
        mini_batch_X = Shuffle_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = Shuffle_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size !=0: #conside the remaining value
        mini_batch_X = Shuffle_X[:, num_complete_minibataches * mini_batch_size : m]
        mini_batch_Y = Shuffle_Y[:, num_complete_minibataches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def data_process():
    X_train_orig, Y_train_orig = load_dataset()
    X_train_flatten = X_train_orig.reshape((X_train_orig.shape[0],-1)).T
    print (X_train_flatten.shape)
    X_train = X_train_flatten / 255
    C = 10
    Y_train = convert_to_one_hot(Y_train_orig, C)
    return X_train, Y_train

def train(learning_rate =0.0001, nums_epoch = 500, minibatch_size = 32, print_cost = True):
    seed = 3
    X_train, Y_train = data_process()
    (n_x, m)= X_train.shape
    print (n_x, m)
    n_y = Y_train.shape[0]
    costs = []
    gpus = [0]
    cuda_gpu = torch.cuda.is_available()
    print (cuda_gpu)

    net = Net(784, 64, 32, 10)
    if(cuda_gpu):
        net = torch.nn.DataParallel(net, device_ids=gpus).cuda()

    print (net)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    #loss_fuc = nn.CrossEntropyLoss()
    loss_fuc = nn.MultiLabelSoftMarginLoss().cuda()

    for epoch in range(nums_epoch):
        epoch_cost = 0
        epoch_accracy = 0
        num_batches = int(m / minibatch_size)
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

        for minibatch in minibatches:
            (minibatchX, minibatchY) = minibatch
            minibatchX = minibatchX.astype(np.float32).T
            minibatchY = minibatchY.astype(np.float32).T
            if(cuda_gpu):
                b_x = Variable(torch.from_numpy(minibatchX).cuda())
                b_y = Variable(torch.from_numpy(minibatchY).cuda())
            else:
                b_x = Variable(torch.from_numpy(minibatchX))
                b_y = Variable(torch.from_numpy(minibatchY))

            output = net(b_x)
            minibatch_cost = loss_fuc(output, b_y)
            optimizer.zero_grad()
            minibatch_cost.backward()
            optimizer.step()
            #print(b_y)
            #print(torch.max(b_y, 1))
            #print(torch.max(output, 1)[1].cpu().data.squeeze() == torch.max(b_y, 1)[1].cpu().data.squeeze())

            correct_prediction = sum(torch.max(output, 1)[1].data.squeeze() == torch.max(b_y, 1)[1].data.squeeze())
            if(cuda_gpu):
                correct_prediction = correct_prediction.cpu().numpy()
            else:
                correct_prediction = correct_prediction.numpy()
            epoch_accracy += correct_prediction / minibatch_size / len(minibatches)
            epoch_cost += minibatch_cost / len(minibatches)

        if print_cost == True and epoch % 100 ==0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                print ("Traing Acc. after epoch %i: %f" % (epoch, epoch_accracy))

        if print_cost == True and epoch%5 == 0:
                costs.append(epoch_cost)

    torch.save({
        'epoch': nums_epoch,
        'state_dict': net.state_dict(),
    }, 'model_net.pkl')
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

if __name__ == "__main__":
    train()
