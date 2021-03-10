from __future__ import division
from __future__ import print_function
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from models import GAT, SpGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, mask, mask_train,val_mask, test_mask = load_data()

# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
else:
    model = GAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=2, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()


features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[mask][mask_train], labels[mask_train])
    acc_train = accuracy(output[mask][mask_train], labels[mask_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features, adj)
    loss_val = F.nll_loss(output[mask][val_mask], labels[val_mask])
    acc_val = accuracy(output[mask][val_mask], labels[val_mask])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {}'.format(loss_train.data.item()),
          'acc_train: {}'.format(acc_train.data.item()),
          'loss_val: {}'.format(loss_val.data.item()),
          'acc_val: {}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[mask][test_mask], labels[test_mask])
    print(classification_report(labels[test_mask].cpu().numpy(),np.argmax(output[mask][test_mask].cpu().detach().numpy(),-1)))
    print(confusion_matrix(labels[test_mask].cpu().numpy(),np.argmax(output[mask][test_mask].cpu().detach().numpy(),-1)))
    acc_test = accuracy(output[mask][test_mask], labels[test_mask])
    print(accuracy_score(labels[test_mask].cpu().numpy(),np.argmax(output[mask][test_mask].cpu().detach().numpy(),-1)))
    print("Test set results:",
          "loss= {:.4f}".format(loss_test),
          "accuracy= {:.4f}".format(acc_test))
    np.save('{}.npy'.format(acc_test), confusion_matrix(labels[test_mask].cpu().numpy(),np.argmax(output[mask][test_mask].cpu().detach().numpy(),-1)))          
    return acc_test

# Train model
t_total = time.time()
l_r = 1e-3
optimizer = optim.Adam(model.parameters(), 
                       lr=l_r, 
                       weight_decay=args.weight_decay)
loss_values = []
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.67)
for epoch in range(1000):
    loss_values.append(train(epoch))
    if epoch%100==0:
        acc_test=compute_test()
        torch.save(model.state_dict(), '{}.pkl'.format(acc_test))
    scheduler.step()
print("Optimization Finished!")
