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
# args.cuda = not args.no_cuda and torch.cuda.is_available()
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
    # mask =mask.cuda()
    # mask_train=mask_train.cuda()
    # val_mask=val_mask.cuda()
    # test_mask=test_mask.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    # print(output.size())
    # print(features[mask])
    loss_train = F.nll_loss(output[mask][mask_train], labels[mask_train])
    # print (labels[mask_train].max())
    acc_train = accuracy(output[mask][mask_train], labels[mask_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
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
    print('true',labels[test_mask].cpu().numpy().max())

    output = model(features, adj)
    loss_test = F.nll_loss(output[mask][test_mask], labels[test_mask])
    # print('true',labels[test_mask].cpu().numpy().max())
    # print('pred',np.argmax(output[mask][test_mask].detach().cpu().numpy(), -1).max())
    # print(classification_report(labels[test_mask].numpy(),np.argmax(output[mask][test_mask].cpu().detach().numpy(),-1)))
    # print(confusion_matrix(labels[test_mask].numpy(),np.argmax(output[mask][test_mask].cpu().detach().numpy(),-1)))
    # acc_test = accuracy(output[mask][test_mask], labels[test_mask])
    # print(accuracy_score(labels[test_mask].cpu().numpy(),np.argmax(output[mask][test_mask].cpu().detach().numpy(),-1)))
    # print("Test set results:",
    #       "loss= {:.4f}".format(loss_test),
    #       "accuracy= {:.4f}".format(acc_test))
    # np.save('{}.npy'.format(acc_test), confusion_matrix(labels[test_mask].cpu().numpy(),np.argmax(output[mask][test_mask].cpu().detach().numpy(),-1)))          
    return

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
l_r = 1e-3
# optimizer = optim.Adam(model.parameters(), 
#                        lr=l_r, 
#                        weight_decay=args.weight_decay)

# for epoch in range(1000):
    
#     loss_values.append(train(epoch))
#     if epoch%100==0:
#         acc_test=compute_test()
#         l_r = l_r/2
#         optimizer = optim.Adam(model.parameters(), 
#                        lr=l_r, 
#                        weight_decay=args.weight_decay)
#         torch.save(model.state_dict(), '{}.pkl'.format(acc_test))
######################
#     if loss_values[-1] < best:
#         best = loss_values[-1]
#         best_epoch = epoch
#         bad_counter = 0
#     else:
#         bad_counter += 1

#     if bad_counter == args.patience:
#         break

#     files = glob.glob('*.pkl')
#     for file in files:
#         epoch_nb = int(file.split('.')[0])
#         if epoch_nb < best_epoch:
#             os.remove(file)

# files = glob.glob('*.pkl')
# for file in files:
#     epoch_nb = int(file.split('.')[0])
#     if epoch_nb > best_epoch:
#         os.remove(file)

print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
# print('Loading {}th epoch'.format(best_epoch))

model.load_state_dict(torch.load('/content/0.7559523809523809.pkl'))
compute_test()
