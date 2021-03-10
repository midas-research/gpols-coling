import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
import pickle
import random

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="data", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))
    G = nx.read_gpickle("graph.gpickle")
    adj = nx.to_numpy_matrix(G)
    adj = sp.coo_matrix(adj, dtype=np.float32)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    labels = torch.LongTensor(np.load('label.npy'))
    features = torch.tensor(np.load('ft_mat.npy'), dtype=torch.float32)
    mask = pickle.load( open( "mask.p", "rb" ) )
    random.shuffle(mask)
    mask_train = range(26768)
    val_mask = range(26768, 30114)
    test_mask = range(30114, 33461)
    return adj, features, labels, mask, mask_train,val_mask, test_mask


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

