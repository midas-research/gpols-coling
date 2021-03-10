import networkx as nx 
import pickle
import os
import numpy as np 
import pandas as pd 

feature_dict = dict()
path_motion = 'motions/'
path_speech = 'speeches/'
i =0
df = pd.read_csv('ParlVote_concat.csv')
for index, row in df.iterrows():
	feature_dict[row['motion_text']] = np.load(path_motion+str(i)+'.npy')
	feature_dict[str(index)] = np.load(path_speech+str(i)+'.npy')
	feature_dict[row['motion_speaker_id']] = np.zeros((1,768))
	feature_dict[row['speaker_id']] = np.zeros((1,768))
	i = i+1

df = pd.read_csv('ParlVote_concat.csv')
lab = dict()
for index, row in df.iterrows():
	lab[str(index)]=row['vote']

texts = range(34000)
texts = [str(i) for i in texts]

i = 0
masker = []
li = []
G = nx.read_gpickle("graph.gpickle")
ft_mat = []
for node in list(G.nodes()):
	if node in texts:
		masker.append(i)
		li.append(lab[node])
	ft_mat.append(feature_dict[node].reshape(768))
	i = i+1

pickle.dump(masker, open( "mask.p", "wb" ))
ft_mat = np.array(ft_mat)
li = np.array(li)
np.save('label.npy', li)
np.save('ft_mat.npy', ft_mat)