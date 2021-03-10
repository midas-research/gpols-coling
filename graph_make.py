import numpy as np 
import pandas as pd 
import networkx as nx
import os

df = pd.read_csv('ParlVote_concat.csv')
G = nx.Graph() 
for index, row in df.iterrows():
	G.add_nodes_from([row['motion_speaker_id'], row['motion_text'], row['speaker_id'], str(row['index1'])])

for index, row in df.iterrows():
	G.add_edges_from([(str(row['index1']), row['motion_text']), (str(row['index1']), row['speaker_id']) ])


mp_dict = dict()
for index, row in df.iterrows():
	if row['motion_party'] in mp_dict.keys():
		mp_dict[row['motion_party']].append(row['motion_speaker_id'])
	else:
		mp_dict[row['motion_party']] =[row['motion_speaker_id']]
	if row['party'] in mp_dict.keys():
		mp_dict[row['party']].append(row['speaker_id'])
	else:
		mp_dict[row['party']] = [row['speaker_id']]

for party_name in mp_dict.keys():
	for mp in mp_dict[party_name]:
		for mp2 in mp_dict[party_name]:
			G.add_edge(mp,mp2)

nx.write_gpickle(G, 'graph'+".gpickle")


