import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, Dataset

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####

class myDataset(Dataset):
	def __init__(self, seqs, labels):
		# seq has 20 + 318 features
		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")
		self.labels = labels
		self.seqs = seqs

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		interventions = np.array(self.seqs.iloc[index, 0])
		vitals = np.array(self.seqs.iloc[index, 1])
		return np.hstack((interventions, vitals)), self.labels[index]

class first_modal(Dataset):
	def __init__(self, seqs, labels):
		# seq has 20 + 318 features
		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")
		self.labels = labels
		self.seqs = seqs

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		interventions = np.array(self.seqs.iloc[index, 0])
		vitals = np.array(self.seqs.iloc[index, 1])
		vitals = np.zeros(vitals.shape)
		return np.hstack((interventions, vitals)), self.labels[index]

class second_modal(Dataset):
	def __init__(self, seqs, labels):
		# seq has 20 + 318 features
		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")
		self.labels = labels
		self.seqs = seqs

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		interventions = np.array(self.seqs.iloc[index, 0])
		interventions = np.zeros(interventions.shape)
		vitals = np.array(self.seqs.iloc[index, 1])
		return np.hstack((interventions, vitals)), self.labels[index]


def collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq_1, label_1), (seq_2, label_2), ... , (seq_N, label_N)]
	where N is minibatch size, seq_i is a Numpy (or Scipy Sparse) array, and label is an int value

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	"""

	batch.sort(key=lambda x: len(x[0]), reverse=True)
	seqs, labels = zip(*batch)
	lengths = [len(seq) for seq in seqs]
	batch_size = len(seqs)
	max_length = max(lengths)
	num_features = max([seq.shape[1] for seq in seqs])
	collated_seqs = torch.zeros([batch_size, max_length, num_features], dtype=torch.float32)
	for i in range(batch_size):
		collated_seqs[i, :lengths[i], :] = torch.FloatTensor(seqs[i])
	seqs_tensor = torch.FloatTensor(collated_seqs)
	lengths_tensor = torch.LongTensor(lengths)
	labels_tensor = torch.LongTensor(labels)

	return (seqs_tensor, lengths_tensor), labels_tensor
