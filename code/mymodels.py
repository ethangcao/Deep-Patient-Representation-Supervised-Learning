import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####

class EFI(nn.Module):
	def __init__(self, encoder):
		super(EFI, self).__init__()
		self.type = encoder
		self.embedding = nn.Sequential(
				nn.Linear(338, 128))
		if self.type == 'Transformer':
			self.encoder = nn.TransformerEncoder(encoder_layer = nn.TransformerEncoderLayer(d_model=128,
																							nhead=2,
																							batch_first=True),
												 num_layers = 2)
		else:
			self.encoder = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True)
		self.net = nn.Sequential(
		nn.Linear(128, 256),
		nn.ReLU(),
		nn.Linear(256, 2))

	def forward(self, input):
		seqs_tensor, lengths_tensor = input
		seqs_tensor = self.embedding(seqs_tensor)
		if self.type == "Transformer":
			mask = torch.ones(seqs_tensor.size()[:2])
			for i in range(len(seqs_tensor)):
				mask[i, :lengths_tensor[i]] = 0
			mask = mask.bool()
			x = self.encoder(seqs_tensor, src_key_padding_mask=mask)
		else:
			x = pack_padded_sequence(seqs_tensor, lengths_tensor, batch_first=True)
			x, hidden = self.encoder(x)
			x, _ = pad_packed_sequence(x, batch_first=True)
		x = x[np.arange(len(x)), lengths_tensor - 1]
		x = self.net(torch.tanh(x))
		return x


class EFE(nn.Module):
	def __init__(self, encoder):
		super(EFE, self).__init__()
		self.type = encoder
		self.embedding1 = nn.Sequential(
			nn.Linear(20, 64))
		self.embedding2 = nn.Sequential(
			nn.Linear(318, 64))
		if self.type == 'Transformer':
			self.encoder = nn.TransformerEncoder(encoder_layer = nn.TransformerEncoderLayer(d_model=128,
																							nhead=2,
																							batch_first=True),
												 num_layers = 2)
		else:
			self.encoder = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True)
		self.net = nn.Sequential(
		nn.Linear(128, 256),
		nn.ReLU(),
		nn.Linear(256, 2))

	def forward(self, input):
		seqs_tensor, lengths_tensor = input
		embedding1 = self.embedding1(seqs_tensor[:,:, :20])
		embedding2 = self.embedding2(seqs_tensor[:,:, 20:])
		x = torch.cat((embedding1, embedding2), 2)

		if self.type == "Transformer":
			mask = torch.ones(seqs_tensor.size()[:2])
			for i in range(len(seqs_tensor)):
				mask[i, :lengths_tensor[i]] = 0
			mask = mask.bool()
			x = self.encoder(x, src_key_padding_mask=mask)
		else:
			x = pack_padded_sequence(x, lengths_tensor, batch_first=True)
			x, hidden = self.encoder(x)
			x, _ = pad_packed_sequence(x, batch_first=True)
		x = x[np.arange(len(x)), lengths_tensor - 1]
		x = self.net(torch.tanh(x))
		return x

class LF(nn.Module):
	def __init__(self, encoder):
		super(LF, self).__init__()
		self.type = encoder
		self.embedding1 = nn.Sequential(
			nn.Linear(20, 128))
		self.embedding2 = nn.Sequential(
			nn.Linear(318, 128))
		if self.type == 'Transformer':
			self.encoder1 = nn.TransformerEncoder(encoder_layer = nn.TransformerEncoderLayer(d_model=128,
																							nhead=2,
																							batch_first=True),
												 num_layers = 2)
			self.encoder2 = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=128,
																						   nhead=2,
																						   batch_first=True),
												  num_layers=2)
		else:
			self.encoder1 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True)
			self.encoder2 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True)
		self.agg = nn.Sequential(
			nn.Linear(256, 128),
			nn.ReLU())
		self.net = nn.Sequential(
		nn.Linear(128, 256),
		nn.ReLU(),
		nn.Linear(256, 2))

	def forward(self, input):
		seqs_tensor, lengths_tensor = input
		embedding1 = self.embedding1(seqs_tensor[:,:, :20])
		embedding2 = self.embedding2(seqs_tensor[:,:, 20:])

		if self.type == "Transformer":
			mask = torch.ones(seqs_tensor.size()[:2])
			for i in range(len(seqs_tensor)):
				mask[i, :lengths_tensor[i]] = 0
			mask = mask.bool()
			x1 = self.encoder1(embedding1, src_key_padding_mask=mask)
			x2 = self.encoder2(embedding2, src_key_padding_mask=mask)
		else:
			x1 = pack_padded_sequence(embedding1, lengths_tensor, batch_first=True)
			x1, hidden = self.encoder1(x1)
			x1, _ = pad_packed_sequence(x1, batch_first=True)

			x2 = pack_padded_sequence(embedding2, lengths_tensor, batch_first=True)
			x2, hidden = self.encoder2(x2)
			x2, _ = pad_packed_sequence(x2, batch_first=True)

		x = torch.cat((x1, x2), 2)
		x = x[np.arange(len(x)), lengths_tensor - 1]
		x = self.agg(x)
		x = self.net(torch.tanh(x))
		return x

class LFpool(nn.Module):
	def __init__(self, encoder):
		super(LFpool, self).__init__()
		self.type = encoder
		self.embedding1 = nn.Sequential(
			nn.Linear(20, 128))
		self.embedding2 = nn.Sequential(
			nn.Linear(318, 128))
		if self.type == 'Transformer':
			self.encoder1 = nn.TransformerEncoder(encoder_layer = nn.TransformerEncoderLayer(d_model=128,
																							nhead=2,
																							batch_first=True),
												 num_layers = 2)
			self.encoder2 = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=128,
																						   nhead=2,
																						   batch_first=True),
												  num_layers=2)
		else:
			self.encoder1 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True)
			self.encoder2 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True)
		self.net = nn.Sequential(
		nn.Linear(128, 256),
		nn.ReLU(),
		nn.Linear(256, 2))

	def forward(self, input):
		seqs_tensor, lengths_tensor = input
		embedding1 = self.embedding1(seqs_tensor[:,:, :20])
		embedding2 = self.embedding2(seqs_tensor[:,:, 20:])

		if self.type == "Transformer":
			mask = torch.ones(seqs_tensor.size()[:2])
			for i in range(len(seqs_tensor)):
				mask[i, :lengths_tensor[i]] = 0
			mask = mask.bool()
			x1 = self.encoder1(embedding1, src_key_padding_mask=mask)
			x2 = self.encoder2(embedding2, src_key_padding_mask=mask)
		else:
			x1 = pack_padded_sequence(embedding1, lengths_tensor, batch_first=True)
			x1, hidden = self.encoder1(x1)
			x1, _ = pad_packed_sequence(x1, batch_first=True)

			x2 = pack_padded_sequence(embedding2, lengths_tensor, batch_first=True)
			x2, hidden = self.encoder2(x2)
			x2, _ = pad_packed_sequence(x2, batch_first=True)

		x1 = x1[np.arange(len(x1)), lengths_tensor - 1]
		x2 = x2[np.arange(len(x2)), lengths_tensor - 1]
		x = torch.maximum(x1, x2)
		x = self.net(torch.tanh(x))
		return x

class SDPRL(nn.Module):
	def __init__(self, encoder):
		super(SDPRL, self).__init__()
		self.type = encoder
		self.embedding1 = nn.Sequential(
			nn.Linear(20, 128))
		self.embedding2 = nn.Sequential(
			nn.Linear(318, 128))
		if self.type == 'Transformer':
			self.encoder1 = nn.TransformerEncoder(encoder_layer = nn.TransformerEncoderLayer(d_model=128,
																							nhead=2,
																							batch_first=True),
												 num_layers = 2)
			self.encoder2 = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=128,
																						   nhead=2,
																						   batch_first=True),
												  num_layers=2)
		else:
			self.encoder1 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True)
			self.encoder2 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=True)
		self.net = nn.Sequential(
		nn.Linear(128, 256),
		nn.ReLU(),
		nn.Linear(256, 2))

	def forward(self, input):
		seqs_tensor, lengths_tensor = input
		embedding1 = self.embedding1(seqs_tensor[:,:, :20])
		embedding2 = self.embedding2(seqs_tensor[:,:, 20:])

		if self.type == "Transformer":
			mask = torch.ones(seqs_tensor.size()[:2])
			for i in range(len(seqs_tensor)):
				mask[i, :lengths_tensor[i]] = 0
			mask = mask.bool()
			x1 = self.encoder1(embedding1, src_key_padding_mask=mask)
			x2 = self.encoder2(embedding2, src_key_padding_mask=mask)
		else:
			x1 = pack_padded_sequence(embedding1, lengths_tensor, batch_first=True)
			x1, hidden = self.encoder1(x1)
			x1, _ = pad_packed_sequence(x1, batch_first=True)

			x2 = pack_padded_sequence(embedding2, lengths_tensor, batch_first=True)
			x2, hidden = self.encoder2(x2)
			x2, _ = pad_packed_sequence(x2, batch_first=True)

		x1 = x1[np.arange(len(x1)), lengths_tensor - 1]
		x2 = x2[np.arange(len(x2)), lengths_tensor - 1]
		x = torch.maximum(x1, x2)
		x = self.net(torch.tanh(x))
		return x