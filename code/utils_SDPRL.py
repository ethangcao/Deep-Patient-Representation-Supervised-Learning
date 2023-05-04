import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def compute_batch_accuracy(output, target):
	"""Computes the accuracy for a batch"""
	with torch.no_grad():

		batch_size = target.size(0)
		_, pred = output.max(1)
		correct = pred.eq(target).sum()

		return correct * 100.0 / batch_size


def train(model, device, data_loader, criterion, optimizer, epoch, train_array0, train_array1, print_freq=10):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()

	model.train()

	end = time.time()

	def collate(seqs, modal=1):
		seqs.sort(key=lambda x: len(x), reverse=True)
		lengths = [len(seq) for seq in seqs]
		batch_size = len(seqs)
		max_length = max(lengths)
		num_features = max([seq.shape[1] for seq in seqs])
		collated_seqs = torch.zeros([batch_size, max_length, num_features], dtype=torch.float32)
		for i in range(batch_size):
			collated_seqs[i, :lengths[i], :] = torch.FloatTensor(seqs[i])
		if modal == 1: seqs_tensor = torch.FloatTensor(collated_seqs)[:, :, :20]
		else: seqs_tensor = torch.FloatTensor(collated_seqs)[:, :, 20:]
		lengths_tensor = torch.LongTensor(lengths)

		return (seqs_tensor, lengths_tensor)

	def encode_input(input, model, modal=1):
		seqs_tensor, lengths_tensor = input
		if modal == 1:
			embedding1 = model.embedding1(seqs_tensor)
			if model.type == "Transformer":
				mask = torch.ones(seqs_tensor.size()[:2])
				for i in range(len(seqs_tensor)):
					mask[i, :lengths_tensor[i]] = 0
				mask = mask.bool()
				x1 = model.encoder1(embedding1, src_key_padding_mask=mask)
			else:
				x1 = pack_padded_sequence(embedding1, lengths_tensor, batch_first=True)
				x1, hidden = model.encoder1(x1)
				x1, _ = pad_packed_sequence(x1, batch_first=True)
		else:
			embedding2 = model.embedding2(seqs_tensor)
			if model.type == "Transformer":
				mask = torch.ones(seqs_tensor.size()[:2])
				for i in range(len(seqs_tensor)):
					mask[i, :lengths_tensor[i]] = 0
				mask = mask.bool()
				x1 = model.encoder2(embedding2, src_key_padding_mask=mask)
			else:
				x1 = pack_padded_sequence(embedding2, lengths_tensor, batch_first=True)
				x1, hidden = model.encoder2(x1)
				x1, _ = pad_packed_sequence(x1, batch_first=True)
		return x1[np.arange(len(x1)), lengths_tensor - 1]

	for i, (input, target) in enumerate(data_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		if isinstance(input, tuple):
			input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
		else:
			input = input.to(device)
		target = target.to(device)

		optimizer.zero_grad()
		output = model(input)
		loss = 0
		for j in range(len(target)):
			input_seq1 = torch.unsqueeze(input[0][j, :, :20], 0)
			input_seq2 = torch.unsqueeze(input[0][j, :, 20:], 0)
			input_len = torch.unsqueeze(input[1][j], 0)
			archor1 = (input_seq1, input_len)
			archor2 = (input_seq2, input_len)
			if target[j] == 1:
				neg_pairs1 = train_array0.sample(n=32, random_state=42, replace=False).values.tolist()
				neg_pairs1 = collate(neg_pairs1, modal=2)
				neg_pairs2 = train_array0.sample(n=32, random_state=42, replace=False).values.tolist()
				neg_pairs2 = collate(neg_pairs2, modal=1)

			else:
				neg_pairs1 = train_array1.sample(n=32, random_state=42, replace=False).values.tolist()
				neg_pairs1 = collate(neg_pairs1, modal=2)
				neg_pairs2 = train_array1.sample(n=32, random_state=42, replace=False).values.tolist()
				neg_pairs2 = collate(neg_pairs2, modal=1)

			anchor1_r = encode_input(archor1, model, modal=1)
			neg_pairs1_r = encode_input(neg_pairs1, model, modal=2)
			archor2_r = encode_input(archor2, model, modal=2)
			neg_pairs2_r = encode_input(neg_pairs2, model, modal=1)

			sim = nn.CosineSimilarity(dim=1, eps=1e-6)
			pos_sim = sim(anchor1_r, archor2_r)/0.1
			anchor1_neg_sim = sim(neg_pairs2_r, anchor1_r)/0.1
			anchor2_neg_sim = sim(neg_pairs1_r, archor2_r)/0.1
			l1 = torch.exp(pos_sim)/torch.sum(torch.exp(anchor1_neg_sim))
			l2 = torch.exp(pos_sim)/torch.sum(torch.exp(anchor2_neg_sim))
			loss += (-torch.log(l1)+-torch.log(l2))

		loss = 0.05*loss + criterion(output, target)
		assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		losses.update(loss.item(), target.size(0))
		accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

		if i % print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
				epoch, i, len(data_loader), batch_time=batch_time,
				data_time=data_time, loss=losses, acc=accuracy))

	return losses.avg, accuracy.avg


def evaluate(model, device, data_loader, criterion, print_freq=10):
	batch_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()

	results = []

	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (input, target) in enumerate(data_loader):

			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			target = target.to(device)

			output = model(input)
			loss = criterion(output, target)

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			losses.update(loss.item(), target.size(0))
			accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

			y_true = target.detach().to('cpu').numpy().tolist()
			y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
			results.extend(list(zip(y_true, y_pred)))

			if i % print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
					i, len(data_loader), batch_time=batch_time, loss=losses, acc=accuracy))

	return losses.avg, accuracy.avg, results