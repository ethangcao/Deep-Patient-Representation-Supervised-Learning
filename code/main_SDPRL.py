import os
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from utils_SDPRL import train, evaluate
from plots import plot_learning_curves, plot_confusion_matrix
from mydatasets import myDataset, first_modal, second_modal, collate_fn
from mymodels import SDPRL

# experiments
task = "mort_icu"  # TODO: tasks contain [mort_icu, los_icu]
encoder = "Transformer"  # TODO: change to different encoder ["LSTM", "Transformer"]

# Set a correct path to the data files that you preprocessed
PATH_TRAIN_SEQS = "../data/processed/seqs.train"
PATH_VALID_SEQS = "../data/processed/seqs.validate"
PATH_TEST_SEQS = "../data/processed/seqs.test"
PATH_TRAIN_LABELS = "../data/processed/"+task+".train"
PATH_VALID_LABELS = "../data/processed/"+task+".validate"
PATH_TEST_LABELS = "../data/processed/"+task+".test"
PATH_OUTPUT = "../output/"+task+"/"
os.makedirs(PATH_OUTPUT, exist_ok=True)

NUM_EPOCHS = 1
BATCH_SIZE = 16
USE_CUDA = True  # Set 'True' if you want to use GPU
NUM_WORKERS = 0

device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
print(device.type)
torch.manual_seed(1)
if device.type == "cuda":
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

# Data loading
print('===> Loading entire datasets')
train_seqs = pickle.load(open(PATH_TRAIN_SEQS, 'rb'))
train_labels = pickle.load(open(PATH_TRAIN_LABELS, 'rb'))
valid_seqs = pickle.load(open(PATH_VALID_SEQS, 'rb'))
valid_labels = pickle.load(open(PATH_VALID_LABELS, 'rb'))
test_seqs = pickle.load(open(PATH_TEST_SEQS, 'rb'))
test_labels = pickle.load(open(PATH_TEST_LABELS, 'rb'))

train_seqs0 = train_seqs.loc[train_labels == 0]
train_seqs1 = train_seqs.loc[train_labels == 1]
train_array0 = train_seqs0.apply(lambda x: np.hstack((np.array(x.iloc[0]), np.array(x.iloc[1]))), axis=1)
train_array1 = train_seqs1.apply(lambda x: np.hstack((np.array(x.iloc[0]), np.array(x.iloc[1]))), axis=1)

train_dataset = myDataset(train_seqs, torch.LongTensor(train_labels.values))
valid_dataset = myDataset(valid_seqs, torch.LongTensor(valid_labels.values))
test_dataset = myDataset(test_seqs, torch.LongTensor(test_labels.values))
test_first_dataset = first_modal(test_seqs, torch.LongTensor(test_labels.values))
test_second_dataset = second_modal(test_seqs, torch.LongTensor(test_labels.values))

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True, num_workers=NUM_WORKERS)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False, num_workers=NUM_WORKERS)
# batch_size for the test set should be 1 to avoid sorting each mini-batch which breaks the connection with patient IDs
test_loader = DataLoader(dataset=test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False, num_workers=NUM_WORKERS)
test_first_loader = DataLoader(dataset=test_first_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False,
							 num_workers=NUM_WORKERS)
test_second_loader = DataLoader(dataset=test_second_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False,
							 num_workers=NUM_WORKERS)

def run(task, fuse, encoder):
	print(task, fuse, encoder)
	model = SDPRL(encoder=encoder)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

	model.to(device)
	criterion.to(device)

	train_losses, train_accuracies = [], []
	valid_losses, valid_accuracies = [], []
	for epoch in range(NUM_EPOCHS):
		train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch, train_array0, train_array1)
		valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)

		train_losses.append(train_loss)
		valid_losses.append(valid_loss)

		train_accuracies.append(train_accuracy)
		valid_accuracies.append(valid_accuracy)

		torch.save(model, os.path.join(PATH_OUTPUT, "{}_{}_{}.pth".format(task, encoder, "SDPRL")))

	best_model = torch.load(os.path.join(PATH_OUTPUT, "{}_{}_{}.pth".format(task, encoder, "SDPRL")))

	plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

	def predict_label(model, device, data_loader):
		model.eval()
		probas = []
		with torch.no_grad():
			for i, (input, target) in enumerate(data_loader):

				if isinstance(input, tuple):
					input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
				else:
					input = input.to(device)
				output = torch.sigmoid(model(input))
				y_pred = output.detach().to('cpu').numpy()
				probas.append(y_pred[0][1])
		return probas

	from sklearn.metrics import roc_auc_score
	from sklearn.metrics import precision_recall_curve
	from sklearn.metrics import auc

	target = pickle.load(open(PATH_TEST_LABELS, 'rb'))
	output = predict_label(best_model, device, test_loader)
	AUROC = roc_auc_score(target, output)
	precision, recall, _ = precision_recall_curve(target, output)
	AUPRC = auc(recall, precision)

	output_first = predict_label(best_model, device, test_first_loader)
	AUROC1 = roc_auc_score(target, output_first)
	precision, recall, _ = precision_recall_curve(target, output_first)
	AUPRC1 = auc(recall, precision)

	output_second = predict_label(best_model, device, test_second_loader)
	AUROC2 = roc_auc_score(target, output_second)
	precision, recall, _ = precision_recall_curve(target, output_second)
	AUPRC2 = auc(recall, precision)

	return [AUROC, AUPRC, AUROC1, AUPRC1, AUROC2, AUPRC2]

result = [str(i) for i in run(task, "SDPRL", encoder)]
with open(os.path.join(PATH_OUTPUT,"output_SDPRL_{}.txt".format(encoder)), 'w') as f:
	f.write(', '.join(result))
	f.write('\n')