import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import train, evaluate
from plots import plot_learning_curves, plot_confusion_matrix
from mydatasets import myDataset, first_modal, second_modal, collate_fn
from mymodels import EFI, EFE, LF, LFpool

# experiments
task = "los_icu"  # TODO: tasks contain [mort_icu, los_icu]
model_classes = {"EFI": EFI, "EFE": EFE, "LF": LF, "LFpool": LFpool}
fuse = "EFI"  # TODO: change to different fuse method [EFI, EFE, LF, LFpool]
encoder = "LSTM"  # TODO: change to different encoder ["LSTM", "Transformer"]

PATH_TRAIN_SEQS = "../data/processed/seqs.train"
PATH_VALID_SEQS = "../data/processed/seqs.validate"
PATH_TEST_SEQS = "../data/processed/seqs.test"
PATH_TRAIN_LABELS = "../data/processed/" + task + ".train"
PATH_VALID_LABELS = "../data/processed/" + task + ".validate"
PATH_TEST_LABELS = "../data/processed/" + task + ".test"
PATH_OUTPUT = "../output/" + task + "/"
os.makedirs(PATH_OUTPUT, exist_ok=True)

NUM_EPOCHS = 2
BATCH_SIZE = 4  # TODO: change batch size to 16
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
train_seqs = pickle.load(open(PATH_VALID_LABELS, 'rb'))
print(train_seqs.describe())