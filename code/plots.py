import matplotlib.pyplot as plt
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	fig, axes = plt.subplots(1, 2, figsize=(15,5))

	axes[0].plot(train_losses,  label='Training Loss')
	axes[0].plot(valid_losses,  label='Validation Loss')
	axes[0].legend(loc="upper right")
	axes[0].set_xlabel("epoch")
	axes[0].set_ylabel("Loss")
	axes[0].set_title('Loss Curve')

	axes[1].plot(train_accuracies, label='Training Accuracy')
	axes[1].plot(valid_accuracies, label='Validation Accuracy')
	axes[1].legend(loc="upper left")
	axes[1].set_xlabel("epoch")
	axes[1].set_ylabel("Accuracy")
	axes[1].set_title('Accuracy Curve')

	fig.savefig('training_curve.png')


def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.

	from sklearn.metrics import confusion_matrix
	import seaborn as sns
	import numpy as np

	y_true, y_pred = zip(*results)
	cf_matrix = confusion_matrix(y_true, y_pred)
	plt.figure(figsize=(10, 10))

	ax = sns.heatmap((cf_matrix / np.sum(cf_matrix, axis=1)).round(2), annot=True, cmap='Blues')

	ax.set_title('Normalized Confusion Matrix')
	ax.set_xlabel('Predicted')
	ax.set_ylabel('Actual')

	ax.xaxis.set_ticklabels(class_names)
	ax.yaxis.set_ticklabels(class_names)
	plt.savefig('confusion.png')
