# Imports
import os
import numpy as np
import pandas as pd
import cv2

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T

# Hyperparameters
test_set_length = 40 		 # Test set (number of images)
train_batch_size = 2  		 # Train batch size
test_batch_size = 1    		 # Test batch size
num_classes = 6        		 # Number of classes
learning_rate = 0.005  		 # Learning rate
num_epochs = 10      	     # Number of epochs
output_dir = "saved_model"   # Output directory to save the model

def create_label_txt(path_to_csv):

	data = pd.read_csv(path_to_csv)
	labels = data['class'].unique()

	labels_dict = {}

	# Creat dictionary from array
	for index, label in enumerate(labels):
		labels_dict.__setitem__(index, label)

	# We need to create labels.txt and write labels dictionary into it
	with open('labels.txt', 'w') as f:
		f.write(str(labels_dict))

	return labels_dict	


def parse_one_annot(path, filename, labels_dict):

	data = pd.read_csv(path)

	class_names = data['class'].unique()
	classes_df = data[data["filename"] == filename]["class"]
	classes_array = classes_df.to_numpy()
	
	boxes_df = data[data["filename"] == filename][["xmin", "ymin", "xmax", "ymax"]]
	boxes_array = boxes_df.to_numpy()
	
	classes = []
	for key, value in labels_dict.items():
		for i in classes_array:
			if i == value:
				classes.append(key)

	# Convert list to tuple
	classes = tuple(classes)

	return boxes_array, classes


class CardsDataset(torch.utils.data.Dataset):

	""" The dataset contains images of playing cards 
		The dataset includes images of king, queen, jack, ten, nine and ace playing cards"""

	def __init__(self, dataset_dir, csv_file, labels_dict, transforms = None):

		self.dataset_dir = dataset_dir
		self.csv_file = csv_file
		self.transforms = transforms
		self.labels_dict = labels_dict
		self.image_names = [file for file in sorted(os.listdir(os.path.join(dataset_dir))) if file.endswith('.jpg') or file.endswith('.JPG')]

	def __getitem__(self, index):

		image_path = os.path.join(self.dataset_dir, self.image_names[index])
		image = cv2.imread(image_path)
		# Convert BGR to RGB
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		box_array, classes = parse_one_annot(self.csv_file, self.image_names[index], self.labels_dict)
		boxes = torch.as_tensor(box_array, dtype = torch.float32)

		labels = torch.tensor(classes, dtype=torch.int64)
		
		image_id = torch.tensor([index])
		area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

		iscrowd = torch.tensor(classes, dtype=torch.int64)
		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		target["image_id"] = image_id
		target["area"] = area
		target["iscrowd"] = iscrowd

		if self.transforms is not None:
			image, target = self.transforms(image, target)

		return image, target

	def __len__(self):

		return len(self.image_names)


def get_model(num_classes):

	# Load an pre-trained object detectin model (in this case faster-rcnn)
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)

	# Number of input features
	in_features = model.roi_heads.box_predictor.cls_score.in_features

	# Replace the pre-trained head with a new head
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	return model


def get_transforms(train):

	transforms = []

	# Convert numpy image to PyTorch Tensor
	transforms.append(T.ToTensor())

	if train:
		# Data augmentation
		transforms.append(T.RandomHorizontalFlip(0.5))

	return T.Compose(transforms)


if __name__ == '__main__':

	# Setting up the device
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	labels_dict = create_label_txt("cards_dataset/train_labels.csv")

	# Define train and test dataset
	dataset = CardsDataset(dataset_dir = "cards_dataset/train/", csv_file = "cards_dataset/train_labels.csv",
							labels_dict = labels_dict, transforms = get_transforms(train = True))

	dataset_test = CardsDataset(dataset_dir = "cards_dataset/train/", csv_file = "cards_dataset/train_labels.csv", 
							labels_dict = labels_dict, transforms = get_transforms(train = False))

	# Split the dataset into train and test
	torch.manual_seed(1)
	indices = torch.randperm(len(dataset)).tolist()
	dataset = torch.utils.data.Subset(dataset, indices[:-test_set_length])
	dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_set_length:])

	# Define train and test dataloaders
	data_loader = torch.utils.data.DataLoader(dataset, batch_size = train_batch_size, shuffle = True,
					num_workers = 4, collate_fn = utils.collate_fn)

	data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = test_batch_size, shuffle = False,
					num_workers = 4, collate_fn = utils.collate_fn)

	print(f"We have: {len(indices)} images in the dataset, {len(dataset)} are training images and {len(dataset_test)} are test images")


	# Get the model using helper function
	model = get_model(num_classes)
	model.to(device = device)

	# Construct the optimizer
	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(params, lr = learning_rate, momentum = 0.9, weight_decay = 0.0005)

	# Learning rate scheduler decreases the learning rate by 10x every 3 epochs
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.1)

	for epoch in range(num_epochs):
		
		train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq = 10)
		lr_scheduler.step()
		# Evaluate on the test dataset
		evaluate(model, data_loader_test, device = device)

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	# Save the model state	
	torch.save(model.state_dict(), output_dir + "/model")
