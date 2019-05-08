import torch
torch.multiprocessing.set_start_method("spawn")
import torchvision
from torch import nn
import numpy as np
from torch.autograd import Variable
from torchvision import datasets,transforms
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
from random import randint
import time
import attack

import cwgan
from cwgan import GoodGenerator
from verify_dataset_cWGAN import verifyDataset, pretrained_model
from models.conwgan import *



# ART libraries
from art.classifiers import PyTorchClassifier
from art.utils import load_dataset
from art.attacks.fast_gradient import FastGradientMethod
from art.attacks.deepfool import DeepFool

a = 0
b_size = 32
total_b = 60000 / b_size
cu = torch.cuda.is_available()
device = torch.device("cuda" if cu else "cpu")
# device = torch.device("cpu")


class CIFAR(nn.Module):
	def __init__(self, features, n_channel, num_classes):
		super(CIFAR, self).__init__()
		assert isinstance(features, nn.Sequential), type(features)
		self.features = features
		self.classifier = nn.Sequential(
			nn.Linear(n_channel, num_classes)
		)
		#print(self.features)
		#print(self.classifier)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for i, v in enumerate(cfg):
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			padding = v[1] if isinstance(v, tuple) else 1
			out_channels = v[0] if isinstance(v, tuple) else v
			conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU()]
			else:
				layers += [conv2d, nn.ReLU()]
			in_channels = out_channels
	return nn.Sequential(*layers)


class Cifar10_Verifier(nn.Module):
	"""
	Modified from LeNet5
	Input - batch_size x 6 x 32 x 32
	Output - batch_size x 2
	"""
	def __init__(self):
		super(Cifar10_Verifier, self).__init__()

		self.convnet = nn.Sequential(
			nn.Conv2d(6, 16, kernel_size=(5, 5)),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2, 2), stride=2),
			nn.Conv2d(16, 32, kernel_size=(5, 5)),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2, 2), stride=2),
			nn.Conv2d(32, 120, kernel_size=(5, 5)),
			nn.ReLU())

		self.fc = nn.Sequential(
			nn.Linear(120, 84),
			nn.ReLU(),
			nn.Linear(84, 2))

	def forward(self, img):
		output = self.convnet(img)
		output = output.view(img.size(0), -1)
		output = self.fc(output)
		return output

def train(num_epochs, verifier, verifier_loader, criterion, optimizer):
	for epoch in range(num_epochs):
		print('epoch: {}'.format(epoch+1))
		train_epoch(verifier, verify_loader, criterion, optimizer)
		tosave = str(epoch+1) + 'epoch_cwgan.pt'
		torch.save(verifier, tosave)

def train_epoch(model, train_loader, criterion, optimizer):
	model.train()
	model.to(device)
	start_time = time.time()
	total_predictions = 0.0
	correct_predictions = 0.0
	running_loss = 0.0
	for batch_idx, (data, target) in enumerate(train_loader):
		optimizer.zero_grad()
		data = data.to(device)
		target = torch.LongTensor(target).to(device)

		outputs = model(data)
		_, predicted = torch.max(F.softmax(outputs, dim=1), 1)
		predicted = predicted.unsqueeze(1)
		correct_predictions += (predicted == target).sum().item()
		total_predictions += target.size(0)
		loss = criterion(outputs, target.squeeze(1))
		running_loss += loss.item()

		loss.backward()
		optimizer.step()

		torch.cuda.empty_cache()
		if batch_idx > 0 and batch_idx % 10 == 0:
			print ("Finished batch " + str(batch_idx))

	acc = (correct_predictions/total_predictions)*100.0
	end_time = time.time()
	running_loss /= len(train_loader)
	print('Training Loss: ', running_loss, 'Accuracy: ', acc, 'Running one epoch took',end_time - start_time, 's')

def test(Generator, verifier, classifier, adv_crafter, x_test, y_test, num_votes, num_tests):
	Generator.eval()
	Generator.to(device)
	verifier.eval()
	verifier.to(device)
	classifier.eval()
	classifier.to(device)
	total_predictions = 0.0
	correct_predictions = 0.0

	print("Start testing.")
	x_test_adv = adv_crafter.generate(x=x_test[:num_tests])
	print("Successfully generated adv examples.")

	# Evaluate the classifier on the adversarial examples
	preds = np.argmax(art_classifier.predict(x_test_adv), axis=1)
	acc = np.sum(preds == np.argmax(y_test[:num_tests], axis=1)) / y_test[:num_tests].shape[0]
	print("Test accuracy on adversarial sample: %.2f%%" % (acc * 100))

	# for each prediction
	for i in range(len(preds)):
		if i % 10 == 0:
			print("Successfully verified " + str(i+1) + " predictions.")
		data = torch.stack([torch.from_numpy(x_test[i])] * num_votes, 0)
		data = data.to(device)
		# (3x2) x H x W
		example = torch.zeros((num_votes, 6, 32, 32))
		example = example.to(device)
		# generate num_votes many images
		for j in range(num_votes):
			noise = cwgan.gen_rand_noise_with_label(preds[i])
			gen_img = cwgan.generate_image(Generator, noise)
			gen_img = gen_img.squeeze(0)
			example[j] = torch.cat((gen_img, data[j]), dim=0)
		outputs = verifier(example)
		predicted = torch.sum(F.softmax(outputs, dim=1)[:, 1])
		if predicted > num_votes / 2:
			same = 1
		else:
			same = 0
		if same == (preds[i] == np.argmax(y_test[i])):
			correct_predictions += 1.0
		total_predictions += 1.0

	print("Accuracy: ", correct_predictions / total_predictions)

############################# MAIN BODY ######################################

Generator = torch.load('cwgan_generator_37600.pt')
Generator.to(device)
# Generator = Generator.to(device)

n_channel = 128
cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
layers = make_layers(cfg, batch_norm=True)
classifier = CIFAR(layers, n_channel=8*n_channel, num_classes=10)
classifier.load_state_dict(torch.load(pretrained_model))
classifier.to(device)

################################ ART ##########################################

(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str('cifar10'))
# 3 x 32 x 32
x_train = x_train.transpose(0,3,1,2)
x_test = x_test.transpose(0,3,1,2)
art_classifier = PyTorchClassifier((min_, max_),model=classifier,loss=nn.CrossEntropyLoss(),
		optimizer=torch.optim.Adam(classifier.parameters()),nb_classes=10,input_shape=(3,32,32),channel_index=1)
art_classifier.fit(x_train, y_train, nb_epochs=0, batch_size=32)
# preds = np.argmax(art_classifier.predict(x_test[:20]), axis=1)
# acc = np.sum(preds == np.argmax(y_test[:20], axis=1)) / y_test.shape[0]
# print("\nTest accuracy: %.2f%%" % (acc * 100))

epsilon = .1  # Maximum perturbation
adv_crafter = DeepFool(art_classifier, max_iter=20)
# x_test_adv = adv_crafter.generate(x=x_test[:2], eps=epsilon)

# Evaluate the classifier on the adversarial examples
# preds = np.argmax(art_classifier.predict(x_test_adv), axis=1)
# acc = np.sum(preds == np.argmax(y_test[:2], axis=1)) / y_test[:2].shape[0]
# print("\nTest accuracy on adversarial sample: %.2f%%" % (acc * 100))

############################# VERIFIER ##########################################

verifier = Cifar10_Verifier().to(device)
# verifier = torch.load('3epoch_cwgan.pt')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(verifier.parameters())
num_epochs = 20
num_votes = 10
num_tests = 500

verify_dataset = verifyDataset(x_train, y_train, x_test, y_test, Generator, art_classifier, adv_crafter, True)
dataloader_args1 = dict(shuffle=True, batch_size=b_size) if cu \
						else dict(shuffle=True, batch_size=b_size)
verify_loader = torch.utils.data.DataLoader(verify_dataset, **dataloader_args1)

############################# TRAINING ##########################################

train(num_epochs, verifier, verify_loader, criterion, optimizer)
# test(Generator, verifier, classifier, adv_crafter, x_test, y_test, num_votes, num_tests)
