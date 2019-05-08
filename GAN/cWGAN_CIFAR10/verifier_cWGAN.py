import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import datasets,transforms
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
from random import randint
import time
import attack
from verify_dataset_cWGAN import verifyDataset, pretrained_model
import cwgan
from cwgan import GoodGenerator
from models.conwgan import *
import numpy as np


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
device = torch.device("cpu")


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

def train_epoch(model, train_loader, criterion, optimizer):
	model.train()
	model.to(device)
	start_time = time.time()
	start_time1 = time.time()
	total_predictions = 0.0
	correct_predictions = 0.0
	running_loss = 0.0
	for batch_idx, (data, target) in enumerate(train_loader):
		print(batch_idx)
		optimizer.zero_grad()
		data = data.to(device)
		target = torch.LongTensor(target).to(device)

		outputs = model(data)
		_, predicted = torch.max(F.softmax(outputs, dim=1), 1)
		# print ('target: ', target.sum())
		# print ('predicted: ', predicted.sum())
		correct_predictions += (predicted == target).sum().item()
		total_predictions += target.size(0)
		loss = criterion(outputs, target)
		running_loss += loss.item()

		loss.backward()
		optimizer.step()

		torch.cuda.empty_cache()
		end_time1 = time.time()
		if batch_idx > 0 and batch_idx % 10 == 0:
			print ('{:.2f} %, about {:.2f} minutes left.'.format(batch_idx*100*b_size/60000, (end_time1 - start_time1)*(total_b-batch_idx)/6000))
			start_time1 = time.time()

	acc = (correct_predictions/total_predictions)*100.0
	end_time = time.time()
	running_loss /= len(train_loader)
	print('Training Loss: ', running_loss, 'Accuracy: ', acc, 'Running one epoch took',end_time - start_time, 's')

def test(Generator, verifier, classifier, test_loader, num_votes):
	# CVAE.eval()
	# CVAE.to(device)
	Generator.eval()
	Generator.to(device)
	verifier.eval()
	verifier.to(device)
	classifier.eval()
	classifier.to(device)
	total_predictions = 0.0
	correct_predictions = 0.0

	total_batch = len(test_loader)

	counter = 0
	for batch_idx, (data, target) in enumerate(test_loader):
		if counter == 10:
			break
		counter += 1
		optimizer.zero_grad()
		data = data.to(device)  # b x 1 x 32 x 32
		data.requires_grad = True
		target = target.long().to(device)

		pred_class = classifier(data)

		# Call FGSM Attack
		loss = F.nll_loss(pred_class, target)
		classifier.zero_grad()
		loss.backward()
		data_grad = data.grad.data

		epsilons = [0, .05, .1, .15, .2, .25, .3]
		epsilon = epsilons[randint(0, len(epsilons) - 1)]
		epsilon = 0
		# perturbed_data = attack.fgsm_attack(data, epsilon, data_grad)
		# data = data.permute(0,2,3,1)[0]
		# perturbed_data = attack.deepfool(data, classifier, 10)
		data = data.detach().numpy()
		perturbed_data = deepfooler.generate(data)

		adv_output = classifier(perturbed_data)
		adv_pred = adv_output.max(1, keepdim=True)[1]

		for i in range(data.size(0)):  # batch size
			label = adv_pred[i].item()
			# gen_img: 1 x 3 x 32 x 32
			data = perturbed_data[i]  # 1 x 32 x 32
			# data: num_votes x 3 x 32 x 32
			data = torch.stack([data] * num_votes, 0)
			example = torch.zeros((num_votes, 6, 32, 32))
			for i in range(len(data)):
				labelnp = np.array([label])
				noise = cwgan.gen_rand_noise_with_label(labelnp)
				gen_img = cwgan.generate_image(Generator, noise)
				# print(type(gen_img))
				# print(gen_img.size())
				# gen_img = Generator(torch.randn(100).view(1, 100, 1, 1), l_onehot.view(1, 10, 1, 1))
				gen_img = gen_img.squeeze(0)
				example[i] = torch.cat((gen_img, data[i]),dim=0)  #  num_votes x 6 x 32 x 32
			outputs = verifier(example)  # num_votes x 2
			predicted = torch.sum(F.softmax(outputs, dim=1)[:, 1])
			if predicted > num_votes/2:
				same = 1
			else:
				same = 0
			if same == (label == target[i].item()):
				correct_predictions += 1.0
			total_predictions += 1.0

		print("Finished batch ", batch_idx, "/", total_batch)

	print("Accuracy: ", correct_predictions / total_predictions)



############################# MAIN BODY ######################################

Generator = torch.load('cwgan_generator_37600.pt', map_location = 'cpu')
# Generator = Generator.to(device)

n_channel = 128
cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
layers = make_layers(cfg, batch_norm=True)
classifier = CIFAR(layers, n_channel=8*n_channel, num_classes=10)
classifier.load_state_dict(torch.load(pretrained_model, map_location = 'cpu'))


################################ ART ##########################################

(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str('cifar10'))
x_train = x_train.transpose(0,3,1,2)
x_test = x_test.transpose(0,3,1,2)
art_classifier = PyTorchClassifier((min_, max_),model=classifier,loss=nn.CrossEntropyLoss(),
		optimizer=torch.optim.Adam(classifier.parameters()),nb_classes=10,input_shape=(3,32,32),channel_index=1)
art_classifier.fit(x_train, y_train, nb_epochs=0, batch_size=32)

# preds = np.argmax(art_classifier.predict(x_test[:20]), axis=1)
# acc = np.sum(preds == np.argmax(y_test[:20], axis=1)) / y_test.shape[0]
# print("\nTest accuracy: %.2f%%" % (acc * 100))

epsilon = .1  # Maximum perturbation
adv_crafter = DeepFool(art_classifier, max_iter=10)
# x_test_adv = adv_crafter.generate(x=x_test[:2], eps=epsilon)

# Evaluate the classifier on the adversarial examples
# preds = np.argmax(art_classifier.predict(x_test_adv), axis=1)
# acc = np.sum(preds == np.argmax(y_test[:2], axis=1)) / y_test[:2].shape[0]
# print("\nTest accuracy on adversarial sample: %.2f%%" % (acc * 100))

############################# VERIFIER ##########################################

verifier = Cifar10_Verifier().to(device)
verifier = torch.load('2epoch_cwgan.pt')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(verifier.parameters())
num_epochs = 20

verify_dataset = verifyDataset(x_train, y_train, x_test, y_test, Generator, art_classifier, adv_crafter, True)

dataloader_args1 = dict(shuffle=True, batch_size=b_size, num_workers=8, pin_memory=True) if cu \
						else dict(shuffle=True, batch_size=b_size)
verify_loader = torch.utils.data.DataLoader(verify_dataset, **dataloader_args1)

# mnist_dataset = MNIST('./', train=False, transform=transforms.ToTensor(), download=True)
# mnist_dataloader = torch.utils.data.DataLoader(mnist_dataset,**dataloader_args1)
# cifar_dataset = CIFAR10('./', train=False, transform=transforms.ToTensor(), download=True)
# cifar_dataloader = torch.utils.data.DataLoader(cifar_dataset,**dataloader_args1)

for epoch in range(num_epochs):
	print('epoch: {}'.format(epoch+1))
	train_epoch(verifier, verify_loader, criterion, optimizer)
	tosave = str(epoch+1) + 'epoch_cwgan.pt'
	torch.save(verifier, tosave)

# num_votes = 5
# test(Generator, verifier, classifier, cifar_dataloader, num_votes)
