import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import datasets,transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
from random import randint
import os
import time
import attack
from verify_dataset_cdcgan import verifyDataset, pretrained_model
from cDCGAN_cifar10 import generator

b_size = 32
total_b = 60000/b_size
num_votes = 100
cu = torch.cuda.is_available()
# device = torch.device("cuda" if cu else "cpu")
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
		optimizer.zero_grad()
		data = data.to(device)
		target = torch.LongTensor(target).to(device)

		outputs = model(data)
		_, predicted = torch.max(F.softmax(outputs, dim=1), 1)
		#print ('target: ', target)
		#print ('predicted: ', predicted)
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
		epsilon = epsilons[randint(1, len(epsilons) - 1)]
		epsilon = 0
		perturbed_data = attack.fgsm_attack(data, epsilon, data_grad)

		adv_output = classifier(perturbed_data)
		adv_pred = adv_output.max(1, keepdim=True)[1]

		for i in range(data.size(0)):  # batch size
			label = adv_pred[i].item()
			# vae_img = conv_cVAE.gen_image(CVAE, [label]*num_votes)  # num_votes x 1 x 32 x 32
			l_onehot = torch.zeros(10)
			l_onehot[label] = 1
			# gen_img: 1 x 3 x 32 x 32
			data = perturbed_data[i]  # 1 x 32 x 32
			# data: num_votes x 3 x 32 x 32
			data = torch.stack([data] * num_votes, 0)
			example = torch.zeros((num_votes, 6, 32, 32))
			for i in range(len(data)):
				gen_img = Generator(torch.randn(100).view(1, 100, 1, 1), l_onehot.view(1, 10, 1, 1))
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

# CVAE = autoencoder().to(device)
# CVAE.load_state_dict(torch.load('curr_model_99.pth', map_location='cpu'))
Generator = generator()
Generator.load_state_dict(torch.load('cifar10_cDCGAN_generator_param.pkl', map_location = 'cpu'))
# Generator = Generator.to(device)

n_channel = 128
cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
layers = make_layers(cfg, batch_norm=True)
classifier = CIFAR(layers, n_channel=8*n_channel, num_classes=10)
classifier.load_state_dict(torch.load(pretrained_model, map_location = 'cpu'))
# classifier = classifier.to(device)

verifier = Cifar10_Verifier().to(device)
verifier = torch.load('cdcgan_weights/9epoch.pt')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(verifier.parameters())
num_epochs = 20

verify_dataset = verifyDataset(Generator, classifier, True)
# verify_dataset = verifyDataset(Generator, classifier, False)
dataloader_args1 = dict(shuffle=True, batch_size=b_size, num_workers = 8, pin_memory=True) if cu \
						else dict(shuffle=True, batch_size=b_size)
verify_loader = torch.utils.data.DataLoader(verify_dataset,**dataloader_args1)

# mnist_dataset = MNIST('./', train=False, transform=transforms.ToTensor(), download=True)
# mnist_dataloader = torch.utils.data.DataLoader(mnist_dataset,**dataloader_args1)
cifar_dataset = CIFAR10('./', train=False, transform=transforms.ToTensor(), download=True)
cifar_dataloader = torch.utils.data.DataLoader(cifar_dataset,**dataloader_args1)

# for epoch in range(num_epochs):
# 	print('epoch: {}'.format(epoch+1))
# 	train_epoch(verifier, verify_loader, criterion, optimizer)
# 	tosave = str(epoch+1) + 'epoch.pt'
# 	torch.save(verifier, tosave)

num_votes = 5
test(Generator, verifier, classifier, cifar_dataloader, num_votes)
