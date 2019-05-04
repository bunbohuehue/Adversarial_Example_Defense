import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from PIL import Image
import time
import scipy.io
import os
import matplotlib.pyplot as plt

cu = torch.cuda.is_available()
device = torch.device("cuda:0" if cu else "cpu")
b_size = 64
total_b = int(60000/b_size)

class encoder(nn.Module):
	def __init__(self):
		super(encoder, self).__init__()
		self.output = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16,
											 kernel_size=5, stride=2, padding=2, bias=True),
								   nn.ReLU(inplace=True),
								   nn.Conv2d(in_channels=16, out_channels=32,
											 kernel_size=5, stride=2, padding=2, bias=True),
								   nn.ReLU(inplace=True))
		self.mean = nn.Linear(7*7*32, 20, bias=True)
		self.std = nn.Linear(7*7*32, 20, bias=True)

	def forward(self, x):
		z = self.output(x)
		z = z.view(z.size(0),-1)
		mean = self.mean(z)
		std = self.std(z)
		return mean, std

class decoder(nn.Module):
	def __init__(self):
		super(decoder, self).__init__()
		self.first = nn.Sequential(nn.Linear(20,7*7*32,bias=True),nn.ReLU(inplace=True))
		self.output = nn.Sequential(nn.ConvTranspose2d(in_channels=32, out_channels=16,
											 kernel_size=5, stride=2, padding=2, output_padding=1, bias=True),
									nn.ReLU(inplace=True),
									nn.ConvTranspose2d(in_channels=16, out_channels=1,
										kernel_size=5, stride=2, padding=2, output_padding=1, bias=True),
									nn.Sigmoid())

	def forward(self, x):
		z = self.first(x)
		z = z.view(z.size(0),32,7,7)
		out = self.output(z)
		return out

class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()
		self.encoder = encoder()
		self.decoder = decoder()

	def forward(self, x, evalMode=False):
		enc_out = self.encoder(x)
		self.zmean = enc_out[0]
		self.zstd = enc_out[1]
		std = self.zstd.mul(0.5).exp_()
		if cu:
			eps = torch.cuda.FloatTensor(std.size()).normal_()
		else:
			eps = torch.FloatTensor(std.size()).normal_()
		eps = Variable(eps)
		z = self.zmean + eps*std
		dec_out = self.decoder(z)
		return dec_out, self.zmean, self.zstd

def train_epoch(model, train_loader, criterion, optimizer):
	model.train()
	model.to(device)
	running_loss = 0.0
	start_time = time.time()
	start_time1 = time.time()
	for batch_idx, (data, target) in enumerate(train_loader):
		optimizer.zero_grad()
		data = data.to(device)
		target = target.long().to(device)

		outputs, mean, std = model(data)
		latent_loss = 0.5 * torch.mean(mean*mean + std*std - torch.log(std*std) - 1.0)
		MSEloss = criterion(outputs, data)
		loss = latent_loss + MSEloss
		running_loss += loss.item()

		loss.backward()
		optimizer.step()

		torch.cuda.empty_cache()
		del data
		del target
		del loss
		end_time1 = time.time()
		if batch_idx > 0 and batch_idx % 100 == 0:
			print ('{:.2f} %, about {:.2f} minutes left.'.format(batch_idx*100*b_size/60000, (end_time1 - start_time1)*(total_b-batch_idx)/6000))
			start_time1 = time.time()

	end_time = time.time()
	running_loss /= len(train_loader)
	print('Training Loss: ', running_loss, 'Running one epoch took',end_time - start_time, 's')

	return running_loss

MNIST = torchvision.datasets.MNIST('./', download=True, transform=torchvision.transforms.ToTensor())
print (len(MNIST))
dataloader_args1 = dict(shuffle=True, batch_size=b_size, num_workers = 8, pin_memory=True) if cu \
						else dict(shuffle=True, batch_size=b_size)

train_loader = torch.utils.data.DataLoader(MNIST, **dataloader_args1)
print ('Finished loading.')

n_epochs = 30
criterion = nn.MSELoss(size_average=False)

model = VAE()
if cu:
	model = model.cuda()

optimizer = optim.Adam(model.parameters(),lr=0.001)

for i in range(n_epochs):
	print (i+1, 'th epoch: ')
	train_loss = train_epoch(model, train_loader, criterion, optimizer)
	print('='*20)

	for batch_idx, (data, target) in enumerate(train_loader):
		model.eval()
		data = data.to(device)
		toshow = (model(data)[0]).detach().numpy()
		plt.figure()
		for j in range(64):
			plt.subplot(8, 8, j + 1)
			plt.imshow(toshow[j][0], cmap='gray')
			plt.xticks([])
			plt.yticks([])

		plt.savefig(str(i+1)+".png")
		break
