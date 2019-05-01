import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import datasets,transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
from random import randint
import os
import time
import attack
from verify_dataset import verifyDataset, pretrained_model
import conv_cVAE

b_size = 64
total_b = 60000/64
num_votes = 100
cu = torch.cuda.is_available()
device = torch.device("cuda:0" if cu else "cpu")
# LeNet Model definition
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

class autoencoder(nn.Module):
	def __init__(self):
		super(autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 16, 5, stride=2, padding=2),  # b, 16, 14, 14
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.Conv2d(16, 32, 5, stride=2, padding=2),  # b, 32, 7, 7
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
		)
		self.mean = nn.Linear(7 * 7 * 32 + 10, 128, bias=True)
		self.std = nn.Linear(7 * 7 * 32 + 10, 128, bias=True)

		self.fc = nn.Sequential(nn.Linear(128 + 10, 7 * 7 * 32, bias=True), nn.ReLU(inplace=True))
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(32, 16, 5, stride=2, padding=2, output_padding=1),  # b, 16, 14, 14
			nn.ReLU(True),
			nn.ConvTranspose2d(16, 1, 5, stride=2, padding=2, output_padding=1),  # b, 1, 28, 28
			nn.Tanh()
		)

	def reparametrize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		if torch.cuda.is_available():
			eps = torch.cuda.FloatTensor(std.size()).normal_()
		else:
			eps = torch.FloatTensor(std.size()).normal_()
		eps = Variable(eps)
		return eps.mul(std).add_(mu)

	def forward(self, x, y):
		x = self.encoder(x)  # b x (7*7*32)
		x = x.view(x.size(0), -1)
		x = torch.cat([x, y], 1)  # b x (7*7*32 + 10)
		mu, logvar = self.mean(x), self.std(x)
		z = self.reparametrize(mu, logvar)
		z = torch.cat([z, y], 1)
		z = self.fc(z)
		x = self.decoder(z.view(z.size(0), 32, 7, 7)), mu, logvar
		return x

class Verifier(nn.Module):
	"""
	Modified from LeNet5
	Input - batch_size x 2 x 28 x 28
	Output - batch_size x 2
	"""
	def __init__(self):
		super(Verifier, self).__init__()

		self.convnet = nn.Sequential(
			nn.Conv2d(2, 6, kernel_size=(5, 5)),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2, 2), stride=2),
			nn.Conv2d(6, 16, kernel_size=(5, 5)),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2, 2), stride=2),
			nn.Conv2d(16, 120, kernel_size=(4, 4)),
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

class Cifar_10_Verifier(nn.Module):
	"""
	Modified from LeNet5
	Input - batch_size x 6 x 32 x 32
	Output - batch_size x 2
	"""
	def __init__(self):
		super(Verifier, self).__init__()

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
		target = target.long().to(device)

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
		if batch_idx > 0 and batch_idx % 100 == 0:
			print ('{:.2f} %, about {:.2f} minutes left.'.format(batch_idx*100*b_size/60000, (end_time1 - start_time1)*(total_b-batch_idx)/6000))
			start_time1 = time.time()

	acc = (correct_predictions/total_predictions)*100.0
	end_time = time.time()
	running_loss /= len(train_loader)
	print('Training Loss: ', running_loss, 'Accuracy: ', acc, 'Running one epoch took',end_time - start_time, 's')

def test(CVAE, verifier, classifier, test_loader, num_votes):
	CVAE.eval()
	CVAE.to(device)
	verifier.eval()
	verifier.to(device)
	classifier.eval()
	classifier.to(device)
	total_predictions = 0.0
	correct_predictions = 0.0

	total_batch = len(test_loader)

	for batch_idx, (data, target) in enumerate(test_loader):
		optimizer.zero_grad()
		data = data.to(device)  # b x 1 x 28 x 28
		data.requires_grad = True
		target = target.long().to(device)

		pred_class = classifier(data)

		# Call FGSM Attack
		loss = F.nll_loss(pred_class, target)
		classifier.zero_grad()
		loss.backward()
		data_grad = data.grad.data

		epsilons = [0, .05, .1, .15, .2, .25, .3]
		#epsilon = epsilons[randint(0, len(epsilons) - 1)]
		epsilon = 0
		perturbed_data = attack.fgsm_attack(data, epsilon, data_grad)

		adv_output = classifier(perturbed_data)
		adv_pred = adv_output.max(1, keepdim=True)[1]

		for i in range(data.size(0)):  # batch size
			label = adv_pred[i].item()
			vae_img = conv_cVAE.gen_image(CVAE, [label]*num_votes)  # num_votes x 1 x 28 x 28
			data = perturbed_data[i]  # 1 x 28 x 28
			data = torch.stack([data] * num_votes, 0)  # num_votes x 1 x 28 x 28
			example = torch.cat((vae_img, data),1)  #  num_votes x 2 x 28 x 28
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

CVAE = autoencoder().to(device)
CVAE.load_state_dict(torch.load('curr_model_99.pth', map_location='cpu'))
classifier = Net().to(device)
classifier.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
#verifier = Verifier()
verifier = torch.load('9674.pt')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(verifier.parameters())
num_epochs = 10

verify_dataset = verifyDataset(CVAE, classifier, True)
verify_dataset = verifyDataset(CVAE, classifier, False)
dataloader_args1 = dict(shuffle=True, batch_size=b_size, num_workers = 8, pin_memory=True) if cu \
						else dict(shuffle=True, batch_size=b_size)
verify_loader = torch.utils.data.DataLoader(verify_dataset,**dataloader_args1)

mnist_dataset = MNIST('./', train=False, transform=transforms.ToTensor(), download=True)
mnist_dataloader = torch.utils.data.DataLoader(mnist_dataset,**dataloader_args1)

# for epoch in range(num_epochs):
#     train_epoch(verifier, verify_loader, criterion, optimizer)
#     tosave = str(epoch+1) + 'epoch.pt'
#     torch.save(verifier, tosave)

num_votes = 10
test(CVAE, verifier, classifier, mnist_dataloader, num_votes)
