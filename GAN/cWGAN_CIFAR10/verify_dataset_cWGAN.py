import attack
import torch
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import cwgan
from random import randint
from torchvision.datasets import CIFAR10
from torchvision import transforms
from models.conwgan import *
import numpy as np

NUM_CLASSES = 10

pretrained_model = "cifar10-d875770b.pth"

class verifyDataset(torch.utils.data.Dataset):
	def __init__(self, generator, classifier, training):
		self.dataset = CIFAR10('./', train=training, transform=transforms.ToTensor(), download=True)
		self.generator = generator
		self.generator.eval()
		self.classifier = classifier
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, i):
		return self.generate(*self.dataset[i])

	def generate(self, data, target):
		epsilons = [0, .05, .1, .15, .2, .25, .3]
		data = torch.FloatTensor(data).unsqueeze(0) #.to(self.device)
		target = torch.tensor([target]) #.to(self.device)
		# Send the data and label 	to the device
		data.requires_grad = True

		# Forward pass the data through the model
		output = self.classifier(data)
		original_pred = output.max(1, keepdim=True)[1]

		loss = F.nll_loss(output, target)
		self.classifier.zero_grad()
		loss.backward()
		data_grad = data.grad.data

		# z = torch.randn(100).view(1, 100, 1, 1)
		# y = torch.zeros(10)

		if (randint(0, 4) > 1):
			# Call FGSM Attack
			epsilon = epsilons[randint(1, len(epsilons) - 1)]
			# generate adversarial example
			adv_example = attack.fgsm_attack(data, epsilon, data_grad)
			# classify adv example
			output = self.classifier(adv_example)

			# Check for success
			adv_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

			label = (adv_pred == target)[0].item()
			# print("label ",label)
			# print("adv pred ",adv_pred)
			labelnp = np.array([adv_pred])
			# print('labelnp ',labelnp)
			noise = cwgan.gen_rand_noise_with_label(labelnp)
			gen_adv_data = cwgan.generate_image(self.generator, noise)
			# gen_adv_data = self.generator(z, y)
			example = torch.cat((gen_adv_data, adv_example),1)
		else:
			# don't call FGSM
			label = (original_pred == target)[0].item()
			# print("label ",label)
			# print("orig pred ",original_pred)
			labelnp = np.array([original_pred])
			# print('labelnp ',labelnp)
			noise = cwgan.gen_rand_noise_with_label(labelnp)
			# y[original_pred] = 1
			# y = y.view(1, 10, 1, 1)
			gen_data = cwgan.generate_image(self.generator, noise)
			example = torch.cat((gen_data, data),1)

		example = example.detach()

		return example.squeeze(0), label
