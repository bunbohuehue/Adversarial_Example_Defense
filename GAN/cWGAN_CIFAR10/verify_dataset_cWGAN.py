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
	def __init__(self, x_train, y_train, x_test, y_test, generator, classifier, adv_crafter, training):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test
		self.generator = generator
		self.generator.eval()
		self.classifier = classifier
		self.adv_crafter = adv_crafter
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

	def __len__(self):
		return len(self.x_train)

	def __getitem__(self, i):
		return self.generate(self.x_train[i], self.y_train[i])

	def generate(self, data, target):
		# TODO: MAKE THIS A BATCHED VERSION
		# epsilons = [0, .05, .1, .15, .2, .25, .3]
		data = np.expand_dims(data, axis=0)

		if (randint(0, 5) > 1):
			# Call Deepfool attack
			x_adv = self.adv_crafter.generate(x=data)
			pred = np.argmax(self.classifier.predict(x_adv), axis=1)
			label = (pred == np.argmax(target, axis=0)).astype(int)

			noise = cwgan.gen_rand_noise_with_label(pred)
			gen_adv_data = cwgan.generate_image(self.generator, noise)

			example = torch.cat((gen_adv_data, torch.from_numpy(x_adv).to(self.device)),1)
		else:
			# don't call
			preds = np.argmax(self.classifier.predict(data), axis=1)
			label = (preds == np.argmax(target, axis=0)).astype(int)
			noise = cwgan.gen_rand_noise_with_label(preds)
			gen_data = cwgan.generate_image(self.generator, noise)
			example = torch.cat((gen_data, torch.from_numpy(data).to(self.device)),1)

		example = example.detach()

		return example.squeeze(0), label
