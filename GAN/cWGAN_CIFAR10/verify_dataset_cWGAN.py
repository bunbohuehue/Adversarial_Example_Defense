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

# import
from art.classifiers import PyTorchClassifier
from art.utils import load_dataset
from art.attacks.fast_gradient import FastGradientMethod
from art.attacks.deepfool import DeepFool

NUM_CLASSES = 10

pretrained_model = "cifar10-d875770b.pth"

class verifyDataset(torch.utils.data.Dataset):
	def __init__(self, x_train, y_train, x_test, y_test, generator, classifier, adv_crafter, training):
		# self.dataset = CIFAR10('./', train=training, transform=transforms.ToTensor(), download=True)
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
		# epsilons = [0, .05, .1, .15, .2, .25, .3]
		data = np.expand_dims(data, axis=0)

		if (randint(0, 4) > 1):
			# Call FGSM Attack or Deepfool attack
			# epsilon = epsilons[randint(1, len(epsilons) - 1)]
			epsilon = 1e-6
			x_adv = self.adv_crafter.generate(x=data, eps=epsilon)
			preds = np.argmax(self.classifier.predict(x_adv), axis=1)
			label = (preds == np.argmax(target, axis=0))

			noise = cwgan.gen_rand_noise_with_label(preds)
			gen_adv_data = cwgan.generate_image(self.generator, noise)

			example = torch.cat((gen_adv_data, torch.from_numpy(x_adv)),1)
		else:
			# don't call FGSM
			preds = np.argmax(self.classifier.predict(data), axis=1)
			label = (preds == np.argmax(target, axis=0))
			noise = cwgan.gen_rand_noise_with_label(preds)
			gen_data = cwgan.generate_image(self.generator, noise)
			example = torch.cat((gen_data, torch.from_numpy(data)),1)

		example = example.detach()

		return example.squeeze(0), label
