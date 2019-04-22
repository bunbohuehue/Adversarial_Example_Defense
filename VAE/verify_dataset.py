import attack
import torch
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import conv_cVAE
from random import randint
from torchvision.datasets import MNIST
from torchvision import transforms

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "../data/lenet_mnist_model.pth"
use_cuda = True


img_transform = transforms.ToTensor()

class verifyDataset(torch.utils.data.Dataset):
	def __init__(self, VAE_model, classifier, training):
		self.dataset = MNIST('./', train=training, transform=img_transform, download=True)
		self.VAE_model = VAE_model
		self.classifier = classifier

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, i):
		return self.generate(*self.dataset[i])

	def generate(self, data, target):
		data = data.unsqueeze(0)
		target = target.unsqueeze(0)
		# Send the data and label to the device
		data.requires_grad = True

		# Forward pass the data through the model
		output = self.classifier(data)
		original_pred = output.max(1, keepdim=True)[1]

		loss = F.nll_loss(output, target)
		self.classifier.zero_grad()
		loss.backward()
		data_grad = data.grad.data

		if (randint(0, 4) > 1):
			# Call FGSM Attack
			epsilon = epsilons[randint(3, len(epsilons) - 1)]
			# generate adversarial example
			adv_example = attack.fgsm_attack(data, epsilon, data_grad)
			# classify adv example
			output = self.classifier(adv_example)

			# Check for success
			adv_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

			label = (adv_pred == target)[0].item()
			gen_adv_data = (conv_cVAE.gen_image(self.VAE_model, adv_pred))
			example = torch.cat((gen_adv_data, adv_example),1)
		else:
			# don't call FGSM
			label = (original_pred == target)[0].item()
			# cVAE generated image
			gen_data = (conv_cVAE.gen_image(self.VAE_model, original_pred))
			example = torch.cat((gen_data, data),1)

		return example.squeeze(0), label
