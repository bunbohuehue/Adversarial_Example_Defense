# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

epsilons = [0, .05, .1, .15, .2, .25, .3]
#epsilons = [0]
pretrained_model = "cifar10-d875770b.pth"
use_cuda=False

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


# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
	datasets.CIFAR10('../data', train=False, download=True, transform=transforms.Compose([
			transforms.ToTensor()])),batch_size=1, shuffle=True)

train_loader = torch.utils.data.DataLoader(
	datasets.CIFAR10('../data', train=True, download=True, transform=transforms.Compose([
			transforms.ToTensor()])),
		batch_size=1, shuffle=True)

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
	# Collect the element-wise sign of the data gradient
	sign_data_grad = data_grad.sign()
	# Create the perturbed image by adjusting each pixel of the input image
	perturbed_image = image + epsilon*sign_data_grad
	# Adding clipping to maintain [0,1] range
	perturbed_image = torch.clamp(perturbed_image, 0, 1)
	# Return the perturbed image
	return perturbed_image

def test( model, device, test_loader, epsilon ):
	# Accuracy counter
	correct = 0
	adv_examples = []

	# Loop over all examples in test set
	idx = 0
	for data, target in test_loader:
		idx += 1
		if idx == 500:
			break
		# Send the data and label to the device
		data, target = data.to(device), target.to(device)
		data.requires_grad = True

		# Forward pass the data through the model
		output = model(data)
		init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

		if init_pred.item() != target.item():
			continue

		loss = F.nll_loss(output, target)
		model.zero_grad()
		loss.backward()
		data_grad = data.grad.data

		# Call FGSM Attack
		perturbed_data = fgsm_attack(data, epsilon, data_grad)

		# Re-classify the perturbed image
		output = model(perturbed_data)

		# Check for success
		final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
		if final_pred.item() == target.item():
			correct += 1
			# Special case for saving 0 epsilon examples
			if (epsilon == 0) and (len(adv_examples) < 5):
				adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
				adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
		else:
			# Save some adv examples for visualization later
			if len(adv_examples) < 5:
				adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
				adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

	# Calculate final accuracy for this epsilon
	#final_acc = correct/float(len(test_loader))
	final_acc = correct/float(10000)
	print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

	# Return the accuracy and an adversarial example
	return final_acc, adv_examples

if __name__ == "__main__":
	# Define what device we are using
	#print("CUDA Available: ",torch.cuda.is_available())
	device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
	# Initialize the network
	n_channel = 128
	cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
	layers = make_layers(cfg, batch_norm=True)
	model = CIFAR(layers, n_channel=8*n_channel, num_classes=10)
	model.to(device)

	# Load the pretrained model
	model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

	# Set the model in evaluation mode. In this case this is for the Dropout layers
	model.eval()

	accuracies = []
	examples = []

	# Run test for each epsilon
	for eps in epsilons:
		acc, ex = test(model, device, test_loader, eps)
		accuracies.append(acc)
		examples.append(ex)

	plt.figure(figsize=(5,5))
	plt.plot(epsilons, accuracies, "*-")
	plt.yticks(np.arange(0, 1.1, step=0.1))
	plt.xticks(np.arange(0, .35, step=0.05))
	plt.title("Accuracy vs Epsilon")
	plt.xlabel("Epsilon")
	plt.ylabel("Accuracy")
	plt.show()

	# Plot several examples of adversarial samples at each epsilon
	cnt = 0
	plt.figure(figsize=(8,10))
	for i in range(len(epsilons)):
		for j in range(len(examples[i])):
			cnt += 1
			plt.subplot(len(epsilons),len(examples[0]),cnt)
			plt.xticks([], [])
			plt.yticks([], [])
			if j == 0:
				plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
			orig,adv,ex = examples[i][j]
			plt.title("{} -> {}".format(orig, adv))
			plt.imshow(ex.transpose((1, 2, 0)))
	plt.tight_layout()
	plt.show()
