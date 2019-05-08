# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import copy
from torch.autograd.gradcheck import zero_gradients

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

def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):

	"""
	   :param image: Image of size HxWx3
	   :param net: network (input: images, output: values of activation **BEFORE** softmax).
	   :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
	   :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
	   :param max_iter: maximum number of iterations for deepfool (default = 50)
	   :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
	"""
	is_cuda = torch.cuda.is_available()

	if is_cuda:
		print("Using GPU")
		image = image.cuda()
		net = net.cuda()
	else:
		print("Using CPU")


	f_image = net.forward(Variable(image[:, :, :, :], requires_grad=True)).data.cpu().numpy()
	I = (np.array(f_image)).argsort()[::-1]

	#I = I[0:num_classes]
	#label = I[0]
	I = I[:,:num_classes]
	label = I[:,:1]


	input_shape = (image.size(0), image.size(1), image.size(2), image.size(3))
	pert_image = copy.deepcopy(image)
	w = np.zeros(input_shape)
	r_tot = np.zeros(input_shape)

	loop_i = 0

	x = Variable(pert_image, requires_grad=True)
	fs = net.forward(x)
	# fs_list = [fs[0,I[k]] for k in range(num_classes)]
	k_i = label

	while k_i == label and loop_i < max_iter:

		pert = np.inf
		fs[0, I[0]].backward(retain_graph=True)
		grad_orig = x.grad.data.cpu().numpy().copy()

		for k in range(1, num_classes):
			zero_gradients(x)

			fs[0, I[k]].backward(retain_graph=True)
			cur_grad = x.grad.data.cpu().numpy().copy()

			# set new w_k and new f_k
			w_k = cur_grad - grad_orig
			f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

			pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

			# determine which w_k to use
			if pert_k < pert:
				pert = pert_k
				w = w_k

		# compute r_i and r_tot
		# Added 1e-4 for numerical stability
		r_i =  (pert+1e-4) * w / np.linalg.norm(w)
		r_tot = np.float32(r_tot + r_i)

		if is_cuda:
			pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
		else:
			pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

		x = Variable(pert_image, requires_grad=True)
		fs = net.forward(x)
		k_i = np.argmax(fs.data.cpu().numpy().flatten())

		loop_i += 1

	r_tot = (1+overshoot)*r_tot

	return r_tot, loop_i, label, k_i, pert_image

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
