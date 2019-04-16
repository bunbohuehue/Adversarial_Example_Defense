import attack
import torch
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
from random import randint

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "../data/lenet_mnist_model.pth"
use_cuda=True

# usage:
# from verify_dataset import verifyDataset
# dataloader_args = dict(shuffle=True, batch_size=64, pin_memory=True
# verify_Dataset = verifyDataset(model, train_loader)
# verify_loader = dataloader.DataLoader(verify_Dataset, **dataloader_args)

class verifyDataset(torch.utils.data.Dataset):
	def __init__(self, model, train_loader):
		device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
		examples, labels = generate(model, device, train_loader)
		self.data = examples
		self.label = labels

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		return self.data[i], self.label[i]


def generate(model, device, test_loader):
	adv_examples = []
	adv_labels = []

	# Loop over all examples in test set
	for data, target in test_loader:

		# Send the data and label to the device
		data, target = data.to(device), target.to(device)
		data.requires_grad = True

		# Forward pass the data through the model
		output = model(data)

		loss = F.nll_loss(output, target)
		model.zero_grad()
		loss.backward()
		data_grad = data.grad.data

		# Call FGSM Attack
		epsilon = epsilons[randint(0, len(epsilons) - 1)]
		# generate adversarial example
		adv_example = attack.fgsm_attack(data, epsilon, data_grad)

		# classify adv example
		output = model(adv_example)

		# Check for success
		final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
		# if predicted label is true
		if final_pred.item() == target.item():
			adv_examples.append((data, adv_example))
			adv_labels.append(True)
		else:
			adv_examples.append((data, adv_example))
			adv_labels.append(True)

	return adv_examples, adv_labels
