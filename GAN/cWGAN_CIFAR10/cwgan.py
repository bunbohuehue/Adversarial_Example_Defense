import os, sys

sys.path.append(os.getcwd())

import time
import numpy as np
import libs as lib
import libs.plot

import pdb

from models.conwgan import *

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import grad
from timeit import default_timer as timer

import torch.nn.init as init

DATA_DIR = './datasets/cifar10'
VAL_DIR = './datasets/cifar10'

IMAGE_DATA_SET = 'cifar10'  # change this to something else, e.g. 'imagenets' or 'raw' if your data is just a folder of raw images.
# If you use lmdb, you'll need to write the loader by yourself, see load_data
TRAINING_CLASS = ['dining_room_train', 'bridge_train', 'restaurant_train', 'tower_train']
VAL_CLASS = ['dining_room_val', 'bridge_val', 'restaurant_val', 'tower_val']
NUM_CLASSES = 10

if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_64x64.py!')

RESTORE_MODE = False  # if True, it will load saved model from OUT_PATH and continue to train
START_ITER = 0  # starting iteration
OUTPUT_PATH = './cwgan/'  # output path where result (.e.g drawing images, cost, chart) will be stored
# MODE = 'wgan-gp'
DIM = 32  # Model dimensionality
CRITIC_ITERS = 5  # How many iterations to train the critic for
GENER_ITERS = 1
N_GPUS = 1  # Number of GPUs
BATCH_SIZE = 100  # Batch size. Must be a multiple of N_GPUS
END_ITER = 100000  # How many iterations to train for
LAMBDA = 10  # Gradient penalty lambda hyperparameter
OUTPUT_DIM = 32 * 32 * 3  # Number of pixels in each iamge
ACGAN_SCALE = 1.  # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 1.  # How to scale generator's ACGAN loss relative to WGAN loss


def weights_init(m):
    if isinstance(m, MyConvo2d):
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)


def load_data(path_to_folder, classes):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    if IMAGE_DATA_SET == 'lsun':
        dataset = datasets.LSUN(path_to_folder, classes=classes, transform=data_transform)
    elif IMAGE_DATA_SET == 'cifar10':
        dataset = datasets.CIFAR10(path_to_folder, train=True, download=True, transform=data_transform)
    else:
        dataset = datasets.ImageFolder(root=path_to_folder, transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=5,
                                                 drop_last=True, pin_memory=True)
    return dataset_loader


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, 3, DIM, DIM)
    alpha = alpha.to(device)

    fake_data = fake_data.view(BATCH_SIZE, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def generate_image(netG, noise=None):
    if noise is None:
        rand_label = np.random.randint(0, NUM_CLASSES, BATCH_SIZE)
        noise = gen_rand_noise_with_label(rand_label)
    with torch.no_grad():
        noisev = noise
    samples = netG(noisev)
    samples = samples.view(BATCH_SIZE, 3, DIM, DIM)

    samples = samples * 0.5 + 0.5

    return samples


def gen_rand_noise_with_label(label=None):
    if label is None:
        label = np.random.randint(0, NUM_CLASSES, BATCH_SIZE)
    # attach label into noise
    noise = np.random.normal(0, 1, (BATCH_SIZE, 128))
    prefix = np.zeros((BATCH_SIZE, NUM_CLASSES))
    prefix[np.arange(BATCH_SIZE), label] = 1
    noise[np.arange(BATCH_SIZE), :NUM_CLASSES] = prefix[np.arange(BATCH_SIZE)]

    noise = torch.from_numpy(noise).float()
    noise = noise.to(device)

    return noise


cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
fixed_label = []
for c in range(BATCH_SIZE):
    fixed_label.append(c % NUM_CLASSES)
fixed_noise = gen_rand_noise_with_label(fixed_label)

if RESTORE_MODE:
    aG = torch.load(OUTPUT_PATH + "generator.pt")
    aD = torch.load(OUTPUT_PATH + "discriminator.pt")
else:
    aG = GoodGenerator(32, 32 * 32 * 3)
    aD = GoodDiscriminator(32, NUM_CLASSES)

    aG.apply(weights_init)
    aD.apply(weights_init)

LR = 1e-4
optimizer_g = torch.optim.Adam(aG.parameters(), lr=LR, betas=(0, 0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=LR, betas=(0, 0.9))

aux_criterion = nn.CrossEntropyLoss()  # nn.NLLLoss()

one = torch.FloatTensor([1])
mone = one * -1
aG = aG.to(device)
aD = aD.to(device)
one = one.to(device)
mone = mone.to(device)

# writer = SummaryWriter()


# Reference: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
def train():
    # writer = SummaryWriter()
    dataloader = load_data(DATA_DIR, TRAINING_CLASS)
    dataiter = iter(dataloader)
    for iteration in range(START_ITER, END_ITER):
        start_time = time.time()
        print("Iter: " + str(iteration))
        start = timer()
        # ---------------------TRAIN G------------------------
        for p in aD.parameters():
            p.requires_grad_(False)  # freeze D

        gen_cost = None
        for i in range(GENER_ITERS):
            # print("Generator iters: " + str(i))
            aG.zero_grad()
            f_label = np.random.randint(0, NUM_CLASSES, BATCH_SIZE)
            noise = gen_rand_noise_with_label(f_label)
            noise.requires_grad_(True)
            fake_data = aG(noise)
            gen_cost, gen_aux_output = aD(fake_data)

            aux_label = torch.from_numpy(f_label).long()
            aux_label = aux_label.to(device)
            aux_errG = aux_criterion(gen_aux_output, aux_label).mean()
            gen_cost = -gen_cost.mean()
            g_cost = ACGAN_SCALE_G * aux_errG + gen_cost
            g_cost.backward()

        optimizer_g.step()
        end = timer()
        # print('---train G elapsed time: {}'.format(end - start))
        # ---------------------TRAIN D------------------------
        for p in aD.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G
        for i in range(CRITIC_ITERS):
            # print("Critic iter: " + str(i))

            start = timer()
            aD.zero_grad()

            # gen fake data and load real data
            f_label = np.random.randint(0, NUM_CLASSES, BATCH_SIZE)
            noise = gen_rand_noise_with_label(f_label)
            with torch.no_grad():
                noisev = noise  # totally freeze G, training D
            fake_data = aG(noisev).detach()
            end = timer();
            # print('---gen G elapsed time: {}'.format(end - start))
            start = timer()
            batch = next(dataiter, None)
            if batch is None:
                dataiter = iter(dataloader)
                batch = dataiter.__next__()
            real_data = batch[0]  # batch[1] contains labels
            real_data.requires_grad_(True)
            real_label = batch[1]
            # print("r_label" + str(r_label))
            end = timer()
            # print('---load real imgs elapsed time: {}'.format(end - start))

            start = timer()
            real_data = real_data.to(device)
            real_label = real_label.to(device)

            # train with real data
            disc_real, aux_output = aD(real_data)
            aux_errD_real = aux_criterion(aux_output, real_label)
            errD_real = aux_errD_real.mean()
            disc_real = disc_real.mean()

            # train with fake data
            disc_fake, aux_output = aD(fake_data)
            # aux_errD_fake = aux_criterion(aux_output, fake_label)
            # errD_fake = aux_errD_fake.mean()
            disc_fake = disc_fake.mean()

            # train with interpolates data
            gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data)

            # final disc cost
            disc_cost = disc_fake - disc_real + gradient_penalty
            disc_acgan = errD_real  # + errD_fake
            (disc_cost + ACGAN_SCALE * disc_acgan).backward()
            w_dist = disc_fake - disc_real
            optimizer_d.step()

            end = timer()
            # print('---train D elapsed time: {}'.format(end - start))
        # ---------------VISUALIZATION---------------------
        # writer.add_scalar('data/gen_cost', gen_cost, iteration)
        # if iteration %200==199:
        #   paramsG = aG.named_parameters()
        #   for name, pG in paramsG:
        #       writer.add_histogram('G.' + name, pG.clone().data.cpu().numpy(), iteration)
        # ----------------------Generate images-----------------

        lib.plot.plot(OUTPUT_PATH + 'time', time.time() - start_time)
        lib.plot.plot(OUTPUT_PATH + 'train_disc_cost', disc_cost.cpu().data.numpy())
        lib.plot.plot(OUTPUT_PATH + 'train_gen_cost', gen_cost.cpu().data.numpy())
        lib.plot.plot(OUTPUT_PATH + 'wasserstein_distance', w_dist.cpu().data.numpy())
        if iteration % 100 == 99:
            val_loader = load_data(VAL_DIR, VAL_CLASS)
            dev_disc_costs = []
            for _, images in enumerate(val_loader):
                imgs = torch.Tensor(images[0])
                imgs = imgs.to(device)
                with torch.no_grad():
                    imgs_v = imgs

                D, _ = aD(imgs_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot(OUTPUT_PATH + 'dev_disc_cost.png', np.mean(dev_disc_costs))
            lib.plot.flush()
            gen_images = generate_image(aG, fixed_noise)
            torchvision.utils.save_image(gen_images, OUTPUT_PATH + 'samples_{}.png'.format(iteration), nrow=NUM_CLASSES,
                                         padding=2)
            grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
            # writer.add_image('images', grid_images, iteration)
            # gen_images = generate_image(iteration, aG, persistant_noise)
            # gen_images = torchvision.utils.make_grid(torch.from_numpy(gen_images), nrow=8, padding=1)
            # writer.add_image('images', gen_images, iteration)
            # ----------------------Save model----------------------
            torch.save(aG, OUTPUT_PATH + "generator.pt")
            torch.save(aD, OUTPUT_PATH + "discriminator.pt")
        lib.plot.tick()


train()

