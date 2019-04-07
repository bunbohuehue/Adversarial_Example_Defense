import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os

import cleverhans

if not os.path.exists('./output_img'):
    os.mkdir('./output_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST('./', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),  # b, 16, 14, 14
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),  # b, 32, 7, 7
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.mean = nn.Linear(7 * 7 * 32, 20, bias=True)
        self.std = nn.Linear(7 * 7 * 32, 20, bias=True)

        self.fc = nn.Sequential(nn.Linear(20, 7 * 7 * 32, bias=True), nn.ReLU(inplace=True))
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

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu, logvar = self.mean(x), self.std(x)
        z = self.reparametrize(mu, logvar)
        z = self.fc(z)
        x = self.decoder(z.view(z.size(0), 32, 7, 7)), mu, logvar
        return x

reconstruction_function = nn.MSELoss(size_average=False)

def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating imagess
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD


model = autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = Variable(img)
        # ===================forward=====================
        recon_batch, mu, logvar = model(img)
        loss = loss_function(recon_batch, img, mu, logvar)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.data.item()))
    if epoch % 1 == 0:
        pic = to_img(recon_batch.data)
        save_image(pic, './output_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './conv_autoencoder.pth')
