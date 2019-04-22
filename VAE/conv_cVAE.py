import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os

if not os.path.exists('./output_img'):
    os.mkdir('./output_img')

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

class Onehot(object):
    def __init__(self, cls_num):
        self.cls_num = cls_num

    def __call__(self, tensor):
        return (torch.arange(0, self.cls_num) == tensor).float()

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

num_epochs = 100
batch_size = 64
learning_rate = 1e-3
hdim = 128

img_transform = transforms.Compose([
    transforms.ToTensor()
    ,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),  # b, 16, 14, 14
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),  # b, 32, 7, 7
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.mean = nn.Linear(7 * 7 * 32 + 10, hdim, bias=True)
        self.std = nn.Linear(7 * 7 * 32 + 10, hdim, bias=True)

        self.fc = nn.Sequential(nn.Linear(hdim + 10, 7 * 7 * 32, bias=True), nn.ReLU(inplace=True))
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
    #print (BCE, KLD)
    return BCE, KLD

def gen_image(model, cls):
    # cls batch_size
    # output batch_size x 1 x 28 x 28
    eps = Variable(torch.randn(len(cls), hdim))
    c = torch.cat([(torch.arange(0, 10) == cls[i]).float().unsqueeze(0) for i in range(len(cls))], 0)
    z = torch.cat([eps, c], 1)
    z = model.fc(z)
    x = model.decoder(z.view(z.size(0), 32, 7, 7))
    return x


def main():
    dataset = MNIST('./', transform=img_transform, target_transform=Onehot(10), download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = autoencoder()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        epochloss = 0.0
        i = 0
        for data in dataloader:
            img, label = data
            img = Variable(img)
            label = Variable(label)
            # ===================forward=====================
            recon_batch, mu, logvar = model(img, label)
            BCE, KLD = loss_function(recon_batch, img, mu, logvar)
            loss = 0.05*BCE+KLD
            if i%100 == 0:
                print(BCE,KLD)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epochloss += loss.data.item()
            i+=1

        # save model
        torch.save(model.state_dict(), './curr_model_{}'.format(epoch))
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, epochloss/len(dataset)))
        if epoch % 1 == 0:
            pic = to_img(recon_batch.data)
            save_image(pic, './output_img/image_{}.png'.format(epoch))
            pic = to_img(torch.cat([gen_image(model, i, 8) for i in range(10)], 0))
            save_image(pic, './output_img/gen_image_{}.png'.format(epoch))

    torch.save(model.state_dict(), './conv_autoencoder.pth')


if __name__ == "__main__":
    main()
