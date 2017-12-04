import os, time, sys

import numpy as np 

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class generator(nn.Module):

    # initializers
    
    def __init__(self, d=128):

        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    # weight_init

    def weight_init(self, mean, std):

        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method

    def forward(self, input):

        # x = F.relu(self.deconv1(input))

        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x



class discriminator(nn.Module):

    # initializers

    def __init__(self, d=128):

        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method

    def forward(self, input):

        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x



def normal_init(m, mean, std):

    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class encoder_fc(nn.Module):

    # initializers
    # it's a basic 4 layered encoder 

    def __init__(self, d=128  ,input_shape=(3,64,64)):

        super(encoder_fc, self).__init__()

        # in , out , kernel , stride , padding 

        n_size = 12288

        self.fc1 = nn.Linear(n_size, 8*d)

        self.fc2 = nn.Linear(8*d, 4*d)
        self.fc3 = nn.Linear(4*d, 2*d)
        self.fc4 = nn.Linear(2*d, 100)

    # weight_init

    def weight_init(self, mean, std):

        for m in self._modules:

            normal_init(self._modules[m], mean, std)

    # forward method

    def forward(self, input):

        #print (input.size())

        x = input.view(input.size(0), -1) # flatten hack 

        #print (x.size())

        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.fc4(x)

    
        return x



def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):

    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
    z_ = Variable(z_.cuda(), volatile=True)

    G.eval()
    
    
    test_images = G(z_)

    test_2 = ((test_images.cpu().data.numpy().transpose(0,2,3,1) + 1) / 2)
    
    #print np.max(test_2) # yeah vals between 0 to 1 hain overall 
    #print np.min(test_2) # yeah the vals are between 0 to 1 overall 

    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):

        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2))

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    plt.savefig(str(num_epoch) + path)

    if show:
        plt.show()
    else:
        plt.close()



G = generator(128)
D = discriminator(128)
e_fc = encoder_fc(128)


G.load_state_dict(torch.load("CelebA_DCGAN_results/generator_param.pkl"))
D.load_state_dict(torch.load("CelebA_DCGAN_results/discriminator_param.pkl"))
e_fc.load_state_dict(torch.load("e_fc_base.pkl"))

G.cuda()
D.cuda()
e_fc.cuda()

img_size = 64


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

data_dir = 'data/expr'# this path depends on your computer
dset = datasets.ImageFolder(data_dir, transform)

criterionMSE = torch.nn.MSELoss()

expr_loader = torch.utils.data.DataLoader(dset, batch_size=4, shuffle=False)

for i, (images, labels) in enumerate(expr_loader):

    images = Variable(images).cuda()

    outputs = e_fc(images)

    outputs = outputs.view(-1,100, 1, 1)

    G.cuda()
    G.eval()

    out_images = G(outputs)

    loss = criterionMSE(out_images, images)


    size_figure_grid = 2

    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(2, 2))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(2*2):

        i = k // 2
        j = k % 2

        ax[i, j].cla()
        ax[i, j].imshow(((out_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2))

    label = 'Reconstructions'

    fig.text(0.5, 0.04, label, ha='center')

    plt.savefig('recons.png')

    
    plt.show()
    
    plt.close()


    size_figure_grid = 2

    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(2, 2))

    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(2*2):

        i = k // 2
        j = k % 2

        ax[i, j].cla()
        ax[i, j].imshow(((images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2))

    label = 'original'

    fig.text(0.5, 0.04, label, ha='center')

    plt.savefig('orig.png')

    
    plt.show()
    
    plt.close()

    print (loss.data[0])











