# generate images .py 
# all the 4 generators to be trained paralleley overall 

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


G = generator(128)
D = discriminator(128)

G.cuda()
D.cuda()


G.load_state_dict(torch.load("CelebA_DCGAN_results/generator_param.pkl"))

D.load_state_dict(torch.load("CelebA_DCGAN_results/discriminator_param.pkl"))

#z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
#z_ = Variable(z_.cuda(), volatile=True)
#G.eval()
#test_images = G(z_)

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


show_result(22)

class encoder_trans(nn.Module):

    # initializers

    def __init__(self, d=128 , input_shape=(3,64,64)):

        super(encoder_trans, self).__init__()

        # in , out , kernel , stride , padding 

        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

        n_size = self._get_conv_output(input_shape)
        
        self.fc1 = nn.Linear(n_size, 100) # d number of outputs for the encoder GAN 


    def _get_conv_output(self, shape):

        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)

        return n_size


    def _forward_features(self, x):

        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)

        return x

    # weight_init

    def weight_init(self, mean, std):

        for m in self._modules:

            normal_init(self._modules[m], mean, std)

    # forward method

    def forward(self, input):

        x = self._forward_features(input)
        x = x.view(x.size(0), -1)
        x = F.linear(self.fc1(x))

        return x



class encoder_deep_trans(nn.Module):

    # initializers

    def __init__(self, d=128 , input_shape=(3,64,64)):

        super(encoder_deep_trans, self).__init__()

        # in , out , kernel , stride , padding 

        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)

        self.conv7 = nn.Conv2d(d*8, d*16, 4, 2, 1)  # addded to the network , not present in the original network
        self.conv7_bn = nn.BatchNorm2d(d*16)


        #self.conv8 = nn.Conv2d(d*8, 1, 4, 1, 0)

        n_size = self._get_conv_output(input_shape)
        
        self.fc8 = nn.Linear(n_size, 100) # d number of outputs for the encoder GAN 


    def _get_conv_output(self, shape):

        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)

        return n_size


    def _forward_features(self, x):

        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv7_bn(self.conv7(x)), 0.2)

        return x

    # weight_init

    def weight_init(self, mean, std):

        for m in self._modules:

            normal_init(self._modules[m], mean, std)

    # forward method

    def forward(self, input):

        x = self._forward_features(input)
        x = x.view(x.size(0), -1)
        x = self.fc8(x)

        return x


# have to see what is the L2 and L1 losses over these networks 

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

# how to run networks on top of certain inputs of the data sections as well 

model = encoder_trans(128)

model_deep = encoder_deep_trans(128)

e_fc = encoder_fc(128)

model_deep.load_state_dict(torch.load("e_deep_trans_base.pkl"))

model_deep.cuda()


batch_size = 128
lr = 0.0002
train_epoch = 120

# data_loader

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):

    x = range(len(hist['enc_losses']))

    y1 = hist['enc_losses']
    y2 = hist['enc_test_losses']

    plt.plot(x, y1, label='enc_train_loss')
    plt.plot(x, y2, label='enc_test_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    plt.show()

    plt.savefig(path)

    plt.close()



img_size = 64


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

data_dir = 'data/celebA_train'# this path depends on your computer
dset = datasets.ImageFolder(data_dir, transform)
train_loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=False)

# vo to train pe hi calc hone na overall 

data_dir2 = 'data/celebA_test'# this path depends on your computer

dset2 = datasets.ImageFolder(data_dir2, transform)
test_loader = torch.utils.data.DataLoader(dset2, batch_size=128, shuffle=True)

e_optimizer = optim.Adam(e_fc.parameters(), lr=lr, betas=(0.5, 0.999))

criterionL1 = torch.nn.L1Loss()
criterionMSE = torch.nn.MSELoss()

optimizer = optim.Adam(e_fc.parameters(), lr=0.001)

# Train the Model save the encoder model and test it side by side on the validation split 
# all the 4 encoders can be trained parallely 

transform3 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

data_dir3 = 'data/expr'# this path depends on your computer
dset3 = datasets.ImageFolder(data_dir3, transform3)

criterionMSE = torch.nn.MSELoss()

expr_loader = torch.utils.data.DataLoader(dset3, batch_size=4, shuffle=False)

print ('calc the vects!')

num_epochs = 1 

train_hist = {}
train_hist['enc_losses'] = []
train_hist['enc_test_losses'] = []


latent = []


for epoch in range(num_epochs):

    enc_losses = []

    for i, (images, labels) in enumerate(train_loader):

        print i 

        #print (epoch , i)

        #print images.shape 
        #print labels 

        images = Variable(images).cuda()
        #labels = Variable(labels).cuda()
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model_deep(images)

        #print (outputs.size())

        outputs = outputs.view(-1,100, 1, 1)

        o_n = outputs.cpu()
        oo = o_n.data.numpy()

        print oo.shape 

        oo = np.reshape(oo ,(128,100)) 

        latent = latent + oo.tolist()

        print np.array(latent).shape 


latent = np.array(latent)

pickle.dump( latent, open( "latent.p", "wb" ) )









