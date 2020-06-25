import random
import math
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from Generator import Generator
from Discriminator import Discriminator

# CUDA
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)


# Random seed
manualSeed = 7777
print('Random Seed:', manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Attributes
dataroot = './data'
batch_size = 128
image_size = 64
G_out_D_in = 3
G_in = 100
G_hidden = 128
D_hidden = 128
attribute = [21]
category = pow(2,len(attribute))

lr_rate = 0.0002
beta1 = 0.5

# Data
def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset(Dataset):
    def __init__(self, txt, attribute=None,transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        attr_num = len(attribute)
        idx = 0
        a = [0,0]
        for line in fh:  #each pic 讀取檔名 特性
            if idx < 2:
                idx += 1
                continue
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            img_name = words[0]
            label = 0
            power = attr_num
            for att in attribute:  #對指定的屬性做label 初步換成整數編碼
                if words[att] == "1":
                    label += pow(2, power-1)
                power -= 1

            if a[0] < 84434 :
                imgs.append(("./data/img_align_celeba/" + img_name, label))
                a[label] += 1
            else :
                if label == 1:
                    imgs.append(("./data/img_align_celeba/"+img_name,label))
                    a[label] += 1

        print(a)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

dataset=MyDataset(txt='./data/list_attr_celeba.txt', attribute=attribute, transform=transforms.Compose([transforms.Resize(image_size),
                                                                        transforms.CenterCrop(image_size),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
# Create the dataLoader
dataLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
'''
def show_batch(imgs):
    grid = vutils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')

for i, (batch_x, batch_y) in enumerate(dataLoader):
    if(i<4):
        print(i, batch_x.size(),batch_y.size())
        show_batch(batch_x[:16])
        plt.axis('off')
        plt.show()
'''
# Weights
def weights_init(m): #網路參數初始化
    classname = m.__class__.__name__
    print('classname:', classname)

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02) #mean , std
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('ConvTranspose') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

# Train
def train(epoch_in):
    # Create the generator
    netG = Generator(G_in, G_hidden, G_out_D_in, category).to(device)
    netG.apply(weights_init) #訪問網路中的模組 把初始參數的函式套用到模組上
    print(netG)

    # Create the discriminator
    netD = Discriminator(G_out_D_in, D_hidden, category).to(device)
    netD.apply(weights_init)
    print(netD)

    # Loss fuG_out_D_intion
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr_rate, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_rate, betas=(beta1, 0.999))
    real_label = 1
    fake_label = 0

    img_list = []
    G_losses = []
    D_losses = []
    epochs = epoch_in
    iters = 0
    print('Start!')
    fixed_noise = torch.randn(10, G_in, 1, 1, device=device)
    gen_labels = torch.ones(10, 1).type(torch.LongTensor)
    gen_att_onehot = torch.zeros(10, category)
    gen_att_onehot = gen_att_onehot.scatter_(1, gen_labels, 1)
    gen_att_onehot_male = gen_att_onehot.view(10, category, 1, 1).to(device)

    gen_labels = torch.zeros(10, 1).type(torch.LongTensor)
    gen_att_onehot = torch.zeros(10, category)
    gen_att_onehot = gen_att_onehot.scatter_(1, gen_labels, 1)
    gen_att_onehot_female = gen_att_onehot.view(10, category, 1, 1).to(device)


    for epoch in range(epochs):
        if (epoch+1) == 15:
            optimizerD.param_groups[0]['lr'] /= 10
            optimizerG.param_groups[0]['lr'] /= 10
        if (epoch+1) == 25:
            optimizerD.param_groups[0]['lr'] /= 10
            optimizerG.param_groups[0]['lr'] /= 10
        for i, (realimg , att_label) in enumerate(dataLoader, 0):
            # Update D network
            ###real img###
            netD.zero_grad()
            real_img = realimg.to(device)
            b_size = realimg.size(0)
            att_onehot = torch.zeros(b_size, category)
            att_onehot = att_onehot.scatter_(1, att_label.view(b_size, 1), 1)  # make one hot label
            att_onehot = att_onehot.view(b_size, category, 1, 1)  # 拉高維度 要與真實圖片維度一樣 channel不用
            att_onehot = att_onehot.expand(-1, -1, image_size, image_size).to(device) #擴展成跟圖片一樣尺寸
            label = torch.full((b_size,), real_label, device=device)# torch.size[b_size]
            output = netD(real_img, att_onehot).squeeze() #b x category x imgsize x imgsize
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            ###fake img###
            noise = torch.randn(b_size, G_in, 1, 1, device=device)
            gen_labels = (torch.rand(b_size, 1) * category).type(torch.LongTensor)
            gen_att_onehot = torch.zeros(b_size, category)
            gen_att_onehot = gen_att_onehot.scatter_(1, gen_labels, 1)
            gen_att_onehot = gen_att_onehot.view(b_size, category, 1, 1).to(device)

            fake_img = netG(noise, gen_att_onehot)
            label.fill_(fake_label)
            genforD_att_onehot = gen_att_onehot.expand(-1, -1, image_size, image_size).to(device)
            output = netD(fake_img.detach(), genforD_att_onehot).squeeze()#detach 不要更新到G
            D_G_z1 = output.mean().item()
            errD_fake = criterion(output, label)
            errD_fake.backward()

            errD = errD_real + errD_fake
            optimizerD.step()

            # Update G network
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake_img, genforD_att_onehot).squeeze()
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, epochs, i, len(dataLoader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if i == (len(dataLoader)-1):
                with torch.no_grad():
                    fake_male = netG(fixed_noise, gen_att_onehot_male).detach().cpu()  # 64x3x64x64
                    fake_female = netG(fixed_noise, gen_att_onehot_female).detach().cpu()  # 64x3x64x64
                    fake = torch.cat((fake_male,fake_female),0)  #兩個batch合併

                   # print(fake.size())
                #將某個mini-batch做成一張圖  並放入list中
                #normalize = true 把 [-1,1] tensor 轉為[0,1]
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True, nrow=10))
                #plotImage(G_losses, D_losses, img_list, epochs)
            iters += 1
        plotImage(G_losses, D_losses, img_list, epoch)

    #torch.save(netD, 'netD.pkl')
    #torch.save(netG, 'netG.pkl')

    return G_losses, D_losses, img_list

# Plot
def plotImage(G_losses, D_losses, img_list, epochs):
    print('Start to plot!!')
    epoch_str = str(epochs)
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss%s.png" % epoch_str)
   # plt.show()

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataLoader))

    # Plot the real images
#    plt.figure(figsize=(15, 15))
#    plt.subplot(1, 2, 1)
#    plt.axis("off")
 #   plt.title("Real Images")
  #  plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
 #   plt.subplot(1, 2, 2)
 #   plt.axis("off")
 #   plt.title("Fake Images")
  #  plt.imshow(np.transpose(img_list[-1], (1, 2, 0))) #c h w  to  h w c 讓imshow畫出圖片
   # plt.savefig("fake_at_%s.png" % epoch_str)"""


    plt.figure()
    plt.axis("off")
    plt.title("Fake Images at %s" %epoch_str)
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.savefig("fake_at_%s.png" % epoch_str)

    #make gif and save
    fig = plt.figure()
    eachfake = []
    for img in img_list:
        eachfake.append([plt.imshow(np.transpose(img, (1, 2, 0)))])

    fakegif = animation.ArtistAnimation(fig, eachfake, interval=400, repeat_delay=3000)
    fakegif.save("fake%s.gif" % epoch_str,writer='pillow')


g_los, d_los, img = train(30)



