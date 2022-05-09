import argparse
import os
from re import T
from networks import Discriminator, Generator
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


class myDataset(Dataset):
    def __init__(self, data_dir, transform):
        '''
        change .png to .jpg if use real pizza images
        '''
        self.data_dir = data_dir
        self.transform = transform
        self.img_names = [
            name
            for name in list(
                filter(lambda x: x.endswith(".png"), os.listdir(self.data_dir)) 
            )
        ]

    def __getitem__(self, index):
        path_img = os.path.join(self.data_dir, self.img_names[index])
        img = Image.open(path_img).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        if len(self.img_names) == 0:
            raise Exception(
                "\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                    self.data_dir
                )
            )
        return len(self.img_names)


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_epochs", type=int, default=100, help="number of epochs of training"
)
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument(
    "--b1",
    type=float,
    default=0.5,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--b2",
    type=float,
    default=0.999,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--n_cpu",
    type=int,
    default=4,
    help="number of cpu threads to use during batch generation",
)
parser.add_argument(
    "--latent_dim", type=int, default=100, help="dimensionality of the latent space"
)
parser.add_argument(
    "--img_size", type=int, default=256, help="size of each image dimension"
)
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=100, help="interval betwen image samples"
)
opt = parser.parse_args()
print(opt)
 
# img_shape = (opt.channels, opt.img_size, opt.img_size)
device = "cuda" if torch.cuda.is_available() else "cpu"


# Loss function

adversarial_loss = torch.nn.BCEWithLogitsLoss()
def adversarial_loss2(i ,j):
    # return F.binary_cross_entropy(i,j)
    return F.mse_loss(i, j)
# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)


dataset = r"D:\pizza\fake"
ng_directory = os.path.join(dataset, "trainAA")
ok_directory = os.path.join(dataset, "valA")

image_transforms = {
    "ng": transforms.Compose(
        [
            transforms.Resize([opt.img_size, opt.img_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    ),
    "ok": transforms.Compose(
        [
            transforms.Resize([opt.img_size, opt.img_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    ),
}

data = {
    "ng": myDataset(data_dir=ng_directory, transform=image_transforms["ng"]),
    "ok": myDataset(data_dir=ok_directory, transform=image_transforms["ok"]),
}

dataloader = DataLoader(data["ng"], batch_size=opt.batch_size, shuffle=True)
ng_data_size = len(data["ng"])
ok_data_size = len(data["ok"])
print("train_size: {:4d}  valid_size:{:4d}".format(ng_data_size, ok_data_size))
# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)


Tensor = torch.cuda.FloatTensor

'''
Training
'''
fh = open(r"D:\pizza\log", 'w')
score = nn.Sigmoid()

for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(dataloader):
        # Sample noise as generator input
        z = torch.randn(opt.batch_size, 3, 256, 256, device=device)
        # Configure input
        real_imgs = Variable(imgs.type(Tensor)).to(device)
       
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        bb = discriminator(real_imgs)
        real = torch.ones_like(bb,device=device)
        # fake = torch.zeros_like(bb,device=device)
        
        real_loss = adversarial_loss(bb, real)
        
        cc = discriminator(generator(z).detach())
        fake_loss = adversarial_loss(cc, torch.zeros_like(cc,device=device))
        d_loss = (real_loss + fake_loss) / 2
       
        d_loss.backward()
        with torch.no_grad():
                Dx = score(bb).mean().item()
        optimizer_D.step()
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()
        
        # Generate a batch of images
        gen_imgs = generator(z)
        
        aa = discriminator(gen_imgs)
        
        # print(aa.shape, valid.shape)
        g_loss = adversarial_loss(aa, torch.ones_like(aa,device=device))
        # g_loss = F.binary_cross_entropy(aa, fake)
        g_loss.backward()
        with torch.no_grad():
                Dgz = score(aa).mean().item()
        optimizer_G.step()

        result =  "Epoch %d/%d, Batch %d/%d, D: %f, G: %f, Dx:%f, D(g(z)): %f\n" % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(),Dx, Dgz)
        fh.write(result)
        print(result)
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # print(result)
            save_image(
                gen_imgs.data[:4],
                 f"./images/epoch{epoch}_batch{i}.png",
                nrow=2,
                normalize=True,
            )
    # if epoch % 5 == 0:

    torch.save(generator.state_dict(), f"./model/generator_epoch{epoch}.pth")
    torch.save(discriminator.state_dict(), f"./model/discriminator_epoch{epoch}.pth")

fh.close()