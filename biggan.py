import torch

from torch import nn
import torch.nn.functional as F

from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)
import torchvision as tv
import torchvision.transforms as tr

from argparse import ArgumentParser

import os, math, sys, random, tqdm
import util

from torch.utils.tensorboard import SummaryWriter

import matplotlib as mpl
mpl.use('Agg')

from skimage import io

# workaround for weird bug (macos only)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Decoder(nn.Module):

    def __init__(self, zdim=256, out_channels=1, prepdepth=3, trunc=0.4):
        super().__init__()

        self.ganz = 128
        self.ganclasses = 1000
        self.trunc = trunc

        self.zdim = zdim

        self.biggan = BigGAN.from_pretrained('biggan-deep-512')
        for param in self.biggan.parameters():
            param.requires_grad=False

        # prep network maps the VAE latent dims to the GAN latent dims
        prep = []
        prep.extend([nn.Linear(zdim, zdim*2), nn.ReLU()])
        for _ in range(prepdepth-2):
            prep.extend([nn.Linear(zdim*2, zdim * 2), nn.ReLU()])

        prep.append(nn.Linear(zdim*2, self.ganz+self.ganclasses))
        self.prep = nn.Sequential(*prep)

    def forward(self, z, cond=None):

        zprep = self.prep(z)

        zgan = zprep[:, :self.ganz]
        zcls = zprep[:, self.ganz:]

        zcls = F.softmax(zcls, dim=1)

        out = self.biggan(zgan, zcls, truncation=self.trunc)

        return (out + 1.0) / 2.0

class Encoder(nn.Module):

    def __init__(self, insize=(3, 512, 512), zdim=256, prepdepth=3):
        super().__init__()

        c, h, w = insize

        # small encoder stack for now
        c1, c2, c3 = 4, 8, 16 # channel sizes
        self.cnns = nn.Sequential(
            nn.Conv2d(c,  c1, kernel_size=3, padding=1), nn.ReLU(),
            # nn.Conv2d(c1, c1, kernel_size=3, padding=1), nn.ReLU(),
            # nn.Conv2d(c1, c1, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1), nn.ReLU(),
            # nn.Conv2d(c2, c2, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(4),
        )
        rh, rw = h // 4**3, w // 4**3
        self.r = rh * rw * c3

        self.toz = nn.Sequential(
            nn.Linear(self.r, self.r // 2), nn.ReLU(),
            nn.Linear(self.r //2, zdim * 2)
        )

    def forward(self, x, cond=None):

        b, c, h, w = x.size()

        x = self.cnns(x)
        x = x.view(b, -1)

        return self.toz(x)


def go(arg):

    if arg.seed < 0:
        seed = random.randint(0, 1000000)
        print('random seed: ', seed)
    else:
        torch.manual_seed(arg.seed)

    tbw = SummaryWriter(log_dir=arg.tb_dir) # Tensorboard logging

    # Load data
    transform = tr.Compose([tr.ToTensor()]) # tr.Resize((512, 512)),
    trainset = tv.datasets.ImageFolder(root=arg.data_dir, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size, shuffle=True, num_workers=2)

    C, H, W = 3, 512, 512

    encoder, decoder = Encoder(insize=(C, H, W), zdim=arg.ls), Decoder(zdim=arg.ls)

    opt = torch.optim.Adam(lr=arg.lr, params=list(encoder.parameters()) + list(decoder.parameters()))

    if torch.cuda.is_available():
        encoder.to('cuda')
        decoder.to('cuda')

    seen = 0
    for e in range(arg.epochs):

        print(f'epoch {e}', end='')
        for i, (images, _) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):

            if arg.limit is not None and i > arg.limit:
                break

            opt.zero_grad()

            b, c, h, w = images.size()
            seen += b

            if torch.cuda.is_available():
                images = images.to('cuda')

            z = encoder(images)

            zmean, zsig = z[:, :arg.ls], z[:, arg.ls:]

            klterm = util.kl_loss(zmean, zsig)
            zsample = util.sample(zmean, zsig)

            out = decoder(zsample)

            assert out.min() >= 0.0
            assert out.max() <= 1.0

            recterm = F.binary_cross_entropy(out, images, reduction='none') # cheat for now
            recterm = recterm.view(b, -1).sum(dim=1)

            loss = (recterm + arg.beta * klterm).mean()

            tbw.add_scalar('biggan/rc-loss', float(recterm.mean().item()), seen)
            tbw.add_scalar('biggan/kl-loss', float(klterm.mean().item()), seen)
            tbw.add_scalar('biggan/train-loss', float(loss.item()), seen)

            loss.backward()
            opt.step()

        # Generate images
        with torch.no_grad():
            b = 9
            z = torch.randn(b, arg.ls, device=util.d())
            output = decoder(z)

            output = output.to('cpu')

            for i in range(b):
                io.imsave(f'sample.{e}.{i}.png', output[i].permute(1, 2, 0).numpy(), check_contrast=False)

if __name__ == "__main__":

    # Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-l", "--lr",
                        dest="lr",
                        help="Learning rate.",
                        default=0.0001, type=float)

    parser.add_argument("-B", "--beta",
                        dest="beta",
                        help="Multiplier for balancing KL and rec loss.",
                        default=1.0, type=float)

    parser.add_argument("-b", "--batch",
                        dest="batch_size",
                        help="Batch size.",
                        default=32, type=int)

    parser.add_argument("-L", "--latent-size",
                        dest="ls",
                        help="Size of the VAE latent space.",
                        default=256, type=int)

    parser.add_argument("--limit",
                        dest="limit",
                        help="Number of batches to train on.",
                        default=None, type=int)

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs over the generated data.",
                        default=30, type=int)

    parser.add_argument("-i", "--iblocks",
                        dest="iblocks",
                        help="Number of trainable transformer blocks inserted.",
                        default=3, type=int)

    parser.add_argument("-D", "--data",
                        dest="data_dir",
                        help="Data dir.",
                        default='./data/images')

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("-T", "--tb-dir", dest="tb_dir",
                        help="Tensorboard logging directory",
                        default='./runs')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("--checkpoint",
                        dest="checkpoint",
                        help="Load a model checkpoint to start from.",
                        default=None, type=str)

    options = parser.parse_args()
    print('OPTIONS ', options)

    go(options)