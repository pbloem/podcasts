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
from util import d, here

import pandas as pd

from torch.utils.tensorboard import SummaryWriter

import matplotlib as mpl
mpl.use('Agg')

from skimage import io

# '1315', 'Science & Medicine'
# '1304', 'Education'
# '1480', 'Software How-To'
# '1318', 'Technology'
# '1448', 'Tech News'
# '1477', 'Natural Sciences'
# '1468',  Educational Technology
PD_GENRES = [1315, 1304, 1480, 1318, 1448, 1477, 1468]

# workaround for weird bug (macos only)
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def sample_gumbel(size, eps=1e-20):
    U = torch.rand(size, device=d())
    return - torch.log(-torch.log(U + eps) + eps)

def gs_sample(logits, temperature=0.5, dim=-1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=dim)

def tobatch(df, g2i, normalize_genres=True, glist=None):

    with torch.no_grad(): # just in case

        # batch of n-hot vectors for genres
        ng = len(g2i)
        genres = torch.zeros(len(df), ng)

        for row in range(len(df)):
            dfgs = [int(g) for g in eval(df.iloc[row]['Genre IDs'])]
            if len(dfgs) == 0:
                print(eval(df.iloc[row]['Genre IDs']))
                raise Exception('Encountered podcast without genres.')
            for intg in dfgs:
                genres[row, g2i[intg]] = 1

        if normalize_genres:
            genres = genres / genres.sum(dim=1, keepdim=True)

    return genres

class NoParam(nn.Module):
    """
    Wraps a module, stopping parameters from being registered
    """

    def __init__(self, mod):

        super().__init__()
        self.mod = [mod]

    def cuda(self): # this doesn't work
        print('NoParam: transferring inner model to CUDA')
        self.mod[0].cuda()

    def forward(self, x, *args, **kwargs):

        return self.mod[0](x, *args, **kwargs)

class IBlock(nn.Module):
    """
    Convolution layer to be inserted into existing architecture.

    """

    def __init__(self, insize, *args, kernel=3, padding=1, mult=0.0, csize=None, cond=[None], **kwargs):
        super().__init__()

        c, h, w = insize
        self.conv = nn.Conv2d(c, c, kernel_size=kernel, padding=padding)

        self.mult = nn.Parameter(torch.tensor([mult]))

        self.cond = cond
        self.cond_out = [None]

        if csize is not None:
            self.to_cond = nn.Linear(csize, c*h*w)

    def forward(self, x):

        b, c, h, w = x.size()

        if self.cond is not None and len(self.cond) > 0 and self.cond[0] is not None:
            cond = self.to_cond(self.cond[0]).view(b, c, h, w)

            self.cond_out[0] = cond

            xc = x + cond
        else:
            xc = x

        r = self.mult * self.conv(xc) + x

        return F.relu(r)

    def clear(self):
        del self.cond_out[0]
        del self.cond_out
        self.cond_out = [None]

class Decoder(nn.Module):

    def __init__(self, zdim=256, out_channels=1, prepdepth=3, trunc=0.4, csize=None, gs_temp=0.1):
        super().__init__()

        self.ganz = 128
        self.ganclasses = 1000
        self.trunc = trunc

        self.zdim = zdim

        self.container = [None]

        biggan = BigGAN.from_pretrained('biggan-deep-512')
        biggan.eval()
        for param in biggan.parameters():
            param.requires_grad=False

        # prep network maps the VAE latent dims to the GAN latent dims
        prep = []
        prep.extend([nn.Linear(zdim + (csize if csize is not None else 0), zdim*2), nn.ReLU()])
        for _ in range(prepdepth - 2):
            prep.extend([nn.Linear(zdim * 2, zdim * 2), nn.ReLU()])

        prep.append(nn.Linear(zdim * 2, self.ganz+self.ganclasses))
        self.prep = nn.Sequential(*prep)

        # b1 = IBlock((2048, 4,   4), csize=csize, cond=self.container) # this one takes too much memory
        b2 = IBlock((1024, 16,  16), csize=csize, cond=self.container)
        b3 = IBlock((512,  64,  64),   csize=csize, cond=self.container)
        b4 = IBlock((128,  256, 256),  csize=csize, cond=self.container)

        self.iblocks = nn.ModuleList([b2, b3, b4])

        biggan.generator.layers.insert(14, b4)
        biggan.generator.layers.insert(8, b3)
        biggan.generator.layers.insert(4, b2)
        # biggan.generator.layers.insert(0, b1)

        self.biggan = NoParam(biggan)
        self.gs_temp = gs_temp

    def forward(self, z, cond=None):

        if cond is not None:
            z = torch.cat([z, cond], dim=-1)

        zprep = self.prep(z)

        zgan = zprep[:, :self.ganz]
        zcls = zprep[:, self.ganz:]

        zcls = gs_sample(zcls, temperature=self.gs_temp, dim=1)

        out = self.biggan(zgan, zcls, truncation=self.trunc)

        return (out + 1.0) / 2.0

    def clear(self):

        del self.container[0]
        del self.container
        self.container = [None]

        for block in self.iblocks:
            block.clear()

class Encoder(nn.Module):

    def __init__(self, insize=(3, 512, 512), zdim=256, prepdepth=3, csize=None):

        super().__init__()

        c, h, w = insize

        self.container = [None]

        mobile = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=True)
        mobile.eval()
        for param in mobile.parameters():
            param.requires_grad = False


        # 0 (32, 256, 256)
        b1 = IBlock((32, 256, 256), csize=csize, cond=self.container)

        # 3, torch.Size([9, 24, 128, 128])
        b2 = IBlock((24, 128, 128), csize=csize, cond=self.container)

        # 13, torch.Size([9, 96, 32, 32])
        b3 = IBlock((96, 32, 32), csize=csize, cond=self.container)

        # 17 torch.Size([9, 320, 16, 16])
        b4 = IBlock((320, 16, 16), csize=csize, cond=self.container)

        features = list(mobile.features)
        # insert from the back so the indices don't change
        features.insert(18, b4)
        features.insert(14, b3)
        features.insert( 4, b2)
        features.insert( 1, b1)

        mobile.features = nn.Sequential(*features)

        # register iblocks for learning
        self.iblocks = nn.ModuleList([b1, b2, b3, b4])

        mobile.classifier = nn.Linear(in_features=1280, out_features=zdim*2, bias=True)
        self.cls = mobile.classifier

        self.mobile = NoParam(mobile)

        # small encoder stack for now
        # c1, c2, c3 = 4, 8, 16 # channel sizes
        # self.cnns = nn.Sequential(
        #     nn.Conv2d(c,  c1, kernel_size=3, padding=1), nn.ReLU(),
        #     # nn.Conv2d(c1, c1, kernel_size=3, padding=1), nn.ReLU(),
        #     # nn.Conv2d(c1, c1, kernel_size=3, padding=1), nn.ReLU(),
        #     nn.MaxPool2d(4),
        #     nn.Conv2d(c1, c2, kernel_size=3, padding=1), nn.ReLU(),
        #     # nn.Conv2d(c2, c2, kernel_size=3, padding=1), nn.ReLU(),
        #     nn.MaxPool2d(4),
        #     nn.Conv2d(c2, c3, kernel_size=3, padding=1), nn.ReLU(),
        #     nn.MaxPool2d(4),
        # )
        # rh, rw = h // 4**3, w // 4**3
        # self.r = rh * rw * c3

    def forward(self, x, cond=None):

        b, c, h, w = x.size()
        if cond is not None:
            self.container[0] = cond

        return self.mobile(x)

    def clear(self):

        del self.container[0]
        del self.container
        self.container = [None]

        for block in self.iblocks:
            block.clear()

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

    df = pd.read_csv(here('./data/df_popular_podcasts.csv'))

    with open(here('./data/genre_IDs.txt')) as file:
        glist = eval(file.read())
        glist = {int(idstr) : name for (idstr, name) in glist}
        rlist = {name : id for (id, name) in glist.items()}

    gs = set()
    for genres in df['Genre IDs']:
        genres = eval(genres)
        for genre in genres:
            g = int(genre)
            gs.add(g)

    i2g = list(gs)
    g2i = {g:i for i, g in enumerate(i2g)}

    encoder, decoder = Encoder(insize=(C, H, W), zdim=arg.ls, csize=len(i2g)), \
                       Decoder(zdim=arg.ls, csize=len(i2g), gs_temp=arg.gs_temp)

    if arg.checkpoint is not None:
        encoder.load_state_dict(torch.load(arg.checkpoint + '.encoder', map_location=torch.device('cpu')))
        decoder.load_state_dict(torch.load(arg.checkpoint + '.decoder', map_location=torch.device('cpu')))


    opt = torch.optim.Adam(lr=arg.lr, params=list(encoder.parameters()) + list(decoder.parameters()))

    if torch.cuda.is_available():
        encoder.to('cuda')
        encoder.mobile.mod[0].to('cuda')

        decoder.to('cuda')
        decoder.biggan.mod[0].to('cuda')

    seen = 0
    for e in range(arg.epochs):

        # Sample images
        with torch.no_grad():
            # Generate 10 titles from the seed
            sgenres = torch.zeros(1, len(i2g))
            for genre in PD_GENRES:
                # print(glist[genre])
                sgenres[0, g2i[genre]] = 1.0
            sgenres = F.softmax(sgenres, dim=-1)
            if torch.cuda.is_available():
                sgenres = sgenres.to('cuda')

            b = 5
            z = torch.randn(b, arg.ls, device=util.d())
            output = decoder(z, cond=sgenres.expand(b, len(i2g)))

            output = output.to('cpu')

            for i in range(b):
                io.imsave(f'sample.{e}.{i}.png', output[i].permute(1, 2, 0).numpy(), check_contrast=False)

        # Check reconstructions
        with torch.no_grad():
            x = next(iter(trainloader))[0]
            dfbatch = df.iloc[:x.size(0)]
            g = tobatch(dfbatch, g2i, glist=glist)

            if torch.cuda.is_available():
                x, g = x.to('cuda'), g.to('cuda')

            z = encoder(x, cond=g)[:, :arg.ls]
            o = decoder(z, cond=g)

            o, x = o.to('cpu'), x.to('cpu')

            c = torch.cat([x, o], dim=-1)
            for i in range(c.size(0)):
                io.imsave(f'recon.{e}.{i}.png', c[i].permute(1, 2, 0).numpy(), check_contrast=False)


        print(f'epoch {e}', end='')
        fr = 0
        for i, (images, _) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
            
            if arg.limit is not None and i >= arg.limit:
                break

            opt.zero_grad()

            b, c, h, w = images.size()
            seen += b
            to = fr + b

            # batch of genres
            dfbatch = df.iloc[fr:to]
            genres = tobatch(dfbatch, g2i, glist=glist)

            if torch.cuda.is_available():
                images, genres = images.to('cuda'), genres.to('cuda')

            z = encoder(images, cond=genres)

            zmean, zsig = z[:, :arg.ls], z[:, arg.ls:]

            klterm = util.kl_loss(zmean, zsig)
            zsample = util.sample(zmean, zsig)

            out = decoder(zsample, cond=genres)

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
            
            fr += b

            del loss, images, out, genres
            encoder.clear()
            decoder.clear()

        torch.save(encoder.state_dict(), './checkpoint.encoder')
        torch.save(decoder.state_dict(), './checkpoint.decoder')

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

    parser.add_argument("--gs-temp",
                        dest="gs_temp",
                        help="Gumbel softmax temperature.",
                        default=0.1, type=float)

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