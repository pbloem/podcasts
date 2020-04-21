import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, modeling_utils

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist

import gc

from util import d, here, contains_inf

import pandas as pd

from modules import TransformerBlock

from torch.utils.tensorboard import SummaryWriter

import random, tqdm, sys, math, gzip, random, json, gzip

from argparse import ArgumentParser
from util import *

import gpt2

import numpy as np

# E = 768
LOG2E = math.log2(math.e)
# CONTEXT = 512
# NHEADS = 12
NUCLEUS_P = 0.9

VAL = 1000
PAD = '.pad'
START, END = '.start', '.end'

class NoParam(nn.Module):
    """
    Wraps a module, stopping parameters from being registered
    """

    def __init__(self, mod):

        super().__init__()
        self.mod = [mod]

    def cuda(self):
        self.mod[0].cuda()

    def forward(self, x, *args, **kwargs):

        return self.mod[0](x, *args, **kwargs)

class GPT2Wrapper(nn.Module):

    def __init__(self, iblocks=3, gptname='distilgpt2', dropout=0.0, csize=None):
        super().__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained(gptname, bos_token=START, pad_token=PAD, eos_token=END)
        model = gpt2.GPT2Model.from_pretrained(gptname)
        model.eval()

        model.set_iblocks(iblocks, csize, heads=8, mask=True, ff_hidden_mult=4, wide=False)

        self.emb = model.config.n_embd
        self.ctx = model.config.n_ctx

        self.model = NoParam(model)
        self.iblocks = nn.ModuleList(list(model.iblocks()))
        # -- store the iblocks separately so they get registered as parameters

        # language model head
        self.lm_head = nn.Linear(model.config.n_embd, model.config.vocab_size, bias=True)

    def forward(self, x, cond=None):

        x = self.model(x, cond=cond)[0]

        return self.lm_head(x)

def sample(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome log-probabilities
    :param temperature: Sampling temperature. 1.0 follows the given distribution,
        0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """

    if temperature == 0.0:
        return lnprobs.argmax()

    if temperature != 1.0:
            lnprobs = lnprobs / temperature
    # Top-p/top-k filtering
    next_token_logits = modeling_utils.top_k_top_p_filtering(lnprobs[None, :], top_p=NUCLEUS_P)
    # Sample

    probs = F.softmax(next_token_logits, dim=-1)
    if contains_inf(probs) or contains_nan(probs):
        raise Exception(f'probs {probs}, logits {next_token_logits}')

    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

    return next_token

    # modeling_utils.top_k_top_p_filtering(lnprobs[None, :], top_p=NUCLEUS_P)
    #
    # p = F.softmax(lnprobs / temperature, dim=0)
    # cd = dist.Categorical(p)
    #
    # return cd.sample()

def load_text(path):
    """
    Load text data.

    :param path:
    :param n_train:
    :param n_valid:
    :param n_test:
    :return:
    """
    with gzip.open(path) if path.endswith('.gz') else open(path) as file:
        s = file.read()
        tr, vl = int(len(s) * 0.6), int(len(s) * 0.8)

        return str(s[:tr]), str(s[tr:vl]), str(s[vl:])

        # X = np.fromstring(file.read(n_train + n_valid + n_test), dtype=np.uint8)
        # trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
        # return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)

def pad_sequences(sequences, token=0):
    """
    Pads a sequences of sequences so that they're all the same length. In-place.
    :param sequences:
    :param token:
    :return:
    """
    mx = max([len(s) for s in sequences])

    for seq in sequences:
        while(len(seq) < mx):
            seq.append(token)

def onehot(categories, max_cat, normalize=False):

    # batch of n-hot vectors for genres

    b = len(categories)
    result = torch.zeros(b, max_cat+1)

    for r in range(b):
        for c in categories[r]:
            result[r, c] = 1

    if normalize:
        result = result / result.sum(dim=1, keepdim=True)

    return result


def go(arg):

    if arg.seed < 0:
        seed = random.randint(0, 1000000)
        print('random seed: ', seed)
    else:
        torch.manual_seed(arg.seed)

    tbw = SummaryWriter(log_dir=arg.tb_dir) # Tensorboard logging

    # load data
    if arg.task == 'coco':

        with open(arg.data + os.sep + 'i2cat_train2017.json') as file:
            i2cat = json.load(file)
        with open(arg.data + os.sep + 'i2cap_train2017.json') as file:
            i2cap = json.load(file)
        with open(arg.data + os.sep + 'labels.json') as file:
            l2i = json.load(file)
            i2l = {v:k for k, v in l2i.items()}

        if arg.final:
            raise Exception('Not implemented yet.')
        else:
            images = list(i2cat.keys())
            images_train = images[:-VAL]
            images_valid = images[-VAL:]

            cats_train, cats_valid = [], []
            caps_train, caps_valid = [], []

            # transform to caption -> categories
            for image in images_train:
                caps = i2cap[image]
                cats = i2cat[image]

                caps_train.extend(caps)
                cats_train.extend([cats] * len(caps))

            for image in images_valid:
                caps = i2cap[image]
                cats = i2cat[image]

                caps_valid.extend(caps)
                cats_valid.extend([cats] * len(caps))

            # sort by length of caption
            pairs = zip(caps_train, cats_train)
            caps_train, cats_train = zip(*sorted(pairs, key=lambda x : len(x[0])))

            pairs = zip(caps_valid, cats_valid)
            caps_valid, cats_valid = zip(*sorted(pairs, key=lambda x : len(x[0])))

            ntrain, nvalid = len(images_train), len(images_valid)
            max_cat = 90

    if arg.task == 'imdb':

        l2i = {'pos':1, 'neg':0}
        i2l = {v: k for k, v in l2i.items()}

        with gzip.open(f'{here()}{os.sep}data{os.sep}imdb{os.sep}imdb.train.json.gz', 'r') as file:
            train = json.load(file)

        with gzip.open(f'{here()}{os.sep}data{os.sep}imdb{os.sep}imdb.test.json.gz', 'r') as file:
            test = json.load(file)

        caps_train = train['pos'] + train['neg']
        cats_train = [1] * len(train['pos']) + [0] * len(train['neg'])

        pairs = zip(caps_train, cats_train)
        caps_train, cats_train = zip(*sorted(pairs, key=lambda x: len(x[0])))

        ntrain, _ = len(caps_train), None
        max_cat = 1

        # TODO split train into train/val, load test properly

    else:
        raise Exception(f'Task {arg.task} not recognized.')

    # create the model
    model = GPT2Wrapper(iblocks=arg.iblocks, gptname=arg.gpt_name, csize=max_cat+1)

    if torch.cuda.is_available():
        model.to('cuda')
        model.model.mod[0].to('cuda')

    model.tokenizer.padding_side = 'right'

    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())

    seen = 0
    for e in range(arg.epochs):

        if e % arg.print_every == 0:

            # Generate some random sequences
            for i in range(3):
                # generate a random category
                random_cat = random.choice(list(l2i.keys()))

                cats = torch.zeros(1, max_cat + 1)
                cats[0, l2i[random_cat]] = 1.0

                # generate and print some random text
                seed = START
                input = torch.tensor(model.tokenizer.encode(seed))

                if torch.cuda.is_available():
                    input, cats = input.to('cuda'), cats.to('cuda')

                outseq = []
                for _ in range(arg.print_size):
                    output = model(input[None, :], cond=cats)
                    c = sample(output[0, -1, :], arg.sampling_temp)
                    outseq.append(c)

                    if c == model.tokenizer.bos_token_id:
                        break

                    input = torch.cat([input, c], dim=0)

                outseq = torch.cat(outseq, dim=0)
                outseq = model.tokenizer.decode(outseq)

                with open(f'random.e{e:03}i{i:02}.txt', 'w') as file:

                    print('chosen category', random_cat, file=file)
                    print('---------------------------------------------', file=file)
                    print(seed, file=file)
                    print(outseq, flush=True, file=file)

        for fr in tqdm.trange(0, ntrain, arg.batch):

            to = min(fr + arg.batch, ntrain)

            bcats = cats_train[fr:to]
            bcaps = caps_train[fr:to]

            # translate captions to tensors
            res = model.tokenizer.batch_encode_plus(bcaps, pad_to_max_length=True, max_length=max([len(s) for s in bcaps]))
            captions = res['input_ids']
            pad_sequences(captions, token=model.tokenizer.pad_token_id)
            captions = torch.tensor(captions)

            b, t = captions.size()
            seen += b

            bos, pad = torch.tensor([[model.tokenizer.bos_token_id]]), torch.tensor([[model.tokenizer.bos_token_id]])
            source = torch.cat([bos.expand(b, 1), captions], dim=1)
            target = torch.cat([captions, pad.expand(b, 1)], dim=1)
            # -- target is the same sequence as source, except one character ahead

            if arg.dropout > 0.0:
                source = source * torch.empty(source.size(1)).bernoulli_(arg.dropout).to(torch.long)[None, :] # token dropout

            # translate categories to n-hots
            cats = onehot(bcats, max_cat=max_cat)

            if torch.cuda.is_available():
                source, target, cats = source.to('cuda'), target.to('cuda'), cats.to('cuda')

            output = model(source, cond=cats)

            loss = F.cross_entropy(output.transpose(2, 1), target, reduction='mean')
            tbw.add_scalar('podcasts/train-loss', float(loss.item()) * LOG2E, seen)

            opt.zero_grad()
            loss.backward()

            # clip gradients
            # - If the total gradient vector has a length > 1, we clip it back down to 1.
            if arg.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

            opt.step()
            # sch.step()

def tobatch(df, tokenizer, g2i, normalize_genres=True, limit=2000, glist=None):

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

        # batch of tokenized text
        strings = []

        for row in range(len(df)):
            name = df.iloc[row]['Name']
            desc = str(df.iloc[row]['Description'])[:limit]

            desc = desc.replace('\n', '')

            if glist is not None:
                dfgs = [int(g) for g in eval(df.iloc[row]['Genre IDs'])]
                dfgs = ', '.join([glist[ig] for ig in dfgs])

                strings.append(f'description: {desc} \ngenres: {dfgs} \ntitle: {name} ||\n')
            else:
                strings.append(f'description: {desc} \ntitle: {name} ||\n')

        ids = []
        for string in strings:
            ids.append(tokenizer.encode(string))

        # pad to max
        mx = min(max([len(id) for id in ids]), 1000)
        ids = [id[:mx] + ( [0] * (mx - len(id)) ) for id in ids]

        # I think zero should work as a pad token
        ids = [torch.tensor(id)[None, :] for id in ids]
        ids = torch.cat(ids, dim=0)

    return ids, genres

if __name__ == "__main__":

    # Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-l", "--lr",
                        dest="lr",
                        help="Learning rate.",
                        default=0.0001, type=float)

    parser.add_argument("-b", "--batch",
                        dest="batch",
                        help="Batch size.",
                        default=16, type=int)

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs over the generated data.",
                        default=30, type=int)

    parser.add_argument("--num_batches",
                        dest="num_batches",
                        help="Number of batches to train for.",
                        default=1_000_000, type=int)

    parser.add_argument("-i", "--iblocks",
                        dest="iblocks",
                        help="Number of trainable transformer blocks inserted.",
                        default=3, type=int)

    parser.add_argument("-t", "--task", dest="task",
                        help="Which task.",
                        default='coco')

    parser.add_argument("-D", "--data", dest="data",
                        help="Data file. Will be read as a string.",
                        default=None)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("-T", "--tb-dir", dest="tb_dir",
                        help="Tensorboard logging directory",
                        default='./runs')

    parser.add_argument("--print-every",
                        dest="print_every",
                        help="How many batches between printing a sample.",
                        default=5, type=int)

    parser.add_argument("--print-seed-size",
                        dest="print_seed_size",
                        help="The size of the sample seed.",
                        default=100, type=int)

    parser.add_argument("--print-size",
                        dest="print_size",
                        help="How many tokens to print in the sample.",
                        default=600, type=int)

    parser.add_argument("--test-every",
                        dest="test_every",
                        help="How many batches between tests.",
                        default=1500, type=int)

    parser.add_argument("--test-subset",
                        dest="test_subset",
                        help="A subset for the validation tests.",
                        default=10000, type=int)

    parser.add_argument("--test-batchsize",
                        dest="test_batchsize",
                        help="Batch size for computing the validation loss. This can be a bit bigger than the training batch size.",
                        default=4, type=int)

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)

    parser.add_argument("--dropout",
                        dest="dropout",
                        help="Word dropout on the input.",
                        default=0.0, type=float)

    parser.add_argument("--sampling-temp",
                        dest="sampling_temp",
                        help="Sampling temperature.",
                        default=1.0, type=float)

    parser.add_argument("--desc-clip",
                        dest="desc_clip",
                        help="What number of characters to clip the description at.",
                        default=2000, type=int)

    parser.add_argument("--checkpoint",
                        dest="checkpoint",
                        help="Load a model checkpoint to start from.",
                        default=None, type=str)

    parser.add_argument("--gpt-name",
                        dest="gpt_name",
                        help="GPT2 model name.",
                        default='distilgpt2', type=str)

    options = parser.parse_args()
    print('OPTIONS ', options)

    go(options)