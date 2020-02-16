import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, modeling_utils

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist

from util import d, here

import pandas as pd

from modules import TransformerBlock

from torch.utils.tensorboard import SummaryWriter

import random, tqdm, sys, math, gzip

from argparse import ArgumentParser
from util import *

import numpy as np

# E = 768
LOG2E = math.log2(math.e)
# CONTEXT = 512
# NHEADS = 12
NUCLEUS_P = 0.9

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

class IBlock(nn.Module):
    """
    Transformer block to be inserted into GPT2 stack. Allows conditionals
    to be registered prior to forward.
    """

    def __init__(self, *args, mult=0.0, **kwargs):

        super().__init__()
        self.block = TransformerBlock(*args, **kwargs)
        self.mult = nn.Parameter(torch.tensor([mult]))

    def set_conditional(self, cond):

        self.cond = cond

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):

        b, l, e = x.size()

        if self.cond is not None:
            assert self.cond.size() == (b, e), f'{self.cond.size()} versus {b, e}'
            xc = x + self.cond[:, None, :]
        else:
            xc = x

        r = self.mult * self.block(xc) +  x

        # print(r.size())
        return r, None, None

    def clear(self):
        del self.cond

class GPT2Wrapper(nn.Module):

    def __init__(self, iblocks=3, gptname='distilgpt2', dropout=0.0, csize=None):
        super().__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained(gptname)
        model = GPT2LMHeadModel.from_pretrained(gptname)
        model.eval()

        emb = model.config.n_embd
        self.ctx = model.config.n_ctx

        self.iblocks = nn.ModuleList([
            IBlock(emb=emb, heads=8, mask=True, ff_hidden_mult=4, dropout=dropout, wide=False) for _ in range(iblocks+1)
        ])

        nb = len(model.transformer.h)      # number of GPT2 blocks
        per = nb // iblocks    # how many blocks to skip between inserted blocks

        h = model.transformer.h # the main stack of transformer blocks
        for i in range(iblocks-1, -1, -1):
            print('inserting block at', i*per)
            block = self.iblocks[i]
            h.insert(i*per, block)
        h.insert(len(h), self.iblocks[-1])

        self.register_buffer(name='head_mask', tensor=torch.ones(len(h), model.config.n_head))
        # We need to create a special head mask because we've changes the number of blocks.

        for param in model.parameters():
            param.requires_grad = False

        self.model = NoParam(model)

        # Out own language model head
        self.headbias = nn.Parameter(torch.zeros(self.tokenizer.vocab_size)) # to token probabilities

        if csize is not None:
            self.to_cond = nn.Sequential(
                nn.Linear(csize, 2*csize), nn.ReLU(),
                nn.Linear(2*csize, emb)
            )

    def forward(self, x, cond=None):

        if cond is not None:
            cond = self.to_cond(cond)

        for block in self.iblocks:
            block.set_conditional(cond)

        x = self.model(x, head_mask=self.head_mask)[0]

        return x + self.headbias

    def clear(self):

        for block in self.iblocks:
            block.clear()


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

    modeling_utils.top_k_top_p_filtering(lnprobs[None, :], top_p=NUCLEUS_P)

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)

    return cd.sample()

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

def go(arg):

    if arg.seed < 0:
        seed = random.randint(0, 1000000)
        print('random seed: ', seed)
    else:
        torch.manual_seed(arg.seed)

    tbw = SummaryWriter(log_dir=arg.tb_dir) # Tensorboard logging

    arg.data = here('data/enwik8.gz') if arg.data is None else arg.data

    str_train, str_val, str_test = load_text(arg.data)
    str_train, str_test = (str_train + str_val, str_test) \
                            if arg.final else (str_train, str_val)

    # create the model
    model = GPT2Wrapper(iblocks=arg.iblocks)

    if torch.cuda.is_available():
        model.to('cuda')
        model.model.mod[0].to('cuda')

    # tokenize the data
    data_train, data_val, data_test = \
        torch.tensor(model.tokenizer.encode(str_train)), \
        torch.tensor(model.tokenizer.encode(str_val)), \
        torch.tensor(model.tokenizer.encode(str_test))

    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())
    # sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))
    # -- linear learning rate warmup

    # training loop
    # -- note: we don't loop over the data, instead we sample a batch of random subsequences each time.
    for i in tqdm.trange(arg.num_batches):

        opt.zero_grad()

        # sample a batch of random subsequences
        starts = torch.randint(size=(arg.batch_size, ), low=0, high=data_train.size(0) - model.ctx - 1)
        seqs_source = [data_train[start  :start+model.ctx  ] for start in starts]
        seqs_target = [data_train[start+1:start+model.ctx+1] for start in starts]

        source = torch.cat([s[None, :] for s in seqs_source ], dim=0).to(torch.long)
        target = torch.cat([s[None, :] for s in seqs_target ], dim=0).to(torch.long)
        # -- target is the same sequence as source, except one character ahead

        if torch.cuda.is_available():
            source, target = source.to('cuda'), target.to('cuda')

        output = model(source)

        loss = F.cross_entropy(output.transpose(2, 1), target, reduction='mean')
        tbw.add_scalar('podcasts/train-loss', float(loss.item()) * LOG2E, i * arg.batch_size)

        loss.backward()

        # clip gradients
        # - If the total gradient vector has a length > 1, we clip it back down to 1.
        if arg.gradient_clipping > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

        opt.step()
        # sch.step()

        model.clear()

        # - validate every {arg.test_every} steps. First we compute the
        #   compression on the validation (or a subset)
        #   then we generate some random text to monitor progress
        if i != 0 and (i % arg.print_every == 0 or i == arg.num_batches - 1):

            with torch.no_grad():

                # generate and print some random text
                seedfr = random.randint(0, data_test.size(0) - arg.print_seed_size)
                input = data_test[seedfr:seedfr + arg.print_seed_size].to(torch.long)

                if torch.cuda.is_available():
                    input = input.cuda()

                # print the seed
                strinput = model.tokenizer.decode(input)
                print(f'[{strinput}]', end='')

                outseq = []
                for _ in range(arg.print_size):
                    output = model(input[None, :])
                    c = sample(output[0, -1, :], arg.sampling_temp)
                    outseq.append(c[None])

                    input = torch.cat([input[1:], c[None]], dim=0)

                outseq = torch.cat(outseq, dim=0)
                outseq = model.tokenizer.decode(outseq)

                print(outseq)

        # val
        if i != 0 and (i % arg.test_every == 0 or i == arg.num_batches - 1):

            with torch.no_grad():

                upto = data_test.size(0) if i == arg.num_batches - 1 else arg.test_subset
                data_sub = data_test[:upto]

                bits, tot = 0.0, 0
                batch = [] # buffer, every time it fills up, we run it through the model

                for current in range(data_sub.size(0)):

                    fr = max(0, current - model.ctx)
                    to = current + 1

                    context = data_sub[fr:to].to(torch.long)
                    if context.size(0) < model.ctx + 1:
                        pad = torch.zeros(size=(model.ctx + 1 - context.size(0),), dtype=torch.long)
                        context = torch.cat([pad, context], dim=0)

                        assert context.size(0) == model.ctx + 1

                    if torch.cuda.is_available():
                        context = context.cuda()

                    batch.append(context[None, :])

                    if len(batch) == arg.test_batchsize or current == data_sub.size(0) - 1:

                        # batch is full, run it through the model
                        b = len(batch)

                        all = torch.cat(batch, dim=0)
                        source = all[:, :-1] # input
                        target = all[:, -1]  # target values

                        output = model(source)

                        lnprobs = output[torch.arange(b, device=d()), -1, target]
                        log2probs = lnprobs * LOG2E # convert from nats to bits

                        bits += - log2probs.sum()
                        batch = [] # empty buffer

                bits_per_byte = bits / data_sub.size(0)

                # print validation performance. 0.92 bit per byte is (currently) state of the art.
                print(f'epoch{i}: {bits_per_byte:.4} bits per byte')
                tbw.add_scalar(f'podcasts/eval-loss', bits_per_byte, i * arg.batch_size)


def tobatch(df, tokenizer, g2i, normalize_genres=True):

    # batch of tokenized text
    strings = []

    for row in range(len(df)):
        name = df.iloc[row]['Name']
        desc = df.iloc[row]['Description']

        desc = desc.replace('\n', '')
        strings.append(f'description: {desc} \n title: {name}')

    ids = []
    for string in strings:
        ids.append(tokenizer.encode(string))

    # pad to max
    mx = max([len(id) for id in ids])
    ids = [id + ( [0] * (mx - len(id)) ) for id in ids]
    # I think zero should work as a pad token

    ids = [torch.tensor(id)[None, :] for id in ids]
    ids = torch.cat(ids, dim=0)

    # batch of n-hot vectors for genres
    ng = len(g2i)
    genres = torch.zeros(ids.size(0), ng)

    for row in range(len(df)):
        dfgs = [int(g) for g in eval(df.iloc[row]['Genre IDs'])]
        for intg in dfgs:
            genres[row, g2i[intg]] = 0

    if normalize_genres:
        genres = genres / genres.sum(dim=1, keepdim=True)

    return ids, genres

def go_pods(arg):

    if arg.seed < 0:
        seed = random.randint(0, 1000000)
        print('random seed: ', seed)
    else:
        torch.manual_seed(arg.seed)

    tbw = SummaryWriter(log_dir=arg.tb_dir) # Tensorboard logging

    df = pd.read_csv('./data/df_popular_podcasts.csv')

    gs = set()
    for genres in df['Genre IDs']:
        genres = eval(genres)
        for genre in genres:
            g = int(genre)
            gs.add(g)

    i2g = list(gs)
    g2i = {g:i for i, g in enumerate(i2g)}

    train, val, test = df.iloc[:8000], df.iloc[8000:9000], df.iloc[9000:]

    # create the model
    model = GPT2Wrapper(iblocks=arg.iblocks, csize=len(i2g))

    if torch.cuda.is_available():
        model.to('cuda')
        model.model.mod[0].to('cuda')

    tok = model.tokenizer
    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())
    # sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))
    # -- linear learning rate warmup

    # training loop
    # -- note: we don't loop over the data, instead we sample a batch of random subsequences each time.
    seen = 0
    for e in range(arg.epochs):
        for fr in tqdm.trange(0, len(train), arg.batch_size):

            to = min(len(train), fr+arg.batch_size)

            dfbatch = df.iloc[fr:to]
            texts, genres = tobatch(dfbatch, tok, g2i)
            b = texts.size(0)
            source = torch.cat([torch.empty(b, 1, dtype=torch.long).fill_(0), texts], dim=1)
            target = torch.cat([texts, torch.empty(b, 1, dtype=torch.long).fill_(0)], dim=1)

            seen += b

            opt.zero_grad()

            if torch.cuda.is_available():
                source, target, genres = source.to('cuda'), target.to('cuda'), genres.to('cuda')

            output = model(source, cond=genres)

            loss = F.cross_entropy(output.transpose(2, 1), target, reduction='mean')
            tbw.add_scalar('podcasts/train-loss', float(loss.item()) * LOG2E, seen)

            loss.backward()

            # clip gradients
            # - If the total gradient vector has a length > 1, we clip it back down to 1.
            if arg.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

            opt.step()
            # sch.step()

            model.clear()

        # - validate every {arg.test_every} steps. First we compute the
        #   compression on the validation (or a subset)
        #   then we generate some random text to monitor progress
        if e != 0 and (e % arg.print_every == 0 or e == arg.num_batches - 1):

            with torch.no_grad():

                # generate and print some random text
                input = tok.encode("description: ")

                if torch.cuda.is_available():
                    input = input.to('cuda')

                # print the seed
                strinput = model.tokenizer.decode(input)
                print(f'[{strinput}]', end='')

                outseq = []
                for _ in range(arg.print_size):
                    output = model(input[None, :])
                    c = sample(output[0, -1, :], arg.sampling_temp)
                    outseq.append(c[None])

                    input = torch.cat([input[1:], c[None]], dim=0)

                outseq = torch.cat(outseq, dim=0)
                outseq = model.tokenizer.decode(outseq)

                print(outseq)

            # val
            # if i != 0 and (i % arg.test_every == 0 or i == arg.num_batches - 1):
            #
            #     with torch.no_grad():
            #
            #         upto = data_test.size(0) if i == arg.num_batches - 1 else arg.test_subset
            #         data_sub = data_test[:upto]
            #
            #         bits, tot = 0.0, 0
            #         batch = [] # buffer, every time it fills up, we run it through the model
            #
            #         for current in range(data_sub.size(0)):
            #
            #             fr = max(0, current - model.ctx)
            #             to = current + 1
            #
            #             context = data_sub[fr:to].to(torch.long)
            #             if context.size(0) < model.ctx + 1:
            #                 pad = torch.zeros(size=(model.ctx + 1 - context.size(0),), dtype=torch.long)
            #                 context = torch.cat([pad, context], dim=0)
            #
            #                 assert context.size(0) == model.ctx + 1
            #
            #             if torch.cuda.is_available():
            #                 context = context.cuda()
            #
            #             batch.append(context[None, :])
            #
            #             if len(batch) == arg.test_batchsize or current == data_sub.size(0) - 1:
            #
            #                 # batch is full, run it through the model
            #                 b = len(batch)
            #
            #                 all = torch.cat(batch, dim=0)
            #                 source = all[:, :-1] # input
            #                 target = all[:, -1]  # target values
            #
            #                 output = model(source)
            #
            #                 lnprobs = output[torch.arange(b, device=d()), -1, target]
            #                 log2probs = lnprobs * LOG2E # convert from nats to bits
            #
            #                 bits += - log2probs.sum()
            #                 batch = [] # empty buffer
            #
            #         bits_per_byte = bits / data_sub.size(0)
            #
            #         # print validation performance. 0.92 bit per byte is (currently) state of the art.
            #         print(f'epoch{i}: {bits_per_byte:.4} bits per byte')
            #         tbw.add_scalar(f'podcasts/eval-loss', bits_per_byte, i * arg.batch_size)



if __name__ == "__main__":

    # Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("--pods",
                        dest="pods",
                        help="Run the podcast experiment.",
                        action="store_true")

    parser.add_argument("-l", "--lr",
                        dest="lr",
                        help="Learning rate.",
                        default=0.0001, type=float)

    parser.add_argument("-b", "--batch",
                        dest="batch_size",
                        help="Batch size.",
                        default=2, type=int)

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

    parser.add_argument("-D", "--data", dest="data",
                        help="Data file. Will be read as a string.",
                        default=None)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Tensorboard logging directory",
                        default='./runs')

    parser.add_argument("--print-every",
                        dest="print_every",
                        help="How many batches between printing a sample.",
                        default=1000, type=int)

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

    parser.add_argument("--sampling-temp",
                        dest="sampling_temp",
                        help="Sampling temperature.",
                        default=1.0, type=float)

    options = parser.parse_args()
    print('OPTIONS ', options)

    if options.pods:
        go_pods(options)
    else:
        go(options)