import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, models, transforms

# from transformers import BertModel, BertTokenizer, BertConfig
import numpy as np
import time
import sys
import os
import warnings
import argparse

from bench import bench, json_bench


class DCGANGenerator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(DCGANGenerator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class DCGANDiscriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(DCGANDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


nz, ngf, nc, ndf = 6, 9, 3, 10


def gen_dcgangen():
    return DCGANGenerator(nz, ngf, nc)


def gen_dcgandiscrim():
    return DCGANDiscriminator(nc, ndf)


class SimpleRLPolicy(nn.Module):
    def __init__(self):
        super(SimpleRLPolicy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


def gen_simplerlpolicy_input(N):
    if N != 1:
        raise Exception("Can only handle BS=1")
    return (torch.rand(1, 4),)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z)  # , mu, logvar


# enc = BertTokenizer.from_pretrained("bert-base-uncased")
# def create_bert_inputs(N):
#  if N != 1:
#    raise Exception("Can only handle batch size 1")
#  text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
#  tokenized_text = enc.tokenize(text)
#
#  masked_index = 8
#  tokenized_text[masked_index] = '[MASK]'
#  indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
#  segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
#
#  tokens_tensor = torch.tensor([indexed_tokens])
#  segments_tensors = torch.tensor([segments_ids])
#  return tokens_tensor, segments_tensors
#
# def create_bert():
#  config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#      num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, torchscript=True)
#  model = BertModel(config)
#  model.eval()
#  model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)
#  tokens_tensor, segments_tensors = create_bert_inputs(1)
#  def m(inp):
#    o1, o2 = model(inp[0], inp[1])
#    return o1
#  traced_model = torch.jit.trace(m, [tuple([tokens_tensor, segments_tensors])])
#  return traced_model

######################
# Benchmarking below #
######################

named_models = [
    # ("bert", create_bert),
    ("DCGANGenerator", gen_dcgangen),
    ("DCGANDiscriminator", gen_dcgandiscrim),
    ("SimpleRLPolicy", lambda: SimpleRLPolicy()),
    ("VariationalAutoEncoder", lambda: VAE()),
    ("resnet18", models.resnet18),
    ("resnext50_32x4d", models.resnext50_32x4d),
    ("mnasnet1_0", models.mnasnet1_0),
    ("squeezenet1_0", models.squeezenet1_0),
    ("densenet161", models.densenet161),
    # ("inception_v3",  models.inception_v3), # this is just too weird a model
    # ("googlenet", models.googlenet), # return value is weird
    ("shufflenet_v2_x1_0", models.shufflenet_v2_x1_0),
    ("mobilenet_v2", models.mobilenet_v2),
    ("wide_resnet50_2", models.wide_resnet50_2),
    ("alexnet", models.alexnet),
    ("vgg16", models.vgg16),
]

overwrite_lambda = {
    # If you *really* want inception, you'll need to mess with training logic
    # "inception_v3": lambda N: (torch.randn(N,3,299,299),)
    # "bert": lambda N: create_bert_inputs(N),
    "DCGANGenerator": lambda N: (torch.rand(N, nz, 1, 1),),
    "DCGANDiscriminator": lambda N: (
        DCGANGenerator(nz, ngf, nc)(torch.rand(N, nz, 1, 1)).detach(),
    ),
    "SimpleRLPolicy": gen_simplerlpolicy_input,
    "VariationalAutoEncoder": lambda N: (torch.rand(N, 1, 28, 28),),
}


class Step(torch.nn.Module):
    def __init__(self, model, cuda, is_train):
        super(Step, self).__init__()
        self.m = model()
        self.is_train = is_train
        if cuda:
            self.m = self.m.cuda()
        if self.is_train:
            self.m.train()
        else:
            self.m.eval()

    def forward(self, inp):
        if self.is_train:
            l = self.m(inp).sum()
            l.backward()
            return l
        else:
            return self.m(inp)


def run(args):
    if args.tensorexpr:
        os.environ["PYTORCH_TENSOREXPR"] = "1"
    else:
        os.environ["PYTORCH_TENSOREXPR"] = "0"

    run_models = []
    if args.models == []:
        run_models = named_models
    else:
        for n, m in named_models:
            if n in args.models:
                run_models.append((n, m))
    for name, model in run_models:
        for batch_size in args.batch_size:
            for is_train in [False] if args.eval else [False, True]:
                full_name = (
                    f"{name}_bs_{batch_size}_"
                    + f"{'train' if is_train else 'eval'}_"
                    + f"{'cuda' if args.cuda else 'cpu'}_"
                    + f"{'te' if args.tensorexpr else 'default'}"
                )
                if args.debug:
                    print(f"Running {full_name}")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        m = torch.jit.script(Step(model, args.cuda, is_train))
                    except:
                        if args.debug:
                            raise
                        continue
                    inp_f = lambda N: (torch.randn(N, 3, 224, 224),)
                    if name in overwrite_lambda:
                        if args.debug:
                            print(f"Input function for {name} overwritten")
                        inp_f = overwrite_lambda[name]
                    if args.cuda:
                        inp_f_tmp = inp_f
                        inp_f = lambda N: tuple([_.cuda() for _ in inp_f_tmp(N)])
                    try:
                        print(
                            json_bench(
                                full_name,
                                m,
                                lambda: inp_f(batch_size),
                                seconds=args.seconds,
                                runs=args.runs,
                            )
                        )
                    except:  # benchmark timeout, omit from results
                        if args.debug:
                            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark a set of models. Prints results in microseconds."
    )
    parser.add_argument("--debug", help="Throw errors", action="store_true")
    parser.add_argument("--cuda", help="Run models with CUDA", action="store_true")
    parser.add_argument(
        "--tensorexpr", help="Use tensorexpr fuser", action="store_true"
    )
    parser.add_argument("--eval", help="Only run forward pass", action="store_true")
    parser.add_argument(
        "--batch_size",
        metavar="SIZE",
        type=int,
        nargs="+",
        help="Batch size or sizes to run",
        default=[1, 32],
    )
    parser.add_argument(
        "--models",
        metavar="MODEL",
        type=str,
        nargs="+",
        help="Models to run",
        default=[],
    )
    parser.add_argument(
        "--seconds",
        help="Time allocated to run each benchmark in seconds",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--runs", help="Number of benchmark loop runs per model", type=int, default=5
    )
    run(parser.parse_args())
