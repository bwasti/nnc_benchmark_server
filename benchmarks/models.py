import torch
import torchvision
from torchvision import datasets, models, transforms
import imageio
import numpy as np
import time
import sys
import os
import warnings
import argparse

from bench import bench, json_bench

named_models = [
    ("resnet18", models.resnet18),
    ("alexnet", models.alexnet),
    ("vgg16", models.vgg16),
    ("squeezenet1_0", models.squeezenet1_0),
    ("densenet161", models.densenet161),
    # ("inception_v3",  models.inception_v3), # this is just too weird a model
    ("googlenet", models.googlenet),
    ("shufflenet_v2_x1_0", models.shufflenet_v2_x1_0),
    ("mobilenet_v2", models.mobilenet_v2),
    ("resnext50_32x4d", models.resnext50_32x4d),
    ("wide_resnet50_2", models.wide_resnet50_2),
    ("mnasnet1_0", models.mnasnet1_0),
]

# If you *really* want inception, you'll need to mess with training logic
# overwrite_lambda = {
#  "inception_v3": lambda N: (torch.randn(N,3,299,299),)
# }

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
        for batch_size in args.batch_sizes:
            for is_train in [False, True]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = model()
                    if args.cuda:
                        m = m.cuda()
                    if is_train:
                        m.train()

                        def step(inp):
                            l = m(inp).sum()
                            l.backward()
                            return l

                    else:
                        m.eval()

                        def step(inp):
                            return m(inp)

                    m = torch.jit.script(step)
                    inp_f = lambda N: (torch.randn(N, 3, 224, 224),)
                    if args.cuda:
                        inp_f_tmp = inp_f
                        inp_f = lambda N: tuple([_.cuda() for _ in inp_f_tmp(N)])
                    print(
                        json_bench(
                            f"{name}_bs_{batch_size}_{'train' if is_train else 'eval'}",
                            m,
                            lambda: inp_f(batch_size),
                            seconds=args.seconds,
                            runs=args.runs,
                        )
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark a set of models. Prints results in microseconds."
    )
    parser.add_argument("--cuda", help="Run models with CUDA", action="store_true")
    parser.add_argument(
        "--tensorexpr", help="Use tensorexpr fuser", action="store_true"
    )
    parser.add_argument("--batch_size", metavar="SIZE", type=int, nargs='+',
                        help="Batch size or sizes to run", default=[1,32])
    parser.add_argument("--models", metavar="MODEL", type=str, nargs='+',
                        help="Models to run", default=[])
    parser.add_argument(
        "--seconds",
        help="Time allocated to run each benchmark in seconds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--runs", help="Number of benchmark loop runs per model", type=int, default=5
    )
    run(parser.parse_args())
