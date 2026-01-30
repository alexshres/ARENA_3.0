# %%

import importlib
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from IPython.core.display import HTML
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor, optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part3_optimization"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

import part3_optimization.tests as tests
from part2_cnns.solutions import Linear, ResNet34, get_resnet_for_feature_extraction
from part3_optimization.utils import plot_fn, plot_fn_with_points
from plotly_utils import bar, imshow, line

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

WORLD_SIZE = min(t.cuda.device_count(), 3)
print(f"{WORLD_SIZE=}")


# %%
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"


def send_receive(rank, world_size):
    dist.init_process_group(backend="gloo",
                            rank=rank,
                            world_size=world_size)
    

    if rank == 0:
        # send tensor to rank 1
        sending_tensor = t.zeros(1)
        print(f"{rank=}, sending {sending_tensor=}")
        dist.send(tensor=sending_tensor, dst=1)

    elif rank == 1:
        # receive tensor from rank 0
        received_tensor = t.ones(1)
        print(f"{rank=}, creating {received_tensor=}")

        # overwrites the received_tensor's data with our 'sending_tensor'
        dist.recv(
            received_tensor, src=0
        )

        print(f"{rank=}, received {received_tensor=}")

    dist.destroy_process_group()

if MAIN:
    world_size = 2
    mp.spawn(
        send_receive,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

# %%
def broadcast(tensor: Tensor, rank: int, world_size: int, src: int=0):
    """Broadcasts averaged gradients from rank=src from one process (rank) 
       to all other processes (rank)"""

    # initialize pipees for processes to communicate
    # actually DONT NEED TO INITIALIZE because it's happening in the test_broadcast()

    # dist.init_process_group(backend="nccl",
    #                         rank=rank,
    #                         world_size=world_size)

    if rank==src:
        # sending averaged gradients 
        for r in range(world_size):
            if r != src:
                print(f"Source rank {rank=} sending tensor to destination rank {r=}")
                dist.send(
                    tensor=tensor,
                    dst=r
                )
    else:
        receive_tensor = t.zeros_like(tensor, dtype=tensor.dtype)
        dist.recv( receive_tensor, src=src)
        tensor.copy_(receive_tensor)
        print(f"Rank {rank=} received tensor from rank {src=}")

    # dist.destroy_process_group()

if MAIN:
    tests.test_broadcast(broadcast, WORLD_SIZE)
# %%

def reduce(tensor, rank, world_size, dst=0,
           op: Literal["sum", "mean"]="sum"):
    """
    Reduces gradient to rank `dst` so this process contains the sum
    or mean of all tensors across processes 
    """
    if rank != dst:
        dist.send(
            tensor=tensor,
            dst=dst
        )
    else:
        for r in range(world_size):
            if r != dst:
                receive_tensor = t.zeros_like(tensor, dtype=tensor.dtype)
                # we want to add current tensor's value for dst to the received
                # interesting can't do 
                # * This is actually because dist.recv(...) is an in-place operation
                # * that return None
                # >>> tensor += dist.recv(...)
                dist.recv(
                    receive_tensor,
                    src=r
                )
                tensor += receive_tensor

        if op == "mean":
            tensor /= world_size

def all_reduce(tensor, rank, world_size, op: Literal["sum", "mean"] = "sum"):
    """
    All_Reduce the tensor across all ranks, using 0 as the initial gathering rank.
    Does the same are `reduce` but broadcasts the result back
    """

    # can i just use reduce and broadcast?
    # reduce all tensors and set dist to rank 0
    reduce(tensor, rank, world_size, dst=0, op=op)
    # broadcast it to all tensors from src=0 (where it was gathered)
    broadcast(tensor, rank, world_size, src=0)

if MAIN:
    tests.test_reduce(reduce, WORLD_SIZE)
    tests.test_all_reduce(all_reduce, WORLD_SIZE)

        



# %%
