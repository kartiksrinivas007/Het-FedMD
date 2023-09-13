import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mpi4py import MPI
from torchvision import datasets, transforms

from aijack.collaborative.fedmd import FedMDAPI, FedMDClient, FedMDServer
from aijack.utils import NumpyDataset

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


training_batch_size = 64
test_batch_size = 64
num_rounds = 5
lr = 0.001
seed = 0
client_size = 2
criterion = F.nll_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fix_seed(seed)
def prepare_dataloader(num_clients, myid, train=True, path=""):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    if train:
        dataset = datasets.MNIST(path, train=True, download=True, transform=transform)
        idxs = list(range(len(dataset.data)))
        random.shuffle(idxs)
        idx = np.array_split(idxs, num_clients, 0)[myid - 1]
        dataset.data = dataset.data[idx]
        dataset.targets = dataset.targets[idx]
        train_loader = torch.utils.data.DataLoader(
            NumpyDataset(
                x=dataset.data.numpy(),
                y=dataset.targets.numpy(),
                transform=transform,
                return_idx=True,
            ),
            batch_size=training_batch_size,
        )
        return train_loader
    else:
        dataset = datasets.MNIST(path, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            NumpyDataset(
                x=dataset.data.numpy(),
                y=dataset.targets.numpy(),
                transform=transform,
                return_idx=True,
            ),
            batch_size=test_batch_size,
        )
        return test_loader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.ln = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = self.ln(x.reshape(-1, 28 * 28))
        output = F.log_softmax(x, dim=1)
        return output
dataloaders = [prepare_dataloader(client_size + 1, c) for c in range(client_size + 1)]
public_dataloader = dataloaders[0]
local_dataloaders = dataloaders[1:]
test_dataloader = prepare_dataloader(client_size, -1, train=False)
clients = [
    FedMDClient(Net().to(device), public_dataloader, output_dim=10, user_id=c)
    for c in range(client_size)
] # voila change the models here sir!
local_optimizers = [optim.SGD(client.parameters(), lr=lr) for client in clients]

server = FedMDServer(clients, Net().to(device))

api = FedMDAPI(
    server,
    clients,
    public_dataloader,
    local_dataloaders,
    F.nll_loss,
    local_optimizers,
    test_dataloader,
    num_communication=2,
)
log = api.run()