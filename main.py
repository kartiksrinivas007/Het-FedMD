import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mpi4py import MPI
from torchvision import datasets, transforms
from aijack.collaborative.fedmd import FedMDAPI, FedMDClient, FedMDServer
from aijack.utils import NumpyDataset, accuracy_torch_dataloader
import argparse
from tqdm import tqdm 
import csv

from utils.data_loaders import *
from utils.utils import *


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# Accuracies = []
# training_batch_size = 64
# test_batch_size = 64
# num_rounds = 5
# lr = 0.001
# seed = 0
# client_size = 2
# criterion = F.nll_loss

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fix_seed(seed)
# def prepare_dataloader(num_clients, myid, train=True, path=""):
#     transform = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
#     )
#     if train:
#         dataset = datasets.MNIST(path, train=True, download=True, transform=transform)
#         idxs = list(range(len(dataset.data)))
#         random.shuffle(idxs)
#         idx = np.array_split(idxs, num_clients, 0)[myid - 1]
#         dataset.data = dataset.data[idx]
#         dataset.targets = dataset.targets[idx]
#         train_loader = torch.utils.data.DataLoader(
#             NumpyDataset(
#                 x=dataset.data.numpy(),
#                 y=dataset.targets.numpy(),
#                 transform=transform,
#                 return_idx=True,
#             ),
#             batch_size=training_batch_size,
#         )
#         return train_loader
#     else:
#         dataset = datasets.MNIST(path, train=False, download=True, transform=transform)
#         test_loader = torch.utils.data.DataLoader(
#             NumpyDataset(
#                 x=dataset.data.numpy(),
#                 y=dataset.targets.numpy(),
#                 transform=transform,
#                 return_idx=True,
#             ),
#             batch_size=test_batch_size,
#         )
#         return test_loader

# args = {
#     "batch_size": training_batch_size,
#     "test_batch_size": test_batch_size,
#     "epochs": num_rounds,
#     "lr": lr,
#     "momentum": 0.5,
#     "no_cuda": False,
# }
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.ln = nn.Linear(28 * 28, 10)

#     def forward(self, x):
#         x = self.ln(x.reshape(-1, 28 * 28))
#         output = F.log_softmax(x, dim=1)
#         return output
# dataloaders = [prepare_dataloader(client_size + 1, c) for c in range(client_size + 1)]
# public_dataloader = dataloaders[0]
# local_dataloaders = dataloaders[1:]
# test_dataloader = prepare_dataloader(client_size, -1, train=False)
# clients = [
#     FedMDClient(Net().to(device), public_dataloader, output_dim=10, user_id=c)
#     for c in range(client_size)
# ] # voila change the models here sir!
# local_optimizers = [optim.SGD(client.parameters(), lr=lr) for client in clients]

# server = FedMDServer(clients, Net().to(device))

# api = FedMDAPI(
#     server,
#     clients,
#     public_dataloader,
#     local_dataloaders,
#     F.nll_loss,
#     local_optimizers,
#     test_dataloader,
#     num_communication=2,
# )
# log = api.run()




def custom_action(api):
    if(api.epoch %2 == 0):
        Accuracies = []
        # field_names = ['Client 0', 'Client 1', 'Client 2', 'Client 3', 'Client 4']
        client_size = len(api.clients)

        for j in tqdm(range(client_size), desc=" Evaluating accuracies", position=0):
            acc_vals = api.score(api.test_loaders[j], api.server_optimizer is None)
            Accuracies.append(acc_vals['clients_score'])
            with open('acc.csv', 'a') as file:
                print("Writing to acc.csv" )
                writer = csv.writer(file)
               
                acc_updated = acc_vals['clients_score'] + [j]
                acc_updated = acc_updated + [api.epoch]
                writer.writerow(acc_updated)

        # now to convert Accuracies to a numpy 
        # 
        # print(Accuracies)
        Accuracies = np.array(Accuracies)
        print(Accuracies.shape)
        # print(Accuracies)
        intra_acc = np.ones(Accuracies.shape[1])
        intra_acc = np.einsum('ii->i', Accuracies)
        Einsteincolsum = np.einsum('ji->i', Accuracies)
        # Einsteincolavg = Einsteincolsum
        mean_inter_acc = (Einsteincolsum - intra_acc)/(client_size - 1)
        
        with open('mean.csv', 'a') as file:
            print("Writing to mean.csv")
            writer = csv.writer(file)
            writer.writerow(intra_acc + ['Intra'] + [api.epoch])
            writer.writerow(mean_inter_acc + ['Inter'] + [api.epoch])

        #calculate mean inter and intra accs.csv
        # for acc in accuracies:

        # with open('acc.csv', 'a') as file:
        #     writer = csv.writer(file, field_names=field_names)
        #     writer.writeheader()
        #     writer.writerows
    else:
        pass


    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fix_seed(0)
    print('Device:', device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='digits', help='Dataset: digits')
    parser.add_argument('--batch', type=int, default=32, help='Batch Size for Training')
    parser.add_argument('--test_batch', type=int, default=32, help='Batch Size for Testing')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate')
    parser.add_argument('--rounds', type=int, default=0, help='Communication Rounds')
    parser.add_argument('--percent', type=float, default=0.1, help='Percentage of Data Used')
    args = parser.parse_args()
    
    
    if(args.dataset == 'digits'):
        client_size = 5
        image_size = [32,32]
        _,_,client_train_loaders, client_test_loaders,_,_,_= prepare_data(args, image_size)
        channels = 3
        num_classes = 10
        print("Works till here")
    else:
        print('Dataset not supported')
        exit()
        
    
    client_model_names = ['AlexNet', 'ConvNet', 'AlexNet', 'ConvNet', 'AlexNet']
    client_models = [get_network(client_model_names[i], channels, num_classes, image_size) for i in range(client_size)]
    clients = [
        FedMDClient(client_models[c].to(device), client_train_loaders[c], output_dim=10, user_id=c, device=device)
        for c in range(client_size)
    ] # voila change the models here sir!
    local_optimizers = [optim.SGD(client_models[client].parameters(), lr=args.lr) for client in range(client_size)]
    server_model = get_network('AlexNet', channels, num_classes, image_size)
    server = FedMDServer(clients, server_model.to(device))
    
    public_dataloader = client_train_loaders[0]
    api = FedMDAPI(
    server,
    clients,
    public_dataloader,
    client_train_loaders,
    F.nll_loss,
    local_optimizers,
    validation_dataloader=client_test_loaders[0],
    num_communication=args.rounds,
    device=device,
    custom_action=custom_action,
    test_loaders=client_test_loaders
)
    log = api.run()
    
    # Accuracies = np.ones((client_size, client_size))
    # trained_models = api.clients
    # print(type(trained_models[0].model))


    # for i in tqdm(range(client_size)):
    #     for j in tqdm(range(client_size)):
    #         acc_i_j = accuracy_torch_dataloader(trained_models[i].model, client_test_loaders[j])
    #         Accuracies[i][j] = acc_i_j
    

    # print(Accuracies)


    # Accuracies = []
    # for j in tqdm(range(client_size), desc=" Evaluating accuracies for : ", position=0):
    #     acc_vals = api.score(client_test_loaders[j], api.server_optimizer is None)
    #     Accuracies.append(acc_vals)
    # print(Accuracies)