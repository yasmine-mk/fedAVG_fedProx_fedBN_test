import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import TensorDataset
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import time
import copy 
from nets.models import DigitModel, Net 
import argparse
from utils import data_utils
import matplotlib.pyplot as plt
import numpy as np
import medmnist 
from medmnist import INFO, Evaluator

NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001
download = True 



def prepare_data():
    # Prepare data
    transform_organamnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
    ])

    transform_organcmnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
    ])

    transform_organsmnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
    ])

    # organaMNIST
    data_flag0 = 'organamnist'
    info0 = INFO[data_flag0]
    DataClass = getattr(medmnist, info0['python_class'])
    organamnist_trainset = DataClass(split='train', transform=transform_organamnist, download=download)
    organamnist_testset = DataClass(split='test', transform=transform_organamnist, download=download)
    
    # organcMNIST
    data_flag1 = 'organcmnist'
    info1 = INFO[data_flag1]
    DataClass = getattr(medmnist, info1['python_class'])
    organcmnist_trainset = DataClass(split='train', transform=transform_organcmnist, download=download)
    organcmnist_testset = DataClass(split='test', transform=transform_organcmnist, download=download)
    
    # organsMNIST
    data_flag2 = 'organsmnist'
    info2 = INFO[data_flag2]
    DataClass = getattr(medmnist, info2['python_class'])
    organsmnist_trainset = DataClass(split='train', transform=transform_organsmnist, download=download)
    organsmnist_testset = DataClass(split='test', transform=transform_organsmnist, download=download)

    
    organamnist_train_loader = data.DataLoader(organamnist_trainset, batch_size=BATCH_SIZE, shuffle=True)
    organamnist_test_loader = data.DataLoader(dataset=organamnist_testset, batch_size=2*BATCH_SIZE, shuffle=False)
    
    organcmnist_train_loader = data.DataLoader(organcmnist_trainset, batch_size=BATCH_SIZE, shuffle=True)
    organcmnist_test_loader = data.DataLoader(dataset=organcmnist_testset, batch_size=2*BATCH_SIZE, shuffle=False)
    
    organsmnist_train_loader = data.DataLoader(organsmnist_trainset, batch_size=BATCH_SIZE, shuffle=True)
    organsmnist_test_loader = data.DataLoader(dataset=organsmnist_testset, batch_size=2*BATCH_SIZE, shuffle=False)
    train_loaders = [organamnist_train_loader, organcmnist_train_loader, organsmnist_train_loader]
    test_loaders  = [organamnist_test_loader, organcmnist_test_loader, organsmnist_test_loader]

    return train_loaders, test_loaders
"""
def train(model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output = model(x)

        loss = loss_fun(output, y)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_iter), correct/num_data
"""
def train(model, train_loader, optimizer, criterion,client_num, task):
    train_correct = 0
    train_total = 0
    model.train()
    for inputs, targets in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        
        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        train_total += targets.size(0)
        train_correct += (predicted == targets).sum().item()
    
    train_acc = train_correct / train_total
    train_loss = loss.item()
    
    return train_loss, train_acc

def train_fedprox(args, model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output = model(x)

        loss = loss_fun(output, y)

        #########################we implement FedProx Here###########################
        # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
        if step>0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            loss += args.mu / 2. * w_diff
        #############################################################################

        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_iter), correct/num_data

def test(model, test_loader, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []

    for data, target in test_loader:
        data = data.to(device).float()
        target = target.to(device).long().squeeze()
        targets.append(target.detach().cpu().numpy())

        output = model(data)
        
        test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()
    
    return test_loss/len(test_loader), correct /len(test_loader.dataset)

################# Key Function ########################
def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed= 1
    np.random.seed(seed)
    torch.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed) 

    test_accs=[]
    print('Device:', device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help ='whether to make a log')
    parser.add_argument('--test', action='store_true', help ='test the pretrained model')
    parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type = int, default= 32, help ='batch size')
    parser.add_argument('--iters', type = int, default=100, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
    parser.add_argument('--mode', type = str, default='fedbn', help='fedavg | fedprox | fedbn')
    parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type = str, default='../checkpoint/digits', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    args = parser.parse_args()

    exp_folder = 'federated_digits'

    args.save_path = os.path.join(args.save_path, exp_folder)
    
    log = args.log
    if log:
        log_path = os.path.join('../logs/digits/', exp_folder)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path,'{}.log'.format(args.mode)), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('    lr: {}\n'.format(args.lr))
        logfile.write('    batch: {}\n'.format(args.batch))
        logfile.write('    iters: {}\n'.format(args.iters))
        logfile.write('    wk_iters: {}\n'.format(args.wk_iters))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}'.format(args.mode))
   
   
    server_model = Net(1,11).to(device)
    loss_fun = nn.CrossEntropyLoss()

    # prepare the data
    train_loaders, test_loaders = prepare_data()

    # name of each client dataset
    datasets = ['organamnist', 'organcmnist', 'organsmnist']
    
    # federated setting
    client_num = len(datasets)
    client_weights = [1/client_num for i in range(client_num)]
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

    if args.test:
        print('Loading snapshots...')
        checkpoint = torch.load('../snapshots/digits/{}'.format(args.mode.lower()))
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower()=='fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
            for test_idx, test_loader in enumerate(test_loaders):
                _, test_acc = test(models[test_idx], test_loader, loss_fun, device)
                print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
        else:
            for test_idx, test_loader in enumerate(test_loaders):
                _, test_acc = test(server_model, test_loader, loss_fun, device)
                print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
        exit(0)

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower()=='fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        resume_iter = int(checkpoint['a_iter']) + 1
        print('Resume training from epoch {}'.format(resume_iter))
    else:
        resume_iter = 0

    # start training
    for a_iter in range(resume_iter, args.iters):
        optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            if args.log: logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters)) 
            
            for client_idx in range(client_num):
                model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
                if args.mode.lower() == 'fedprox':
                    if a_iter > 0:
                        train_fedprox(args, model, train_loader, optimizer, loss_fun, client_num, device)
                    else:
                        train(model, train_loader, optimizer, loss_fun, client_num, device)
                else:
                    train(model, train_loader, optimizer, loss_fun, client_num, device)
         
        # aggregation
        server_model, models = communication(args, server_model, models, client_weights)
        # report after aggregation
        for client_idx in range(client_num):
                model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
                train_loss, train_acc = test(model, train_loader, loss_fun, device) 
                print(' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx] ,train_loss, train_acc))
                if args.log:
                    logfile.write(' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx] ,train_loss, train_acc))\

        # start testing
        epoch_test_acc=0.0
        for test_idx, test_loader in enumerate(test_loaders):
            test_loss, test_acc = test(models[test_idx], test_loader, loss_fun, device)
            print(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(datasets[test_idx], test_loss, test_acc))
            epoch_test_acc+=test_acc
            if args.log:
                logfile.write(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}\n'.format(datasets[test_idx], test_loss, test_acc))
        epoch_test_acc=epoch_test_acc/len(test_loaders)
        test_accs.append(epoch_test_acc)
        print("test_accs: ", test_accs)
    # define the x-axis as the number of epochs
    epochs = range(len(test_accs))

    # plot the accuracies for each model

    plt.plot(epochs, test_accs, label='fedBN')
    

    # add a legend to the plot
    plt.legend()

    # add x and y axis labels and a title
    plt.xlabel('Epochs')
    plt.ylabel('Testing Accuracy %')
    #plt.title('Comparison of Testing Accuracies for Three Models')

    # save the plot as a PNG file
    plt.savefig('test_accuracies.png')

    # display the plot
    plt.show()

    # Save checkpoint
    print(' Saving checkpoints to {}...'.format(SAVE_PATH))
    if args.mode.lower() == 'fedbn':
        torch.save({
            'model_0': models[0].state_dict(),
            'model_1': models[1].state_dict(),
            'model_2': models[2].state_dict(),
            'model_3': models[3].state_dict(),
            'model_4': models[4].state_dict(),
            'server_model': server_model.state_dict(),
        }, SAVE_PATH)
    else:
        torch.save({
            'server_model': server_model.state_dict(),
        }, SAVE_PATH)

    if log:
        logfile.flush()
        logfile.close()


