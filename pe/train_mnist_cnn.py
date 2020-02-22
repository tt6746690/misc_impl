import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchvision 
import torchvision.transforms as tv_transforms

import models
import datasets
from utils import makedirs_exists_ok, seed_rng, set_cuda_visible_devices, load_weights_from_file


def train(
    target_type,
    model_name,
    data_root,
    model_root,
    log_root,
    seed,
    image_size,
    batch_size,
    gpu_id,
    n_workers,
    load_weights,
    lr,
    n_epochs,
    log_interval):

    makedirs_exists_ok(data_root)
    makedirs_exists_ok(model_root)
    makedirs_exists_ok(log_root)

    writer = SummaryWriter(log_root)
    writer.flush()

    seed_rng(seed)
    device = set_cuda_visible_devices(gpu_id)

    ##############################################################
    # data
    ##############################################################

    transforms = tv_transforms.Compose([
        tv_transforms.Resize(image_size),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.ColorMNIST(
            root=data_root, download=True, train=True, transform=transforms),
        batch_size=batch_size, shuffle=True, num_workers=n_workers)
    test_loader = torch.utils.data.DataLoader(
        datasets.ColorMNIST(
            root=data_root, download=True, train=False, transform=transforms),
        batch_size=batch_size, shuffle=True, num_workers=n_workers)

    if target_type == 'digit':
        n_classes = 10
        criterion = nn.CrossEntropyLoss()
        select_target = lambda y_digit, y_color: y_digit
    else:
        n_classes = 1
        criterion = nn.BCEWithLogitsLoss()
        select_target = lambda y_digit, y_color: (y_color < 0.5).float().view(-1, 1)

    model = models.MnistCNN(3, n_classes, 32).to(device)
    load_weights_from_file(model, load_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    ##############################################################
    # training
    ##############################################################

    for epoch in range(n_epochs):
        for it, (x, y_digit, y_color) in enumerate(train_loader):
            
            y = select_target(y_digit, y_color)
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            ##############################################################
            # print
            ##############################################################
            
            global_step = epoch*len(train_loader)+it
            writer.add_scalar('loss/training', loss, global_step)
            
            if it % log_interval == log_interval-1:
                print(f'[{epoch+1}/{n_epochs}]\t'
                      f'[{(it+1)*batch_size}/{len(train_loader.dataset)} ({100.*(it+1)/len(train_loader):.0f}%)]\t'
                      f'loss={loss.item():.4}')

        test_loss, correct = evaluate(model, test_loader, device, target_type)

        print(f'[{epoch+1}/{n_epochs}]\t'
                f'Average Loss: {test_loss:.4}\t'
                f'Accuracy: {correct}/{len(test_loader.dataset)} ({correct:.0f}%)')
                
        torch.save(model.state_dict(), os.path.join(model_root, f'mnist_cnn_{epoch}.pt'))


def evaluate(model, test_loader, device, target_type):
    
    if target_type == 'digit':
        select_target = lambda y_digit, y_color: y_digit
        criterion = F.cross_entropy
        compute_decision = lambda output: output.argmax(dim=1, keepdim=True)
    else:
        select_target = lambda y_digit, y_color: (y_color < 0.5).float().view(-1, 1)
        criterion = F.binary_cross_entropy_with_logits
        compute_decision = lambda output: (torch.sigmoid(output) > 0).float()

    model.eval()

    with torch.no_grad():
        test_loss = 0
        correct   = 0
        for x, y_digit, y_color in test_loader:
            y = select_target(y_digit, y_color)
            x, y  = x.to(device), y.to(device)
            output = model(x)
            test_loss += criterion(output, y, reduction='sum').item()
            pred = compute_decision(output)
            correct += pred.eq(y.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        correct = 100. * correct / len(test_loader.dataset)

    model.train()

    return test_loss, correct

        
if __name__ == '__main__':

    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default='../data', help="data folder")
    parser.add_argument("--model_root", type=str, default='./models/mnist_cnn', help="model folder")
    parser.add_argument("--log_root", type=str, default=f'./logs/mnist_cnn', help="log folder")
    parser.add_argument("--model_name", type=str, default='mnist_cnn', help="name of the model")
    parser.add_argument("--load_weights", type=str, default='', help="optional .pt model file to initialize generator with")
    parser.add_argument("--seed", type=int, default=0, help="rng seed")
    parser.add_argument("--image_size", type=int, default=32, help="image size of the inputs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--gpu_id", type=str, default='0', help="gpu id assigned by cluster")
    parser.add_argument("--n_workers", type=int, default=8, help="number of CPU workers for processing input data")
    parser.add_argument("--learning_rate", type=float, dest='lr', default=0.0002, help="rng seed")
    parser.add_argument("--epochs", type=int, dest='n_epochs', default=5, help="number of epochs")
    parser.add_argument("--log_interval", type=int, dest='log_interval', default=100,  help="number of batches to print loss / plot figures")
    parser.add_argument("--target_type", type=str, dest='target_type', default='digit', help="in {'digit', 'color'}")

    args = parser.parse_args()

    if args.target_type not in ['digit', 'color']:
        print('target type should be either "digit" or "color"')

    args.model_name = f'{args.model_name}_{args.target_type}'

    args.model_root = os.path.join(args.model_root, args.model_name)
    args.log_root   = os.path.join(args.log_root,   args.model_name)

    train(**vars(args))