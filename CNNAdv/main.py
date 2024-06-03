import cnn
import pickle as pkl
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
from PIL import Image
import numpy as np
import os, glob, torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, SubsetRandomSampler
from torch.utils.data import random_split
from sklearn.model_selection import KFold
import torch.optim as optim
import dataloader as dt
from cnn import DSNN
from cnn import ImprovedMyCNN
import sys

from tqdm import tqdm
import warnings
from torch.utils.tensorboard import SummaryWriter
writer_path = os.path.join(os.getcwd(),'runs/test1')
writer = SummaryWriter(writer_path)

def train_one_epoch(model, device, train_loader, optimizer, loss_func):
    model.train()
    train_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

def evaluate(model, device, test_loader, loss_func):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def main(
        results_path,
        network_config: dict,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        n_updates: int = 50_000,
        device: str = "cuda",
        train_mode : str = 'standard',
        dataset_name : str = "mnist"
):
    device = torch.device(device)
    if "cuda" in device.type and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = dt.NotationDataset(os.path.join(os.getcwd(), dataset_name), transform)
    num_classes = len(os.listdir(os.path.join(os.getcwd(), dataset_name)))

    writer = SummaryWriter(results_path)

    if train_mode == 'cross_validation':
        k_folds = 5
        kfold = KFold(n_splits=k_folds, shuffle=True)
        for fold, (train_ids, test_ids) in tqdm(enumerate(kfold.split(dataset)), total=k_folds, desc="Folds"):
            print(f'FOLD {fold}')
            print('--------------------------------')

            train_set = SubsetRandomSampler(train_ids)
            test_set = SubsetRandomSampler(test_ids)

            train_loader = DataLoader(dataset, batch_size=64, sampler=train_set, pin_memory=True)
            test_loader = DataLoader(dataset, batch_size=64, sampler=test_set, pin_memory=True)

            model = ImprovedMyCNN(num_classes).to(device)
            loss_func = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            for epoch in tqdm(range(n_updates), desc="Training"):
                train_loss = train_one_epoch(model, device, train_loader, optimizer, loss_func)
                test_loss, test_accuracy = evaluate(model, device, test_loader, loss_func)
                writer.add_scalar("Fold{}/Training Loss".format(fold), train_loss, epoch)
                writer.add_scalar("Fold{}/Test Accuracy".format(fold), test_accuracy, epoch)
                print(f'Epoch {epoch+1}/{n_updates}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    elif train_mode == 'standard':
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        print(f"classes :{num_classes}")
        model = ImprovedMyCNN(num_classes).to(device)
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        print(model)
        for epoch in tqdm(range(n_updates), desc="Training"):
            train_loss = train_one_epoch(model, device, train_loader, optimizer, loss_func)
            test_loss, test_accuracy = evaluate(model, device, test_loader, loss_func)
            writer.add_scalar("Training Loss", train_loss, epoch)
            writer.add_scalar("Test Accuracy", test_accuracy, epoch)
            print(f'Epoch {epoch+1}/{n_updates}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    writer.close()
    


            

                




if __name__ == '__main__':
    import json
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    config_file = os.path.join(os.getcwd(), "working_config.json")
    with open(config_file) as cf:
        config = json.load(cf)
    main(**config)

    writer.flush()