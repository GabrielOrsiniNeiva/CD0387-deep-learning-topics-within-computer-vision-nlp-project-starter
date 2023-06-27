import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import smdebug.pytorch as smd
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
    
import argparse

def test(model, test_loader, loss_criterion, hook):
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs
        labels=labels
        outputs=model(inputs)
        loss=loss_criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)

    return total_acc, total_loss

def train(model, train_loader, loss_criterion, optimizer, hook):
    model.train()
    hook.set_mode(smd.modes.TRAIN)
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data, target
        optimizer.zero_grad()
        output = model(data)
        loss = loss_criterion(output, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    average_loss = running_loss / len(train_loader)
    return model, average_loss

def net():
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))

    return model

def create_data_loaders(data, batch_size):
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    testing_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = torchvision.datasets.ImageFolder(data + '/train', transform=training_transform)
    testset = torchvision.datasets.ImageFolder(data + '/test', transform=testing_transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def save_model(model, model_dir):
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

def main():
    # Define Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = net()

    # Creating loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    # Initialize hook for debbuging
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(loss_criterion)
    
    # Catching data from S3
    train_loader, test_loader = create_data_loaders(args.data_path, args.batch_size)

    # Train the model
    
    for epoch in range(args.epochs):
        model, train_loss = train(model, train_loader, loss_criterion, optimizer, hook)
        test_acc, test_loss = test(model, test_loader, loss_criterion, hook)
        
        print('Train Epoch: {}\tTrain Loss: {:.4f}\Test Loss: {:.4f}'.format(epoch, train_loss, test_loss))

    # Save the trained model
    save_model(model, args.model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add any training arguments that you might need
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--learning_rate', type=float, default=0.001)

    parser.add_argument('--batch_size', type=int, default=64)

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model_path", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data_path", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    args = parser.parse_args()

    main()