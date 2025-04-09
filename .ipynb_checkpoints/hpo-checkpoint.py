# Importing required libraries and modules
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import os
import logging
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set up the logger to display messages on the console at DEBUG level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def test(model, test_loader, criterion, device, epoch_no):
    """
    Evaluate the model on the test dataset.
    Args:
        model: The trained model to be tested.
        test_loader: DataLoader providing batches from the test dataset.
        criterion: Loss function (e.g., cross entropy loss).
        device: Device to run the model on (CPU).
        epoch_no: The current epoch number for logging purposes.
    """
    logger.info(f"Epoch: {epoch_no} - Testing Model on Complete Testing Dataset")
    # Set the model for evaluation 
    model.eval()
    running_loss = 0
    running_corrects = 0

    # No gradient calculation needed during evaluation for performance gain
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs to specified device
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Perform forward pass through the model to get predictions
            outputs = model(inputs)
            # Compute the loss using the specified criterion
            loss = criterion(outputs, labels)
            # Get predicted class by selecting the index with the maximum score
            pred = outputs.argmax(dim=1, keepdim=True)
            running_loss += loss.item() * inputs.size(0) #Calculating the running loss
            running_corrects += pred.eq(labels.view_as(pred)).sum().item() #Calculating running corrects

        total_loss = running_loss / len(test_loader.dataset)
        total_acc = running_corrects/ len(test_loader.dataset)
        logger.info( "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            total_loss, running_corrects, len(test_loader.dataset), 100.0 * total_acc
        ))


def train(model, train_loader, criterion, optimizer, device, epoch_no):
    """
    Train the model for one epoch on the training dataset.
    Args:
      model: The model to be trained.
      train_loader: DataLoader providing batches from the training dataset.
      criterion: Loss function (e.g., cross entropy loss).
      optimizer: Optimization algorithm (e.g., AdamW).
      device: Device to run the model on (CPU).
      epoch_no: The current epoch number for logging purposes.
      
    Returns:
      The updated model after training for one epoch.
    """
    logger.info(f"Epoch: {epoch_no} - Training Model on Complete Training Dataset")
    model.train()
    running_loss = 0
    running_corrects = 0
    running_samples = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        pred = outputs.argmax(dim=1,  keepdim=True)
        running_loss += loss.item() * inputs.size(0) #Calculating the running loss
        running_corrects += pred.eq(labels.view_as(pred)).sum().item() #Calculating running corrects
        running_samples += len(inputs) #Counting the number of running samples
        loss.backward()
        optimizer.step()
        if running_samples % 500 == 0:
            logger.info("\nTrain set:  [{}/{} ({:.0f}%)]\t Loss: {:.2f}\tAccuracy: {}/{} ({:.2f}%)".format(
                running_samples,
                len(train_loader.dataset),
                100.0 * (running_samples / len(train_loader.dataset)),
                loss.item(),
                running_corrects,
                running_samples,
                100.0*(running_corrects/ running_samples)
            ))
    total_loss = running_loss / len(train_loader.dataset)
    total_acc = running_corrects/ len(train_loader.dataset)
    logger.info( "\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        total_loss, running_corrects, len(train_loader.dataset), 100.0 * total_acc
    ))
    return model


def net():

    """
    Create and modify the pre-trained ResNet50 model for the specific task.
    Returns:
      A modified ResNet50 model with its final layer replaced to suit 133-class classification.
    """
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 256),
                             nn.ReLU(inplace=True),
                             nn.Linear(256, 133),
                             nn.ReLU(inplace=True))
    return model


def create_data_loaders(batch_size, data):
    """
    Create and return data loaders for training and testing datasets.
    Args:
        batch_size: Size of each batch.
        data: Root directory containing the data organized into 'train' and 'test' folders.
    Returns:
      A tuple of (train_loader, test_loader).
    """

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor(),
    ])

    train_data = ImageFolder(root=os.path.join(data, "train"), transform=transform)
    test_data = ImageFolder(root=os.path.join(data, "test"), transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader


def main(args):
    """
    Main function to set up device, model, data loaders, loss, optimizer, and execute training and testing loops.
    Args:
      args: Parsed command-line arguments.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device {device}")

    model=net()
    model = model.to(device)
    train_loader, test_loader = create_data_loaders(args.batch_size, args.data)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.fc.parameters(
    ), lr=args.lr, eps=args.eps, weight_decay=args.weight_decay)

    for epoch_no in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch_no} - Starting Training phase.")
        model = train(model, train_loader, criterion, optimizer, device, epoch_no)
        logger.info(f"Epoch {epoch_no} - Starting Testing phase.")
        test(model, test_loader, criterion, device, epoch_no)
    
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")
    logger.info(f"Output Dir  Path: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (0.1 default value)")
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="no of epoch to train. Default value is 2")
    parser.add_argument("--eps", type=float, default=1e-8, metavar="EPS", help="eps (default: 1e-8)")
    parser.add_argument("--weight_decay", type=float, default=1e-2, metavar="WEIGHT-DECAY", help="weight decay coefficient (default 1e-2)")
    parser.add_argument("--batch_size", type=int, default=64, metavar='N', help='input batch size with default(64)')
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--data", type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args = parser.parse_args()
    main(args)