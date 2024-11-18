"""
  *  @copyright (c) 2020 Charan Karthikeyan P V, Nagireddi Jagadesh Nischal
  *  @file    train.py
  *  @author  Charan Karthikeyan P V, Nagireddi Jagadesh Nischal
  *
  *  @brief Main file to train and evaluate the model.  
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import RandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from network_model import model_cnn
from data_extractor import Features
import utils
import argparse

# Global arrays for plotting graphs
loss_vals = []
train_step = []
val_step = []
val_losses = []

def train_model(args, model, dataset_train, dataset_val, writer):
    """Train the model with the input data and save it."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    step = 0
    imgs_per_batch = args.batch_size

    optimizer.zero_grad()
    for epoch in range(args.nb_epoch):
        sampler = RandomSampler(dataset_train, replacement=True, num_samples=args.samples_per_epoch)
        for i, sample_id in enumerate(sampler):
            # Load data
            data = dataset_train[sample_id]
            label = data['steering_angle']
            img_pth, label = utils.choose_image(label)

            # Image preprocessing
            img = cv2.imread(data[img_pth])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))  # Resize to (224, 224)
            img = img.transpose((2, 0, 1))  # Convert to [C, H, W]
            img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]

            # Convert to PyTorch tensors
            img = torch.tensor(img, dtype=torch.float32, device='cuda').unsqueeze(0)  # Add batch dimension
            label = torch.tensor(label, dtype=torch.float32, device='cuda').view(-1, 1)  # Ensure shape is (batch_size, 1)

            # Forward pass
            out_vec = model(img)
            loss = criterion(out_vec, label)

            # Backward pass
            loss.backward()
            if step % imgs_per_batch == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Log training progress
            if step % 20 == 0:
                print(f'Epoch: {epoch} | Iter: {i} | Step: {step} | Train Loss: {loss.item():.8f}')
                train_step.append(step)
                loss_vals.append(loss.item())
                writer.add_scalar('Train/Loss', loss.item(), step)

            # Validate and save model checkpoint
            if step % 5000 == 0:
                val_loss = eval_model(model, dataset_val, num_samples=1470)
                val_losses.append(val_loss)
                val_step.append(step)
                writer.add_scalar('Validation/Loss', val_loss, step)
                print(f'Epoch: {epoch} | Iter: {i} | Step: {step} | Val Loss: {val_loss:.8f}')
                model.train()  # Resume training mode

                # Save checkpoint
                if not os.path.exists(args.model_dir):
                    os.makedirs(args.model_dir)
                reflex_pth = os.path.join(args.model_dir, f'model_{step}')
                torch.save(model.state_dict(), reflex_pth)

            step += 1


def eval_model(model, dataset, num_samples):
    """Evaluate the trained model."""
    model.eval()
    criterion = nn.MSELoss()
    val_loss = 0
    count = 0
    sampler = RandomSampler(dataset)
    torch.manual_seed(0)
    for step, sample_id in enumerate(tqdm(sampler)):
        if step == num_samples:
            break

        data = dataset[sample_id]
        img_pth, label = utils.choose_image(data['steering_angle'])
        img = cv2.imread(data[img_pth])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))  # Resize to (224, 224)
        img = img.transpose((2, 0, 1))  # Convert to [C, H, W]
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Convert to PyTorch tensors
        img = torch.tensor(img, dtype=torch.float32, device='cuda').unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32, device='cuda').view(-1, 1)

        # Forward pass
        out_vec = model(img)
        loss = criterion(out_vec, label)

        val_loss += loss.data.item()
        count += 1

    return val_loss / count


def main(args):
    """Main function to run training."""
    model = model_cnn()
    if torch.cuda.is_available():
        model = model.cuda()

    # TensorBoard setup
    writer = SummaryWriter(log_dir='log/')

    print('Creating model ...')
    print('Creating data loaders ...')
    dataset = Features(args.data_dir)
    train_size = int(args.train_size * len(dataset))
    test_size = len(dataset) - train_size
    dataset_train, dataset_val = torch.utils.data.dataset.random_split(dataset, [train_size, test_size])

    train_model(args, model, dataset_train, dataset_val, writer)

    # Plot the loss graphs
    plt.plot(train_step, loss_vals)
    plt.xlabel("Train Steps")
    plt.ylabel("Train Loss")
    plt.savefig('train_loss.png')
    plt.clf()

    plt.plot(val_step, val_losses)
    plt.xlabel("Validation Steps")
    plt.ylabel("Validation Loss")
    plt.savefig('val_loss.png')
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='data')
    parser.add_argument('-m', help='model directory', dest='model_dir', type=str, default='models')
    parser.add_argument('-t', help='train size fraction', dest='train_size', type=float, default=0.8)
    parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=10)
    parser.add_argument('-s', help='samples per epoch', dest='samples_per_epoch', type=int, default=20000)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=40)
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)

    args = parser.parse_args()
    main(args)
