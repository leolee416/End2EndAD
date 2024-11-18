import os
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler
import matplotlib.pyplot as plt
from network_model import model_cnn, resnet_model
from data_extractor import Features
import utils
import argparse
import cv2
from torch.utils.tensorboard import SummaryWriter

# Global declaration of the arrays to plot the graphs.
loss_vals = []
train_step = []
val_step = []
val_losses = []

def train_model(args, model, dataset_train, dataset_val):
    # Imports the training model.
    model.train()
    # Declaration of the optimizer and the loss model.
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    step = 0  # steps initialization
    imgs_per_batch = args.batch_size  # gets the batch size from the argument parameters
    optimizer.zero_grad()

    # Initialize SummaryWriter for TensorBoard logging
    writer = SummaryWriter("log/")

    for epoch in range(args.nb_epoch):  # runs for the number of epochs set in the arguments
        sampler = RandomSampler(dataset_train, replacement=True, num_samples=args.samples_per_epoch)
        for i, sample_id in enumerate(sampler):
            data = dataset_train[sample_id]
            label = data['steering_angle']
            img_pth, label = utils.choose_image(label)

            # Data augmentation and processing steps           
            img = cv2.imread(data[img_pth])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = utils.preprocess(img)
            img, label = utils.random_flip(img, label)
            img, label = utils.random_translate(img, label, 100, 10)
            img = utils.random_shadow(img)
            img = utils.random_brightness(img)

            img = np.expand_dims(img, axis=0)  # 添加批量维度 (1, H, W, C)
            img = Variable(torch.cuda.FloatTensor(img))
            label = np.array([label]).astype(float)
            label = Variable(torch.cuda.FloatTensor(label))
            img = img.permute(0, 3, 1, 2)

            # Training and loss calculation
            out_vec = model(img)
            loss = criterion(out_vec, label)

            loss.backward()
            if step % imgs_per_batch == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Status update for the network
            if step % 100 == 0:
                writer.add_scalar('train_loss', loss.item(), step)  # Log training loss
                train_step.append(step)
                loss_vals.append(loss.item())

            if step % 5000 == 0:
                val_loss = eval_model(model, dataset_val, num_samples=1470)
                writer.add_scalar('val_loss', val_loss, step)  # Log validation loss
                val_losses.append(val_loss)
                val_step.append(step)
                print(f"Epoch: {epoch}, Step: {step}, Val Loss: {val_loss:.8f}")
                model.train()  # Resume training after validation

                # Save the model
                if not os.path.exists(args.model_dir):
                    os.makedirs(args.model_dir)

                reflex_pth = os.path.join(args.model_dir, f"model_{step}")
                torch.save(model.state_dict(), reflex_pth)

            step += 1

    writer.close()  # Ensure logging is complete

def eval_model(model, dataset, num_samples):
    model.eval()
    criterion = nn.MSELoss()
    step = 0
    val_loss = 0
    count = 0
    sampler = RandomSampler(dataset)
    torch.manual_seed(0)
    for sample_id in tqdm(sampler):
        if step == num_samples:
            break

        data = dataset[sample_id]
        img_pth, label = utils.choose_image(data['steering_angle'])
        # Image preprocessing and augmentation
        img = cv2.imread(data[img_pth])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = utils.preprocess(img)
        img, label = utils.random_flip(img, label)
        img, label = utils.random_translate(img, label, 100, 10)
        img = utils.random_shadow(img)
        img = utils.random_brightness(img)

        # 修复警告：将单个 img 转为 NumPy 数组，并添加批量维度
        img = np.expand_dims(img, axis=0)  # 添加批量维度
        img = Variable(torch.cuda.FloatTensor(img))
        img = img.permute(0, 3, 1, 2)
        label = np.array([label]).astype(float)
        label = Variable(torch.cuda.FloatTensor(label))

        out_vec = model(img)

        loss = criterion(out_vec, label)

        batch_size = 4
        val_loss += loss.data.item()
        count += batch_size
        step += 1

    val_loss = val_loss / float(count)
    return val_loss

def main(args):
    # Build and import the network model.
    model = model_cnn()
    if torch.cuda.is_available():
        model = model.cuda()

    print('Creating model ...')
    print('Creating data loaders ...')
    dataset = Features(args.data_dir)
    train_size = int(args.train_size * len(dataset))
    test_size = len(dataset) - train_size
    dataset_train, dataset_val = torch.utils.data.dataset.random_split(dataset, [train_size, test_size])

    train_model(args, model, dataset_train, dataset_val)

    # Plot the loss graphs from the training file.
    plt.plot(train_step, loss_vals)
    plt.xlabel("Train Steps")
    plt.ylabel("Train Loss")
    plt.savefig(f'train_loss({args.nb_epoch},{args.learning_rate},{args.keep_prob}).png')
    plt.clf()

    # Plot the loss graphs from the validation step
    plt.plot(val_step, val_losses)
    plt.xlabel("Validation Steps")
    plt.ylabel("Validation Loss")
    plt.savefig(f'val_loss({args.nb_epoch},{args.learning_rate},{args.keep_prob}).png')
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
