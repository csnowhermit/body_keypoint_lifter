from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time

from dataset import Keypoint_Dataset
from model_body import SimpleBaseline
from eval_util import mpjpe, compute_errors

'''
    CUDA_VISIBEL_DEVICES=0 nohup python -u train.py >> train.log &
'''

os.environ['CUDA_VISIBEL_DEVICES'] = '0'

writer = SummaryWriter()

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Batch size during training
batch_size = 1024
num_epochs = 200
lr = 1e-4
ngpu = 1

root_path = "./data/"

train_dataset = Keypoint_Dataset(root_path + "data_train.npy", label_path=root_path + "label_train.npy")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = Keypoint_Dataset(root_path + "data_val.npy", label_path=root_path + "label_val.npy")
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

face_loss_func = nn.L1Loss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleBaseline(input_dim=33*2, out_dim=33*3)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
scheduler = lr_scheduler.StepLR(optimizer, step_size=len(train_dataloader) * 10, gamma=0.9)    # 每10 epoch衰减到0.9

best_eval_loss = 999999    # 最好的loss，val loss mean只要比该值小，则保存

total_step = len(train_dataloader)
model.train()
for epoch in range(num_epochs):
    start = time.time()

    for i, (data, label) in enumerate(train_dataloader):
        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)

        output = model(data)

        loss = mpjpe(output.view(-1, 33, 3), label.view(-1, 33, 3))

        loss.backward()
        optimizer.step()
        scheduler.step()

        curr_step = epoch * total_step + i
        writer.add_scalar("train/MPJPE/body_loss", loss * 1000.0, curr_step)
        writer.flush()

        if (i % 10) == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, spend time: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), time.time() - start))
            start = time.time()

    model.eval()
    with torch.no_grad():
        val_loss = []
        pa_val_loss = []
        start = time.time()
        for i, (data, label) in enumerate(val_dataloader):
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)

            output = model(data)

            loss = mpjpe(output.view(-1, 33, 3), label.view(-1, 33, 3))

            pa_body_loss = compute_errors(label.detach().cpu().numpy().reshape(-1, 33, 3) * 1000.0,
                                          output.detach().cpu().numpy().reshape(-1, 33, 3) * 1000.0)

            val_loss.append(loss.item() * 1000.0)
            for item in pa_body_loss[1]:
                pa_val_loss.append(item)

        curr_val_loss = np.mean(val_loss)
        print('Epoch [{}/{}], val_loss: {:.6f}, time: {:.4f}'.format(epoch + 1, num_epochs, curr_val_loss, time.time() - start))

        writer.add_scalar("val/MPJPE/body_loss", curr_val_loss, epoch)
        writer.add_scalar("val/PA_MPJPE/body_loss", np.mean(pa_val_loss), epoch)
        writer.flush()

        if os.path.exists(os.path.join(root_path, "checkpoint/")) is False:
            os.makedirs(os.path.join(root_path, "checkpoint/"))

        if curr_val_loss < best_eval_loss:    # 只要损失下降就保存
            best_eval_loss = curr_val_loss    # 保存当前的loss为最好
            torch.save({
                    "curr_epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_eval_loss": best_eval_loss,
                    "lr": scheduler.get_last_lr()
                },root_path + 'checkpoint/body_{}_loss_{:.6f}_{:.6f}.pt'.format(epoch, curr_val_loss, np.mean(pa_val_loss)))

    model.train()
