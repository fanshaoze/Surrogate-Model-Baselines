# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import random

import torch
import torch.utils.data
from models import *
from config import get_args

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dataset import CustomDataset


def random_input(args, data_point_num):
    return torch.rand(args.input_size, data_point_num)


def data_preprocessing(raw_data):
    """
    data preprocessing, data split and convert to tensor
    :param raw_data: the raw data that need to be convert to a tensor
    :return:
    """
    training_data_tensor, validation_data_tensor, test_data_tensor = {}, {}, {}
    return training_data_tensor, validation_data_tensor, test_data_tensor


def train_one_epoch(training_dataloader, optimizer, epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting

    for i, data in enumerate(training_dataloader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        # print(outputs)
        # print(labels)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        running_loss += loss.item()
        if i % _args.batch_size == _args.batch_size-1:
            last_loss = running_loss / _args.batch_size  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def tensors_2_dataloader(data_tensors, args):
    data_loaders = []
    for data_tensor in data_tensors:
        data_loaders.append(torch.utils.data.DataLoader(data_tensor,
                                                        batch_size=args.batch_size,
                                                        num_workers=args.num_workers))

    return data_loaders


# main process

_args = get_args()
raw_data = torch.rand(10, _args.input_dim)
labels = torch.rand(10, _args.output_dim)

# need shuffle

train_end_idx = int(len(raw_data) * _args.train_ratio)
val_end_idx = int(len(raw_data) * _args.train_ratio) + int(len(raw_data) * _args.val_ratio)
test_end_idx = len(raw_data)
tensors = [CustomDataset(raw_data[:train_end_idx], labels[:train_end_idx]),
           CustomDataset(raw_data[train_end_idx:val_end_idx], labels[train_end_idx:val_end_idx]),
           CustomDataset(raw_data[val_end_idx:test_end_idx], labels[val_end_idx:test_end_idx])]

training_loader, validation_loader, testing_loader = tensors_2_dataloader(data_tensors=tensors, args=_args)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/trainer_{}'.format(timestamp))

current_epoch = 0
best_vloss = 1_000_000.

# RSE,
# loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.NLLLoss()

model = Model(args=_args)
model_path = 'DL_model'
#  SGD, RMSprop, or Adam and their relationship
# weight decay: parameter of the L2 penalty
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(_args.epoch_number):
    print('EPOCH {}:'.format(current_epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(training_dataloader=training_loader, optimizer=optimizer,
                               epoch_index=current_epoch, tb_writer=writer)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss, i = 0.0, 0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    # count is i + 1
    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': avg_loss, 'Validation': avg_vloss},
                       current_epoch + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        torch.save(model.state_dict(), model_path)

    current_epoch += 1

saved_model = Model(args=_args)
saved_model.load_state_dict(torch.load(model_path))

# if __name__ == '__main__':
#     # auto_train(args)
