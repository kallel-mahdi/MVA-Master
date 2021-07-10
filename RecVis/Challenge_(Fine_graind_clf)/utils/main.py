import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
from config import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD
# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading
from core.dataset import get_loaders


train_loader,val_loader = get_loaders()

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script

from core import model as mdl
model  = mdl.attention_net(topN=PROPOSAL_NUM)
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

#optimizer = optim.SGD(model.parameters(), lr= 0.001, momentum=0.9, weight_decay=WD)
# from torchtools.optim import RangerLars # Over9000

# optimizer = RangerLars(model.parameters())

from torchtools.optim import Lookahead
optimizer = optim.SGD(model.parameters(), lr= 0.001, momentum=0.9, weight_decay=WD)
optimizer = Lookahead(base_optimizer=optimizer, k=5, alpha=0.5)

def train(epoch):
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        if use_cuda:
            data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        ### Compute loss####
        raw_logits, concat_logits, part_logits, _, top_n_prob = model(data)
        part_loss = mdl.list_loss(part_logits.view(BATCH_SIZE * PROPOSAL_NUM, -1),
                                    label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(BATCH_SIZE, PROPOSAL_NUM)
        raw_loss = criterion(raw_logits, label)
        concat_loss = criterion(concat_logits, label)
        rank_loss = mdl.ranking_loss(top_n_prob, part_loss)
        partcls_loss = criterion(part_logits.view(BATCH_SIZE * PROPOSAL_NUM, -1),
                                 label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))

        total_loss = raw_loss + rank_loss + concat_loss + partcls_loss
        ##########
        total_loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), concat_loss.data.item()))

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
      for data, label in val_loader:
          if use_cuda:
              data, label = data.cuda(), label.cuda()
          output = model(data)
          # sum up batch loss
          criterion = torch.nn.CrossEntropyLoss(reduction='mean')
          raw_logits, concat_logits, part_logits, _, top_n_prob = model(data)
          part_loss = mdl.list_loss(part_logits.view(BATCH_SIZE * PROPOSAL_NUM, -1),
                                      label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(BATCH_SIZE, PROPOSAL_NUM)
          raw_loss = criterion(raw_logits, label)
          concat_loss = criterion(concat_logits, label)
          rank_loss = mdl.ranking_loss(top_n_prob, part_loss)
          partcls_loss = criterion(part_logits.view(BATCH_SIZE * PROPOSAL_NUM, -1),
                                  label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))

          total_loss = raw_loss + rank_loss + concat_loss + partcls_loss
          validation_loss += concat_loss.data.item()
          # get the index of the max log-probability
          pred = concat_logits.data.max(1, keepdim=True)[1]
          correct += pred.eq(label.data.view_as(pred)).cpu().sum()

      validation_loss /= len(val_loader.dataset)
      print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
          validation_loss, correct, len(val_loader.dataset),
          100. * correct / len(val_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation()
    model_file = args.experiment + '/model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
