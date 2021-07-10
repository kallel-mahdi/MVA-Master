import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch

import core.model as mdl
from config import *
from core.dataset import get_test
#from main import validation

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

state_dict = torch.load(args.model)
model = mdl.attention_net(topN=PROPOSAL_NUM)
model.load_state_dict(state_dict)
model.eval()


if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

val_loader = get_test()

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



validation()


