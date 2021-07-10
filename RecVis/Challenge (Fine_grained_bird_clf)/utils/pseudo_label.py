# Based on https://github.com/peimengsui/semi_supervised_mnist
from tqdm import tqdm_notebook
import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch
import torch.optim as optim
import core.model as mdl
from config import *
from core.dataset import get_loaders,get_unlabel


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')                    
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

state_dict = torch.load(args.model)
model = mdl.attention_net(topN=PROPOSAL_NUM)
model.load_state_dict(state_dict)
criterion = torch.nn.CrossEntropyLoss(reduction='mean')

from torchtools.optim import Lookahead
optimizer = optim.SGD(model.parameters(), lr= 0.001, momentum=0.9, weight_decay=WD)
optimizer = Lookahead(base_optimizer=optimizer, k=5, alpha=0.5)


if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

T1 = 31
T2 = 70
af = 3

train_loader,val_loader = get_loaders()
unlabeled_loader = get_unlabel()

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



def alpha_weight(step):
    if step < T1:
        return 0.0
    elif step > T2:
        return af
    else:
         return ((step-T1) / (T2-T1))*af
     
def semisup_train():
    
    
    # Instead of using current epoch we use a "step" variable to calculate alpha_weight
    # This helps the model converge faster
    step = 30
    
    for epoch in tqdm_notebook(range(T1,T2)):
        for batch_idx, batch_data in enumerate(unlabeled_loader):
            
            with torch.no_grad():
              x_unlabeled,y = batch_data
              # Forward Pass to get the pseudo labels
              x_unlabeled = x_unlabeled.cuda()
              model.eval()
              _, output_unlabeled, _, _, _ = model(x_unlabeled)
              _, pseudo_label = torch.max(output_unlabeled, 1)

            model.train()          
           
            # Now calculate the unlabeled loss using the pseudo label
            optimizer.zero_grad()
            #######
            raw_logits, concat_logits, part_logits, _, top_n_prob = model(x_unlabeled)
            part_loss = mdl.list_loss(part_logits.view(BATCH_SIZE * PROPOSAL_NUM, -1),
                                        pseudo_label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(BATCH_SIZE, PROPOSAL_NUM)
            raw_loss = criterion(raw_logits, pseudo_label)
            concat_loss = criterion(concat_logits, pseudo_label)
            rank_loss = mdl.ranking_loss(top_n_prob, part_loss)
            partcls_loss = criterion(part_logits.view(BATCH_SIZE * PROPOSAL_NUM, -1),
                                    pseudo_label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))

            loss = raw_loss + rank_loss + concat_loss + partcls_loss
            total_loss = alpha_weight(step)* loss
            ##########
            total_loss.backward()
            optimizer.step()
            
            #For every 150 batches train one epoch on labeled data 
            if batch_idx % 300 == 0:
                
                # Normal training procedure
                train(epoch)        
                validation()
                model_file = args.experiment + '/model_' + str(epoch) + '.pth'
                torch.save(model.state_dict(), model_file)
                # Now we increment step by 1
                step += 1
                #break
                

semisup_train()