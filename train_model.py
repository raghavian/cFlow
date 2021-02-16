import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from data.dataset import LIDC
from models.cflownet import cFlowNet
import torch.nn as nn
from models.unet import Unet
from utils.utils import l2_regularisation,ged
import time
from utils.tools import makeLogFile, writeLog, dice_loss
import pdb
import argparse
import sys
import os

torch.manual_seed(42)
np.random.seed(42)


parser = argparse.ArgumentParser()
parser.add_argument('--flow', action='store_true', default=False, help=' Train with Flow model')
parser.add_argument('--glow', action='store_true', default=False, help=' Train with Glow model')
parser.add_argument('--data', type=str, default='data/lidc/',help='Path to data.')
parser.add_argument('--probUnet', action='store_true', default=False, help='Train with Prob. Unet')
parser.add_argument('--unet', action='store_true', default=False, help='Train with Det. Unet')
parser.add_argument('--singleRater', action='store_true', default=False, help='Train with single rater')
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=96, help='Batch size')
parser.add_argument('--num_flows', type=int, default=4, help='Num flows')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LIDC(data_dir=args.data)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.2 * dataset_size))

np.random.shuffle(indices)
valid_indices, test_indices, train_indices = indices[:split], indices[split:2*split], indices[2*split:]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
valid_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler)
test_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler)

print("Number of train/valid/test patches:", (len(train_indices),len(valid_indices),len(test_indices)))

fName = time.strftime("%Y%m%d_%H_%M")

if args.singleRater:
    print("Using a single rater..")
    fName = fName+'_1R'
else:
    print("Using all experts...")
    fName = fName+'_4R'

if args.flow:
    print("Using flow based model with %d steps"%args.num_flows)
    fName = fName+'_flow'
    net = cFlowNet(input_channels=1, num_classes=1, 
			num_filters=[32,64,128,256], latent_dim=6, 
        	no_convs_fcomb=4, num_flows=args.num_flows, 
			norm=True,flow=args.flow)
elif args.glow:
    print("Using Glow based model with %d steps"%args.num_flows)
    fName = fName+'_glow'
    net = cFlowNet(input_channels=1, num_classes=1,
			num_filters=[32,64,128,256], latent_dim=6, 
			no_convs_fcomb=4, norm=True,num_flows=args.num_flows,
			flow=args.flow,glow=args.glow)
elif args.probUnet:
    print("Using probUnet")
    fName = fName+'_probUnet'
    net = cFlowNet(input_channels=1, num_classes=1, 
			num_filters=[32,64,128,256], latent_dim=6, 
        	no_convs_fcomb=4, norm=True,flow=args.flow)
elif args.unet:
    print("Using Det. Unet")
    fName = fName+'_Unet'
    net = Unet(input_channels=1, num_classes=1, 
			num_filters=[32,64,128,256], apply_last_layer=True, 
            padding=True, norm=True, 
			initializers={'w':'he_normal', 'b':'normal'})
    criterion = nn.BCELoss(size_average=False)
else:
    print("Choose a model.\nAborting....")
    sys.exit()

if not os.path.exists('logs'):
    os.mkdir('logs')

logFile = 'logs/'+fName+'.txt'
makeLogFile(logFile)

net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
nTrain = len(train_loader)
nValid = len(valid_loader)
nTest = len(test_loader)

minLoss = 1e8

convIter=0
convCheck = 20

for epoch in range(args.epochs):
    trLoss = []
    vlLoss = []
    vlGed = [0]
    klEpoch = [0]
    recEpoch = [0]
    kl = torch.zeros(1)
    recLoss = torch.zeros(1)
    dGED = 0
    t = time.time()
    for step, (patch, masks) in enumerate(train_loader): 
        patch = patch.to(device)
        masks = masks.to(device)
        if args.singleRater or args.unet:
            rater = 0
        else:
            # Choose a random mask
            rater = torch.randperm(4)[0]
        mask = masks[:,[rater]]
        if not args.unet:
            net.forward(patch, mask, training=True)
            _,_,_,elbo = net.elbo(mask,use_mask=False,analytic_kl=False)
            reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior)
            loss = -elbo + 1e-5 * reg_loss
        else:
            pred = torch.sigmoid(net.forward(patch,False))
            loss = criterion(target=mask,input=pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trLoss.append(loss.item())
        
        if (step+1) % 5 == 0:
            with torch.no_grad():
                for idx, (patch, masks) in enumerate(valid_loader): 
                    patch = patch.to(device)
                    masks = masks.to(device)
                        # Choose a random mask
                    mask = masks[:,[rater]]
                    if not args.unet:
                        net.forward(patch, mask, training=True)
                        _,recLoss, kl, elbo = net.elbo(mask,use_mask=False,
													analytic_kl=False)
                        reg_loss = l2_regularisation(net.posterior) + \
                                l2_regularisation(net.prior)
                        loss = -elbo + 1e-5 * reg_loss
                        klEpoch.append(kl.item())
                        recEpoch.append(recLoss.item())
                    else:
                        pred = torch.sigmoid(net.forward(patch, False))
                        loss = criterion(target=mask, input=pred)
                    vlLoss.append(loss.item())
                    break
                print ('Epoch [{}/{}], Step [{}/{}], TrLoss: {:.4f}, VlLoss: {:.4f}, RecLoss: {:.4f}, kl: {:.4f}, GED: {:.4f}'
                    .format(epoch+1, args.epochs, step+1, nTrain, trLoss[-1], vlLoss[-1], recLoss.item(),\
                            kl.item(), vlGed[-1]))
    epValidLoss =  np.mean(vlLoss)
    if (epoch+1) % 1 == 0 and epValidLoss > 0 and epValidLoss < minLoss:
        convIter = 0
        minLoss = epValidLoss
        print("New min: %.2f\nSaving model..."%(minLoss))
        torch.save(net.state_dict(),'../models/'+fName+'.pt')
    else:
        convIter += 1
    writeLog(logFile, epoch, np.mean(trLoss),
                       epValidLoss,np.mean(recEpoch), np.mean(klEpoch), time.time()-t)

    if convIter == convCheck:
        print("Converged at epoch %d"%(epoch+1-convCheck))
        break
    elif np.isnan(epValidLoss):
        print("Nan error!")
        break

