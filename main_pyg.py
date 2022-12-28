from re import I
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from gnn import GNN

from tqdm import tqdm
import argparse
import time
import random
import numpy as np
import pandas as pd
import os

from dataset import PygGraphPropPredDataset

### importing utils
from eval_helper import Evaluator

multicls_criterion = torch.nn.MSELoss()

def train(model, device, loader, optimizer):
    model.train()

    loss_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        print("training batch shape.. ",batch.x.shape,batch)
        # ideally the control should do following checks
        # 1. check  if batch.x.shape[0] ==1
        # 2. batch has only one graph 
        # if so pass 
        if batch.x.shape[0] == 1:
            pass
        else:
            output = model(batch)
            loss = 0
            optimizer.zero_grad()
            for inx,y in enumerate(batch.y):
                notNanInx = torch.isnan(y)
                notTrueInx = [b for b in range(notNanInx.size(0)) if notNanInx[b].item() == False]
                loss += multicls_criterion(output[inx][notTrueInx].to(torch.float32), batch.y[inx][notTrueInx])

            loss.backward()
            optimizer.step()

            loss_accum += loss.item()

    print('Average training loss: {}'.format(loss_accum / (step + 1)))

def eval(model, device, loader, evaluator, isLogging=False,**kwargs):
    model.eval()
    landmarks_ref_list = []
    landmarks_pred_list = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        print("batch shape.. ",batch.y.shape)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                output = model(batch)
                pred_to_eval = []
                target_to_eval = []
                for inx,y in enumerate(batch.y):
                    notNanInx = torch.isnan(y)
                    notTrueInx = [b for b in range(notNanInx.size(0)) if notNanInx[b].item() == False]
                    # print("----",output[inx])
                    pred_to_eval.append(output[inx][notTrueInx].tolist())
                    target_to_eval.append(batch.y[inx][notTrueInx].tolist())

            landmarks_ref_list.extend(target_to_eval)
            landmarks_pred_list.extend(pred_to_eval)

    input_dict = {"landmark_ref": landmarks_ref_list, "landmark_pred": landmarks_pred_list}

    if isLogging and (kwargs["split"] is not None) and (kwargs["split_idx"] is not None):
        if kwargs["filedir"] is None:
            filedir = ""
        else:
            filedir =kwargs["filedir"] 
    
        torch.save({"landmarks_pred":landmarks_pred_list, "image_id":kwargs["split_idx"]}, os.path.join(filedir,"pred_landmarks_{split}.pt".format(split=kwargs["split"])))

    return evaluator.eval(input_dict)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-code2 data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gcn',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gcn-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of epochs to train (default: 25)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="pyg_face",
                        help='dataset name (default: pyg_face)')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                    help='Training data split ratio, Test and Val dataset are splitted equally from the rest')

    parser.add_argument('--seed',type=int,default=42)

    parser.add_argument('--rootdir', type=str, default="",
                        help='root directory to build the dataset')
    
    parser.add_argument('--filedir', type=str, default="",
                        help='filename to output result (default: )')

    parser.add_argument('--metric', type=str, default="mse",
                        help='evaluation metric to be used')
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # change here to minimize the full dataset size
    dataset = PygGraphPropPredDataset(root=args.rootdir,name = args.dataset)
    num_node_attr = dataset[0].x.shape[-1]
    print("num node attr... ",num_node_attr)
    num_edge_attr = dataset[0].edge_attr.shape[-1]

    num_landmarks = dataset.num_landmarks()

    split_idx= dict()

    
    print('Using random split')
    perm = torch.randperm(len(dataset))
    #  num_train, num_valid, num_test = 7047,1,1
    # num_train, num_valid, num_test =int(len(dataset)*0.75), int(len(dataset)*0.15), int(len(dataset)*0.1)
    num_train, num_valid, num_test =int(len(dataset)*args.split_ratio), int(len(dataset)*(1-args.split_ratio)/2), int(len(dataset)*(1-args.split_ratio)/2)
    split_idx['train'] = perm[:num_train]
    split_idx['valid'] = perm[num_train:num_train+num_valid]
    split_idx['test'] = perm[num_train+num_valid:]


    ### automatic evaluator. takes dataset name as input
    #  Set this either to MSE or R2
    # include keyword arg squared = True if mse otherwise rmse
    eval_metric = args.metric
    evaluator = Evaluator(type=eval_metric,num_landmarks=num_landmarks)

    print("landmarks ",num_landmarks)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.gnn == 'gin':
        model = GNN(num_landmarks = num_landmarks, num_layer = args.num_layer, input_node_dim=num_node_attr,input_edge_dim = num_edge_attr, gnn_type = 'gin', emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(num_landmarks = num_landmarks, num_layer = args.num_layer, input_node_dim=num_node_attr,input_edge_dim = num_edge_attr, gnn_type = 'gin', emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(num_landmarks = num_landmarks, num_layer = args.num_layer, input_node_dim=num_node_attr,input_edge_dim = num_edge_attr, gnn_type = 'gcn', emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(num_landmarks = num_landmarks, num_layer = args.num_layer, input_node_dim=num_node_attr,input_edge_dim = num_edge_attr, gnn_type = 'gcn', emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f'#Params: {sum(p.numel() for p in model.parameters())}')

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer)

        print('Evaluating...')

        if epoch is args.epochs:
            train_perf = eval(model, device, train_loader, evaluator,isLogging=True,filedir=args.filedir,split="train",split_idx=split_idx["train"])
            valid_perf = eval(model, device, valid_loader, evaluator,isLogging=True,filedir=args.filedir,split="valid",split_idx=split_idx["valid"])
            test_perf = eval(model, device, test_loader, evaluator,isLogging=True,filedir=args.filedir,split="test",split_idx=split_idx["test"])
        else:
            train_perf = eval(model, device, train_loader, evaluator)
            valid_perf = eval(model, device, valid_loader, evaluator)
            test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf)
        valid_curve.append(valid_perf)
        test_curve.append(test_perf)


    
    best_val_epoch = np.argmax(np.array(valid_curve))
    best_train = max(train_curve)
    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    
    result_dict = {'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}
    torch.save(result_dict, os.path.join(args.filedir,"metric_output.pt"))


if __name__ == "__main__":
    main()