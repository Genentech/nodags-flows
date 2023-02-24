import os
import argparse
from re import L 
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader 
from matplotlib import pyplot as plt 

from lib.learnStructure import contractResFlow

from data.generateDataset import Dataset 
from data.torchDataset import experimentDataset 
from utils import *

def saveArgs(args, dataPath):
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)

    with open(os.path.join(dataPath, 'config.txt'), mode='w') as f:
        f.write('n = {}\n'.format(args.n))
        f.write('exp_dens = {}\n'.format(args.exp_dens))
        f.write('n_samples = {}\n'.format(args.n_samples))
        f.write('mode = {}\n'.format(args.mode))
        f.write('batch_size = {}\n'.format(args.batch_size))
        f.write('gen_model = {}\n'.format(args.gen_model))
        f.write('l1_reg = {}\n'.format(args.l1_reg))
        f.write('lambda_c = {}\n'.format(args.lambda_c))
        f.write('n_lip_iter = {}\n'.format(args.n_lip_iter))
        f.write('fun_type = {}\n'.format(args.fun_type))
        f.write('n_hidden = {}\n'.format(args.n_hidden))
        f.write('lip_const = {}\n'.format(args.lip_const))
        f.write('act_fun = {}\n'.format(args.act_fun))
        f.write('lr = {}\n'.format(args.lr))
        f.write('epochs = {}\n'.format(args.epochs))
        f.write('optim = {}\n'.format(args.optim))
        f.write('store_fig = {}\n'.format(args.store_fig))
        f.write('inline = {}\n'.format(args.inline))
        f.write('upd_lip = {}\n'.format(args.upd_lip))
        f.write('dp = {}\n'.format(args.dp))        

def run_iteration(args, n_nodes, n_exp, exp_n=1):
    dataset_gen = Dataset(n_nodes=n_nodes, 
                        expected_density=args.exp_dens, 
                        n_samples=args.n_samples, 
                        n_experiments=n_exp, 
                        mode=args.mode, 
                        sem_type=args.gen_model)
    dataset = dataset_gen.generate()
    graph = dataset_gen.graph
    generative_model = dataset_gen.gen_model

    training_dataset = [experimentDataset(data, dataset_gen.targets[i]) for i, data in enumerate(dataset)]
    train_dataloader = [DataLoader(training_data, batch_size=args.batch_size) for training_data in training_dataset]

    resblock = contractResFlow(n_nodes=n_nodes,
                               batch_size=args.batch_size,
                               l1_reg=args.l1_reg,
                               lambda_c=args.lambda_c,
                               n_lip_iter=args.n_lip_iter,
                               fun_type=args.fun_type,
                               lip_const=args.lip_const,
                               act_fun=args.act_fun,
                               lr=args.lr,
                               epochs=args.epochs, 
                               optim=args.optim,
                               v=False,
                               inline=args.inline,
                               upd_lip=args.upd_lip,
                               full_input=False,
                               n_hidden=args.n_hidden
                               )
    loss, logdetgrad = resblock.train(train_dataloader, training_dataset, return_loss=True)

    if args.store_fig:
        resblock.store_figure(graph, generative_model, gid=exp_n)

    if args.gen_model == 'lin':
        W = np.abs(generative_model.weights) > 0
    elif args.gen_model == 'nnl':
        W = np.abs(get_adj_from_single_func(generative_model.f)) > 0
    baseline, area = resblock.get_auprc(W)
    
    return loss, logdetgrad, area, baseline

def experiment(args, data_output_path='experiments/data', store_data=True, return_data=False):
    n_nodes_list = [5, 10, 20]
    if args.mode == 'indiv-node':
        n_exp_list = [5, 10, 20]
    else:
        n_exp_list = [7, 12, 22]
    
    loss_data = pd.DataFrame(columns=['5', '10', '20'])
    logdetgrad_data = pd.DataFrame(columns=['5', '10', '20'])
    area_data = pd.DataFrame(columns=['5', '10', '20'])
    baseline_data = pd.DataFrame(columns=['5', '10', '20'])

    for i, n_nodes in enumerate(n_nodes_list):
        print("# nodes: {}, # experiments: {}".format(n_nodes, n_exp_list[i]))
        loss_n_list = list()
        logdetgrad_n_list = list()
        area_n_list = list()
        baseline_n_list = list()
        for n in range(args.n):
            loss, logdetgrad, area, baseline = run_iteration(args, n_nodes, n_exp_list[i], exp_n = n)
            print()
            loss_n_list.append(loss)
            logdetgrad_n_list.append(logdetgrad)
            area_n_list.append(area)
            baseline_n_list.append(baseline)
        print()
        print()
        loss_data[str(n_nodes)] = loss_n_list
        logdetgrad_data[str(n_nodes)] = logdetgrad_n_list
        area_data[str(n_nodes)] = area_n_list
        baseline_data[str(n_nodes)] = baseline_n_list

    # bplot_data = [np.array(area_data[str(n_nodes)]) for n_nodes in n_nodes_list]
    # labels = [str(n_nodes) for n_nodes in n_nodes_list]

    # _ = plt.boxplot(bplot_data, labels=labels)
    # plt.xlabel('# nodes')
    # plt.ylabel('AUROC')
    
    # figname = args.fun_type
    # if args.upd_lip:
    #     figname += "_upd_lip"
    # figname += "_bplot.png"
    # plt.savefig(os.path.join(data_output_path, figname))

    if store_data:
        loss_data.to_csv(os.path.join(data_output_path, 'loss_data.csv'))
        logdetgrad_data.to_csv(os.path.join(data_output_path, 'logdetgrad_data.csv'))
        area_data.to_csv(os.path.join(data_output_path, 'area_data.csv'))
        baseline_data.to_csv(os.path.join(data_output_path, 'baseline_data.csv'))
    if return_data:
        return loss_data, logdetgrad_data, area_data, baseline_data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--exp_dens', type=int, default=1)
    parser.add_argument('--n_samples', type=int, default=5000)
    parser.add_argument('--gen_model', type=str, choices=['lin' ,'nnl'], default='lin')
    parser.add_argument('--mode', type=str, choices=['indiv-node', 'no-constraint', 'sat-pair-condition'], default='indiv-node')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--l1_reg', action='store_true', default=False)
    parser.add_argument('--lambda_c', type=float, default=1e-2)
    parser.add_argument('--n_lip_iter', type=int, default=5)
    parser.add_argument('--fun_type', type=str, choices=['mul-mlp', 'lin-mlp', 'nnl-mlp'], default='mul-mlp')
    parser.add_argument('--lip_const', type=float, default=0.9)
    parser.add_argument('--act_fun', type=str, choices=['tanh', 'relu', 'sigmoid'], default='tanh')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--optim', type=str, choices=['adam', 'sgd'], default='sgd')
    parser.add_argument('--store_fig', action='store_true', default=False)
    parser.add_argument('--inline', action='store_true', default=False)
    parser.add_argument('--upd_lip', action='store_true', default=False)
    parser.add_argument('--dp', type=str, default='experiments/data')
    parser.add_argument('--n_hidden', type=int, default=1)

    args = parser.parse_args()
    print("GPU: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("Device Name: {}".format(torch.cuda.get_device_name(0)))
    saveArgs(args, args.dp)

    experiment(args, data_output_path=args.dp)



