import numpy as np 
import os 
import argparse 
import pandas as pd
import torch 
import telegram_send as ts

from models.resblock_trainer import resflow_train_test_wrapper

from utils import * 

from baselines.llc import LLCClassWrapper
from baselines.notears.linear import NotearsClassWrapper


def saveArgs(args):
    if not os.path.exists(args.dop):
        os.makedirs(args.dop)

    with open(os.path.join(args.dop, 'config.txt'), mode='w') as f:
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
        f.write('dip = {}\n'.format(args.dip))    
        f.write('dop = {}\n'.format(args.dop))
        f.write('model = {}\n'.format(args.model))    
        f.write('n_factors = {}\n'.format(args.n_factors))
        f.write('lg = {}\n'.format(args.lg))

def run_train_iteration(args, n_nodes, train_datasets, train_intervention_sets, W):
    # possible model choices = ['resflow', 'llc']
    
    if args.model == 'nodags':

        # training_dataset = [experimentDataset(data, intervention_sets[i]) for i, data in enumerate(datasets)]
        # train_dataloader = [DataLoader(training_data, batch_size=args.batch_size) for training_data in training_dataset]
        model = resflow_train_test_wrapper(n_nodes=n_nodes,
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
                                n_hidden=args.n_hidden,
                                n_factors=args.n_factors,
                                dag_input=args.dag_input
                                )
        # model.train(datasets, intervention_sets)
    elif args.model == 'llc':
        model = LLCClassWrapper()
    elif args.model == 'notears':
        model = NotearsClassWrapper(lambda1=0.01)

    model.train(train_datasets, train_intervention_sets, batch_size=args.batch_size)

    baseline, area = model.get_auprc(W)

    return baseline, area, model


def experiment(args):
    if not args.lg:
        n_nodes_list = [5, 10, 20]
        area_data = pd.DataFrame(columns=['5', '10', '20'])
        baseline_data = pd.DataFrame(columns=['5', '10', '20'])
        pred_nll_data = pd.DataFrame(columns=['5', '10', '20'])

        intervention_sizes = [2, 3, 4]
    else:
        n_nodes_list = [200]
        area_data = pd.DataFrame(columns=['200'])
        baseline_data = pd.DataFrame(columns=['200'])
        pred_nll_data = pd.DataFrame(columns=['200'])

        intervention_sizes = [2, 3, 4]


    for i, n_nodes in enumerate(n_nodes_list):
        print("# nodes: {}".format(n_nodes))
        area_n_list = list()
        baseline_n_list = list()
        pred_nll_list = list()

        for n in range(args.n):
            # Loading the training data
            datapath = os.path.join(args.dip, 'training_data/nodes_{}/graph_{}'.format(n_nodes, n))
            intervention_sets = np.load(os.path.join(datapath, 'intervention_sets.npy'), allow_pickle=True)
            datasets = [np.load(os.path.join(datapath, 'dataset_{}.npy'.format(t))) for t in range(len(intervention_sets))]
            W = np.load(os.path.join(datapath, 'weights.npy'))
            
            # Training the model
            baseline, area, model = run_train_iteration(args, n_nodes, datasets, intervention_sets, np.abs(W) > 0)
            baseline_n_list.append(baseline)
            area_n_list.append(area)

            # Loading the testing data
            pred_nll = 0
            for n_targets in intervention_sizes:
                val_datapath = os.path.join(args.dip, 'validation_data/nodes_{}/graph_{}/n_inter_{}'.format(n_nodes, n, n_targets))
                val_datasets = [np.load(os.path.join(val_datapath, 'dataset_{}.npy'.format(t))) for t in range(10)]
                val_intervention_sets = np.load(os.path.join(val_datapath, 'intervention_sets.npy'), allow_pickle=True)

                # Compute the holdout likelihood
                predicted_nll_list = model.predictLikelihood(val_datasets, val_intervention_sets)
                pred_nll += np.mean(predicted_nll_list)
            
            pred_nll = pred_nll / len(intervention_sizes)
            pred_nll_list.append(pred_nll)

        ts.send(messages=["val_st_{}--# nodes: {}, avg_area: {}".format(args.id, n_nodes, np.mean(area_n_list))])
        area_data[str(n_nodes)] = area_n_list 
        baseline_data[str(n_nodes)] = baseline_n_list
        pred_nll_data[str(n_nodes)] = pred_nll_list

    if not os.path.exists(args.dop):
        os.makedirs(args.dop)

    area_data.to_csv(os.path.join(args.dop, 'area_data.csv'))
    baseline_data.to_csv(os.path.join(args.dop, 'baseline_data.csv'))
    pred_nll_data.to_csv(os.path.join(args.dop, 'pred_nll_data.csv'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--id", type=int, default=1)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--exp_dens', type=int, default=1)
    parser.add_argument('--n_samples', type=int, default=5000)
    parser.add_argument('--gen_model', type=str, choices=['lin' ,'nnl'], default='lin')
    parser.add_argument('--mode', type=str, choices=['indiv-node', 'no-constraint', 'sat-pair-condition'], default='indiv-node')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--l1_reg', action='store_true', default=False)
    parser.add_argument('--lambda_c', type=float, default=1e-2)
    parser.add_argument('--n_lip_iter', type=int, default=5)
    parser.add_argument('--fun_type', type=str, choices=['mul-mlp', 'lin-mlp', 'nnl-mlp', 'gst-mlp'], default='mul-mlp')
    parser.add_argument('--lip_const', type=float, default=0.9)
    parser.add_argument('--act_fun', type=str, choices=['tanh', 'relu', 'sigmoid', 'selu', 'gelu'], default='tanh')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--optim', type=str, choices=['adam', 'sgd'], default='sgd')
    parser.add_argument('--store_fig', action='store_true', default=False)
    parser.add_argument('--inline', action='store_true', default=False)
    parser.add_argument('--upd_lip', action='store_true', default=False)
    parser.add_argument('--dip', type=str, default='../training_data')
    parser.add_argument('--dop', type=str, default='validation/data/baseline/llc')
    parser.add_argument('--n_hidden', type=int, default=1)
    parser.add_argument('--model', type=str, choices=['nodags', 'llc', 'notears'], default='nodags')
    parser.add_argument('--n_factors', type=int, default=10)
    parser.add_argument('--lg', action='store_true', default=False) # for large graphs
    parser.add_argument('--dag-input', action='store_true', default=False) # add --dag-input to CL arguments when the input data is dag
    parser.add_argument('--lin-logdet', action='store_true', default=False) # add --lin-logdet when you want to use explicit logdet calculation in the linear case. 

    args = parser.parse_args()
    print("GPU: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("Device Name: {}".format(torch.cuda.get_device_name(0)))
    saveArgs(args)

    ts.send(messages=["Started Validation experiment\nID: {}\nModel: {}\ndip: {}".format(args.id, args.model, args.dip)])

    experiment(args)
    




