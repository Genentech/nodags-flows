import numpy as np 
import os 
import argparse 
import pandas as pd
import torch 
import telegram_send as ts

from utils import *

from models.resblock_trainer import resflow_train_test_wrapper
from baselines.llc import LLCClassWrapper
from baselines.notears.linear import NotearsClassWrapper
from baselines.golem import GolemClassWrapper

METRICS = ['auprc', 'val_nll', 'shd', 'cond_mean_err']

def saveConfig(args):
    if not os.path.exists(args.dop):
        os.makedirs(args.dop)

    with open(os.path.join(args.dop, "config.txt"), mode='w') as f:
        f.write("General Arguments\n")
        f.write("--------------------\n")
        f.write("n-graphs: {}\n".format(args.n_graphs))
        f.write("n-nodes: {}\n".format(args.n_nodes))
        f.write("m: {}\n".format(args.m))
        f.write("dag-input: {}\n".format(args.dag_input))
        f.write("dip: {}\n".format(args.dip))
        f.write("dop: {}\n".format(args.dop))
        f.write("inline: {}\n".format(args.inline))
        f.write("\n")
        f.write("Optimization config arguments\n")
        f.write("-------------------------------\n")
        f.write("bs: {}\n".format(args.bs))
        f.write("l1-reg: {}\n".format(args.l1_reg))
        f.write("lam-c: {}\n".format(args.lam_c))
        f.write("lr: {}\n".format(args.lr))
        f.write("epochs: {}\n".format(args.epochs))
        f.write("optim: {}\n".format(args.optim))
        f.write("lam-dag: {}\n".format(args.lam_dag))
        f.write("\n")
        f.write("NODAGs-Flow model config arguments\n")
        f.write("------------------------------------\n") 
        f.write("fun-type: {}\n".format(args.fun_type))
        f.write("n-lip-iter: {}\n".format(args.n_lip_iter))
        f.write("lip-const: {}\n".format(args.lip_const))
        f.write("act-fun: {}\n".format(args.act_fun))
        f.write("upd-lip: {}\n".format(args.upd_lip))
        f.write("n-hidden: {}\n".format(args.n_hidden))
        f.write("lin-logdet: {}\n".format(args.lin_logdet))

def run_iter(args, iter):

    # Load training data
    datapath = os.path.join(args.dip, 'training_data/nodes_{}/graph_{}'.format(args.n_nodes, iter))
    intervention_sets = np.load(os.path.join(datapath, 'intervention_sets.npy'), allow_pickle=True)
    datasets = [np.load(os.path.join(datapath, 'dataset_{}.npy'.format(t))) for t in range(len(intervention_sets))]
    W = np.load(os.path.join(datapath, 'weights.npy'))

    # Initialize the mode and train on data
    if args.m == 'nodags':
        model = resflow_train_test_wrapper(n_nodes=args.n_nodes,
                                batch_size=args.bs,
                                l1_reg=args.l1_reg,
                                lambda_c=args.lam_c,
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
                                n_factors=5,
                                dag_input=args.dag_input,
                                thresh_val=args.thresh
                                )
        model.train(datasets, intervention_sets, batch_size=args.bs)
    elif args.m == 'llc':
        model = LLCClassWrapper(thresh_val=args.thresh)
        model.train(datasets, intervention_sets)
    elif args.m == 'notears':
        model = NotearsClassWrapper(lambda1=args.lam_c)
        model.train(datasets, intervention_sets, batch_size=args.bs)
    elif args.m == 'golem':
        model = GolemClassWrapper(n_nodes=args.n_nodes, enforce_dag=True)
        model.train(datasets, intervention_sets, batch_size=args.bs, inline=args.inline, lambda_c=args.lam_c, lambda_dag=args.lam_dag, l1_reg=args.l1_reg)
    
    _, auprc = model.get_auprc(np.abs(W) > 0)
    
    # Evaluating the predictive power of the proposed model 
    intervention_sizes = [2, 3]
    pred_nll = 0
    cond_mean_err = 0
    for n_targets in intervention_sizes:
        val_datapath = os.path.join(args.dip, 'validation_data/nodes_{}/graph_{}/n_inter_{}'.format(args.n_nodes, iter, n_targets))
        val_intervention_sets = np.load(os.path.join(val_datapath, "intervention_sets.npy"), allow_pickle=True)
        val_datasets = [np.load(os.path.join(val_datapath, 'dataset_{}.npy'.format(t))) for t in range(len(val_intervention_sets))]    

        predicted_nll_list = model.predictLikelihood(val_datasets, val_intervention_sets) 
        pred_nll += np.mean(predicted_nll_list)

        predictions = model.forwardPass(val_datasets)
        er_cond_mean = np.mean([np.abs(dataset - cond_mean).mean() for dataset, cond_mean in zip(val_datasets, predictions)])
        cond_mean_err += er_cond_mean

    cond_mean_err /= len(intervention_sizes)
    pred_nll /= len(intervention_sizes)

    # Threshold the learn't graph
    model.threshold()

    shd = model.get_shd(np.abs(W) > 0)

    return auprc, pred_nll, shd, cond_mean_err
    
def run_benchmark(args):
    
    benchmark_data = pd.DataFrame(columns=['metrics'] + [n for n in range(args.n_graphs)])
    benchmark_data['metrics'] = METRICS

    for iter in range(args.n_graphs):
        auprc, val_nll, shd, cond_mean_err = run_iter(args, iter)
        graph_data = [auprc, val_nll, shd, cond_mean_err]
        benchmark_data[iter] = graph_data
    
    if not os.path.exists(args.dop):
        os.makedirs(args.dop)
    
    benchmark_data.to_csv(os.path.join(args.dop, 'synth_benchmark_data.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("-n-graphs", type=int, default=3)
    parser.add_argument("-n-nodes", type=int, default=20)
    parser.add_argument("-m", type=str, choices=["nodags", "llc", "notears", "golem"], default='nodags')
    parser.add_argument("--dag-input", action="store_true", default=False)
    parser.add_argument("-dip", type=str, default="../training_data/linear_data")
    parser.add_argument("-dop", type=str, default="results/exp_1")
    parser.add_argument("--inline", action="store_true", default=False)
    parser.add_argument("--thresh", type=float, default=0.05)

    # Optimization config arguments
    parser.add_argument("-bs", type=int, default=512)
    parser.add_argument("--l1-reg", action="store_true", default=False)
    parser.add_argument("-lam-c", type=float, default=1e-2)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-epochs", type=int, default=100)
    parser.add_argument("-optim", type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument("-lam-dag", type=float, default=500)
    
    # NODAGs-Flow model config arguments
    parser.add_argument("-fun-type", type=str, choices=['lin-mlp', 'mul-mlp', 'nnl-mlp', 'gst-mlp'], default='lin-mlp')
    parser.add_argument("-n-lip-iter", type=int, default=5)
    parser.add_argument("-lip-const", type=float, default=0.99)
    parser.add_argument("-act-fun", type=str, choices=["tanh", "relu", "sigmoid", "selu", "gelu", "none"], default="none")
    parser.add_argument("--upd-lip", action="store_true", default=False)
    parser.add_argument("-n-hidden", type=int, default=0)
    parser.add_argument("--lin-logdet", action="store_true", default=False)

    args = parser.parse_args()
    saveConfig(args)

    print("Running synthetic benchmark on {}".format(args.m))
    run_benchmark(args)

