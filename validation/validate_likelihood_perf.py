import numpy as np 
import argparse
import pandas as pd
import os

from data.generateDataset import Dataset
from data.graph import DirectedGraphGenerator
from data.structuralModels import linearSEM, nonlinearSEM

from validation.validate_predictive_performance import resblockPredictor

def createConfig(args):
    if not os.path.exists(args.dp):
        os.makedirs(args.dp)
        print("Folder created since it doesn't exist")

    with open(os.path.join(args.dp, 'config.txt'), mode='w') as f:
        f.write('n = {}\n'.format(args.n_graphs))
        f.write('exp_dens = {}\n'.format(args.exp_dens))
        f.write('n_samples = {}\n'.format(args.n_samples))
        f.write('mode = {}\n'.format(args.mode))
        f.write('gen_model = {}\n'.format(args.mode))
        f.write('n_hidden = {}\n'.format(args.n_hidden))
        f.write('batch_size = {}\n'.format(args.batch_size))
        f.write('l1_reg = {}\n'.format(args.l1_reg))
        f.write('lambda_c = {}\n'.format(args.lambda_c))
        f.write('n_lip_iter = {}\n'.format(args.n_lip_iter))
        f.write('fun_type = {}\n'.format(args.fun_type))
        f.write('lip_const = {}\n'.format(args.lip_const))
        f.write('act_fun = {}\n'.format(args.act_fun))
        f.write('lr = {}\n'.format(args.lr))
        f.write('epochs = {}\n'.format(args.epochs))
        f.write('optim = {}\n'.format(args.optim))
        f.write('inline = {}\n'.format(args.inline))
        f.write('upd_lip = {}\n'.format(args.upd_lip))
        f.write('full_input= {}\n'.format(args.full_input))
        f.write('dp = {}\n'.format(args.dp))        

def run_iter(args, n_nodes=5):
    
    # Generate the graph and the generative model
    graph_generator = DirectedGraphGenerator(nodes=n_nodes, expected_density=args.exp_dens)
    graph = graph_generator()
    if args.gen_model == 'lin':
        gen_model_linear = linearSEM(graph)
    elif args.gen_model == 'nnl':
        gen_model_linear = nonlinearSEM(graph, lip_const=0.99, n_hidden=args.n_hidden)

    # Initialize and train the model
    predictor = resblockPredictor(graph, gen_model_linear, model_args=args)
    predictor.fit(n_samples=args.n_samples, exp_dens=1)

    gt_nll_avg = 0
    pred_nll_avg = 0

    intervention_sizes = [1, 2, 3, 4]
    for n_targets in intervention_sizes:
        dataset_gen = Dataset(n_nodes=n_nodes, expected_density=1, n_samples=args.n_samples, n_experiments=args.n_exp, mode=args.mode, graph_provided=True, graph=graph, gen_model_provided=True, gen_model=gen_model_linear, min_targets=n_targets, max_targets=n_targets)
        datasets = dataset_gen.generate(interventions=True, fixed_interventions=True, return_latents=False)

        predicted_nll = predictor.predictLikelihood(datasets, dataset_gen.targets)
        gt_nll = [gen_model_linear.computeNLL(datasets[i], dataset_gen.targets[i]) for i in range(args.n_exp)]

        obs_set_size = n_nodes - n_targets
        gt_nll_avg += np.mean(gt_nll) / obs_set_size
        pred_nll_avg += np.mean(predicted_nll) / obs_set_size
    
    return gt_nll_avg/len(intervention_sizes), pred_nll_avg/len(intervention_sizes)

def run_experiment(args):

    createConfig(args)

    n_nodes_list = [5, 10, 20]
    gt_nll_df = pd.DataFrame(columns=[str(n) for n in n_nodes_list])
    pred_nll_df = pd.DataFrame(columns=[str(n) for n in n_nodes_list])
    for n_nodes in n_nodes_list:
        print("Number of nodes: {}".format(n_nodes))
        print()
        gt_nll_list = list()
        pred_nll_list = list()
        for _ in range(args.n_graphs):
            gt_nll, pred_nll = run_iter(args, n_nodes)
            gt_nll_list.append(gt_nll)
            pred_nll_list.append(pred_nll)
        gt_nll_df[str(n_nodes)] = gt_nll_list 
        pred_nll_df[str(n_nodes)] = pred_nll_list

    gt_nll_df.to_csv(os.path.join(args.dp, 'gt_nll_df.csv'))
    pred_nll_df.to_csv(os.path.join(args.dp, 'pred_nll_df.csv')) 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_graphs', type=int, default=10)
    parser.add_argument('--exp_dens', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=5000)
    parser.add_argument('--gen_model', type=str, choices=['lin' ,'nnl'], default='lin')
    parser.add_argument('--n_exp', type=int, default=10)
    parser.add_argument('--mode', type=str, choices=['indiv-node', 'no-constraint', 'sat-pair-condition'], default='no-constraint')
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
    parser.add_argument('--inline', action='store_true', default=False)
    parser.add_argument('--upd_lip', action='store_true', default=False)
    parser.add_argument('--full_input', action='store_true', default=False)
    parser.add_argument('--dp', type=str, default='experiments/data')
    parser.add_argument('--n_hidden', type=int, default=1)

    args = parser.parse_args()  
    
    run_experiment(args)
