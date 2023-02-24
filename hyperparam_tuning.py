import numpy as np 
import os 
import pandas as pd 
from ax.service.managed_loop import optimize
import argparse
import telegram_send as ts

from models.resblock_trainer import resflow_train_test_wrapper 

from utils import *

def get_arch_and_nodes(config):
    arch = config[:7]
    dag = False 
    if config[8] == 'd':
        dag = True 
        nodes = int(config[12:])
    else:
        nodes = int(config[8:])
    return arch, dag, nodes

def get_datasets(data_dir):
    intervention_sets = np.load(os.path.join(data_dir, "intervention_sets.npy"), allow_pickle=True)
    datasets = [np.load(os.path.join(data_dir, "dataset_{}.npy".format(t)), allow_pickle=True) for t in range(len(intervention_sets))]
    weights = np.abs(np.load(os.path.join(data_dir, "weights.npy"))) > 0

    return (datasets, intervention_sets, weights)

def get_best_parameters(config, train_info, valid_info, lin_logdet=False, auprc_metric=False):
    # train_info = (train_data_sets, train_intervention_sets)
    # valid_info = (valid_data_sets, valid_intervention_sets)
    
    arch, dag, nodes = get_arch_and_nodes(config)
    print(config, arch, dag, nodes)
    print()

    if arch == 'nnl-mlp':
        act_fun = 'selu'
    else:
        act_fun = 'none'

    def train_evaluate(parameterization):
        ts.send(messages=["In train_evaluate"])
        nodags_wrapper = resflow_train_test_wrapper(
            n_nodes=nodes,
            batch_size=1024,
            l1_reg=True,
            lambda_c=parameterization.get("lambda_c"),
            fun_type=arch,
            act_fun=act_fun,
            epochs=100,
            lr=parameterization.get("lr"),
            optim='adam',
            inline=False, 
            n_hidden=parameterization.get("n_hidden"),
            lin_logdet=lin_logdet,
            dag_input=dag,
            n_lip_iter=parameterization.get("n_lip_iter")
        )
    
        nodags_wrapper.train(train_info[0], train_info[1], batch_size=1024)

        if auprc_metric:
            _, auprc = nodags_wrapper.get_auprc(train_info[2])
            return {"auprc": (auprc, 0.0)}

        pred_nll_list = nodags_wrapper.predictLikelihood(valid_info[0], valid_info[1])
        return {"val_nll": (np.mean(pred_nll_list), 0.0)}
    
    if auprc_metric:
        metric_name = 'auprc'
        minimize = False
    else:
        metric_name = "vall_nll"
        minimize = True

    best_parameters, _, _, _ = optimize(
        parameters=[
            {
                "name": "lambda_c",
                "type": "range",
                "bounds": [1e-4, 1e+2],
                "log_scale": True
            },
            {
                "name": "lr",
                "type": "range",
                "bounds": [1e-4, 1e+0],
                "log_scale": True
            },
            {
                "name": "n_hidden",
                "type": "choice",
                "value_type": "int",
                "values": [0, 1, 2]
            },
            {
                "name": "n_lip_iter",
                "type": "choice",
                "value_type": "int",
                "values": [5, 10, 20]
            },
        ],
        experiment_name="hyper_tune",
        objective_name=metric_name,
        evaluation_function=train_evaluate,
        minimize=minimize,
    )

    return best_parameters

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='lin-mlp-10')
    parser.add_argument('--dop', type=str, default='results/part_2/hyper_tune')
    parser.add_argument('--lin-det', action='store_true', default=False)
    parser.add_argument('--metric', type=str, choices=['val-nll', 'auprc'], default='val-nll')

    args = parser.parse_args()

    data_train_path = {
        "lin-mlp-10": "/storage/home/hcoda1/8/msethuraman7/projects/datasets/linear_data/training_data/nodes_10/graph_4",
        "lin-mlp-20": "/storage/home/hcoda1/8/msethuraman7/projects/datasets/linear_data/training_data/nodes_20/graph_4",
        "lin-mlp-dag-10": "/storage/home/hcoda1/8/msethuraman7/projects/datasets/dags/linear_data/training_data/nodes_10/graph_4",
        "lin-mlp-dag-20": "/storage/home/hcoda1/8/msethuraman7/projects/datasets/dags/linear_data/training_data/nodes_20/graph_4",
        "gst-mlp-10": "/storage/home/hcoda1/8/msethuraman7/projects/datasets/linear_data/training_data/nodes_10/graph_4",
        "gst-mlp-20": "/storage/home/hcoda1/8/msethuraman7/projects/datasets/linear_data/training_data/nodes_20/graph_4",
        "gst-mlp-dag-10": "/storage/home/hcoda1/8/msethuraman7/projects/datasets/dags/linear_data/training_data/nodes_10/graph_4",
        "gst-mlp-dag-20": "/storage/home/hcoda1/8/msethuraman7/projects/datasets/dags/linear_data/training_data/nodes_20/graph_4",
        "nnl-mlp-10": "/storage/home/hcoda1/8/msethuraman7/projects/datasets/nonlinear_data_selu/training_data/nodes_10/graph_4",
        "nnl-mlp-20": "/storage/home/hcoda1/8/msethuraman7/projects/datasets/nonlinear_data_selu/training_data/nodes_20/graph_4"
    }

    data_valid_path = {
        "lin-mlp-10": "/storage/home/hcoda1/8/msethuraman7/projects/datasets/linear_data/validation_data/nodes_10/graph_4/n_inter_2",
        "lin-mlp-20": "/storage/home/hcoda1/8/msethuraman7/projects/datasets/linear_data/validation_data/nodes_20/graph_4/n_inter_2",
        "lin-mlp-dag-10": "/storage/home/hcoda1/8/msethuraman7/projects/datasets/dags/linear_data/validation_data/nodes_10/graph_4/n_inter_2",
        "lin-mlp-dag-20": "/storage/home/hcoda1/8/msethuraman7/projects/datasets/dags/linear_data/validation_data/nodes_20/graph_4/n_inter_2",
        "gst-mlp-10": "/storage/home/hcoda1/8/msethuraman7/projects/datasets/linear_data/validation_data/nodes_10/graph_4/n_inter_2",
        "gst-mlp-20": "/storage/home/hcoda1/8/msethuraman7/projects/datasets/linear_data/validation_data/nodes_20/graph_4/n_inter_2",
        "gst-mlp-dag-10": "/storage/home/hcoda1/8/msethuraman7/projects/datasets/dags/linear_data/validation_data/nodes_10/graph_4/n_inter_2",
        "gst-mlp-dag-20": "/storage/home/hcoda1/8/msethuraman7/projects/datasets/dags/linear_data/validation_data/nodes_20/graph_4/n_inter_2",
        "nnl-mlp-10": "/storage/home/hcoda1/8/msethuraman7/projects/datasets/nonlinear_data_selu/validation_data/nodes_10/graph_4/n_inter_2",
        "nnl-mlp-20": "/storage/home/hcoda1/8/msethuraman7/projects/datasets/nonlinear_data_selu/validation_data/nodes_20/graph_4/n_inter_2"
    }

    parameter_list = ['lambda_c', 'lr', 'n_hidden', 'n_lip_iter']
    best_parameters_df = pd.DataFrame(columns=['param']+[args.config])

    best_parameters_df['param'] = parameter_list

    ts.send(messages=["Starting Hyper parameterization tuning\nconifg:{}".format(args.config)])

    # for config in data_train_path:
    train_info = get_datasets(data_train_path[args.config])
    valid_info = get_datasets(data_valid_path[args.config])

    use_auprc=False
    if args.metric == 'auprc':
        use_auprc = True
    
    best_parameters = get_best_parameters(args.config, train_info, valid_info, lin_logdet=args.lin_det, auprc_metric=use_auprc)
    best_parameters_df[args.config] = list(best_parameters.values())

    best_parameters_df.to_csv(os.path.join(args.dop, "{}.csv".format(args.config)))

