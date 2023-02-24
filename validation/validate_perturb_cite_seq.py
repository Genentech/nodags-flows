import numpy as np 
import os
import argparse 
import pandas as pd

from models.resblock_trainer import resflow_train_test_wrapper

def run_perturb_seq_exp(args):

    # Loading the training data
    train_path = os.path.join(args.dip, "training_data")

    obs_data_mean = np.load(os.path.join(args.dip, "obs_mean.npy"))

    training_interventions = np.load(os.path.join(train_path, "intervention_sets.npy"))
    training_datasets = [np.load(
        os.path.join(train_path, "dataset_{}.npy".format(t))
    ) for t in range(len(training_interventions))]

    # centering the data
    training_datasets = [td - obs_data_mean for td in training_datasets]

    # Training the model
    n_nodes = training_datasets[0].shape[1]
    nodags_wrapper = resflow_train_test_wrapper(
        n_nodes=n_nodes,
        batch_size=64,
        l1_reg=True,
        lambda_c=args.rc,
        n_lip_iter=5,
        fun_type=args.fun_type,
        act_fun=args.act_fun,
        epochs=args.epochs,
        lr=args.lr,
        lip_const=args.lc,
        optim='adam',
        inline=args.inline,
        n_hidden=args.nh,
        lin_logdet=False,
        dag_input=args.dag_input,
        thresh_val=0.05,
        upd_lip=True,
        centered=True
    )


    nodags_wrapper.train(training_datasets, training_interventions, batch_size=args.bs)

    # Loading validation data
    valid_path = os.path.join(args.dip, "validation_data")

    validation_interventions = np.load(os.path.join(valid_path, "intervention_sets.npy"))
    validation_datasets = [
        np.load(os.path.join(valid_path, "dataset_{}.npy".format(t))) for t in range(len(validation_interventions))
    ]

    # centering the data
    validation_datasets = [vd - obs_data_mean for vd in validation_datasets]

    # Measuring holdout likelihood and Conditional mean
    val_nll = np.mean(
        nodags_wrapper.predictLikelihood(validation_datasets, validation_interventions)
    )

    cm_list = nodags_wrapper.predictConditionalMean(validation_datasets, validation_interventions)
    gt_cm_list = [dat.mean(axis=0) for dat in validation_datasets]
    cm_err = np.mean([np.abs(gt - cm) for gt, cm in zip(gt_cm_list, cm_list)])
    
    benchmark_data = pd.DataFrame(columns=['metrics', 'results'])
    benchmark_data['metrics'] = ['val_nll', 'val_mae']
    benchmark_data['results'] = [val_nll, cm_err]

    if not os.path.exists(args.dop):
        os.makedirs(args.dop)
    
    benchmark_data.to_csv(os.path.join(args.dop, "benchmark_data.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--bs", type=int, default=1024)
    parser.add_argument("--rc", type=float, default=1e-1)
    parser.add_argument("--fun-type", type=str, choices=["lin-mlp", "nnl-mlp", "gst-mlp"], default='gst-mlp')
    parser.add_argument("--act-fun", type=str, default='none')
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lc", type=float, default=0.99)
    parser.add_argument("--inline", action="store_true", default=False)
    parser.add_argument("--nh", type=int, default=0)
    parser.add_argument('--dag-input', action='store_true', default=False)
    parser.add_argument('--dip', type=str, default='../perturb_cite_seq_data/nodags_data/control')
    parser.add_argument('--dop', type=str, default="results/perturb_seq")
    
    args = parser.parse_args()

    run_perturb_seq_exp(args)
