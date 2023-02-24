import numpy as np 
import torch 

def auprc(W_gt, model):
    return model.get_auprc(W_gt)

def nll(datasets, intervention_sets, predicted=True, predictor=None, gt=False, gen_model=None):
    if predicted:
        nll = predictor.predictLikelihood(datasets, intervention_sets)
    elif gt:
        nll = gen_model.predictLikelihood(datasets, intervention_sets)
    
    n_nodes = datasets[0].shape[1]

    holdout_nll = np.mean(nll) / n_nodes
    return holdout_nll 

def expected_mean(datasets, latents, intervention_sets, predictor, n_iter=50, init_provided=False):
    predictions = predictor.predict(datasets, latents, intervention_sets, n_iter=n_iter, init_provided=False, x_init=datasets)

    expected_values = list()
    for prediction in predictions:
        expected_values.append(prediction.mean(axis=0))
    
    return expected_values
            

