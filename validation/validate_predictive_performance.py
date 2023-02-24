import numpy as np 
import pandas as pd
import torch 
import math 
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt 

from lib.learnStructure import contractResFlow

from data.generateDataset import Dataset
from data.torchDataset import experimentDataset

def standard_normal_logprob(z, noise_scale=0.5):
    logZ = -0.5 * math.log(2 * math.pi * noise_scale**2)
    return logZ - z.pow(2) / (2 * noise_scale**2)

def computeNLL(latent, intervention_set, logdetgrad):
    observed_set = np.setdiff1d(np.arange(latent.shape[1]), intervention_set)
    logpe = standard_normal_logprob(latent[:, observed_set]).sum(1, keepdim=True)
    logpx = logpe + logdetgrad
    return -torch.mean(logpx).detach().cpu().numpy()


class resblockPredictor:
    def __init__(self, graph, generative_model, model_provided=False, trained_model=None, model_args=None):
        self.graph = graph
        self.generative_model = generative_model 
        self.model_args = model_args
        if model_provided:
            self.model = trained_model 
        else:
            self.model = contractResFlow(n_nodes=len(graph.nodes),
                                    batch_size=model_args.batch_size,
                                    l1_reg=model_args.l1_reg,
                                    lambda_c=model_args.lambda_c,
                                    n_lip_iter=model_args.n_lip_iter,
                                    fun_type=model_args.fun_type,
                                    lip_const=model_args.lip_const,
                                    act_fun=model_args.act_fun,
                                    lr=model_args.lr,
                                    epochs=model_args.epochs, 
                                    optim=model_args.optim,
                                    v=False,
                                    inline=model_args.inline,
                                    upd_lip=model_args.upd_lip,
                                    full_input=model_args.full_input, 
                                    n_hidden=model_args.n_hidden)


    def fit(self, n_samples, exp_dens, return_loss=False, return_data=False):
        training_dataset_gen = Dataset(n_nodes=len(self.graph.nodes), 
                          expected_density=exp_dens,
                          n_samples=n_samples,
                          n_experiments=len(self.graph.nodes),
                          mode='indiv-node', 
                          graph_provided=True,
                          graph=self.graph,
                          gen_model_provided=True,
                          gen_model=self.generative_model)
        dataset = training_dataset_gen.generate()

        training_dataset = [experimentDataset(data, training_dataset_gen.targets[i]) for i, data in enumerate(dataset)]
        train_dataloader = [DataLoader(training_data, batch_size=self.model_args.batch_size) for training_data in training_dataset]

        if return_loss:
            loss, logdetgrad = self.model.train(train_dataloader, training_dataset, return_loss=True)
            if return_data:
                return loss, logdetgrad, training_dataset_gen.targets, dataset 
            else:
                return loss, logdetgrad
        else:
            self.model.train(train_dataloader, training_dataset)
            if return_data:
                return training_dataset_gen.targets, dataset
    
    def predict(self, latents, intervention_sets, n_iter=10, init_provided=False, x_init=None):
        pred_datasets = list()
        i = 0
        for latent, intervention_set in zip(latents, intervention_sets):
            lat_t = torch.tensor(latent).float().to(self.model.device)
            
            data_pred = self.model.model.predict_from_latent(lat_t, n_iter, intervention_set, init_provided=init_provided, x_init=x_init[i])
            i += 1
            data_pred = data_pred.detach().cpu().numpy()
            pred_datasets.append(data_pred)
        return pred_datasets

    def predictScore(self, datasets, latents, intervention_sets, n_iter=10, init_provided=False, x_init=None):
        pred_datasets = self.predict(latents, intervention_sets, n_iter, init_provided, x_init)
        scores = list()
        for dataset, pred_dataset, intervention_set in zip(datasets, pred_datasets, intervention_sets):
            n = dataset.shape[0]
            d = dataset.shape[1]
            rmse = ((1/(d*n))*np.linalg.norm(dataset - pred_dataset, 'fro')**2)**0.5
            scores.append(rmse)
        
        return pred_datasets, scores
    
    def predictLikelihood(self, datasets, intervention_sets):
        likelihood_list = list()

        for dataset, intervention_set in zip(datasets, intervention_sets):
            data_t = torch.tensor(dataset).float().to(self.model.device)
            latents, logdetgrad = self.model.model(data_t, intervention_set, logdet=True, neumann_grad=False)
            nll = computeNLL(latents, intervention_set, logdetgrad)
            likelihood_list.append(nll.item())
        
        return likelihood_list
