import os 
import argparse
from re import L 
import torch 
import torch.nn as nn 
import numpy as np 
import math

from utils import *

from torch.utils.data import DataLoader 
from torch.optim import Adam, SGD 

from datagen.generateDataset import Dataset
from datagen.graph import DirectedGraphGenerator
from datagen.structuralModels import linearSEM
from datagen.torchDataset import experimentDataset

def standard_normal_logprob(z, noise_scales):
    logZ = -0.5 * torch.log(2 * math.pi * (noise_scales**2))
    return logZ - z.pow(2) / (2 * (noise_scales**2))

def computeNLL(lat, observed_set, logdetgrad, noise_scale=0.5):
    logpe = standard_normal_logprob(lat[:, observed_set], noise_scales=noise_scale).sum(1, keepdim=True)
    logpx = logpe + logdetgrad
    return -torch.mean(logpx).detach().cpu().numpy()

def dag_constraint(W):
    return torch.trace(torch.matrix_exp(W * W)) - W.shape[0]

def compute_loss(model, x, intervention_set, l1_reg=True, lambda_c=1e-2, enforce_dag=True, lambda_dag=1e-2, noise_scale=0.5):
    e = model(x, intervention_set)
    observed_set = np.setdiff1d(np.arange(x.shape[1]), intervention_set)
    lat_var = torch.exp(noise_scale[intervention_set])
    U = torch.zeros(x.shape[1], x.shape[1], device=x.device)
    U[observed_set, observed_set] = 1

    W = model.causal_mech.weight
    I = torch.eye(x.shape[1], device=x.device)

    likeli_loss = -0.5*torch.log((e[:, observed_set]**2).sum(dim=1)).sum() + torch.log(torch.abs(torch.det(I - U @ W)))
    # logpe = standard_normal_logprob(e[:, observed_set], noise_scales=lat_var).sum(1, keepdim=True)
    # likeli_loss = logpe.mean() + torch.log(torch.abs(torch.det(I - U @ W)))
    loss = -1 * likeli_loss
    h_w = dag_constraint(W)
    if l1_reg:
        loss += lambda_c * W.abs().sum()
    if enforce_dag:
        loss += lambda_dag * h_w
    
    return loss, h_w

class SEM(nn.Module):
    def __init__(self, n_nodes, bias=False):
        super(SEM, self).__init__()
        self.n_nodes = n_nodes
        self.causal_mech = nn.Linear(in_features=n_nodes, out_features=n_nodes, bias=bias)
    
    def forward(self, x, intervention_set):
        observed_set = np.setdiff1d(np.arange(self.n_nodes), intervention_set)
        U = torch.zeros(x.shape[1], x.shape[1], device=x.device)
        U[observed_set, observed_set] = 1

        return x - self.causal_mech(x) @ U

class GolemClassWrapper:
    def __init__(self, 
            n_nodes,
            enforce_dag=True,
            var=None,
            noise_scale=0.5
        ):

        self.n_nodes = n_nodes
        self.enforce_dag = enforce_dag
        self.sem = SEM(self.n_nodes)
        self.threshold_value = None

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        self.sem = self.sem.to(self.device)

        if var == None:
            self.var = nn.Parameter(0.5*torch.ones(self.n_nodes).float().to(self.device))
        else:
            var = noise_scale

    
    def train(self, datasets, intervention_sets, batch_size=64, l1_reg=True, lambda_c=1e-2, epochs=100, optim='adam', lr=1e-3, lambda_dag=1e-2, inline=False):

        training_dataset = [experimentDataset(data, intervention_sets[i]) for i, data in enumerate(datasets)]
        train_dataloader = [DataLoader(training_data, batch_size=batch_size) for training_data in training_dataset]
        if inline:
            print("Starting Training")
        if optim == 'sgd':
            optimizer = SGD(self.sem.parameters(), lr=lr)
        else:
            optimizer = Adam(self.sem.parameters(), lr=lr)

        loss_rep = 0
        count = 0

        for epoch in range(epochs):
            for exp, dataloader in enumerate(train_dataloader):
                for batch, x in enumerate(dataloader):
                    optimizer.zero_grad()

                    intervention_set = intervention_sets[exp]
                    x = x.float().to(self.device)
                    loss, h_w = compute_loss(self.sem, x, intervention_set, l1_reg=l1_reg, lambda_c=lambda_c, enforce_dag=self.enforce_dag, lambda_dag=lambda_dag, noise_scale=self.var)

                    loss_rep += loss.item()
                    count += 1 

                    if batch % 20 == 0:
                        if inline:
                            loss_rep /= count 
                            count = 0
                            print("Exp: {}/{}, Epoch: {}/{}, Batch: {}/{}, Loss: {}, h(W): {}".format(exp+1, len(train_dataloader), epoch+1, epochs, batch, len(dataloader), loss_rep, h_w), end="\r", flush=True)
                            loss_rep = 0

                    loss.backward()
                    optimizer.step()
        
        # print()
        # self.threshold()
    
    def threshold(self):
        W = self.get_adjacency()

        def acyc(t):
            return float(is_acyclic(np.abs(W) >= t)) - 0.5

        thresh = bisect(acyc, 0, 5)
        assert acyc(thresh) > 0

        self.threshold_value = thresh

        W_t = (W >= thresh) * W 
        self.sem.causal_mech.weight.data = torch.from_numpy(W_t.T).float().to(self.device)
    
    def get_adjacency(self):
        return self.sem.causal_mech.weight.detach().cpu().numpy().T
    
    def get_auprc(self, W_gt, n_points=50):
        baseline, area = compute_auprc(W_gt, np.abs(self.get_adjacency()), n_points=n_points)
        return baseline, area 

    def get_shd(self, W_gt):
        W_est = np.abs(self.get_adjacency()) > 0
        shd, _ = compute_shd(W_gt, W_est)
        return shd
    
    def forwardPass(self, datasets):
        predictions = list()
        for data in datasets:
            data_t = torch.tensor(data).float().to(self.device)
            pred = self.sem(data_t, intervention_set=[None])
            predictions.append(pred.detach().cpu().numpy())
        
        return predictions
    
    def predictLikelihood(self, datasets, intervention_sets, noise_scale=0.5):
        likelihood_list = list() 
        for dataset, intervention_set in zip(datasets, intervention_sets):
            data_t = torch.tensor(dataset).float().to(self.device)
            lat = self.sem(data_t, intervention_set)
            observed_set = np.setdiff1d(np.arange(self.n_nodes), intervention_set)
            W = self.sem.causal_mech.weight
            U = torch.zeros(data_t.shape[1], data_t.shape[1], device=data_t.device)
            U[observed_set, observed_set] = 1

            I = torch.eye(data_t.shape[1], device=data_t.device)

            logdetgrad = torch.log(torch.abs(torch.det(I - U @ W)))
            lat_var = torch.exp(self.var[observed_set])
            nll = computeNLL(lat, observed_set, logdetgrad, noise_scale=lat_var)
            likelihood_list.append(nll.item()/self.n_nodes)
        
        return likelihood_list

    def predict(self, latents, intervention_sets, x_inits):
        pred_list = list()
        n_nodes = self.n_nodes
        W = self.get_adjacency()
        for latent, intervention_set, x_init in zip(latents, intervention_sets, x_inits):
            observed_set = np.setdiff1d(np.arange(n_nodes), intervention_set)
            U, I = np.zeros((n_nodes, n_nodes)), np.zeros((n_nodes, n_nodes))
            U[observed_set, observed_set] = 1
            I[intervention_set, intervention_set] = 1
            data_pred = (latent @ U + x_init @ I) @ np.linalg.inv(np.eye(n_nodes) - U @ W.T).T
            pred_list.append(data_pred)
        
        return pred_list
    
    def predictConditionalMean(self, datasets, intervention_sets, noise_scale=0.5):
        latents = [np.random.randn(datasets[i].shape[0], datasets[i].shape[1]) * noise_scale for i in range(len(datasets))]
        pred_list = self.predict(latents, intervention_sets, x_inits=datasets)

        return [pred.mean(axis=0) for pred in pred_list]

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    graph_gen = DirectedGraphGenerator(nodes=5, expected_density=2, enforce_dag=True)
    graph = graph_gen()

    gen_model = linearSEM(graph, contractive=True)
    data_gen = Dataset(n_nodes=5, 
                    expected_density=1,
                    n_samples=5000,
                    n_experiments=5,
                    graph_provided=True,
                    graph=graph,
                    gen_model_provided=True,
                    gen_model=gen_model,
                    contractive=False)

    datasets = data_gen.generate(interventions=False)
    intervention_sets = [[None]]

    golem_wrapper = GolemClassWrapper(n_nodes=5, enforce_dag=True)

    golem_wrapper.train(datasets, 
        intervention_sets, 
        batch_size=64, 
        inline=True, 
        lambda_c=1, 
        lambda_dag=10, 
        l1_reg=True)

    print("AUPRC: {}".format(golem_wrapper.get_auprc(np.abs(gen_model.weights) > 0)))

    print(golem_wrapper.get_adjacency())