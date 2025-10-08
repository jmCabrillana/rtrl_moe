import numpy as np
import pandas as pd
import random
import time
from pathlib import Path

from google.cloud import bigquery
from google.oauth2 import service_account

import torch
from torch.autograd import grad, Variable
from torch.func import jacrev, functional_call

class Param2Vec():
    def __init__(self, model, param_filter='', num_parameters=None):
        """
        get a list of trainable variables
        """
        self.param_list = []
        self.name_list = []
        self.size_list = []
        for name, p in model.named_parameters():
            if p.requires_grad and name.startswith(param_filter):
                self.param_list.append(p)
                self.name_list.append(name)
                self.size_list.append(p.size())

        if num_parameters is not None:
            new_param_list, new_name_list, new_size_list = [], [], []
            cnt_parameters = 0
            while cnt_parameters < num_parameters and len(self.param_list) > 0 :
                i = torch.randint(len(self.param_list), (1,)).item()
                new_param_list.append(self.param_list.pop(i)) 
                new_name_list.append(self.name_list.pop(i))
                size_list = self.size_list.pop(i)
                new_size_list.append(size_list)
                cnt_parameters += np.prod(size_list)

            self.param_list = new_param_list
            self.name_list = new_name_list
            self.size_list = new_size_list
            for name, p in model.named_parameters():
                if name.startswith(param_filter) and name not in self.name_list:
                    p.requires_grad = False
            

    def merge(self, var_list, device='cpu'):
        """
        merge a list of variables to a vector
        """
        assert len(var_list) == len(self.size_list)
        theta_list = []
        for i in range(len(var_list)):
            if var_list[i] is not None:
                theta_list.append(var_list[i].flatten())
            else:
                theta_list.append(torch.zeros(self.size_list[i], device=device).flatten())
        return torch.cat(theta_list)

    def merge_matrix(self, var_list, device='cpu'):
        """
        merge a list of variables to a matrix
        """
        assert len(var_list) == len(self.size_list)
        theta_list = []
        batch_size = len(var_list[0])
        for i in range(len(var_list)):
            if var_list[i] is not None:
                var_tmp = var_list[i].flatten(start_dim=1)
                theta_list.append(var_tmp)
            else:
                zeros = torch.zeros(batch_size, np.prod(self.size_list[i]), device=device)
                theta_list.append(zeros)

        return torch.cat(theta_list, dim=1)

    def split(self, var_vec):
        """
        split a vec to a list
        """
        var_list = []
        count = 0
        for i in range(len(self.size_list)):
            prod_size = np.prod(self.size_list[i])
            var_list.append(var_vec[count:(count+prod_size)].reshape(self.size_list[i]))
            count += prod_size
        return var_list


class RTRL_partial():
    def __init__(self, model, param_state_filter, num_parameters=None, slice_indexes=None):

        # Init params
        self.num_parameters = num_parameters
        self.slice_indexes=slice_indexes
        self.param_state_filter = param_state_filter
        self.nn_param_state = Param2Vec(model, param_filter=param_state_filter, num_parameters=num_parameters)
        self.nn_param = Param2Vec(model)
        
        # Iterative sensitivity
        self.d_state_d_theta = None
        self.table_save = None
        self.len_state_all = None

    def step(self, model, x, state_old, loss, **kwargs):
        """
        x: input 
        state_old: previous recurrent state of size (state)
        state_new: previous recurrent state of size (state)
        """
        device = state_old.device
        if self.slice_indexes is None:
            self.slice_indexes = list(range(len(state_old)))

        # Update online gradient 

        delta_theta = grad(loss, self.nn_param.param_list, retain_graph=True, allow_unused=True)

        for i in range(len(self.nn_param.param_list)):
            if delta_theta[i] is not None:
                if self.nn_param.param_list[i].grad is None:
                    self.nn_param.param_list[i].grad = delta_theta[i]
                else:
                    self.nn_param.param_list[i].grad += delta_theta[i]

        # RTRL for state operations

        delta_s = grad(loss, state_old, retain_graph=True)[0][self.slice_indexes]

        state_old_slice = state_old[self.slice_indexes]
        len_state = len(state_old_slice)
        self.len_state_all = len(state_old)
        len_param_state = len(self.nn_param_state.merge(self.nn_param_state.param_list, device=device))

        if self.d_state_d_theta is None:
            self.d_state_d_theta = torch.zeros(len_state, len_param_state, device=device)

        # Jacobians

        dict_param_state = dict(zip(self.nn_param_state.name_list, self.nn_param_state.param_list))
        dict_param_state = {k:v.detach() for k, v in dict_param_state.items()}
        state_old_detached = state_old_slice.detach()

        def fmodel(params, old_s): #functional version of 
            state_old_clone = state_old.clone().detach()
            state_old_clone[self.slice_indexes] = old_s
            _, _, _, state_new = functional_call(model, params, (x, state_old_clone), kwargs)
            return state_new[self.slice_indexes]
        
        ## ds/dtheta: dim (S/C) T
        d_state_new_d_theta_tmp = jacrev(fmodel, argnums=(0))(dict_param_state, state_old_detached)
        torch.cuda.empty_cache() 
        d_state_new_d_theta = self.nn_param_state.merge_matrix(list(d_state_new_d_theta_tmp.values()), 
                                                                    device=device)

        ## ds_new/ds_old: dim (S/C) S
        d_state_new_d_state_old =  jacrev(fmodel, argnums=(1))(dict_param_state, state_old_detached)
        torch.cuda.empty_cache() 


        # Recursion formula and compute gradient

        g_t1 = delta_s.unsqueeze(0).mm(self.d_state_d_theta)  # + delta_theta_vec # Already done in online update
        g_t1_list = self.nn_param_state.split(g_t1[0])
        self.d_state_d_theta = d_state_new_d_state_old.mm(self.d_state_d_theta) + d_state_new_d_theta

        for i in range(len(self.nn_param_state.param_list)):
            if self.nn_param_state.param_list[i].grad is None:
                self.nn_param_state.param_list[i].grad = g_t1_list[i]
            else:
                self.nn_param_state.param_list[i].grad += g_t1_list[i]

    def __init__(self, model):

        self.nn_param = Param2Vec(model)