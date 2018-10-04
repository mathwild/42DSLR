#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:43:19 2018

@author: louislimnavong
"""

import numpy as np
from scipy import optimize 

from louisdataset import MyDataSet
from louis_get_matrix import get_DictX, get_dummies, get_Y, full_one_hot_encoder

class LogRegModel : 
    
    def __init__(self):
        pass
       
    def fit(self, Y_full, X_train): # Y_full is a dictionary 
        self.coef = dict()
        SubDict = get_dummies(self.dict_list(), Y_full.keys())
        Categories = list(set(Y_full))
        for i in Categories: 
            Y_train = get_Y(SubDict, i)
            self.fit_binary(Y_train, X_train)
    
    def fit_binary(self, Y_train, X_train):
        m, p = X_train.shape
        intercept = np.ones(m)
        X_one = np.column_stack((intercept,X_train))
        n, d = X_one.shape
        init_w = np.zeros(d)
        res = optimize.minimize(LogRegModel.neg_loglikelihood,init_w, method = 'BFGS', args = (Y_train,X_one))
        self.coef['TEST'] = res.x
        return res.x

    @staticmethod
    def neg_loglikelihood(beta, Y, X):
        # sum without NAs
        return -np.nansum(Y*np.matmul(X,beta) - np.log(1+np.exp(np.matmul(X,beta))))
    
name_Y = 'Hogwarts House'
name_subY = 'Gryffindor'

if __name__ == '__main__':
    
    ###
    dataset_train = MyDataSet().read_csv('resources/dataset_train.csv')
    dataset_train_dict = dataset_train.dict_list()
    
    # getting X
    DictX = get_DictX(dataset_train_dict, name_Y)
    X = full_one_hot_encoder(DictX)
    # getting Y
    SubDict = get_dummies(dataset_train_dict, name_Y)
    Y = get_Y(SubDict, name_subY)
    ###
    
    
    model = LogRegModel()
    log_params = model.fit(name_Y, X_train = X)
    #print(log_params)
    
