#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 17:36:29 2018

@author: louislimnavong
"""

import numpy as np
from scipy import optimize 

from mydataset import MyDataSet
from preprocessing import get_dummies, full_one_hot_encoder, to_matrix


class LogRegModel : 
    
    def __init__(self):
        pass
       
    def fit(self, Y_full, X_train): # Y_full is a dictionary 
        self.coef = dict()
        Y_name = str(list(Y_full.keys())[0])
        SubDict = get_dummies(Y_full, Y_name)
        Categories = set(Y_full[Y_name])
        for i in Categories: 
            # Y_train = get_Y(SubDict, i)
            Y_train = SubDict[i]
            self.fit_binary(Y_train, X_train, i)
    
    def fit_binary(self, Y_train, X_train, name):
        m, p = X_train.shape
        intercept = np.ones(m)
        X_one = np.column_stack((intercept,X_train))
        n, d = X_one.shape
        init_w = np.zeros(d)
        res = optimize.minimize(LogRegModel.neg_loglikelihood,init_w, method = 'BFGS', args = (Y_train,X_one))
        self.coef[name] = res.x
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
    # getting X
    DictX = dataset_train[['Best Hand','Arithmancy', 'Astronomy']]
    DictX_encod = full_one_hot_encoder(DictX)
    X = to_matrix(DictX_encod)
    # getting Y
    Y = dataset_train[name_Y]
    ###
    
    
    model = LogRegModel()
    model.fit(Y, X_train = X)
    print(model.coef)
    
