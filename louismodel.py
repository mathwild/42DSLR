#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:43:19 2018

@author: louislimnavong
"""

import pandas as pd
import numpy as np
from scipy import optimize 

from louisdataset import MyDataSet
from louis_get_matrix import getDictX, get_dummies, get_Y, full_one_hot_encoder

class LogRegModel : 
    
    def __init__(self):
        pass
       
    def get_LogReg_coef(self, Y_train, X_train):
        n, d = X_train.shape
        init_w = np.zeros(d)
        res = optimize.minimize(neg_loglikelihood,init_w, method = 'BFGS', args = (Y_train,X_train))
        self.coef = res.x
        return res.x

    @staticmethod
    def neg_loglikelihood(beta, Y, X):
        # sum without NAs
        return -np.nansum(Y*np.matmul(X,beta) - np.log(1+np.exp(np.matmul(X,beta))))
    
name_Y = 'Hogwarts House'
name_subY = 'Gryffindor'

if __name__ == '__main__':
    dataset_train = MyDataSet().read_csv('resources/dataset_train.csv')
    dataset_train_dict = dataset_train.dict_list()
    
    # getting X
    DictX = get_DictX(dataset_train_dict, name_Y)
    X = full_one_hot_encoder(DictX)
    # getting Y
    SubDict = get_dummies(dataset_train_dict, name_Y)
    Y = get_Y(SubDict, name_subY)
    
    print(get_LogReg_coef(Y, X))
