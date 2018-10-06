#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:43:19 2018

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
    
#    def predict(self, x_test):
#        max_proba = 0
#        for key, value in self.coef.items():
#            proba = np.exp(np.matmul(x_test,value))/(1+(np.exp(np.matmul(x_test,value))))
#            if proba > max_proba:
#                max_proba = proba
#                prediction = key
#            else:
#                continue 
#        return prediction
        

    
name_Y = 'Hogwarts House'
name_subY1 = 'Gryffindor'
name_subY2 = 'Ravenclaw'

if __name__ == '__main__':
    
    ### TRAIN ###
    dataset_train = MyDataSet().read_csv('resources/dataset_train.csv')
    # getting X
    DictX = dataset_train[['Best Hand','Arithmancy', 'Astronomy']]
    DictX_encod = full_one_hot_encoder(DictX)
    X = to_matrix(DictX_encod)
    # getting Y
    Y = dataset_train[name_Y]
    ###
    
    ### TEST ### 
    dataset_test = MyDataSet().read_csv('resources/dataset_test.csv')
    # getting X
    DictX_test = dataset_test[['Best Hand','Arithmancy', 'Astronomy']]
    DictX_encod_test = full_one_hot_encoder(DictX_test)
    X_test = to_matrix(DictX_encod_test)
    # adding intercept
    m, p = X_test.shape
    intercept = np.ones(m)
    X_test = np.column_stack((intercept,X_test))
    ###
    
    ### FIT ### 
    model = LogRegModel()
    model.fit(Y, X_train = X)
    ###
    
    indiv = X_test[:10]
    coef1 = model.coef[name_subY1]
    proba1 = np.exp(np.matmul(indiv,coef1))/(1+(np.exp(np.matmul(indiv,coef1))))
    coef2 = model.coef[name_subY2]
    proba2 = np.exp(np.matmul(indiv,coef2))/(1+(np.exp(np.matmul(indiv,coef2))))
    pairwise_max = np.maximum(proba1, proba2)
    
    print(pairwise_max)
