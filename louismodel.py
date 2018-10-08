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
    
            
    def predict(self, X):
        m, p = X.shape
        intercept = np.ones(m)
        X = np.column_stack((intercept,X))
        houses = set(self.coef.keys())
        pairwise_max = np.zeros(m)
        
        for house in houses: 
            #print(house)
            coef = self.coef[house]
            proba = np.exp(np.matmul(X,coef))/(1+(np.exp(np.matmul(X,coef))))
            pairwise_max = np.maximum(pairwise_max, proba)
        #print(pairwise_max)
        labels = pairwise_max
        for house in houses: 
            coef = self.coef[house]
            boolean = pairwise_max == np.exp(np.matmul(X,coef))/(1+(np.exp(np.matmul(X,coef))))
            labels = [labels[i] if labels[i] in houses else house if elements == True else labels[i] for i, elements in enumerate(boolean)]
        self.prediction = labels
    
name_Y = 'Hogwarts House'

if __name__ == '__main__':
    
    ### TRAIN ###
    dataset_train = MyDataSet().read_csv('resources/dataset_train.csv')
    DictX = dataset_train[['Best Hand','Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
                           'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']]
    DictX_encod = full_one_hot_encoder(DictX)
    X = to_matrix(DictX_encod)
    Y = dataset_train[name_Y]
    ###
    
    ### FIT ### 
    model = LogRegModel()
    model.fit(Y, X_train = X)
    ###
    
    ### TEST ### 
    dataset_test = MyDataSet().read_csv('resources/dataset_test.csv')
    
    # getting X
    DictX_test = dataset_test[['Best Hand','Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
                           'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']]
    DictX_encod_test = full_one_hot_encoder(DictX_test)
    #X_test = to_matrix(DictX_encod_test)
    X_test = X.copy()
    
    #getting Y
    #Y_test = dataset_test[name_Y]
    Y_test = dataset_train[name_Y]
    ### 
    
    indiv = X_test[:]
    model.predict(indiv)
    
    print(model.prediction)
    
