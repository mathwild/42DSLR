#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:48:45 2018

@author: louislimnavong
"""

## should we get this as a class ? ##

from louisdataset import MyDataSet
import numpy as np

def get_DictX(full_dataframe, Y): # e.g. create_dict_list(dataset_train.df)
    
    # temporary fix: can put the variables as arguments
    DictX = full_dataframe.copy()
    del DictX['First Name']
    del DictX['Last Name']
    del DictX['Birthday']
    del DictX['Index'] 
    del DictX[Y]
    return DictX

def get_dummies(full_dataframe, variable):
    Dict = {variable:full_dataframe[variable]}.copy()
    Categories = list(set(Dict[variable])) 
    for i in Categories: 
        Dict[i] = [(element == i)*1 for element in Dict[variable]]
    del Dict[variable]
    return Dict

def get_Y(full_dict, Y):
    return np.array(full_dict[Y].copy())

# for a given feature 
def one_hot_encoder(dataframe, column_to_encode) : 
    Encoded_Dict = dataframe.copy()
    NewCategories = list(set(Encoded_Dict[column_to_encode]))
    NewCategories.pop() # remove the last element 
    for i in NewCategories: 
        Encoded_Dict[i] = [(element == i)*1 for element in Encoded_Dict[column_to_encode]]
    del Encoded_Dict[column_to_encode]
    return Encoded_Dict

# for every feature 
def full_one_hot_encoder(dataframe) : # dataframe = NewDict
    keys_str = [keys for keys in dataframe.keys() if type(list(dataframe[keys])[0]) == str]
    Full_Dict = dataframe.copy()
    for key in keys_str:
        Full_Dict = one_hot_encoder(Full_Dict, key)
    return np.column_stack(list(Full_Dict.values())) # return the dictionary as a matrix for the LogReg

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
    
    print(X)
    print(Y)
    