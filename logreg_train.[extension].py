#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    File name: logreg_train.[extension].py
    Description: Train logistic regression.
    Author: Mathilde DUVERGER, Alessandro GIRELLI, Louis LIMNAVONG
    Date created: 2018/10/08
    Python Version: 3.6
"""

from mydataset import MyDataSet
from logregmodel import LogRegModel
from preprocessing import full_one_hot_encoder, to_matrix
import sys

path = str(sys.argv[1])
print(path)
dataset_train = MyDataSet().read_csv(path)

# getting X
DictX = dataset_train[['Best Hand', 'Astronomy', 'Herbology',
                       'Defense Against the Dark Arts',
                       'Muggle Studies', 'Ancient Runes',
                       'History of Magic', 'Transfiguration', 
                       'Charms', 'Flying']]

DictX_encod = full_one_hot_encoder(DictX)
X = to_matrix(DictX_encod)
# getting Y
Y = dataset_train['Hogwarts House']
###

model = LogRegModel()
model.fit(Y, X_train=X)
coefs = model.coef

file = open('coefs.txt', 'w')
file.write(str(coefs))
file.close()
