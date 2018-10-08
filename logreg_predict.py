#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    File name: logreg_train.[extension].py
    Description: Predict logistic regression.
    Author: Mathilde DUVERGER, Alessandro GIRELLI, Louis LIMNAVONG
    Date created: 2018/10/08
    Python Version: 3.6
"""

from mydataset import MyDataSet
from logregmodel import LogRegModel
from preprocessing import full_one_hot_encoder, to_matrix
import sys
import csv 

path = str(sys.argv[1])
dataset_test = MyDataSet().read_csv(path)

# getting X
DictX_test = dataset_test[['Best Hand', 'Astronomy', 'Herbology',
                       'Defense Against the Dark Arts',
                       'Muggle Studies', 'Ancient Runes',
                       'History of Magic', 'Transfiguration', 
                       'Charms', 'Flying']]

DictX_encod_test = full_one_hot_encoder(DictX_test)
X_test = to_matrix(DictX_encod_test)
###

dataset_train = MyDataSet().read_csv('resources/dataset_train.csv')

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
model.fit(Y, X)
model.predict(X_test)
with open('houses.csv', mode='w') as houses_file:
        house_writer = csv.writer(houses_file, delimiter=',')
        house_writer.writerow(('Index', 'Hogwarts House'))
        for i, prediction in enumerate(model.prediction): 
            house_writer.writerow((i,prediction))