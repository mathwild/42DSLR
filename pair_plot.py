#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    File name: scatter_plot.[extension].py
    Description: Scatter plot for MyDataSet instance.
    Author: Mathilde DUVERGER
    Date created: 2018/10/08
    Python Version: 3.6
"""

from mydataset import MyDataSet

dataset_train = MyDataSet().read_csv('resources/dataset_train.csv')
dataset_train.plot_pair()
