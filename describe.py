#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    File name: describe.[extension].py
    Description: Describe input CSV DataSet.
    Author: Mathilde DUVERGER
    Date created: 2018/10/08
    Python Version: 3.6
"""

from mydataset import MyDataSet
import sys

path = str(sys.argv[1])
dataset_train = MyDataSet().read_csv(path)
dataset_train.describe()
