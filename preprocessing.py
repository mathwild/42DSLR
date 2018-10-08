#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    File name: preprocessing.py
    Description: Functions to preprocess MyDataSet instance before applying a
    model.
    Author: Louis LIMNAVONG, Mathilde DUVERGER
    Date created: 2018/10/04
    Python Version: 3.6
"""

from mydataset import MyDataSet
import numpy as np


def get_dummies(dataset, col_to_dummy):
    """
        Summary
        -------
        Get dummies for given column.

        Parameters
        ----------
        dataset: 'dict'
            MyDataSet dictionary to create dummies for.
        col_to_dummy: 'str'
            Column name to get dummies for.

        Returns
        --------
        Dict: 'dict'
            MyDataSet dictionary without column to dummy and with new dummified 
            columns.
        """
    Dict = dataset.copy()
    Categories = list(set(Dict[col_to_dummy]))
    for i in Categories:
        Dict[i] = [(element == i)*1 for element in Dict[col_to_dummy]]
    del Dict[col_to_dummy]
    return Dict


def one_hot_encoder(dataset, col_to_encode):
    """
        Summary
        -------
        Get one hot encoding for given column.

        Parameters
        ----------
        dataset: 'dict'
            MyDataSet dictionary to create dummies for.
        col_to_encode: 'str'
            Column name to encode.

        Returns
        --------
        Encoded_Dict: 'dict'
                MyDataSet dictionary without column to dummy and with new
                encoded columns.
        """
    Encoded_Dict = dataset.copy()
    NewCategories = list(set(Encoded_Dict[col_to_encode]))
    NewCategories.pop()  # remove the last element
    for i in NewCategories:
        Encoded_Dict[i] = [(element == i)*1 for element in Encoded_Dict[col_to_encode]]
    del Encoded_Dict[col_to_encode]
    return Encoded_Dict


def full_one_hot_encoder(dataset):
    """
        Summary
        -------
        Get full one hot encoding for all categorical columns.

        Parameters
        ----------
        dataset: 'dict'
            MyDataSet dictionary to create encode for.

        Returns
        --------
        Full_Dict: 'dict'
                MyDataSet dictionary without column to dummy and with new 
                dummified columns.
        """
    keys_str = [key for key in dataset.keys() if all(isinstance(x, (str)) for x in dataset[key])]
    Full_Dict = dataset.copy()
    for key in keys_str:
        Full_Dict = one_hot_encoder(Full_Dict, key)
    return Full_Dict


def to_matrix(full_dict, select=None):
    """
        Summary
        -------
        Transform MyDataSet dictionary into a matrix.

        Parameters
        ----------
        full_dict: 'dict'
            MyDataSet dictionary to create dummies for
        select: 'str', default None
            Column name to select.

        Returns
        --------
        'numpy array'
        Array or matrix from MyDataSet dictionary.
        """
    if select is None:
        return np.column_stack(list(full_dict.values()))
    else:
        return np.array(full_dict[select].copy())


if __name__ == '__main__':
    dataset_train = MyDataSet().read_csv('resources/dataset_train.csv')

    # getting X
    DictX = dataset_train[['Best Hand','Arithmancy', 'Astronomy']]
    DictX_encod = full_one_hot_encoder(DictX)
    X = to_matrix(DictX_encod)
    # getting Y
    DictY = dataset_train['Hogwarts House']
    DictY_dum = get_dummies(DictY, 'Hogwarts House')
    Y = to_matrix(DictY_dum, 'Ravenclaw')