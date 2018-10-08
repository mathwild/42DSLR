#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    File name: logregmodel.py
    Description: Class for fitting a logistic regression.
    Author: Louis LIMNAVONG, Alessandro GIRELLI
    Date created: 2018/10/08
    Python Version: 3.6
"""

import numpy as np
from scipy import optimize
from mydataset import MyDataSet
from preprocessing import get_dummies, full_one_hot_encoder, to_matrix


class LogRegModel:
    """
    Summary
    -------
    Fit a logistic regression.
    """
    def __init__(self):
        pass

    def fit(self, Y_full, X_train):  # Y_full is a dictionary
        """
        Summary
        -------
        Function to fit a logistic regression.

        Parameters
        ----------
        self: LogRegModel instance
        Y_full: 'dict'
            Dictionary of response variable.
        X_train: 'numpy matrix'
            Matrix of covariates.
        """
        self.coef = dict()
        Y_name = str(list(Y_full.keys())[0])
        SubDict = get_dummies(Y_full, Y_name)
        Categories = set(Y_full[Y_name])
        for cat in Categories:
            Y_train = SubDict[cat]
            self.fit_binary(Y_train, X_train, cat)

    def fit_binary(self, Y_train, X_train, name):
        """
        Summary
        -------
        Function to fit a binary logistic regression.

        Parameters
        ----------
        self: LogRegModel instance
        Y_train: 'numpy array'
            Response variable vector.
        X_train: 'numpy matrix'
            Matrix of covariates.

        Returns
        -------
        res.x: 'list'
            Coefficients of the features.
        """
        m, p = X_train.shape
        intercept = np.ones(m)
        X_one = np.column_stack((intercept, X_train))
        n, d = X_one.shape
        init_w = np.zeros(d)
        res = optimize.minimize(LogRegModel.neg_loglikelihood, init_w,
                                method='BFGS', args=(Y_train, X_one))
        self.coef[name] = res.x
        return res.x

    @staticmethod
    def neg_loglikelihood(beta, Y, X):
        """
        Summary
        -------
        Loss function of the logistic regression.

        Parameters
        ----------
        beta: 'numpy array'
            Parameters of the logistic regression.
        Y: 'numpy array'
            Response variable vector.
        X: 'numpy matrix'
            Matrix of covariates.

        Returns
        -------
        Loss function.
        """
        # sum without NAs
        return -np.nansum(Y*np.matmul(X, beta)
                          - np.log(1+np.exp(np.matmul(X, beta))))

    def predict(self, X):
        """
        Summary
        -------
        Predict response variable.

        Parameters
        ----------
        self: LogRegModel instance.
        X: 'numpy matrix'
            Matrix of covariates.

        Returns
        -------
        prediction: 'list'
            Response variable prediction.
        """
        m, p = X.shape
        intercept = np.ones(m)
        X = np.column_stack((intercept, X))
        houses = set(self.coef.keys())
        pairwise_max = np.zeros(m)

        for house in houses:
            coef = self.coef[house]
            proba = np.exp(np.matmul(X, coef))/(1+(np.exp(np.matmul(X, coef))))
            pairwise_max = np.maximum(pairwise_max, proba)
        labels = pairwise_max

        for house in houses:
            coef = self.coef[house]
            boolean = pairwise_max == (np.exp(np.matmul(X, coef)) /
                                       (1+(np.exp(np.matmul(X, coef)))))
            labels = [labels[i] if labels[i] in houses else house
                      if elements is True else labels[i]
                      for i, elements in enumerate(boolean)]

        self.prediction = labels
        return labels

    def accuracy_score(self, Y_true):
        """
        Summary
        -------
        Accuracy score of prediction.

        Parameters
        ----------
        self: LogRegModel instance.
        Y_true: 'list'
            Observed response variable.

        Returns
        -------
        accuracy: 'float'
            Accuracy of prediction.
        """
        return (len([i for i, j in zip(self.prediction, Y_true) if i == j])
                / len(Y_true))


if __name__ == '__main__':
    ###
    dataset_train = MyDataSet().read_csv('resources/dataset_train.csv')
    # getting X
    DictX = dataset_train[['Best Hand', 'Arithmancy', 'Astronomy', 'Herbology',
                           'Defense Against the Dark Arts', 'Divination',
                           'Muggle Studies', 'Ancient Runes',
                           'History of Magic', 'Transfiguration', 'Potions',
                           'Care of Magical Creatures', 'Charms', 'Flying']]
    DictX_encod = full_one_hot_encoder(DictX)
    X = to_matrix(DictX_encod)
    # getting Y
    Y = dataset_train['Hogwarts House']
    ###

    model = LogRegModel()
    model.fit(Y, X_train=X)
    preds = model.predict(X)
    print(model.prediction)
