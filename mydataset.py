#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    File name: mydataset.py
    Description: Class for storing a DataSet.
    Author: Mathilde DUVERGER
    Date created: 2018/10/03
    Python Version: 3.6
"""

import numpy as np
import matplotlib.pyplot as plt


class MyDataSet:

    """
    Summary
    -------
    Stores a DataSet.
    """

    def __init__(self):
        pass

    def __getitem__(self, key_tup):
        if type(key_tup) == tuple:
            row = key_tup[0]
            col = key_tup[1]
            return self.df[col][row]
        else:
            if type(key_tup) == list:
                return {key: self.df[key] for key in key_tup}
            else:
                return {key_tup: self.df[key_tup]}

    def read_csv(self, path):
        """
        Summary
        -------
        Read a DataSet from a CSV file.

        Parameters
        ----------
        self: MyDataSet instance
        path: 'str'
            Path to the CSV file.

        Returns
        --------
        self: Instance of MyDataSet class.
        """

        with open(path, 'r') as file:
            line_count = 0
            for line in file:
                if line_count == 0:
                    colNames = line.strip().split(',')
                    dataframe = dict.fromkeys(colNames, )
                    line_count += 1
                elif line_count == 1:
                    words = line.strip().split(',')
                    words = [MyDataSet.words_convert(x) for x in words]
                    for i in range(len(words)):
                        dataframe[colNames[i]] = [words[i]]
                    line_count += 1
                else:
                    words = line.strip().split(',')
                    words = [MyDataSet.words_convert(x) for x in words]
                    for i in range(len(words)):
                        dataframe[colNames[i]] += [words[i]]
                    line_count += 1
        self.df = dataframe
        return self

    @staticmethod
    def words_convert(item):
        """
        Summary
        -------
        Convert str to its appropriate type.

        Parameters
        ----------
        item: 'str'
            String to be converted.

        Returns
        --------
        item: 'int' or 'float' or 'str'
            Input item converted into its appropriate type
        """
        if item == '':
            return np.nan
        else:
            try:
                return int(item)
            except ValueError:
                try:
                    return float(item)
                except ValueError:
                    return item

    def get_cond(self, cond_col, cond_val, get_col, dropna=True):
        """
        Summary
        -------
        Function to access values in MyDataSet instance given a condition.

        Parameters
        ----------
        self: MyDataSet instance
        cond_col: 'str'
            Column name to condition on.
        cond_val: 'str' or 'int' or 'float'
            Value of cond_col to condition on (equality).
        get_col: 'str'
            Column name to get.
        dropna: 'bool', default True
            Remove nan values from output if True.

        Returns
        --------
        'list'
        List of values from get_col conditioned on cond_col values.
        """
        idx = [i for i, x in enumerate(self.df[cond_col]) if x == cond_val]
        if dropna is True:
            return [self.df[get_col][x] for x in idx if
                    str(self.df[get_col][x]) != 'nan']
        else:
            return [self.df[get_col][x] for x in idx]

    def describe(self):
        keys_float = [key for key in self.df.keys() if
                      all(isinstance(x, (float)) for x in self.df[key])]
        print('\t', end='')
        for feature in keys_float:
            print((feature+'           ')[:14], end='\t')
        print()
        rows = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
        for row in rows:
            print(row, end='\t')
            for feature in keys_float:
                self.describe_rows_val(row, feature)
            print()

    def describe_rows_val(self, row, feature):
        if row == 'Count':
            print(format(self.column_count(feature), '.0f'), end='\t \t')
        elif row == 'Mean':
            print(format(self.column_mean(feature), '.2f'), end='\t \t')
        elif row == 'Std':
            print(format(self.standard_deviation(feature), '.2f'), end='\t \t')
        elif row == 'Min':
            print(format(self.column_minimum(feature), '.2f'), end='\t \t')
        elif row == '25%':
            print(format(self.quartiles(feature, 0.25), '.2f'), end='\t \t')
        elif row == '50%':
            print(format(self.quartiles(feature, 0.5), '.2f'), end='\t \t')
        elif row == '75%':
            print(format(self.quartiles(feature, 0.75), '.2f'), end='\t \t')
        elif row == 'Max':
            print(format(self.column_maximum(feature), '.2f'), end='\t \t')

    def column_count(self, feature):
        """
        Summary
        -------
        Count non nan values for given feature in the DataSet.

        Parameters
        ----------
        self: MyDataSet instance
        feature: 'str'
            Column name.

        Returns
        --------
        'int'
        Count.
        """
        column = self.df[feature]
        count = 0
        for value in column:
            if str(value) != 'nan':
                count = count + 1
            else:
                continue
        return count

    def column_mean(self, feature):
        """
        Summary
        -------
        Mean of given feature in the DataSet.

        Parameters
        ----------
        self: MyDataSet instance
        feature: 'str'
            Column name.

        Returns
        --------
        'float'
        Mean.
        """
        column = self.df[feature]
        count = 0
        total = 0
        for value in column:
            if str(value) != 'nan':
                count = count + 1
                total = total + value
            else:
                continue
        return total/count

    def standard_deviation(self, feature):
        """
        Summary
        -------
        Standard deviation of given feature in the DataSet.

        Parameters
        ----------
        self: MyDataSet instance
        feature: 'str'
            Column name.

        Returns
        --------
        'float'
        Standard deviation.
        """
        column = self.df[feature]
        count = 0
        total = 0
        for value in column:
            if str(value) != 'nan':
                count = count + 1
                total = total + value
            else:
                continue
        mean = total/count

        variance = 0
        for value in column:
            if str(value) != 'nan':
                res = value - mean
                square = res*res
                variance = variance + square
            else:
                continue
        return (variance/(count-1))**(1/2)

    def column_minimum(self, feature):
        """
        Summary
        -------
        Minimum of given feature in the DataSet.

        Parameters
        ----------
        self: MyDataSet instance
        feature: 'str'
            Column name.

        Returns
        --------
        'int' or 'float'
        Minimum.
        """
        column = self.df[feature]
        mini = np.inf
        for value in column:
            if mini > value:
                mini = value
            else:
                continue
        return mini

    def column_maximum(self, feature):
        """
        Summary
        -------
        Maximum of given feature in the DataSet.

        Parameters
        ----------
        self: MyDataSet instance
        feature: 'str'
            Column name.

        Returns
        --------
        'int' or float'
        Maximum.
        """
        column = self.df[feature]
        maxi = - np.inf
        for value in column:
            if maxi < value:
                maxi = value
            else:
                continue
        return maxi

    def quartiles(self, feature, quart):
        """
        Summary
        -------
        Quartiles of given feature in the DataSet.

        Parameters
        ----------
        self: MyDataSet instance
        feature: 'str'
            Column name.
        quart: 'float'
            Quartile to calculate (0.5 for median).

        Returns
        --------
        'float'
        Quartile.
        """
        mylist = self.df[feature]
        func_list = mylist.copy()
        func_list = [x for x in func_list if str(x) != 'nan']
        func_list.sort()
        quart_idx = quart*len(func_list)
        if quart_idx.is_integer():
            result = (func_list[int(quart_idx)] + func_list[int(quart_idx)-1])/2
            return result
        else:
            result = func_list[int(quart_idx)]
            return result

    def plot_hist(self):
        """
        Summary
        -------
        Plot a histogram of all numeric columns in DataSet.

        Parameters
        ----------
        self: MyDataSet instance
        """
        class_list = [key for key in self.df.keys() if all(isinstance(x, (float)) for x in self.df[key])]
        num_cols = int(len(class_list)/2) + 1
        fig, axes = plt.subplots(2, num_cols, sharey=True, figsize=(15, 6))
        i, j = [0, 0]
        for class_name in class_list:
            for house in set(self.df['Hogwarts House']):
                axes[i, j].hist(self.get_cond('Hogwarts House', house, class_name), alpha=0.7)
                axes[i, j].set_title(class_name, fontsize=12)
            if j < num_cols - 1:
                j += 1
            else:
                i += 1
                j = 0

        plt.show()

    def plot_scatter(self):
        """
        Summary
        -------
        Plot a scatter matrix for all numeric columns in DataSet.

        Parameters
        ----------
        self: MyDataSet instance
        """
        class_list = [key for key in self.df.keys() if all(isinstance(x, (float)) for x in self.df[key])]
        num_cols = len(class_list)
        fig, axes = plt.subplots(num_cols, num_cols, figsize=(30, 30))
        i = 0
        for class_name_1 in class_list:
            j = 0
            for class_name_2 in class_list:
                if i == j:
                    axes[i, j].text(x=0.1, y=1/2, s=class_name_1, fontsize=15)
                else:
                    axes[i, j].scatter(x=self.df[class_name_1], y=self.df[class_name_2])
                axes[i, j].set_xticklabels([])
                axes[i, j].set_yticklabels([])
                j += 1
            i += 1
        plt.show()

    def plot_pair(self):
        """
        Summary
        -------
        Plot a pair plot of all numeric columns in DataSet.

        Parameters
        ----------
        self: MyDataSet instance
        """
        class_list = [key for key in self.df.keys() if all(isinstance(x, (float)) for x in self.df[key])]
        num_cols = len(class_list)
        fig, axes = plt.subplots(num_cols, num_cols, figsize=(60, 60))
        i = 0
        for class_name_1 in class_list:
            j = 0
            for class_name_2 in class_list:

                if j == 0:
                    axes[i, j].set_ylabel(class_name_1)

                if i == j:
                    for house in set(self.df['Hogwarts House']):
                        axes[i, j].hist(self.get_cond('Hogwarts House', house, class_name_1), alpha=0.7)
                else:
                    for house in set(self.df['Hogwarts House']):
                            sc_x = self.get_cond('Hogwarts House', house, class_name_1, dropna=False)
                            sc_y = self.get_cond('Hogwarts House', house, class_name_2, dropna=False)
                            axes[i, j].scatter(x=sc_x, y=sc_y)
                axes[i, j].set_xticklabels([])
                axes[i, j].set_yticklabels([])

                if i == num_cols-1:
                    axes[i, j].set_xlabel(class_name_2)

                j += 1
            i += 1
        plt.show()


if __name__ == '__main__':
    dataset_train = MyDataSet().read_csv('resources/dataset_train.csv')
    dataset_train.describe()
