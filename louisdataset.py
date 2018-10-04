#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:25:37 2018

@author: louislimnavong
"""

import numpy as np
from prettytable import PrettyTable


class MyDataSet:

    def __init__(self):
        pass
    
    def __getitem__(self, key_tup):
        if len(key_tup) == 2:
            row = key_tup[0]
            col = key_tup[1]
            if type(row)==int :
                return self.df[col][row]
            elif type(row)==slice :
                print('Sorry we are not ready for slices yet...')
            else :
                print('For conditionnal arguments please use the get_cond function.')
        else:
            return self.df[key_tup]

    def read_csv(self, path):
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
                    index = words[0]
                    for i in range(len(words)):
                        dataframe[colNames[i]] = {index: words[i]}
                    line_count += 1
                else:
                    words = line.strip().split(',')
                    words = [MyDataSet.words_convert(x) for x in words]
                    index = words[0]
                    for i in range(len(words)):
                        dataframe[colNames[i]][index] = words[i]
                    line_count += 1
        self.df = dataframe
        return self
    
    @staticmethod
    def words_convert(item):
        if item == '' :
            return np.nan
        else:
            try:
                return int(item)
            except ValueError:
                try:
                    return float(item)
                except ValueError:
                    return item

    def get_cond(self, cond_col, cond_val, get_col, cond='equal'):
        idx = [key for key, value in self.df[cond_col].items() if value == cond_val]
        return [self.df[get_col][x] for x in idx if str(self.df[get_col][x]) != 'nan']

    def describe(self):
        keys_float = [keys for keys in self.df.keys() if type(list(self.df[keys].values())[0]) == float]
        t = PrettyTable(['',*keys_float[:8]])
        t.add_row(['count', *[self.column_count(keys) for keys in keys_float[:8]]])
        t.add_row(['mean', *[self.column_mean(keys) for keys in keys_float[:8]]])
        t.add_row(['std', *[self.standard_deviation(keys) for keys in keys_float[:8]]])
        t.add_row(['min', *[self.column_minimum(keys) for keys in keys_float[:8]]])
        t.add_row(['25%', *[self.quartiles(keys, 0.25) for keys in keys_float[:8]]])
        t.add_row(['50%', *[self.quartiles(keys, 0.5) for keys in keys_float[:8]]])
        t.add_row(['75%', *[self.quartiles(keys, 0.75) for keys in keys_float[:8]]])
        t.add_row(['max', *[self.column_maximum(keys) for keys in keys_float[:8]]])

        t2 = PrettyTable(['',*keys_float[8:]])
        t2.add_row(['count', *[self.column_count(keys) for keys in keys_float[8:]]])
        t2.add_row(['mean', *[self.column_mean(keys) for keys in keys_float[8:]]])
        t2.add_row(['std', *[self.standard_deviation(keys) for keys in keys_float[8:]]])
        t2.add_row(['min', *[self.column_minimum(keys) for keys in keys_float[8:]]])
        t2.add_row(['25%', *[self.quartiles(keys, 0.25) for keys in keys_float[8:]]])
        t2.add_row(['50%', *[self.quartiles(keys, 0.5) for keys in keys_float[8:]]])
        t2.add_row(['75%', *[self.quartiles(keys, 0.75) for keys in keys_float[8:]]])
        t2.add_row(['max', *[self.column_maximum(keys) for keys in keys_float[8:]]])

        print(t)
        print(t2)

    def column_count(self, feature):
        column = self.df[feature]
        count = 0
        for key,value in column.items():
            if str(value) != 'nan':
                count = count + 1
            else:
                continue
        return count

    def column_mean(self, feature):
        column = self.df[feature]
        count = 0
        total = 0
        for key,value in column.items():
            if str(value) != 'nan':
                count = count + 1
                total = total + value
            else:
                continue
        return total/count

    def standard_deviation(self, feature):
        column = self.df[feature]
        count = 0
        total = 0
        for key,value in column.items():
            if str(value) != 'nan':
                count = count + 1
                total = total + value
            else:
                continue
        mean = total/count

        variance = 0
        for key, value in column.items():
            if str(value) != 'nan':
                res = value - mean
                square = res*res
                variance = variance + square
            else:
                continue
        return (variance/(count-1))**(1/2)

    def column_minimum(self, feature):
        column = self.df[feature]
        mini = np.inf
        for key,value in column.items():
            if mini > value:
                mini = value
            else:
                continue
        return mini

    def column_maximum(self, feature):
        column = self.df[feature]
        maxi = - np.inf
        for key,value in column.items():
            if maxi < value:
                maxi = value
            else:
                continue
        return maxi

    def quartiles(self, feature, quart):
        mylist = list(self.df[feature].values())
        func_list = mylist.copy()
        func_list = [x for x in func_list if str(x) != 'nan']
        func_list.sort()
        quart_idx = quart*len(func_list)
        if quart_idx.is_integer() :
            result = (func_list[int(quart_idx)] + func_list[int(quart_idx)-1])/2
            return result
        else:
            result = func_list[int(np.floor(quart_idx))]
            return result
        
    ### UPDATES FROM LOUIS 
    
    def dict_list(self): # on self.df
        newDict = dict.fromkeys(self.df.keys())
        for i in self.df.keys():
            newDict[i] = list(self.df[i].values())
        self.df_dict = newDict
        #return newDict # necessary ? 


if __name__ == '__main__':
    dataset_train = MyDataSet().read_csv('resources/dataset_train.csv')
    # dataset_train.describe()
    dataset_train.dict_list()