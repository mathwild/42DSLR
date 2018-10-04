from mydataset import MyDataSet
import numpy as np

def get_dummies(dataset, col_to_dummy):
    Dict = dataset.copy()
    Categories = list(set(Dict[col_to_dummy])) 
    for i in Categories: 
        Dict[i] = [(element == i)*1 for element in Dict[col_to_dummy]]
    del Dict[col_to_dummy]
    return Dict

def one_hot_encoder(dataset, col_to_encode) : 
    Encoded_Dict = dataset.copy()
    NewCategories = list(set(Encoded_Dict[col_to_encode]))
    NewCategories.pop() # remove the last element 
    for i in NewCategories: 
        Encoded_Dict[i] = [(element == i)*1 for element in Encoded_Dict[col_to_encode]]
    del Encoded_Dict[col_to_encode]
    return Encoded_Dict

def full_one_hot_encoder(dataset) : 
    keys_str = [key for key in dataset.keys() if all(isinstance(x, (str)) for x in dataset[key])]
    Full_Dict = dataset.copy()
    for key in keys_str:
        Full_Dict = one_hot_encoder(Full_Dict, key)
    return Full_Dict

def to_matrix(full_dict, select=None):
    if select==None:
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