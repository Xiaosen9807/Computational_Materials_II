import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import torch
import random
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import torch.optim as optim
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
import re
import copy
from string import digits


def setup_seed(seed):
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed
randomseed = setup_seed(9807)

periodic_table = ('H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                  'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
                  'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
                  'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
                  'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
                  'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
                  'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Uue')


def indexNumber(path=''):
    kv = []
    nums = []
    beforeDatas = re.findall('\d', path)
    for num in beforeDatas:
        indexV = []
        times = path.count(num)
        if(times > 1):
            if(num not in nums):
                indexs = re.finditer(num, path)
                for index in indexs:
                    iV = []
                    i = index.span()[0]
                    iV.append(num)
                    iV.append(i)
                    kv.append(iV)
            nums.append(num)
        else:
            index = path.find(num)
            indexV.append(num)
            indexV.append(index)
            kv.append(indexV)
    # Sort by numeric position
    indexSort = []
    resultIndex = []
    for vi in kv:
        indexSort.append(vi[1])
    indexSort.sort()
    for i in indexSort:
        for v in kv:
            if(i == v[1]):
                resultIndex.append(v)
    return resultIndex


def get_element(compoundname):

    withoutnum = compoundname.translate(str.maketrans('', '', digits))
    posiofelements = []
    for i in range(len(withoutnum)):
        if withoutnum[i].isupper():
            posiofelements.append(i)
    # print(posiofelements)
    elementsincompund = []
    count = 0
    for j in posiofelements:
        elementsincompund.append(withoutnum[count:j])
        count = j
    elementsincompund.append(
        withoutnum[posiofelements[-1]:])  # Add the last one
    # Remove the first(which is an empty list)
    elementsincompund = elementsincompund[1:]
    return(elementsincompund)


def EleEncoder(com_name):
    
    name_sub = {}
    ele = get_element(com_name)
    ind = indexNumber(com_name)
    if len(ind) == 0:
        for i in ele:
            name_sub[i] = 1
    elif len(ind) > 1:
        count = 0
        index = []
        while count < len(ind)-1:

            if ind[count+1][1] == ind[count][1]+1:
                index.append((str(ind[count][0])+str(ind[count+1][0])))
                count += 2
            else:
                index.append((str(ind[count][0])))
                count += 1
        if ind[-1][1] != ind[-2][1]+1:
            index.append(str(ind[-1][0]))

        for i in range(len(ele)):
            name_sub[ele[i]] = int(index[i]) / int(max(index))
    elif len(ind) == 1:
        for i in ele:
            name_sub[i] = 1
    #print(name_sub)
    encoder = []
    if len(name_sub.keys()) == 4:
        for i in range(8):
            if i%2 == 0:
                elenum = periodic_table.index(list(name_sub.keys())[int(i/2)])
                encoder.append(elenum)
            else:
                encoder.append(list(name_sub.values())[int((i-1)/2)])
    if len(name_sub.keys()) == 3:
        encoder.append(periodic_table.index(list(name_sub.keys())[0]))
        encoder.append(list(name_sub.values())[0])
        encoder.append(periodic_table.index(list(name_sub.keys())[1]))
        encoder.append(list(name_sub.values())[1])
        encoder.append(periodic_table.index(list(name_sub.keys())[2]))
        encoder.append(list(name_sub.values())[2]/2)
        encoder.append(periodic_table.index(list(name_sub.keys())[2]))
        encoder.append(list(name_sub.values())[2]/2)
    if len(name_sub.keys()) == 2:
        encoder.append(periodic_table.index(list(name_sub.keys())[0]))
        encoder.append(list(name_sub.values())[0]/2)
        encoder.append(periodic_table.index(list(name_sub.keys())[0]))
        encoder.append(list(name_sub.values())[0]/2)
        encoder.append(periodic_table.index(list(name_sub.keys())[1]))
        encoder.append(list(name_sub.values())[1]/2)
        encoder.append(periodic_table.index(list(name_sub.keys())[1]))
        encoder.append(list(name_sub.values())[1]/2)
    if len(name_sub.keys()) == 1:
        encoder.append(periodic_table.index(list(name_sub.keys())[0]))
        encoder.append(list(name_sub.values())[0]/4)
        encoder.append(periodic_table.index(list(name_sub.keys())[0]))
        encoder.append(list(name_sub.values())[0]/4)
        encoder.append(periodic_table.index(list(name_sub.keys())[0]))
        encoder.append(list(name_sub.values())[0]/4)
        encoder.append(periodic_table.index(list(name_sub.keys())[0]))
        encoder.append(list(name_sub.values())[0]/4)
            
    return encoder


# from 尝试一下 combine_2
def dataset(valuetype='modulus', index=1, Encoder='EleEncoder'):

    load_dict = np.load('originaldata_file.npy', allow_pickle=True).item()
    load_dict_copy = copy.deepcopy(load_dict)
    for i in load_dict_copy.keys():
        if len(get_element(i))>4:
            del load_dict[i]


    symbol = []
    hall = []
    for i in range(1):
        point_group = []
        pretty_formula_encode = []
        wrongcomp = []
        crystal_system = []
        oxide_type = []
        Bulk_modulus = []
        Shear_modulus = []
        for key in load_dict.keys():
            symbol.append(load_dict[key]['spacegroup']['symbol'])
            hall.append(load_dict[key]['spacegroup']['hall'])
            point_group.append(load_dict[key]['spacegroup']['point_group'])
            crystal_system.append(load_dict[key]['spacegroup']['crystal_system'])
            if Encoder == 'LabelEncoder':
                pretty_formula_encode.append(load_dict[key]['pretty_formula'])
            elif Encoder == 'EleEncoder':
                try:
                    pretty_formula_encode.append(EleEncoder(key))
                except:
                    wrongcomp.append(key)

            oxide_type.append(load_dict[key]['oxide_type'])
            Bulk_modulus.append(load_dict[key]['elasticity']['G_Voigt_Reuss_Hill'])
            Shear_modulus.append(
                load_dict[key]['elasticity']['K_Voigt_Reuss_Hill'])
    Symbol = np.array(symbol)
    Hall = np.array(hall)
    Point_group = np.array(point_group)
    Pretty_formula_encode = np.array(pretty_formula_encode)
    Crystal_system = np.array(crystal_system)
    Oxide_type = np.array(oxide_type)
    Bulk_modulus = np.array(Bulk_modulus)
    Shear_modulus = np.array(Shear_modulus)

    encoder = LabelEncoder()
    Symbol = encoder.fit_transform(Symbol)
    Hall = encoder.fit_transform(Hall)
    Point_group = encoder.fit_transform(Point_group)
    Crystal_system = encoder.fit_transform(Crystal_system)
    Oxide_type = encoder.fit_transform(Oxide_type)

    # print(Pretty_formula_encode.shape)
    data_0 = []
    count_00 = 0
    for key in load_dict:
        x_0 = [Pretty_formula_encode[count_00],
               load_dict[key]['energy_per_atom'],
               load_dict[key]['volume']/load_dict[key]['nsites'],
               Symbol[count_00],
               Hall[count_00], Point_group[count_00],
               Crystal_system[count_00],
               load_dict[key]['formation_energy_per_atom'],
               load_dict[key]['density'], Oxide_type[count_00],
               load_dict[key]['nsites']]
        count_00 += 1

        if valuetype == 'eigvalues':
            # y_0 = [[load_dict[key]['elasticity']['elastic_tensor'][0][0], load_dict[key]['elasticity']['elastic_tensor'][0][1], 0],
            #        [load_dict[key]['elasticity']['elastic_tensor'][0][1],
            #         load_dict[key]['elasticity']['elastic_tensor'][0][0], 0],
            #        [0, 0, load_dict[key]['elasticity']['elastic_tensor'][3][3]]
            #        ]
            y_0 = load_dict[key]['elasticity']['elastic_tensor']
            # print(y_0)
            if np.linalg.eigvals(y_0)[index+1] <= 1e-6:
                y_0 = [0.]
            else:
                y_0 = [np.linalg.eigvals(y_0)[index+1]]

        elif valuetype == 'realvalues':
            if index == 0:
                # DNN 0.60, RFR 0.57
                if load_dict[key]['elasticity']['elastic_tensor'][0][0] <= 1e-6:
                    y_0 = [0.]
                else:
                    y_0 = [load_dict[key]['elasticity']
                           ['elastic_tensor'][0][0]]
            elif index == 1:
                if load_dict[key]['elasticity']['elastic_tensor'][0][1] <= 1e-6:
                    y_0 = [0.]
                else:
                    y_0 = [load_dict[key]['elasticity']
                           ['elastic_tensor'][0][1]]
            elif index == 2:
                if load_dict[key]['elasticity']['elastic_tensor'][3][3] <= 1e-6:
                    y_0 = [0.]
                else:
                    y_0 = [load_dict[key]['elasticity']
                           ['elastic_tensor'][3][3]]
                # print(valuetype)

        elif valuetype == 'modulus':
            if index == 0:
                if load_dict[key]['elasticity']['G_Voigt_Reuss_Hill'] <= 1e-6:
                    y_0 = [0.]
                else:
                    y_0 = [load_dict[key]['elasticity']['G_Voigt_Reuss_Hill']]
            if index == 1:
                if load_dict[key]['elasticity']['K_Voigt_Reuss_Hill'] <= 1e-6:
                    y_0 = [0.]
                else:
                    y_0 = [load_dict[key]['elasticity']['K_Voigt_Reuss_Hill']]

        data_0.append((x_0, y_0))

    input_1 = []
    input_2 = []
    #xx_0 = []
    yy_0 = []
    input_1 = []
    for i in range(len(data_0)):

        if isinstance(data_0[i][1][0], complex) is False:
            input_1.append(data_0[i][0][0])
            input_2.append(data_0[i][0][1:])

            # xx_0.append(data_0[i][0])
            yy_0.append(data_0[i][1])
    input_1 = np.array(input_1, dtype = float)
    input_2 = np.array(input_2)

        # if Encoder == 'EleEncoder':
        #     xx_0 = np.array(xx_0, dtype=np.float64)
        # else:
        #     xx_0 = np.array(xx_0)

        # yy_0 = np.array(yy_0)

    # Remove outliers
    yaxis = []
    for i in range(len(yy_0)):
        yaxis.append(yy_0[i][0])

    std_0 = np.std(yaxis)
    mean_0 = np.mean(yaxis)
    lst_0 = []
    count_0 = 0
    for i in range(len(yaxis)):
        count_0 += 1
        if abs(yaxis[i] - mean_0) > std_0 * 2:
            lst_0.append(i)

    yy_0 = np.delete(yy_0, lst_0, axis=0)
    input_1 = np.delete(input_1, lst_0, axis=0)
    input_2 = np.delete(input_2, lst_0, axis=0)
    len(data_0[0][0][1:])

    #print(input_1.shape)
   #X = np.concatenate((input_1, input_2), axis=1)
    return input_1, input_2, yy_0




