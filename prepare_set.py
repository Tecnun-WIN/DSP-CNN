# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 12:29:00 2023

@author: jfodopsokou
"""
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

from os import listdir
from os.path import isfile, join, splitext
import scipy.io

import matplotlib.pyplot as plt
import seaborn as sns

def make_set(source_path, result_path, file_name):
    mypath = source_path

    all_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    files_name = [string for string in all_files if splitext(string)[-1]=='.mat']

    columns = [a for a in range(0,1601)] + ['Tag_names', 'Interval']
    df = pd.DataFrame(columns = columns)
    for name in files_name:    
        tag_name = name[:13]
        interval = name[-13:-6]
        data = scipy.io.loadmat(mypath[:-1] + name)['dataMags'].T

        df_dummy = pd.DataFrame(data)

        df_dummy['Tag_names'] = [tag_name]*data.shape[0]
        df_dummy['Interval'] = [interval]*data.shape[0]

        data = np.concatenate((df.values, df_dummy.values), axis=0)

        df = pd.DataFrame(columns=columns, data=data)

    df.reset_index(drop=True, inplace=True)

    pca = PCA()
    X_pca = pca.fit_transform(df.values[:,:1601])

    plt.figure(figsize=(12,8))

    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], 
                    hue=df.Tag_names,
                    palette = sns.color_palette("hls", np.unique(df.Tag_names, return_counts=False).shape[0]),
                    legend='full',
                    style=df.Interval)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    df.to_csv(result_path + file_name, index=False)
    
    return df

def load_matlab_file_v7(file, dataMags=False, freq=False):
    '''

    Parameters
    ----------
    file : h5py
        h5py reference of the dataset.
    dataMags : boolean, optional
        If True the magnitud of the measurements is returned. The default is False.
    freq : boolean, optional
        If True the vector of frequencies in frequency domain is return

    Returns
    -------
    Dictionary
        Dictionary containing the data.

    '''
    dataset = {}
    dataset = dict()
    for key in file.keys():
        if not key.startswith('#'):
            #print(key)
            if key == 'labels' or key == 'intervals':
                ref = file[key]
                my_list = []
                
                for item in ref:
                    my_list += [''.join([chr(char[0]) for char in file[item[0]]])]
            
                dataset[key] = np.array(my_list)
            else:
                #print(key)
                dataset[key] = np.array(file[key]).T
        
    return dataset