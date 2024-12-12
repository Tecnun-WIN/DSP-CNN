# -*- coding: utf-8 -*-

#import cupy as cp

"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random, keyboard, time

def filter_cwt(X, thr):
    
    X_copy = X.copy() # Convert input array to CuPy array
    
    X_copy[X_copy < thr] = 0
    
    return X_copy # Convert the filtered array back to NumPy array and return

def make_confusion_matrix_final(
    cf, ax=None,
    cbar=True,
    cmap='Blues',
    annot=True,
    xticklabels='auto',
    yticklabels='auto',
    xyplotlabels=True,
    sum_stats=True,
):
    '''
    Pretty plot of sklearn confusion matrix
    
    cf : confusion matrix
    figsize : Tuple representing the figure size
    cbar : If True, show the color bar. Default value is True. 
    cmap : Colormap of the values displayed. Default value is 'Blues'
    annot : If True, write the data value in each cell. If an array-like with the same shape as data, then use this to annotate heatmap instead of data.
    xticklabels : List of strings containing the categories to be displayed on the x axis. Default value is 'auto' (densely plot non-overlapping lables)
    yticklabels : List of strings containing the categories to be displayed on the y axis. Default value is 'auto' (densely plot non-overlapping lables)
    xyplotlabels : If True, show 'True Label' and 'Predicted Label' on the figure. Default value: True
    sumstats : If Tue, display summary statistics below the figure
    '''
    
    if sum_stats:
        accuracy = np.trace(cf) / float(np.sum(cf))
        stats_text = '\nAccuracy={:0.3f}'.format(accuracy)
    else:
        stats_text = ''
    #fig = plt.figure(figsize=figsize)
    
    sns.heatmap(cf, cmap=cmap, cbar=cbar, annot=annot, xticklabels=xticklabels, yticklabels=yticklabels, ax=ax)
    
    if xyplotlabels:
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label' + stats_text)
    else:
        ax.set_xlabel(stats_text)

def testing(le, model, X_test, y_test):
    #predict_proba = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        predict_proba = model.predict_proba(X_test)
        y_pred = np.argmax(predict_proba, axis=1)
        
    else:
        predict_proba = model.predict(X_test)
        y_pred = np.argmax(predict_proba, axis=1)
    
    categories = le.inverse_transform(np.union1d(np.unique(y_pred), np.unique(y_test)))
    labels_test = le.inverse_transform(y_test)
    labels_pred = le.inverse_transform(y_pred)
    cf = confusion_matrix(labels_test, labels_pred)
    
    fig, ax = plt.subplots(figsize=(10,8))
    return make_confusion_matrix_final(cf, ax=ax, cbar=False, xticklabels=categories, yticklabels=categories)

def testing_bit_wise(X_test, y_test, model, show_cm=True, figsize=(10,8)):
    y_pred = model.predict(X_test)
    y_pred = np.squeeze(np.array(y_pred)) #Squeeze is used to remove the last dimension
    y_pred = y_pred.T
    
    y_test_cm = [''.join(map(str, row)) for row in y_test] #Making a labels list made of strings
    
    y_test_cm = [''.join(map(str, row)) for row in y_test]
    y_pred_cm = [''.join(map(str, row)) for row in y_pred.round().astype('uint8')]

    categories = np.union1d(np.unique(y_pred_cm), np.unique(y_test_cm))

    cm = confusion_matrix(y_test_cm, y_pred_cm)
    
   
    proba_array = y_pred.ravel().reshape(-1,1)
    proba_array = np.concatenate((proba_array, proba_array.round().astype('uint8')), axis=1)
    bool_idx = proba_array[:,1] == y_test.ravel()
    pred_qual = np.array(['Wrong'] * proba_array.shape[0]).reshape(-1,1)
    pred_qual[bool_idx] = 'Right'

    df_pred = pd.DataFrame(data=proba_array, columns=['Probabilities', 'Bits'])
    df_pred['Probabilities'] = df_pred['Probabilities'].astype(float)
    df_pred['Bits'] = df_pred['Bits'].astype('uint8')

    df_pred['Quality'] = pred_qual.astype(str)
    
    # Plotting confusion matrix
    if show_cm:
        _, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios':[0.75, 0.25]})
        make_confusion_matrix_final(cm, xticklabels=categories, yticklabels=categories, ax=axes[0], cbar=False)
        sns.boxplot(data=df_pred, x='Bits', y='Probabilities', hue='Quality', ax=axes[1])
    
    return df_pred, y_pred, y_test_cm, y_pred_cm


def explore_dataset(X, freq, title):
    paused = False
    fig, ax = plt.subplots(figsize=(10, 6))
    
    while True:
        ax.clear()
        rand_idx = random.randint(0, len(X)-1)
        #print(rand_idx)
        signal = X[rand_idx]
        
        ax.plot(freq, signal)
        ax.set_title(title[rand_idx])
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Magnitude (dB)')
        
        plt.pause(1)
        
        if keyboard.is_pressed(' '):
            paused = not paused
            while paused:
                if keyboard.is_pressed(' '):
                    paused = False
                time.sleep(0.1)
        elif keyboard.is_pressed('q'):
            break
            
        time.sleep(0.1)
        
        plt.close(fig)