# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 12:13:42 2023

@author: jfodopsokou
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns


def prob_well_bad(estimator, X, y, n_splits=5, stripplot=False, legend=False,
                  random_state=None, mode='testing',
                  figsize=(20,6), title='', 
                  x_rot=0, height=5, aspect=1.5):
    '''
    

    Parameters
    ----------
    estimator : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    n_splits : TYPE, optional
        DESCRIPTION. The default is 5.
    stripplot : TYPE, optional
        DESCRIPTION. The default is False.
    legend : TYPE, optional
        DESCRIPTION. The default is False.
    random_state : TYPE, optional
        DESCRIPTION. The default is None.
    mode : TYPE String
        DESCRIPTION. The default value is 'testing', Possibel values: 'testing','training'
        If training, the splitting strategy is used to make cross-validation. 
        If testing, the function only make a test using the actual parameters of the model.
    figsize : TYPE, optional
        DESCRIPTION. The default is (20,6).
    title : TYPE, optional
        DESCRIPTION. The default is ''.
    x_rot : TYPE, optional
        DESCRIPTION. The default is 0.
    height : TYPE, optional
        DESCRIPTION. The default is 5.
    aspect : TYPE, optional
        DESCRIPTION. The default is 1.5.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    '''
    
    
    if mode == 'training':
        
        data = np.empty((0,3))
        estimator['clf'].set_params(random_state=random_state)
        
        kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        for train_index, test_index in kf.split(X):
            estimator.fit(X[train_index], y[train_index])
            
            if hasattr(estimator, 'predict_proba'): #checking if the model has predict_proba attrib (sklearn)
                prob = estimator.predict_proba(X[test_index]).max(axis=1).reshape(-1,1)
                y_pred = estimator.predict(X[test_index])
                good = (y_pred == y[test_index]).reshape(-1,1)
            else: #if the model does not have 'predict_proba' attribut (keras), the 'predict' attrib return probabilities
                proba_mat = estimator.predict(X[test_index]) #matrix of the probabilities   
                prob =proba_mat.max(axis=1).reshape(-1,1)
                y_pred = np.argmax(proba_mat, axis=1)
                good = (y_pred == y[test_index]).reshape(-1,1)
            
            data_dum = np.concatenate((prob, good,
                                       y_pred.reshape(-1,1)),
                                       #y[test_index].reshape(-1,1)), 
                                       axis=1)
            data = np.concatenate((data, data_dum))
    
    else:
        if hasattr(estimator, 'predict_proba'):
            prob = estimator.predict_proba(X).max(axis=1).reshape(-1,1)
            y_pred = estimator.predict(X)
            good = (y_pred == y).reshape(-1,1)
        else:
            proba_mat = estimator.predict(X)
            prob = proba_mat.max(axis=1).reshape(-1,1)
            #print(prob.shape)
            y_pred = np.argmax(proba_mat, axis=1).reshape(-1,1)
            #print(y_pred.dtype)
            good =( y_pred.ravel() == y.ravel()).reshape(-1,1)
            #print(y.dtype)
        
        data = np.concatenate((prob, good, 
                               y_pred.reshape(-1,1)), #y.reshape(-1,1)), 
                              axis=1)
        
    columns = ['probabilities', 'good_predictions', 'labels']
    resume_for_box_plot = pd.DataFrame(columns=columns, data = data)
    
    '''
    prob = estimator.predict_proba(X).max(axis=1).reshape(-1,1)
    good = (estimator.predict(X) == y).reshape(-1,1)
    data = np.concatenate((prob, good, y.reshape(-1,1)), axis=1)
    columns = ['probabilities', 'good_predictions', 'labels']
    resume_for_box_plot = pd.DataFrame(columns=columns, data = data)
    '''
    resume_for_box_plot['labels'] = resume_for_box_plot['labels'].astype('int')
    
    resume_for_box_plot.sort_values(by='labels', inplace=True)
    
    plt.figure(figsize=figsize)
    fig = sns.catplot(x="labels", y="probabilities",
                hue="good_predictions",
                data=resume_for_box_plot, kind="box",
                orient='v',
                height=height, aspect=aspect, legend=legend)
    
    if stripplot: 
        #ax=
        sns.stripplot(x="labels", y="probabilities",
                    hue="good_predictions",
                    data=resume_for_box_plot,
                    palette='dark', orient='v',
                    #color='black',
                    size=5, 
                    #legend=False
                    )
    
    plt.xticks(rotation=x_rot, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('Probabilities', fontdict={'fontsize': 16, 'fontweight': 'normal'})
    plt.xlabel('Predicted label', fontdict={'fontsize': 16, 'fontweight': 'normal'})
    plt.title(title, fontdict={'fontsize': 14, 'fontweight': 'bold'})
    
    return fig, good, y, y_pred
    
def make_confusion_matrix(cf,cf2,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,
                          x_rotation=0,
                          y_rotation=0,
                          ax=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        predictions = cf.flatten()
        dropped = predictions - cf2.flatten()
        group_counts = ["{0:0.0f}".format(value) for value in predictions]
        
        for i in range(len(group_counts)):
            if predictions[i] != 0:
                group_counts[i] +=  " ({0:0.0f})".format(dropped[i])
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf[:,:cf.shape[0]]))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            #acc_bef = np.diagonal(cf).sum() / cf.sum() #accuracy before filtering
            acc_after = np.trace(cf2) / cf2.sum() #accuracy after filtering
            good_conserv = np.trace(cf2) / np.trace(cf)
            bad_conserv = (cf2.sum() - np.trace(cf2)) / (cf.sum() - np.trace(cf))
            conserv = cf2.sum() / cf.sum()
            stats_text = "\n\nConserved good predictions ratio={:0.4f}\nConserved bad predictions ratio={:0.4f}".format(good_conserv, bad_conserv)
            stats_text += "\nConverved predictions ratio={:0.4f}".format(conserv)
            stats_text += "\nAccuracy before filtering={:0.4f}\nAccuracy after filtering={:0.4f}".format(accuracy, acc_after)
            #stats_text = "\n\nAccuracy before filtering={:0.4f}\nAccuracy after filtering={:0.4f}\nConserved Good Prediction ratio={:0.4\nConserved Bad Prediction ratio={:0.4f}}".format(accuracy, acc_after, good_conserv, bad_conserv)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    if ax==None:
        fig = plt.figure(figsize=figsize)
        
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,
                annot_kws={'fontsize':9, 'fontweight': 'bold'},
                xticklabels=categories[:cf.shape[1]],
                yticklabels=categories[:cf.shape[0]],
                ax=ax)
    plt.xticks(rotation=x_rotation)
    plt.yticks(rotation=y_rotation)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title, fontdict={'fontsize': 14, 'fontweight': 'bold'})
    
    if ax==None:
        return fig
    else:
        return ax

    
def make_filtered_confusion_matrix(estimator, X, y,
                                   thresholds, extr_thr = 0.95, extr_bool=False,
                                   title='', figsize=(15,12), 
                                   x_rotation=90, y_rotation=0,
                                   ax=None):

    categories = estimator.classes_
    le = LabelEncoder()
    le.fit(estimator.classes_)

    #filtering
    matrix = np.concatenate(
    (y.reshape(-1,1), estimator.predict_proba(X).max(axis=1).reshape(-1,1)), axis=1)
    matrix = np.concatenate((matrix, np.asarray([thresholds[i] for i in le.transform(y)]).reshape(-1,1)), axis=1)
    matrix = np.concatenate((matrix, np.asarray([extr_thr]*matrix.shape[0]).reshape(-1,1)), axis=1)

    if not extr_bool:
        upper = matrix[:,-3] > matrix[:,-2]
    else:
        upper = matrix[:,-3] > matrix[:,-1]
        
        
    pred = estimator.predict(X[upper])
    
    '''
    cf_matrix below is inicialized to ensure that all the categories will be
    represented on both x and y axis of the confusion matrix. cf_matrix2 is 
    inicialized the same way. This also ensure that cf_matrix and cf_matrix2
    have the same shape.
    '''
    cf_matrix = np.zeros((len(categories), len(categories)))

    for cont,res in zip(y,estimator.predict(X)):
        cf_matrix[le.transform(np.array(cont).ravel()), le.transform(np.array(res).ravel())] += 1

    cf_matrix2 = np.zeros((len(categories), len(categories)))


    cf_matrix2 = np.zeros((len(categories), len(categories)))

    for cont,res in zip(y[upper],pred):
        cf_matrix2[le.transform(np.array(cont).ravel()), le.transform(np.array(res).ravel())] += 1

    '''
    unassigned = np.zeros((len(np.unique(categories)),1))
    cf_matrix_sum = cf_matrix2.sum(axis=1)

    nb_per_class = np.unique(y, return_counts=True)[1]

    for i in range(0,len(np.unique(y))):
        unassigned[i] = nb_per_class[i] - cf_matrix_sum[i]

    cf_matrix_ext = np.concatenate((cf_matrix2,unassigned),axis=1)
    '''
    
    cf_matrix_ext = cf_matrix2
    make_confusion_matrix(cf_matrix, cf_matrix_ext,
                          categories=categories,
                          cbar = False,
                          percent = False,
                          sum_stats=False,
                          figsize=figsize,
                          title=title,
                          y_rotation=y_rotation,
                          x_rotation=x_rotation,
                          ax=ax)
    
    #the next line gave some problem. Look ''ax''
    #ax.set_xticklabels(np.unique(y), rotation = x_rotation, fontdict={'fontsize': 10, 'fontweight': 'bold'})
    #ax.set_yticklabels(np.unique(y), rotation = y_rotation, fontdict={'fontsize': 10, 'fontweight': 'bold'})
    
    return upper