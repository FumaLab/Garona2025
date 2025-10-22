#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 13:53:00 2022

@author: fitzvero
Adapted by: Mariana Natalino
"""

############### import all the necessary libraries ###################
    
import FlowCytometryTools
from FlowCytometryTools import FCMeasurement
from FlowCytometryTools import ThresholdGate, PolyGate
from FlowCytometryTools import PolyGate
    
from pylab import *
import itertools
import glob
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import matplotlib
#    matplotlib.use('webagg')
import pandas as pd 
import os
import arrow
import pickle
import csv
import numpy as np

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde

from matplotlib.offsetbox import AnchoredText

##############################################################################

#from mpl_toolkits.axes_grid1.inset_locator import inset_axes
params = {'legend.fontsize': 'x-large',
#         'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
           'axes.titlesize':'x-large',
           'xtick.labelsize':'x-large',
           'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

#% matplotlib inline
    
############################## FUNCTIONS #####################################


     ############## create new directory to store plots ####################
    

def Newpath(my_root_path, 
          subfolder):
    '''

    Parameters
    ----------
    my_root_path : path to main DataFolder
        
    subfolder : path to subfolder (if you have several days/conditions...)
                !!! [0] to choose the files in the first folder
                write subfolders = [subfolder1,
                                    subfolder2,
                                    subfolder3]

    Returns
    -------
    newDataPath

    '''
    
    newDataPath = my_root_path + 'Analysis/'
    
    isExist = os.path.exists(newDataPath)
    print (newDataPath)
    
    if not isExist:
        # Create new storage directory
        os.mkdir(newDataPath)

    return newDataPath

def AnalysisFitness_fromFile(newDataPath, 
                             labels, 
                             colorlabels, 
                             SampleList_CumGenerations,
                             SampleList_lnRatio,
                             well_ind):
     '''
        
        Parameters
        ----------
        my_root_path : TYPE
            DESCRIPTION.
        subfolder : TYPE
            DESCRIPTION.
        Labels : TYPE
            DESCRIPTION.
        colorlabels : TYPE
            DESCRIPTION.
        SampleList_CumGenerations : TYPE
            DESCRIPTION.
        SampleList_lnRatio : TYPE
            DESCRIPTION.
        well_ind : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
    '''
        
############################### Linear Regression ############################
    
##### check whether you have the same number of lists of dataframes as you have samples

     print(str(len(SampleList_CumGenerations)) + '=' + str(len(labels))) 
  
#store all variables from linear regression for each individual replica in list
   
     PredictionsList = []
     dataframe_output_list = []
     dataframe_output_values = pd.DataFrame(columns = ['Labels', 
                                                      'Mean_relFitness', 
                                                      'Std_relFitness'])
    #meanCoefValuesList = []
    #stdCoefValuesList = []
    
    
     for i in range(0,len(well_ind)):
         L = well_ind[i]
         Gen_sample = SampleList_CumGenerations[i]
         Ratio_sample = SampleList_lnRatio[i]
         dataframe_output = pd.DataFrame(columns = ['labels', 
                                                   'Score', 
                                                   'relFitness', 
                                                   'Intercept'])
        
         PredictionsValues = []

         fig_data = plt.figure()                                   
                
         for l in range(0,len(L)):
             R_Gen = 'CumGen' + str(L[l])
             R_Ratio = 'lnRatio' + str(L[l])
             Gen_sample_replica = Gen_sample[R_Gen]
             Ratio_sample_replica = Ratio_sample[R_Ratio]
             npG = Gen_sample_replica.to_numpy()
             npGr = np.array(npG).reshape(-1, 1)
             npR = Ratio_sample_replica.to_numpy()
             npRr = np.array(npR).reshape(-1, 1)
            
############ linear regression model for each individual replica #############           
            
             lm = linear_model.LinearRegression()                  
             model = lm.fit(npGr,npRr)
             predictions = lm.predict(npGr)
             Score = lm.score(npGr,Ratio_sample_replica)
             Coef = lm.coef_
             Intercept = lm.intercept_
            
################# store variables of linear regression model #################
            
             PredictionsValues.append(predictions)                 
             temp = pd.DataFrame([[L[l], Score, Coef, Intercept]], 
                                 columns = ['labels', 'Score', 'relFitness', 'Intercept'])
             dataframe_output = dataframe_output.append(temp)
 
########## plot cumGenerations over the lnRatios for each replica/sample ######
                # and linear regression for each replica/sample #
            
             plt.plot(npGr,npRr, colorlabels[l], markersize=6)     
             #plt.ylim(-2,1)
             #plt.xlim(-5,25)
             plt.plot(npGr, predictions, color='blue',linestyle='dashed', linewidth=0.5)
             #plt.title(labels[i], fontdict=None, loc='center', pad=None)
            
        
         PredictionsList.append(PredictionsValues)
         dataframe_output_list.append(dataframe_output)
         a = dataframe_output['relFitness'].mean()
         b = dataframe_output['relFitness'].std()
         c = dataframe_output['Score'].mean()
         df_values_temp = pd.DataFrame([[labels[i], a, b, c]], 
                                 columns = ['Labels', 'Mean_relFitness', 'Std_relFitness', 'Mean_R2'])
        
         dataframe_output_values = dataframe_output_values.append(df_values_temp)
    
##### save single Values: labels, Score, relFitness, Intercept to csv file ####
        
         dataframe_output.to_csv(newDataPath + labels[i] +"SingleValues_output.csv", index=False)

        
         #plt.ylim(-2,1)
         plt.xlim(-5,25)
         plt.title(labels[i], fontdict=None, loc='center', pad=None)
         plt.xlabel("Generations")
         plt.ylabel("lnRatio")
         fig_data.savefig(newDataPath + labels[i] +'.pdf')        #save plot
        
 
#### save Mean_Values: Mean_rel_Fitness, Std_relFitness and Mean_R2 to csv file ####    
    
     dataframe_output_values.to_csv(newDataPath +"output_MeanValues.csv", index=True)
 
     return dataframe_output_values, dataframe_output_list
 
########################### PLOT RELATIVE FITNESS ############################

def PlotrelFitness(newDataPath, 
                   dataframe_output_values, 
                   dataframe_output_list):
    
    
    ############## Plot relative Fitness ####################
    
     df = pd.DataFrame()

     df['list'] = pd.DataFrame(list(range(1,len(dataframe_output_values)+1)))
    
     print(df['list'])
     
     fig_relFitness = plt.figure(1)
     ax = fig_relFitness.add_subplot(111)
     
     ax.errorbar(df['list'], 
                 dataframe_output_values['Mean_relFitness'], 
                 yerr=dataframe_output_values['Std_relFitness'], 
                 fmt='o', color='Black', elinewidth=1,capthick=1,errorevery=1, alpha=1, ms=4, capsize = 5)
     #plt.bar(df['labels'], df['MEAN'],tick_label = df['labels'])##Bar plot
     #plt.xlabel('labels') ## Label on X axis
     ax.set_ylabel('Relative Fitness') ##Label on Y axis
     ax.set_ylim(-0.45,0.05)
     ax.set_xticks(df['list'])
     ax.set_xticklabels(dataframe_output_values['Labels'], rotation = 90)
    
     for i in range(0,len(dataframe_output_list)):
         ax.plot([df['list'][i]] * len(dataframe_output_list[i]),
                 dataframe_output_list[i]['relFitness'], '.')
    
     plt.tight_layout()
     fig_relFitness.savefig(newDataPath +'Plot_relFitness.pdf')
    
    
    
    
    
       #revtransformedColorRef_gated = ColorRef_gated.transform('hlog', direction='reverse', 
       #                                                        channels=['FSC PMT-A', 'SSC (BV)-A', 
       #                                                        col_FCS], b=500.0)   
        
    # my_list_of_SamplesReplicas = []
    # k = len(Number_Rows) * len(Number_Columns)

    # for i in range(0,k):
    #     x = (0+i)*3
    #     y = x+3
    #     S = Big_datacountsI.iloc[x:y]
    #     my_list_of_SamplesReplicas.append(S)
    
    # print ('This is an example:' + str(my_list_of_SamplesReplicas[5]))
