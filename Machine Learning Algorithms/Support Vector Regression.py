# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 08:44:24 2022

@author: mcva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import glob

path = os.getcwd()
xlsx_files = glob.glob(os.path.join(path, "*.xlsx"))


for f in xlsx_files:
    # Import data
    
    data = pd.read_excel(f, index_col=0, header=0)
    data.columns = ['obsWT','obsAirT','discharge','radiation','max_temp','min_temp','month','day']
    
    
    #Define data
    X = data.iloc[:, 1:]
    y = data.loc[:, ['obsWT']]
    
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, train_size=0.7, test_size=0.3, shuffle=False, stratify=None)
    
    XTrain = XTrain
    yTrain = yTrain.values.ravel()
    XTest = XTest
    yTest = yTest.values.ravel()

    XScaler = StandardScaler()
    XScaler.fit(XTrain)
    XTrainScaled = XScaler.transform(XTrain)
    XTestScaled = XScaler.transform(XTest)
    
    yScaler = StandardScaler()
    yScaler.fit(yTrain.reshape(-1, 1))
    yTrainScaled = yScaler.transform(yTrain.reshape(-1, 1)).ravel()
    yTestScaled = yScaler.transform(yTest.reshape(-1, 1)).ravel()
    
    #Fitting the SVR model
    regressor = SVR(kernel = 'rbf', degree = 1, C= 1, gamma=0.1, epsilon=0.1)
    regressor.fit(XTrainScaled,yTrainScaled)
    
    
    #Observed value versus predicted value
    
    yEstTrain = yScaler.inverse_transform(regressor.predict(XTrainScaled).reshape(-1, 1)).ravel()
    yEstTest = yScaler.inverse_transform(regressor.predict(XTestScaled).reshape(-1, 1)).ravel()
    
    
    results_train = pd.DataFrame({'obs': yTrain, 'est': yEstTrain}, index = XTrain.index)
    results_test = pd.DataFrame({'obs': yTest, 'est': yEstTest }, index = XTest.index)
    
       
    #Export results to Excel
    writer = pd.ExcelWriter('1-'+os.path.basename(f)+'train'+'.xlsx')
    results_train.to_excel(writer,'results_train')
    writer.save()
    
    writer = pd.ExcelWriter('2-'+os.path.basename(f)+'test'+'.xlsx')
    results_test.to_excel(writer,'results_test')
    writer.save()
    
    
    # Metrics
    mae = np.mean(np.abs(yTrain-yEstTrain))
    mse  = np.mean((yTrain-yEstTrain)**2)
    mseTes = np.mean((yTest-yEstTest)**2)
    maeTes = np.mean(np.abs(yTest-yEstTest))
    meantrain = np.mean(yTrain)
    ssTest = (yTrain-meantrain)**2
    r2=(1-(mse/(np.mean(ssTest))))
    meantest = np.mean(yTest)
    ssTrain = (yTest-meantest)**2
    r2Tes=(1-(mseTes/(np.mean(ssTrain))))
    
    
    # Plot results
    print("NN MAE: %f (All), %f (Test) " % (mae, maeTes))
    print ("NN MSE: %f (All), %f (Test) " % (mse, mseTes))
    print ("NN R2: %f (All), %f (Test) " % (r2, r2Tes))
    
