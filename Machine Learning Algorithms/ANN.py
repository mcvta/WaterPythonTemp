import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from neupy import layers, algorithms
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
    
    
    #===============================================================================
    # Neural network
    #===============================================================================
    
    # Define neural network
    

    cgnet = algorithms.Momentum(
        network=[
            layers.Input(XTrain.shape[1]),
            layers.PRelu(41) >> Dropout(0.919),
            layers.PRelu(10) >> Dropout(0.919),
            layers.Linear(1),
        ],
        step=algorithms.step_decay(
            initial_value=0.0492,
            reduction_freq=121.32,
        ),
        loss='mse',
        batch_size=17,
        #regularizer = algorithms.l2(0.000217),
        shuffle_data=True,
        verbose=True,
        show_epoch=500,
    )

    XScaler = StandardScaler()
    XScaler.fit(XTrain)
    XTrainScaled = XScaler.transform(XTrain)
    XTestScaled = XScaler.transform(XTest)
    
    yScaler = StandardScaler()
    yScaler.fit(yTrain.reshape(-1, 1))
    yTrainScaled = yScaler.transform(yTrain.reshape(-1, 1)).ravel()
    yTestScaled = yScaler.transform(yTest.reshape(-1, 1)).ravel()
    
    # Train
    cgnet.train(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, epochs=764)
    yEstTrain = yScaler.inverse_transform(cgnet.predict(XTrainScaled).reshape(-1, 1)).ravel()
    yEstTest = yScaler.inverse_transform(cgnet.predict(XTestScaled).reshape(-1, 1)).ravel()
    
    
    
    results_train = pd.DataFrame({'obs': yTrain, 'est': yEstTrain}, index = XTrain.index).fillna(0)
    results_test = pd.DataFrame({'obs': yTest, 'est': yEstTest }, index = XTest.index).fillna(0)
    
       
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
    
    



