import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from hyperopt import hp
from hyperopt import STATUS_OK
from pprint import pprint
import os
import glob
import smogn
import random
from random import randrange
from sklearn.ensemble import RandomForestRegressor
path = os.getcwd()
xlsx_files = glob.glob(os.path.join(path, "*.xlsx"))



def nashsutcliffe(evaluation, simulation):
    """
    Nash-Sutcliffe model efficinecy
        .. math::
         NSE = 1-\\frac{\\sum_{i=1}^{N}(e_{i}-s_{i})^2}{\\sum_{i=1}^{N}(e_{i}-\\bar{e})^2} 
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :return: Nash-Sutcliff model efficiency
    :rtype: float
    """
    if len(evaluation) == len(simulation):
        s, e = np.array(simulation), np.array(evaluation)
        # s,e=simulation,evaluation
        mean_observed = np.nanmean(e)
        # compute numerator and denominator
        numerator = np.nansum((e - s) ** 2)
        denominator = np.nansum((e - mean_observed)**2)
        # compute coefficient
        return 1 - (numerator / denominator)

    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan


k_out=[]
samp_method_out=[]
rel_thres_out=[]
rel_xtrm_type_out=[]
rel_coef_out=[]

mae_out=[]
nse_out=[]

for f in xlsx_files:
    for i in range(100): #number of new sintetic datasets
        # Import data
        data = pd.read_excel(f, index_col=0, header=0)
        data.columns = ['obsWT','obsAirT','discharge','radiation','max_temp','min_temp','month','day']
        
        
        #Define data
        X = data.iloc[:, 1:]
        y = data.loc[:, ['obsWT']]
        
        
        #Split data
        XTrain, XTest, yTrain, yTest = train_test_split(X, y, train_size=0.7, test_size=0.3, shuffle=False, stratify=None)
        
        data_train = pd.DataFrame(np.column_stack([XTrain, yTrain]), columns=['obsAirT','discharge','radiation','max_temp','min_temp','month','day','obsWT'])

        #SMOGN parameters optimization range
        k=randrange(1, 10)
        d = {'extreme':'extreme', 'balance':'balance'}
        samp_method = random.choice(list(d.values()))
        rel_thres =random.uniform(0, 1) #real number between 0 and 1
        
        c = {'high':'high', 'both':'both'}
        rel_xtrm_type = random.choice(list(c.values()))
        rel_coef = random.uniform(0.01, 0.4)
        
        
        k_out.append(k)
        samp_method_out.append(samp_method)
        rel_thres_out.append(rel_thres)
        rel_xtrm_type_out.append(rel_xtrm_type)
        rel_coef_out.append(rel_coef)

        #run SMOGN
        try:
            data_smogn = smogn.smoter(
            
            data = data_train, 
            y = 'obsWT', 
            k = k, 
            samp_method = samp_method,
            rel_thres = rel_thres, #It specifies the threshold of rarity. The higher the threshold, the higher the over/under-sampling boundary. The inverse is also true, where the lower the threshold, the lower the over/under-sampling boundary. 
            rel_method = 'auto',
            rel_xtrm_type = rel_xtrm_type,
            rel_coef = rel_coef) #It specifies the box plot coefficient used to automatically determine extreme and therefore rare "minority" values in y
            X = data_smogn.drop(columns=['obsWT'])
            y = data_smogn[['obsWT']]
            smogn_out=pd.concat([X,y],axis=1)
            
            #Export results to Excel
            writer = pd.ExcelWriter(os.path.basename(f)+ 'SMOGN_out'+str(i)+'.xlsx')
            smogn_out.to_excel(writer,'train')
            writer.save()
            
            #Define train and test datasets
            XTrain = smogn_out.loc[:, smogn_out.columns != 'obsWT']
            yTrain = smogn_out.loc[:, ['obsWT']].values.ravel()            
            XTest = XTest
            yTest = yTest.values.ravel()
            
            #===============================================================================
            # Retrieve best hyperopt parameters
            #===============================================================================
            
            def getBestModelfromTrials(trials):
                valid_trial_list = [trial for trial in trials if STATUS_OK == trial['result']['status']]
                losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
                index_having_minumum_loss = np.argmin(losses)
                best_trial_obj = valid_trial_list[index_having_minumum_loss]
                return best_trial_obj['result']['Trained_Model']
            
            #===============================================================================
            # Define parameters in hyperopt
            #===============================================================================
            
            
            def uniform_int(name, lower, upper):
                # `quniform` returns:
                # round(uniform(low, high) / q) * q
                return hp.quniform(name, lower, upper, q=1)
            
            def loguniform_int(name, lower, upper):
                # Do not forget to make a logarithm for the
                # lower and upper bounds.
                return hp.qloguniform(name, np.log(lower), np.log(upper), q=1)
            
            
            
            parameter_space = {'max_features':hp.choice('max_features', ['auto','sqrt']),
            'bootstrap':hp.choice('bootstrap', [0, 1]),
            'n_estimators': hp.uniform('n_estimators', 50, 2000),
            'max_depth': hp.uniform('max_depth', 10, 1000),
            'min_samples_split': hp.uniform('min_samples_split', 2, 10),}
            
            
                
            #===============================================================================
            # Construct a function that we want to minimize
            #===============================================================================
            
            score_out=[]
            parameters_out=[]
            maetrain_out=[]
            maeTest_out=[]
            
            def train_network(parameters):
                parameters_out.append(parameters)
                print("Parameters:")
                pprint(parameters)
                print()    
                
                max_features = parameters['max_features']
                bootstrap = parameters['bootstrap']
                n_estimators = int(parameters['n_estimators'])
                max_depth = int(parameters['max_depth'])
                min_samples_split = int(parameters['min_samples_split'])
                
                XScaler = StandardScaler()
                XScaler.fit(XTrain)
                XTrainScaled = XScaler.transform(XTrain)
                XTestScaled = XScaler.transform(XTest)
            
                yScaler = StandardScaler()
                yScaler.fit(yTrain.reshape(-1, 1))
                yTrainScaled = yScaler.transform(yTrain.reshape(-1, 1)).ravel()
                yTestScaled = yScaler.transform(yTest.reshape(-1, 1)).ravel()
            
            
                regressor = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, max_features = max_features , bootstrap = bootstrap)
                regressor.fit(XTrainScaled,yTrainScaled) 
            
            
                # Train
                
                score = regressor.score(XTestScaled, yTestScaled)
            
            
                yEstTrain = yScaler.inverse_transform(regressor.predict(XTrainScaled).reshape(-1, 1)).ravel()
                yEstTest = yScaler.inverse_transform(regressor.predict(XTestScaled).reshape(-1, 1)).ravel()
                maetrain = np.mean(np.abs(yTrain-yEstTrain))
                maeTest = np.mean(np.abs(yTest-yEstTest))
                  
                print("Final score: {}".format(score))
                #print("Accuracy: {:.2%}".format(accuracy))
                   
                maetrain_out.append(maetrain)
                maeTest_out.append(maeTest)
                
                score_out.append(score)
                
                return {'loss':maeTest, 'status': STATUS_OK, 'Trained_Model':regressor}
            
            #===============================================================================
            # run hyperparameter optimization
            #===============================================================================
            import inspect
            import sklearn
            import hyperopt
            from functools import partial
            
            # Object stores all information about each trial.
            # Also, it stores information about the best trial.
            trials = hyperopt.Trials()
            
            tpe = partial(
                hyperopt.tpe.suggest,
            
                # Sample 1000 candidate and select candidate that
                # has highest Expected Improvement (EI)
                n_EI_candidates=10,
            
                # Use 20% of best observations to estimate next
                # set of parameters
                gamma=0.2,
            
                # First 20 trials are going to be random
                n_startup_jobs=10,
            )
            
            hyperopt.fmin(
                train_network,
            
                trials=trials,
                space=parameter_space,
            
                # Set up TPE for hyperparameter optimization
                algo=tpe,
            
                # Maximum number of iterations. Basically it trains at
                # most 200 networks before selecting the best one.
                max_evals=100,
            )
            
            
            # Export parameters
            df2 = pd.DataFrame (parameters_out)
            df2.to_csv(os.path.basename(f)+'parameters'+str(i)+'.csv')
   
            bootstrap=0
            model = getBestModelfromTrials(trials)
            parameters=model.get_params()
            
            bootstrap=(parameters['bootstrap'])
            max_depth=round(parameters['max_depth'])
            max_features=parameters['max_features']
            min_samples_split=parameters['min_samples_split']
            n_estimators=parameters['n_estimators']
          
            
            
            #===============================================================================
            # run random forest model
            #===============================================================================
            
            
            XScaler = StandardScaler()
            XScaler.fit(XTrain)
            XTrainScaled = XScaler.transform(XTrain)
            XTestScaled = XScaler.transform(XTest)
            
            yScaler = StandardScaler()
            yScaler.fit(yTrain.reshape(-1, 1))
            yTrainScaled = yScaler.transform(yTrain.reshape(-1, 1)).ravel()
            yTestScaled = yScaler.transform(yTest.reshape(-1, 1)).ravel()
            
            regressor = RandomForestRegressor(n_estimators =n_estimators, max_depth =max_depth, min_samples_split = min_samples_split, max_features = max_features , bootstrap = bootstrap)
            regressor.fit(XTrainScaled,yTrainScaled)
            
            #Actual value and the predicted value
            
            yEstTrain = yScaler.inverse_transform(regressor.predict(XTrainScaled).reshape(-1, 1)).ravel()
            yEstTest = yScaler.inverse_transform(regressor.predict(XTestScaled).reshape(-1, 1)).ravel()
            
            
            results_train = pd.DataFrame({'obs': yTrain, 'est': yEstTrain}, index = XTrain.index)
            results_test = pd.DataFrame({'obs': yTest, 'est': yEstTest }, index = XTest.index)
            
               
            #Export results to Excel
            writer = pd.ExcelWriter(os.path.basename(f)+ 'trainRF'+str(i)+'.xlsx')
            results_train.to_excel(writer,'train')
            writer.save()
            
            writer = pd.ExcelWriter(os.path.basename(f)+ 'testRF'+str(i)+'.xlsx')
            results_test.to_excel(writer,'test')
            writer.save()
            
            #coefficients.to_excel(writer,'coefficients')
            
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
            
            nse=nashsutcliffe(yTest, yEstTest)
            
            mae_out.append(maeTes)
            nse_out.append(nse)
            # Plot results

               
            print("NN MAE: %f (All), %f (Test) " % (mae, maeTes))
            print ("NN MSE: %f (All), %f (Test) " % (mse, mseTes))
            print ("NN R2: %f (All), %f (Test) " % (r2, r2Tes))
                
                
            
            #===============================================================================
            # Predict with unseen data
            #===============================================================================
            
            dataR = pd.read_excel(f, index_col=0, header=0)
            dataR.columns = ['obsWT','obsAirT','discharge','radiation','max_temp','min_temp','month','day']
            
            
            #Define data
            XR = dataR.iloc[:, 1:]
            yR = dataR.loc[:, ['obsWT']]
            
            
            XTestR = XR
            yTestR = yR.values.ravel()
            XTestScaledR = XScaler.transform(XTestR)
            yTestScaledR = yScaler.transform(yTestR.reshape(-1, 1)).ravel()
            yTestPredictedR = yScaler.inverse_transform(regressor.predict(XTestScaledR).reshape(-1, 1)).ravel()
            
            results2 = pd.DataFrame({'obs': yTestR, 'est': yTestPredictedR}, index = XTestR.index)
            
            
            #===============================================================================
            #Export results to Excel
            writer = pd.ExcelWriter(os.path.basename(f)+'results'+'.xlsx')
            results2.to_excel(writer,'results')
            writer.save()
            #==============================================================================

        
        except:
              pass
    
    model_out = pd.DataFrame( {'mae_out': mae_out, 'nse_out': nse_out})
    writer = pd.ExcelWriter('A -'+os.path.basename(f)+'model_out'+str(i)+'.xlsx')
    model_out.to_excel(writer,'mae_all')
    writer.save()
    
    
    SMOGN_parameters_out = pd.DataFrame( {'k': k_out,'samp_method': samp_method_out, 'rel_thres': rel_thres_out,'rel_xtrm_type': rel_xtrm_type_out, 'rel_coef': rel_coef_out})
    #Export results to Excel
    writer = pd.ExcelWriter(os.path.basename(f)+'SMOGN_parameters_out'+str(i)+'.xlsx')
    SMOGN_parameters_out.to_excel(writer,'results')
    writer.save()
