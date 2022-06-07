import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from hyperopt import hp
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor

# Import data
data = pd.read_excel('st2.xlsx', index_col=0, header=0)
data.columns = ['obsWT','obsAirT','discharge','radiation','max_temp','min_temp','month','day']


#Define data
X = data.iloc[:, 1:]
y = data.loc[:, ['obsWT']]

#XTrain, XTest, yTrain, yTest = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)
XTrain, XTest, yTrain, yTest = train_test_split(X, y, train_size=0.7, test_size=0.3, shuffle=False, stratify=None)

XTrain = XTrain
yTrain = yTrain.values.ravel()
XTest = XTest
yTest = yTest.values.ravel()

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



parameter_space = {
   
    'max_features':hp.choice('max_features', ['auto','sqrt']),
       
    
    'bootstrap':hp.choice('bootstrap', [
    {
        'bootstrap': True,
        
    }, {
        'bootstrap': False,
    },
    ]),
    
    
    'n_estimators': hp.uniform('n_estimators', 50, 2000),
    
    'max_depth': hp.uniform('max_depth', 10, 1000),
    'min_samples_split': hp.uniform('min_samples_split', 2, 10),
    #'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
    
}


    
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
    max_depth = parameters['max_depth']
    min_samples_split = int(parameters['min_samples_split'])
    #min_samples_leaf = parameters['min_samples_leaf']
    
    XScaler = StandardScaler()
    XScaler.fit(XTrain)
    XTrainScaled = XScaler.transform(XTrain)
    XTestScaled = XScaler.transform(XTest)

    yScaler = StandardScaler()
    yScaler.fit(yTrain.reshape(-1, 1))
    yTrainScaled = yScaler.transform(yTrain.reshape(-1, 1)).ravel()
    yTestScaled = yScaler.transform(yTest.reshape(-1, 1)).ravel()


    rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, max_features = max_features , bootstrap = bootstrap, random_state = 42) 
    rf.fit(XTrainScaled,yTrainScaled) 


    # Train
    
    score = rf.score(XTestScaled, yTestScaled)


    yEstTrain = yScaler.inverse_transform(rf.predict(XTrainScaled).reshape(-1, 1)).ravel()
    yEstTest = yScaler.inverse_transform(rf.predict(XTestScaled).reshape(-1, 1)).ravel()
    maetrain = np.mean(np.abs(yTrain-yEstTrain))
    maeTest = np.mean(np.abs(yTest-yEstTest))
      
    #y_predicted = yScaler.inverse_transform(cgnet.predict(XTestScaled).reshape(-1, 1)).ravel()
    #accuracy = accuracy_score(np.array(yTestScaled.argmax(axis=1)), np.array(y_predicted.argmax(axis=1)))

    print("Final score: {}".format(score))
    #print("Accuracy: {:.2%}".format(accuracy))
       
    maetrain_out.append(maetrain)
    maeTest_out.append(maeTest)
    
    score_out.append(score)
    
    return (score)

#===============================================================================
# run hyperparameter optimization
#===============================================================================

import hyperopt
from functools import partial

# Object stores all information about each trial.
# Also, it stores information about the best trial.
trials = hyperopt.Trials()

tpe = partial(
    hyperopt.tpe.suggest,

    # Sample 1000 candidate and select candidate that
    # has highest Expected Improvement (EI)
    n_EI_candidates=1000,

    # Use 20% of best observations to estimate next
    # set of parameters
    gamma=0.2,

    # First 20 trials are going to be random
    n_startup_jobs=20,
)

hyperopt.fmin(
    train_network,

    trials=trials,
    space=parameter_space,

    # Set up TPE for hyperparameter optimization
    algo=tpe,

    # Maximum number of iterations. Basically it trains at
    # most 200 networks before selecting the best one.
    max_evals=200,
)



df1 = pd.DataFrame (score_out)
df2 = pd.DataFrame (parameters_out)
df3 = pd.DataFrame (maetrain_out)
df4 = pd.DataFrame (maeTest_out)

df1.to_csv('score_out.csv') 
df2.to_csv('parameters_out.csv')
df3.to_csv('maetrain_out.csv')
df4.to_csv('maeTest_out.csv')
