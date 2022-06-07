import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from hyperopt import hp
from pprint import pprint
from sklearn.svm import SVR

# Import data
data = pd.read_excel('st65.xlsx', index_col=0, header=0)
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
   
    'C':hp.choice('C', [0.1,1,100,1000]),
    'kernel':hp.choice('kernel', ['rbf','poly','sigmoid','linear']),
    'degree':hp.choice('degree', [1,2,3,4,5,6]),
    'gamma':hp.choice('gamma', [1, 0.1, 0.01, 0.001, 0.0001]), 
    'epsilon':hp.choice('epsilon', [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10])
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
    
    C = parameters['C']
    kernel = parameters['kernel']
    degree = parameters['degree']
    gamma = parameters['gamma']
    epsilon = parameters['epsilon']
    
    XScaler = StandardScaler()
    XScaler.fit(XTrain)
    XTrainScaled = XScaler.transform(XTrain)
    XTestScaled = XScaler.transform(XTest)

    yScaler = StandardScaler()
    yScaler.fit(yTrain.reshape(-1, 1))
    yTrainScaled = yScaler.transform(yTrain.reshape(-1, 1)).ravel()
    yTestScaled = yScaler.transform(yTest.reshape(-1, 1)).ravel()


    regressor = SVR(kernel = kernel, degree = degree, C= C, gamma=gamma, epsilon=epsilon)
    regressor.fit(XTrainScaled,yTrainScaled) 


    # Train
    
    score = regressor.score(XTestScaled, yTestScaled)


    yEstTrain = yScaler.inverse_transform(regressor.predict(XTrainScaled).reshape(-1, 1)).ravel()
    yEstTest = yScaler.inverse_transform(regressor.predict(XTestScaled).reshape(-1, 1)).ravel()
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
