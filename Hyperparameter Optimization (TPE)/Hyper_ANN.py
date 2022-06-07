import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from neupy import layers, algorithms
from hyperopt import hp
from pprint import pprint


# Import data
data = pd.read_excel('st5.xlsx', index_col=0, header=0)
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
   
    'layers': hp.choice('layers', [{
        'n_layers': 1,
        'n_units_layer': [
            uniform_int('n_units_layer_11', 10, 50),
        ],
    }, {
        'n_layers': 2,
        'n_units_layer': [
            uniform_int('n_units_layer_21', 10, 50),
            uniform_int('n_units_layer_22', 10, 50),
        ],
    }]),
    'act_func_type': hp.choice('act_func_type', [
        layers.Relu,
        layers.PRelu,
        layers.Elu,
        layers.Tanh,
        layers.Sigmoid

    ]),
    

    'epochs_choice':hp.choice('training_parameters', [
    {
        'regularization': True,
        'n_epochs':[hp.quniform('n_epochs1', 500, 1000, q=1)],
    }, {
        'regularization': False,
        'n_epochs':[hp.quniform('n_epochs2', 20, 300, q=1)],
    },
    ]),
    
    'dropout': hp.uniform('dropout', 0, 1.0),
    'batch_size': loguniform_int('batch_size', 5, 20),
    #step
    'initial_value':hp.uniform('initial_value', 0.001,0.1), 
    'reduction_freq':hp.uniform('reduction_freq', 10,200), 
    #regularization
    'decay_rate':hp.uniform('decay_rate',0.0001,0.001),
    #epochs_choice
    'regularization':'regularization'
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
    
    initial_value = parameters['initial_value']
    reduction_freq = parameters['reduction_freq']
    decay_rate = parameters['decay_rate']
    batch_size = int(parameters['batch_size'])
    proba = parameters['dropout']
    activation_layer = parameters['act_func_type']
    
    layer_sizes = [int(n) for n in parameters['layers']['n_units_layer']]
    
    n_epochs = [int(n) for n in parameters['epochs_choice']['n_epochs']]
    
    regularization = parameters['regularization']

    if regularization:
        regularizer = algorithms.l2(decay_rate)
    else:
        regularizer = False 
        
        
    network = layers.Input(XTrain.shape[1])
    
    for layer_size in layer_sizes:
        network=Input(XTrain.shape[1])
        network = network >> activation_layer(layer_size)
    network = network >> layers.Dropout(proba) >> layers.Linear(1)
    
    
    cgnet = algorithms.Momentum(network,
        
    step=algorithms.step_decay(
        initial_value=initial_value,
        reduction_freq=reduction_freq,
    ),
    loss='mse',
    batch_size = batch_size,
        
    regularizer = regularizer,
    
    shuffle_data=True,
    verbose=True,
    show_epoch=1000,
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
    cgnet.train(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, epochs=n_epochs[0])
    score = cgnet.score(XTestScaled, yTestScaled)


    yEstTrain = yScaler.inverse_transform(cgnet.predict(XTrainScaled).reshape(-1, 1)).ravel()
    yEstTest = yScaler.inverse_transform(cgnet.predict(XTestScaled).reshape(-1, 1)).ravel()
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
    n_startup_jobs=50,
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
