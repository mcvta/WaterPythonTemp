# WaterPythonTemp


This repository includes the python code of four models that were used to predict the water temperature of 83 rivers with limiting forcing data (with 98% of missing data). The results of this study are described in the following manuscript: 
Almeida, MC and Coelho PS.: Modeling river water temperature with limiting forcing data:

- Random Forest (_vide_ [sklearn webpage](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html))
-	Artificial Neural Network (Momentum algorithm) (_vide_ [neupy webpage](http://neupy.com/modules/generated/neupy.algorithms.Momentum.html))
-	Support Vector Regression (_vide_ [sklearn webpage](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html]))
-	Multiple Regression (_vide_ [sklearn webpage](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html]))


The machine learning models hyperparameter optimization was implemented with the Tree-structured Parzen Estimators algorithm (TPE) (Bergstra et al 2011). The python code implementation of TPE with the Hyperot algorithm (Bergstra et al 2013) is also available.


## Predictor variables

- Mean daily air temperature (°C);
- Discharge (m<sup>3</sup>s<sup>-1</sup>);
- Mean daily Global radiation (Jm<sup>-2</sup>);
- Maximum day air temperature(°C);
- Minimum day air temperature (°C);
- Month of the year (e.g. 1, 2, 3,..., 12);
- Day of the year (e.g. 1, 2, 3,..., 365).

## Hyperparameter optimization
It is easy to find the model parameters in the code. Nonetheless, in the folowing table we have included the models parameters that are optimized with the TPE algorithm.

#### Table1. Model parameters and optimization range
Model|	Prior distribution|	Parameter     |	Optimization range
---- | ------------------ | ------------- | ------------------ 
Random Forest   |uniform           |	'n_estimators'|	[50, 2000]
Random Forest   |uniform           |	'max_depth'   |	[10, 1000]
Random Forest   |uniform           |	'min_samples_split'|[2, 10]
Random Forest   |-                 |	'max_features'|	[auto, sqrt]
Random Forest   |-                 |	'bootstrap'   |	[True, False]
ANN             |	categorical      |	'n_layers'	  | [1, 2]
ANN             |uniform integer   |	'n_units_layer' | [10, 50]
ANN          	  |categorical        |	'act_func_type' |	['Relu', 'PRelu', 'Elu', 'Tanh', 'Sigmoid']
ANN             |categorical        |	'regularization'|	[True, False]
ANN          	  |quantized distribution|	'n_epochs'|	With regularization: [500, 1000]; without regularization: [20, 300]
ANN         	  |uniform            |	'dropout'     |	[0, 1.0]
ANN         	  |loguniform         |	'batch_size'  |	[5, 20]
ANN             |uniform            |	'initial_value'| [0.001, 0.1]
ANN             |uniform            |	'reduction_freq'|	[10, 200]
ANN             |uniform            |	'decay_rate' (regularization)|[0.0001, 0.001]
SVR             |	Categorical       |	'C'|[0.1,1,100,1000]
SVR             |Categorical        |	'kernel'|	['rbf','poly','sigmoid','linear']
SVR         	  |Categorical        |'degree' |	[1,2,3,4,5,6]
SVR             |Categorical        |	'gamma' |	[1, 0.1, 0.01, 0.001, 0.0001]
SVR             |Categorical        |	'epsilon'|	[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]


## How to run the models
1 - Instal neupy from the [neupy webpage](http://neupy.com/pages/installation.html)
2 - Create an empty folder;
3 - In this folder include the python code file (e.g. Hyper_ANN.py) and the input file (e.g. st1.xlsx)
4 - Run the code. The output includes 

