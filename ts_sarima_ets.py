# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:44:19 2019

@author: Jie.Hu
"""



# grid search sarima hyperparameters for monthly car sales dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error



#==============================================================================
data = pd.read_csv('monthly-car-sales.csv', header=0, index_col=0)
series = data.values

plt.plot(data)


def train_test_split(data, n):
	return data[:-n], data[-n:]

train, test = train_test_split(data, 12)

###############################################################################
# create a set of exponential smoothing configs to try
def exp_smoothing_configs(seasonal=[None]):
	models = list()
	# define config lists
	t_params = ['add', 'mul', None]
	d_params = [True, False]
	s_params = ['add', 'mul', None]
	p_params = seasonal
	b_params = [True, False]
	r_params = [True, False]
	# create config instances
	for t in t_params:
		for d in d_params:
			for s in s_params:
				for p in p_params:
					for b in b_params:
						for r in r_params:
							cfg = [t,d,s,p,b,r]
							models.append(cfg)
	return models

cfg_list = exp_smoothing_configs(seasonal=[0,6,12])

cfg_list0 = cfg_list[0:10]
        
        
# one-step Holt Winter’s Exponential Smoothing forecast
def exp_smoothing_forecast(train, test, config):
    t,d,s,p,b,r = config
    # define model
    #history = np.array(history)
    model = ExponentialSmoothing(train, trend=t, damped=d, seasonal=s, seasonal_periods=p)
    # fit model
    model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
	# make one step forecast
    yhat1 = model_fit.predict(start = 0, end= len(train))
    yhat2 = model_fit.predict(start = len(train), end= len(train)+len(test)-1)
    mape1 = float(np.abs(sum(yhat1) - sum(train))/sum(train))
    mape2 = float(np.abs(sum(yhat2) - sum(test))/sum(test))
    return (mape1, mape2)

lst = []
for i in cfg_list:
    try:
        lst.append(exp_smoothing_forecast(train['Car Sales'], test['Car Sales'], i))
    except:
        lst.append((1,1))        

# collect all solutions
dct1 = {tuple(k): v for k, v in zip(cfg_list, lst)}     
# get optium solution
sorted(dct1.items(), key=lambda x: sum(x[1]), reverse=False)[:5]


# final model
y_hat_avg = test.copy()
model = ExponentialSmoothing(train, trend='add', damped=True, seasonal='add', seasonal_periods=12)
model_fit = model.fit(optimized=True, use_boxcox=True, remove_bias=False)
y_hat_avg['Holt_Winter'] = model_fit.forecast(12)

# graph
plt.figure(figsize=(12,7))
plt.plot( train, label='Train')
plt.plot(test, label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()

rms = np.sqrt(mean_squared_error(test, y_hat_avg['Holt_Winter']))
print(rms)

###############################################################################        
# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
	models = list()
	# define config lists
	p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal
	# create config instances
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models

cfg_list = sarima_configs(seasonal=[0,6,12])

cfg_list0 = cfg_list[0:10]

# one-step sarima forecast
def sarima_forecast(train, test, config):
    order, sorder, trend = config
	# define model
    model = SARIMAX(train, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
	# fit model
    model_fit = model.fit(disp=False)
	# make one step forecast
    yhat1 = model_fit.predict(start = 0, end= len(train))
    yhat2 = model_fit.predict(start = len(train), end= len(train)+len(test)-1)
    mape1 = float(np.abs(sum(yhat1) - sum(train))/sum(train))
    mape2 = float(np.abs(sum(yhat2) - sum(test))/sum(test))
    return (mape1, mape2)

#y_pred = sarima_forecast(train, test, cfg_list0)

lst = []
for i in cfg_list:
    try:
        lst.append(sarima_forecast(train['Car Sales'].values, test['Car Sales'].values, i))
    except:
        lst.append((1,1))

# collect all solutions
dct2 = {tuple(k): v for k, v in zip(cfg_list, lst)}    
# get optium solution
sorted(dct2.items(), key=lambda x: sum(x[1]), reverse=False)[:5]

# final model
y_hat_avg = test.copy()
model = SARIMAX(train, order=(0,0,0), seasonal_order=(1,1,1,12), trend='c', enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(disp=False)
y_hat_avg['SARIMA'] = model_fit.forecast(12)

# graph
plt.figure(figsize=(12,7))
plt.plot( train, label='Train')
plt.plot(test, label='Test')
plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
plt.legend(loc='best')
plt.show()

rms = np.sqrt(mean_squared_error(test['Car Sales'], y_hat_avg['SARIMA']))
print(rms)


      