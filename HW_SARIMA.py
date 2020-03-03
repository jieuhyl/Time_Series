
# coding: utf-8

# ## Forecasting with the HW and SARIMA

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error


df = pd.read_csv('../Data/airline_passengers.csv',index_col='Month',parse_dates=True)
df.index.freq = 'MS'
df.head()


# In[2]:


df.tail()


# In[3]:


df.info()


# ## Train Test Split

# In[4]:


train = df.loc[:'1957-12-01'] # Goes up to but not including 108
test = df.loc['1958-01-01':]


# ## Exponential Weighted Moving Average

# In[5]:


# EWMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
span = 12
alpha = 2/(span+1)

df['EWMA'] = df['Thousands of Passengers'].ewm(alpha=alpha, adjust=False).mean()

model_ewma = SimpleExpSmoothing(train['Thousands of Passengers'])
model_ewma = model_ewma.fit(smoothing_level=alpha, optimized=True)
test_predictions = model_ewma.forecast(len(test))


# In[6]:


df['EWMA'].plot(legend=True,label='EWMA',figsize=(12,8))
train['Thousands of Passengers'].plot(legend=True,label='TRAIN')
test['Thousands of Passengers'].plot(legend=True,label='TEST')
test_predictions.plot(legend=True, label='SES');


# ## Holt-Winters 
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

cfg_list0 = cfg_list[0:10]# one-step Holt Winter’s Exponential Smoothing forecast
def exp_smoothing_forecast(train, test, config):
    t,d,s,p,b,r = config
    # define model
    model = ExponentialSmoothing(train, trend=t, damped=d, seasonal=s, seasonal_periods=p)
    # fit model
    model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
    # make forecast
    yhat = model_fit.forecast(len(test))
    rmse = np.sqrt(mean_squared_error(test, yhat))
    return (rmse)lst = []
for i in cfg_list:
    try:
        lst.append(exp_smoothing_forecast(train['Thousands of Passengers'], test['Thousands of Passengers'], i))
    except:
        lst.append(0)# collect all solutions
dct1 = {tuple(k): v for k, v in zip(cfg_list, lst)}     
# get top 5 solution
dct11 = dict([(x,y) for x, y in dct1.items() if y != 0])
sorted(dct11.items(), key=lambda x: x[1], reverse=False)[:5]
# In[7]:


model_hw = ExponentialSmoothing(train['Thousands of Passengers'], trend='add', damped=False, seasonal='mul', seasonal_periods=12)
model_hw = model_hw.fit(optimized=True, use_boxcox=False, remove_bias=False)
test_predictions = model_hw.forecast(len(test))


# In[8]:


train['Thousands of Passengers'].plot(legend=True,label='TRAIN')
test['Thousands of Passengers'].plot(legend=True,label='TEST',figsize=(12,8))
test_predictions.plot(legend=True,label='PREDICTION');


# In[9]:


# final model
model_hw = ExponentialSmoothing(df['Thousands of Passengers'], trend='add', damped=False, seasonal='mul', seasonal_periods=12)
model_hw = model_hw.fit(optimized=True, use_boxcox=False, remove_bias=False)
forecast_predictions1 = model_hw.forecast(36)


# In[10]:


df['Thousands of Passengers'].plot(legend=True,label='DATA', figsize=(12,8))
forecast_predictions1.plot(legend=True, label='HW');


# ## SARIMA
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

cfg_list0 = cfg_list[0:10]# one-step sarima forecast
def sarima_forecast(train, test, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(train, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make forecast
    yhat = model_fit.forecast(len(test))
    rmse = np.sqrt(mean_squared_error(test, yhat))
    return (rmse)lst = []
for i in cfg_list:
    try:
        lst.append(sarima_forecast(train['Thousands of Passengers'], test['Thousands of Passengers'].values, i))
    except:
        lst.append(0)# collect all solutions
dct2 = {tuple(k): v for k, v in zip(cfg_list, lst)}    
# get top 5 solution
dct22 = dict([(x,y) for x, y in dct2.items() if y != 0])
sorted(dct22.items(), key=lambda x: x[1], reverse=False)[:5]
# In[11]:


model_sarima = SARIMAX(train['Thousands of Passengers'], order=(0,1,0), seasonal_order=(1,0,0,12), trend='t', enforce_stationarity=False, enforce_invertibility=False)
model_sarima = model_sarima.fit(disp=False)
test_predictions = model_sarima.forecast(len(test))


# In[12]:


train['Thousands of Passengers'].plot(legend=True,label='TRAIN')
test['Thousands of Passengers'].plot(legend=True,label='TEST',figsize=(12,8))
test_predictions.plot(legend=True,label='PREDICTION');


# In[13]:


# final model
model_sarima = SARIMAX(df['Thousands of Passengers'], order=(0,1,0), seasonal_order=(1,0,0,12), trend='t', enforce_stationarity=False, enforce_invertibility=False)
model_sarima = model_sarima.fit(disp=False)
forecast_predictions2 = model_sarima.forecast(36)


# In[14]:


df['Thousands of Passengers'].plot(legend=True, label='DATA', figsize=(12,8))
forecast_predictions2.plot(legend=True, label='SARIMA');


# In[15]:


df['Thousands of Passengers'].plot(legend=True,label='DATA', figsize=(12,8))
forecast_predictions1.plot(legend=True, label='HW')
forecast_predictions2.plot(legend=True, label='SARIMA');

