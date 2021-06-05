#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


A = pd.read_csv("/users/ranjeetgaikwad/desktop/data science class/AirPassengers.csv")


# In[4]:


A.head()


# In[10]:


A.Date = pd.to_datetime(A.Month, format = "%Y-%m")


# In[11]:


A.head()


# In[12]:


A.index = A.Date


# In[13]:


A.head()


# In[14]:


A = A.drop(labels = ["Month"], axis = 1)


# In[15]:


A.head(2)


# In[16]:


A.plot()


# In[20]:


from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
fig = seasonal_decompose(A, model = "multiplicative")
fig.plot()
plt.show()


# # Dickey-fuller test

# In[21]:


A.head()


# In[22]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit_transform(A[["#Passengers"]])


# In[23]:


A[["#Passengers"]].mean()


# In[24]:


import numpy as np
log_pass = np.log(A[["#Passengers"]])
rm = log_pass.rolling(window=12).mean()
A['diff'] = log_pass-rm


# In[25]:


B = A[A['diff'].isnull()==False]


# In[26]:


B = B.drop(labels = ['#Passengers'], axis = 1)


# In[27]:


B.columns= ["Pass"]


# In[29]:


B.head()


# In[30]:


from statsmodels.tsa.stattools import adfuller
x = adfuller(pd.Series(B['Pass']))
if(x[1]<0.05):
    print("Stationary")
else:
    print("Non stationary")


# In[32]:


from statsmodels.tsa.stattools import adfuller
x = adfuller(pd.Series(A['#Passengers']))
if(x[1]<0.05):
    print("Stationary")
else:
    print("Non stationary")


# In[ ]:




