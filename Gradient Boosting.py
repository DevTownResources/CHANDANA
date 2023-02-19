#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error


# In[2]:


boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)


# In[3]:


X


# In[4]:


y


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[7]:


regressor = GradientBoostingRegressor(max_depth=2, n_estimators=50, learning_rate=1.0,)
regressor.fit(X_train, y_train)


# In[8]:


y_pred = regressor.predict(X_test)

mean_absolute_error(y_test, y_pred)


# In[9]:


mean_squared_error(y_test,y_pred)


# In[ ]:




