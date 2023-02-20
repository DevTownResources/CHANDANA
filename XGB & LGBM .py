#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install xgboost


# In[3]:


pip install lightgbm


# ## Import necessary libraries

# In[8]:


import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

import xgboost as xgb
import lightgbm as lgb

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# In[10]:


credit_card = pd.read_csv('credit_card_clean.csv')   ## imported our dataset
credit_card


# In[11]:


pd.set_option('max_columns', None)


# In[12]:


credit_card


# ## Data Understanding  (Initial Analysis)

# In[13]:


credit_card.shape


# In[14]:


credit_card.isna().sum()


# In[15]:


credit_card.dtypes


# ## Data preparation

# In[16]:


le = LabelEncoder()
credit_card['sex_encoded'] = le.fit_transform(credit_card['SEX'])
credit_card['education_encoded'] = le.fit_transform(credit_card['EDUCATION'])
credit_card['marriage_encoded'] = le.fit_transform(credit_card['MARRIAGE'])


# In[18]:


credit_card.head(10)


# ## Remove Unwanted features

# In[19]:


credit_card.drop(labels=['ID', 'SEX', 'EDUCATION','MARRIAGE'],axis =1, inplace = True)


# In[20]:


credit_card.head(10)


# ## gender (1 = male, 2 = female)
# ## education( 1= graduate school, 2=university, 3= high school, 4=others, 5 & 6 = unknown
# ## marriage (1= married, 2 = single, 3= others)
# 
# ## Target variable 
# ## default,payment.next.month: Default payment ( 1= yes, 0= no)
# ## Pay == repayment status of each months  
# ##  1= august
# ##  2 = july 
#  
#  ## 3 = july -1 = june 
# ## bill amnt = amount of bill statement of each months ( same goes for bill amnt each month)

# ## Model Building 

# In[22]:


X = credit_card.drop(labels='DEFAULT', axis = 1)


# In[23]:


y = credit_card['DEFAULT']


# In[25]:


X


# In[26]:


y


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, stratify = y)


# In[29]:


X_train.shape, y_train.shape


# ## Test Data

# In[31]:


X_test.shape, y_test.shape


# ## Model Traning
# ## Initializing all alogorithms for comparison

# In[34]:


rf_classifier = RandomForestClassifier()
gb_classifier = GradientBoostingClassifier()
adboost = AdaBoostClassifier()
xgb_classifier = XGBClassifier()
lgb_classifier = LGBMClassifier()


# ## traning models

# In[35]:


rf_classifier.fit(X_train, y_train)


# In[37]:


adboost.fit(X_train, y_train)


# In[38]:


gb_classifier.fit(X_train, y_train)


# In[39]:


xgb_classifier.fit(X_train, y_train)


# In[40]:


lgb_classifier.fit(X_train, y_train)


# In[42]:


y_train_pred_rf = rf_classifier.predict(X_train) ## training data
y_train_pred_adb = adboost.predict(X_train)
y_train_pred_gb = gb_classifier.predict(X_train)
y_train_pred_xgb = xgb_classifier.predict(X_train)
y_train_pred_lgb = lgb_classifier.predict(X_train)


# In[43]:


## testing data
y_test_pred_rf = rf_classifier.predict(X_test) 
y_test_pred_adb = adboost.predict(X_test)
y_test_pred_gb = gb_classifier.predict(X_test)
y_test_pred_xgb = xgb_classifier.predict(X_test)
y_test_pred_lgb = lgb_classifier.predict(X_test)


# In[ ]:


## model evaluation


# In[47]:


print('RF Accuracy:',accuracy_score(y_test,y_test_pred_rf))
print('RF Precision:' , precision_score(y_test,y_test_pred_rf))
print('RF Recall:' , recall_score(y_test,y_test_pred_rf))


# In[48]:


print('Adaboost Accuracy:',accuracy_score(y_test,y_test_pred_adb))
print('Adaboost Precision:' , precision_score(y_test,y_test_pred_adb))
print('Adaboost Recall:' , recall_score(y_test,y_test_pred_adb))


# In[49]:


print(' GBM Accuracy:',accuracy_score(y_test,y_test_pred_gb))
print('GBM Precision:' , precision_score(y_test,y_test_pred_gb))
print(' GBM Recall:' , recall_score(y_test,y_test_pred_gb))


# In[50]:


print(' XGBM Accuracy:',accuracy_score(y_test,y_test_pred_xgb))
print('XGBM Precision:' , precision_score(y_test,y_test_pred_xgb))
print(' XGBM Recall:' , recall_score(y_test,y_test_pred_xgb))


# In[51]:


print(' LGBM Accuracy:',accuracy_score(y_test,y_test_pred_lgb))
print('LGBM Precision:' , precision_score(y_test,y_test_pred_lgb))
print('LGBM Recall:' , recall_score(y_test,y_test_pred_lgb))


# In[52]:


xgb_imp_features = xgb_classifier.feature_importances_


# In[53]:


lgb_imp_features = xgb_classifier.feature_importances_


# In[54]:


X_train.columns


# In[56]:


feature_importance = pd.DataFrame({'Features': X_train.columns, 
                                   'xgb_Features': xgb_imp_features,
                                   'lgb_Features': lgb_imp_features})


# In[58]:


feature_importance


# In[59]:


xg_imp = feature_importance.sort_values(by ='xgb_Features', ascending = False)
xg_imp


# In[60]:


lgb_imp = feature_importance.sort_values(by ='lgb_Features', ascending = False)
lgb_imp


# In[61]:


import seaborn as sns
from matplotlib import pyplot as plt

plt.figure(figsize=(10,8))
sns.barplot(x='xgb_Features', y='Features', data = xg_imp)
plt.title('XGB IMPORTANT FEATURES')
plt.show()


# In[62]:


import seaborn as sns
from matplotlib import pyplot as plt

plt.figure(figsize=(10,8))
sns.barplot(x='lgb_Features', y='Features', data = lgb_imp)
plt.title('LGB IMPORTANT FEATURES')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




