#!/usr/bin/env python
# coding: utf-8

# In[8]:


pip install xgboost


# In[9]:


pip install lightgbm


# In[10]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,roc_curve

import xgboost as xgb
import lightgbm as lgb

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# In[22]:


credit_card = pd.read_csv('credit_card_clean.csv')
credit_card


# In[23]:


pd.set_option('max_columns',None)


# In[24]:


credit_card


# ## Data Understanding  (Initial Analysis) 

# In[25]:


credit_card.shape


# In[26]:


credit_card.isna().sum()


# In[27]:


credit_card.dtypes


# ## Data Preparation

# In[28]:


le = LabelEncoder()
credit_card['sex_encoded']       = le.fit_transform(credit_card['SEX'])
credit_card['education_encoded'] = le.fit_transform(credit_card['EDUCATION'])
credit_card['marriage_endoded']  = le.fit_transform(credit_card['MARRIAGE'])


# In[29]:


credit_card.head(10)


# ## Remove unwanted features

# In[30]:


credit_card.drop(labels=['ID','SEX','EDUCATION','MARRIAGE'],axis=1,inplace=True)


# In[31]:


credit_card.head(10)


# ## Model Building

# In[32]:


X = credit_card.drop(labels='DEFAULT',axis = 1)
y = credit_card[['DEFAULT']]


# In[33]:


X


# In[34]:


y


# In[35]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, stratify=y) 
#that preserves the same proportions of samples in each class as observed in the original dataset.


# In[36]:


X_train.shape,y_train.shape


# In[37]:


#Test data
X_test.shape,y_test.shape


# ## Model Traning
# ## Initializing all algorithmns for comparison

# In[38]:


rf_classifier      = RandomForestClassifier()
adboost_classifier = AdaBoostClassifier() 
gb_classifer       = GradientBoostingClassifier()
xgb_classifer      = XGBClassifier()
lgb_classifer      = LGBMClassifier()


# ## Training models

# In[39]:


rf_classifier.fit(X_train,y_train)


# In[40]:


adboost_classifier.fit(X_train,y_train)


# In[41]:


gb_classifer.fit(X_train,y_train)


# In[42]:


xgb_classifer.fit(X_train,y_train)


# In[44]:


lgb_classifer.fit(X_train,y_train)


# ## training data

# In[45]:


y_train_pred_rf  = rf_classifier.predict(X_train)
y_train_pred_adb = adboost_classifier.predict(X_train)
y_train_pred_gb  = gb_classifer.predict(X_train)
y_train_pred_xgb = xgb_classifer.predict(X_train)
y_train_pred_lgb = lgb_classifer.predict(X_train)


# ## test data

# In[46]:


y_test_pred_rf  = rf_classifier.predict(X_test)
y_test_pred_adb = adboost_classifier.predict(X_test)
y_test_pred_gb  = gb_classifer.predict(X_test)
y_test_pred_xgb = xgb_classifer.predict(X_test)
y_test_pred_lgb = lgb_classifer.predict(X_test)


# ## model evaluation

# In[48]:


print('RF Accuracy  : ',accuracy_score(y_test,y_test_pred_rf))
print('RF Precision : ',precision_score(y_test,y_test_pred_rf))
print('RF Recall    : ', recall_score(y_test,y_test_pred_rf))


# In[49]:


print('AdaBoost Accuracy  : ',accuracy_score(y_test,y_test_pred_adb))
print('AdaBoost Precision : ',precision_score(y_test,y_test_pred_adb))
print('AdaBoost Recall    : ', recall_score(y_test,y_test_pred_adb))


# In[50]:


print('Gradient Boost Accuracy  : ',accuracy_score(y_test,y_test_pred_gb))
print('Gradient Precision : ',precision_score(y_test,y_test_pred_gb))
print('Gradient Recall    : ', recall_score(y_test,y_test_pred_gb))


# In[47]:


print('XGBM Boost Accuracy  : ',accuracy_score(y_test,y_test_pred_xgb))
print('XGBM Precision       : ',precision_score(y_test,y_test_pred_xgb))
print('XGBM Recall          : ', recall_score(y_test,y_test_pred_xgb))


# In[51]:


print('LGBM Boost Accuracy  : ',accuracy_score(y_test,y_test_pred_lgb))
print('LGBM Precision       : ',precision_score(y_test,y_test_pred_lgb))
print('LGBM Recall          : ', recall_score(y_test,y_test_pred_lgb))


# In[ ]:





# In[52]:


xgb_imp_features = xgb_classifer.feature_importances_


# In[55]:


lgb_imp_features = lgb_classifer.feature_importances_


# In[56]:


X_train.columns


# In[57]:


feature_importance = pd.DataFrame({ 'Features'    : X_train.columns,
                                    'xgb_Features': xgb_imp_features,
                                    'lgb_Features': lgb_imp_features})
feature_importance


# In[58]:


xg_imp = feature_importance.sort_values(by = 'xgb_Features',ascending=False)
xg_imp


# In[59]:


lgb_imp = feature_importance.sort_values(by = 'lgb_Features',ascending=False)
lgb_imp


# In[61]:


import seaborn as sns
from matplotlib import pyplot as plt

plt.figure(figsize=(10,8))
sns.barplot(x='xgb_Features',y='Features',data=xg_imp)
plt.title('XGB IMPORTANT FEATURES')
plt.show()


# In[62]:


import seaborn as sns
from matplotlib import pyplot as plt

plt.figure(figsize=(10,8))
sns.barplot(x='lgb_Features',y='Features',data=lgb_imp)
plt.title('LGB IMPORTANT FEATURES')
plt.show()


# In[ ]:




