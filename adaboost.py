#!/usr/bin/env python
# coding: utf-8

# ## Import Necessary Libraries

# In[2]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


# In[3]:


brst_cancer = load_breast_cancer()

X = pd.DataFrame(brst_cancer.data, columns=brst_cancer.feature_names)
y = pd.Categorical.from_codes(brst_cancer.target, brst_cancer.target_names) 


# In[4]:


X


# In[5]:


encoder = LabelEncoder()    
binary_encoded_y = pd.Series(encoder.fit_transform(y))


# In[6]:


train_X, test_X, train_y, test_y = train_test_split(X, binary_encoded_y, random_state=1)


# In[7]:


classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1,),n_estimators=50,)

classifier.fit(train_X, train_y)


# In[8]:


predictions = classifier.predict(test_X)


# In[9]:


confusion_matrix(test_y, predictions)


# In[10]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
clf.score(X, y)


# In[ ]:




