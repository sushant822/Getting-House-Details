#!/usr/bin/env python
# coding: utf-8

# #### Testing ML Model

# In[1]:


import joblib


# In[4]:


filename = '/data/LogisticRegression.sav'


# In[5]:


joblib_LR_model = joblib.load(filename)
joblib_LR_model


# In[6]:


test_data = [[3, 2, 1, 1650, 1, 30, 1800, 1, 1, 35, 60, 50, 1, 0, 0]]
Ypredict = joblib_LR_model.predict(test_data)  
Ypredict


# In[ ]:




