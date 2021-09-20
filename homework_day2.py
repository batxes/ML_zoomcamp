#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

df = pd.read_csv("AB_NYC_2019.csv")
df


# In[3]:


strings = list(df.dtypes[df.dtypes == "object"].index)
strings
for col in strings:
    df[col] = df[col].str.lower().str.replace(" ","_")


# In[17]:


categorical_values = ['latitude',
'longitude',
'price',
'minimum_nights',
'number_of_reviews',
'reviews_per_month',
'calculated_host_listings_count',
'availability_365']
df = df[categorical_values]
df.isnull().sum()


# In[5]:


np.median(df.minimum_nights.values)


# In[6]:


np.random.seed(42)


# In[7]:


n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test


# In[8]:


idx = np.arange(n)
np.random.shuffle(idx)
df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]


# In[9]:


len(df_train),len(df_val),len(df_test)


# In[11]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = np.log1p(df_train.price.values)
y_val = np.log1p(df_val.price.values)
y_test = np.log1p(df_test.price.values)
del df_train['price']
del df_val['price']
del df_test['price']


# In[12]:


df_train.head()


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.histplot(df.price, bins=50)


# In[14]:


price_logs = np.log1p(df.price)


# In[15]:


sns.histplot(price_logs, bins=50)


# In[56]:


base = ['latitude',
'longitude',
'minimum_nights',
'number_of_reviews',
'reviews_per_month',
'calculated_host_listings_count',
'availability_365']

def prepare_X_filling_0(df):
    df = df.copy()
    features = base.copy()
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

def prepare_X_train_filling_mean(df):
    df = df.copy()
    features = base.copy()
    df_num = df[features]
    for c in features:
        mean = np.nanmean(df_num[c].values)
        df_num[c] = df_num[c].fillna(mean)
    X = df_num.values
    return X


# In[57]:


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones,X])
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]

def train_linear_regression_regularized(X, y, r=0.01): #r cant be very high or very low, we need to TUNE it
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones,X])
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])
    
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]

#RMSE
def rmse (y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)


# In[61]:


#linear regression filling missing values with mean

X_train = prepare_X_train_filling_mean(df_train)
w0, w = train_linear_regression(X_train, y_train)

#validation
X_val = prepare_X_filling_0(df_val)
y_pred = w0 + X_val.dot(w)

score = rmse(y_val, y_pred)
score.round(2)


# In[62]:


#linear regression filling missing values with 0

X_train = prepare_X_filling_0(df_train)
w0, w = train_linear_regression(X_train, y_train)

#validation
X_val = prepare_X_filling_0(df_val)
y_pred = w0 + X_val.dot(w)

score = rmse(y_val, y_pred)
score.round(2)


# In[66]:


#regularized

r = [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]

for x in r:
    X_train = prepare_X_filling_0(df_train)
    w0, w = train_linear_regression_regularized(X_train, y_train,x)

    #validation
    X_val = prepare_X_filling_0(df_val)
    y_pred = w0 + X_val.dot(w)

    score = rmse(y_val, y_pred)
    print (x, score)


# In[78]:


seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
rmse_scores = []
for s in seeds:
    np.random.seed(s)
    idx = np.arange(n)
    np.random.shuffle(idx)
    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train+n_val]]
    df_test = df.iloc[idx[n_train+n_val:]]
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    y_train = np.log1p(df_train.price.values)
    y_val = np.log1p(df_val.price.values)
    y_test = np.log1p(df_test.price.values)
    del df_train['price']
    del df_val['price']
    del df_test['price']
    
    X_train = prepare_X_filling_0(df_train)
    w0, w = train_linear_regression(X_train, y_train)

    #validation
    X_val = prepare_X_filling_0(df_val)
    y_pred = w0 + X_val.dot(w)

    score = rmse(y_val, y_pred)
    rmse_scores.append(score)


print (round(np.std(rmse_scores),3))


# In[81]:


np.random.seed(9)
idx = np.arange(n)
np.random.shuffle(idx)
df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = np.log1p(df_train.price.values)
y_val = np.log1p(df_val.price.values)
y_test = np.log1p(df_test.price.values)
del df_train['price']
del df_val['price']
del df_test['price']

df_full_train = pd.concat([df_train, df_val])
df_full_train = df_full_train.reset_index(drop=True)
X_full_train = prepare_X_filling_0(df_full_train)
y_full_train = np.concatenate([y_train, y_val])


w0, w = train_linear_regression_regularized(X_full_train, y_full_train,0.001)

#test
X_test = prepare_X_filling_0(df_test)
y_pred = w0 + X_test.dot(w)
score = rmse(y_test, y_pred)
score


# In[ ]:




