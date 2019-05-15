#!/usr/bin/env python
# coding: utf-8

# ## EU solar generation using
# ---
# 30 years EU solar generation data download [here](https://www.kaggle.com/sohier/30-years-of-european-solar-generation/downloads/30-years-of-european-solar-generation.zip/2)
# ---

# In[2]:


import os
import pandas as pd
import numpy as np
import datetime as dt

import matplotlib.pyplot as plt
import matplotlib.dates as pltdt

import seaborn as sns
sns.set_style('darkgrid')


# ### load raw data

# In[3]:


# rawdatafilepath=
solarcountry =pd.read_csv('./dataset/EMHIRESPV_TSh_CF_Country_19862015.csv')
solarcountry.head(5)


# In[4]:


solarcountry.shape


# re time lable
# show generation curve of someday among all

# In[5]:


t=pd.date_range('1/1/1986',periods=262968,freq='H')
solarcountry['hour']=t
solarcountry.set_index('hour',inplace=True)
solarcountry['2015-12-01'].plot()
plt.legend(bbox_to_anchor=(1,1),loc=2,ncol=2,borderaxespad=0)


# show generation curve of somemonth among all

# In[6]:


solarcountry['2015-12'].plot()
plt.legend(bbox_to_anchor=(1.05,1),loc=2,ncol=2,borderaxespad=0)


# ## per DAY

# In[7]:


solarcountry['day']=solarcountry.index.map(lambda x:x.strftime('%Y-%m-%D'))
group_day=solarcountry.groupby('day').mean()
group_day.plot()
plt.legend(bbox_to_anchor=(1.05,1),loc=2,ncol=2,borderaxespad=0)


# ## per MONTH

# In[8]:


solarcountry['month']=solarcountry.index.map(lambda x:x.strftime('%Y-%m'))
solarcountry['montho']=solarcountry.index.map(lambda x:x.strftime('%m'))
group_month=solarcountry.groupby('month').mean()
group_month.plot()
plt.legend(bbox_to_anchor=(1.05,1),loc=2,ncol=2,borderaxespad=0)


# ## per YEAR

# In[9]:


solarcountry['year']=solarcountry.index.map(lambda x:x.strftime('%Y'))
group_year=solarcountry.groupby('year').mean()
group_year.plot()
plt.legend(bbox_to_anchor=(1.05,1),loc=2,ncol=2,borderaxespad=0)


# ## choose some somecountry
# eg:FR

# In[10]:


FR_heatmap=solarcountry.pivot_table(index='montho',columns='year',values='FR')
FR_heatmap.sort_index(level=0,ascending=True,inplace=True)
sns.heatmap(FR_heatmap,vmin=0.09,vmax=0.29,cmap='inferno',linewidth=0.5)


# In[11]:


sns.clustermap(FR_heatmap,cmap='inferno',standard_scale=1)


# ## view from time-series perspective

# In[12]:


FR_ts_d=solarcountry.filter(['month','year','FR'],axis=1)
FR_ts_d.plot()


# In[13]:


FR_ts_m=FR_ts_d.groupby('month').mean()
FR_ts_m.plot()


# In[14]:


FR_ts_y=FR_ts_d.groupby('year').mean()
FR_ts_y.plot()


# # prediction of solar generation

# 1. first try RNN model to predict next hour's solar generation,
# 

# In[15]:


fr_nn=solarcountry.filter(['hour','FR'],axis=1)

fr_nn=fr_nn.reset_index()
fr_nn['hour']=pd.to_datetime(fr_nn['hour'])

start=pd.Timestamp('2015-12-01')
split=pd.Timestamp('2015-12-22')
fr_nn=fr_nn[fr_nn['hour']>=start]

fr_nn=fr_nn.set_index('hour')

fr_nn.plot()


# In[16]:


train=fr_nn.loc[:split,['FR']]
test=fr_nn.loc[split:,['FR']]
tr_pl=train
te_pl=test
ax=tr_pl.plot()
te_pl.plot(ax=ax)


# train set is up to 22nd December,colored in **blue**, and test set is colored in **orange**
# 
# DATA standardscaler 

# In[17]:


from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
train_sc=sc.fit_transform(train)
test_sc=sc.transform(test)

# X_train=train_sc[:-1]
# y_train=X_train[1:]
# X_train=X_train[:-1]

# X_test=test_sc[:-1]
# y_test=X_test[1:]
# X_test=X_test[:-1]


# ### Rolling Windows

# In[18]:


train_df=pd.DataFrame(train_sc,columns=['FR'],index=train.index)
test_df=pd.DataFrame(test_sc,columns=['FR'],index=test.index)

for i in range(1,25):
    train_df['shift {}'.format(i)]=train_df['FR'].shift(i,freq='H')
    test_df['shift {}'.format(i)]=test_df['FR'].shift(i,freq='H')

train_df


# drop NaN

# In[19]:


X_train=train_df.dropna().drop('FR',axis=1)
y_train=train_df.dropna()[['FR']]

X_test=test_df.dropna().drop('FR',axis=1)
y_test=test_df.dropna()[['FR']]
X_train


# In[20]:


X_train.shape


# to np.array

# In[21]:


X_train=X_train.values
y_train=y_train.values

X_test=X_test.values
y_test=y_test.values


# ## Predictive Model
# 
# re-dimension dataset for LSTM layer

# In[22]:


X_train_w=X_train.reshape(X_train.shape[0],1,24)
X_test_w=X_test.reshape(X_test.shape[0],1,24)
X_train_w.shape


# In[23]:


from keras.models import Sequential
from keras.layers import Dense,LSTM,Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras.backend as K


# In[24]:


K.clear_session()

eps=50
bs=1

in_sh=(1,24)
hidden_1=12
hidden_2=12
outputs=1

model=Sequential()
model.add(LSTM(hidden_1,input_shape=in_sh))
model.add(Dense(hidden_2,activation='relu'))
model.add(Dense(outputs))
model.compile(optimizer='adam',loss='mean_squared_error')
model.summary()


# In[25]:


early_stop=EarlyStopping(monitor='loss',patience=1,verbose=1)
history=model.fit(X_train_w,y_train,epochs=eps,batch_size=bs,verbose=1,callbacks=[early_stop])


# In[26]:


y_pred=model.predict(X_test_w)

y_pred_raw=sc.inverse_transform(y_pred)
y_test_raw=sc.inverse_transform(y_test)


# In[27]:


plt.plot(y_test_raw)
plt.plot(y_pred_raw)


# In[28]:


# history_h=history.history
plt.plot(history.epoch,np.array(history.history['loss']),label='train_loss')
plt.legend()


# In[30]:


import tensorflow as tf
import tensorboard

writer=tf.summary.FileWriter('./EUsolargraph')
writer.add_graph(model.graph)

