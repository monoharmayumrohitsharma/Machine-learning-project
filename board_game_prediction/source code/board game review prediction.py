
# coding: utf-8

# In[ ]:


import sys
import pandas
import matplotlib
import seaborn
import sklearn


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[ ]:


games = pandas.read_csv('Desktop\games.csv')


# In[ ]:


print(games.columns)
print(games.shape)


# In[ ]:


plt.hist(games['average_rating'])
plt.show()


# In[ ]:


print(games[games["average_rating"]==0].iloc[0])
print(games[games["average_rating"]>0].iloc[0])


# In[ ]:


games = games[games["users_rated"]>0]
games = games.dropna(axis=0)

plt.hist(games['average_rating'])
plt.show()


# In[ ]:


corrmat = games.corr()
fig = plt.figure(figsize = (12,9))
sns.heatmap(corrmat,vmax= .8,square=True)
plt.show()


# In[ ]:


columns=games.columns.tolist()

columns=[c for c in columns if c not in ["bayes_average_rating","average_rating","type","name","id"]]
target="average_rating"



# In[ ]:


#from sklearn.cross_validation import train_test_split

train= games.sample(frac=0.8,random_state=1)
test = games.loc[~games.index.isin(train.index)]


print(train.shape)
print(test.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

LR = LinearRegression()

LR.fit(train[columns],train[target])


# In[ ]:


predictions = LR.predict(test[columns])

mean_squared_error(predictions, test[target])


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

RFR = RandomForestRegressor(n_estimators = 100, min_samples_leaf =20, random_state=1)

RFR.fit(train[columns],train[target])


# In[ ]:


predictions = RFR.predict(test[columns])

mean_squared_error(predictions, test[target])


# In[ ]:


test[columns].iloc[2]


# In[ ]:


rating_LR = LR.predict(test[columns].iloc[2].values.reshape(1,-1))
rating_RFR = RFR.predict(test[columns].iloc[2].values.reshape(1,-1))

print(rating_LR)
print(rating_RFR)


# In[ ]:


test[target].iloc[2]

