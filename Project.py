#!/usr/bin/env python
# coding: utf-8

# In[27]:


import sys
import json
import io
import numpy as np
import sklearn
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# In[28]:


train_data = pd.read_json('train.json')
test_data = pd.read_json('test.json')


# In[29]:


plt.style.use('ggplot')
train_data['cuisine'].value_counts().plot(kind='bar')


# In[30]:


cv = CountVectorizer()
train_data['concat_ingredients'] = train_data['ingredients'].map(';'.join)
test_data['concat_ingredients'] = test_data['ingredients'].map(';'.join)
train_data.head()


# In[31]:


## This Function helpfull for lower casing and stripping accents also
X = cv.fit_transform(train_data['concat_ingredients'].values)
X_test = cv.transform(test_data['concat_ingredients'].values)


# In[32]:


id_test = test_data['id']
Y = train_data['cuisine']
Y.head()


# In[33]:


## Naive Bayes Classifier
NaiveModel = MultinomialNB().fit(X,Y)


# In[34]:


## Random Forest CLassifier
Model1 = RandomForestClassifier(max_depth=40, n_estimators=20).fit(X,Y)


# In[35]:


## SGD Classifier
Model2 = SGDClassifier(loss='modified_huber', penalty='l2' , alpha=0.0001 , max_iter=10, tol=1e-3, random_state=65).fit(X,Y)


# In[36]:


print("predicting")
Naive_PredictedY1s = NaiveModel.predict(X)
Random_PredictedY1s = Model1.predict(X)
SGD_PredictedY1s = Model2.predict(X)


# In[37]:


print("Naive Bayes Accurracy : %f " % np.mean ( Naive_PredictedY1s == Y))
print("Random Forest Accurracy : %f " % np.mean ( Random_PredictedY1s == Y))
print("SGD classifier Accurracy : %f " % np.mean ( SGD_PredictedY1s == Y))


# In[38]:


print(classification_report(Naive_PredictedY1s, Y))


# In[39]:


print(classification_report(Random_PredictedY1s, Y))


# In[40]:


print(classification_report(SGD_PredictedY1s, Y))


# In[41]:


## Predicting the test data
Predicted_Cuisines = Model1.predict(X_test)


# In[42]:


## Output Predictions
out = io.open('submit.csv','w')
out.write(u'id , cuisine\n')
for i in range(9944):
    out.write('%s,%s\n' % (id_test[i], Predicted_Cuisines[i]))


# In[43]:


## This section is to predict the user input ingredients

ing_array = ["baking powder;eggs;all-purpose flour;raisins;milk;white sugar"]
no_of_ingredients = input("Total Number Of Ingredients: ")
no_of_ingredients = int(no_of_ingredients)

ingredient = ""

for i in range(no_of_ingredients):
    ing = input("Enter Ingredient " + str(i) + " : ")
    ingredient = ingredient + ing + ";"

## Predicting User input ingredients

ing_array.append(ingredient)
User_in = cv.transform(ing_array)
Predicted = Model1.predict(User_in)
print("")
print("The predicted cuisine for input ingredients is : "+Predicted[1])


# In[ ]:




