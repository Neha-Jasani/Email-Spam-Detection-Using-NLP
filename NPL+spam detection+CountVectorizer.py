#!/usr/bin/env python
# coding: utf-8

# # NLP +Spam Detection

# In[15]:


import pandas as pd


# In[16]:


df_sm = pd.read_csv('spam.csv')


# In[17]:


df_sm.head()


# In[18]:


df_sm = df_sm.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis =1)


# In[19]:


df_sm.columns = ['label','message']


# In[20]:


df_sm.shape


# In[21]:


df_sm.describe()


# In[22]:


df_sm.info()


# # DAta preprocessing

# In[23]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[24]:


# instance of stemmer
ps = PorterStemmer()


# In[25]:


corpus = []


# In[26]:


for i in range(0,len(df_sm)):
    review = re.sub('[^a-zA-Z]',' ',df_sm['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[27]:


print(corpus)


# In[28]:


from sklearn.feature_extraction.text import CountVectorizer


# In[29]:


# make an instance of vector if not none
# biult a vocabulary that only consider the top max_features(2500) ordered by term frequency
# across the corpus
cv = CountVectorizer(max_features=2500)


# In[30]:


# preparing target
df_sm.label.value_counts()


# In[31]:


# mapping label to 1 and 0
df_sm['target'] = df_sm['label'].map({'ham':0,'spam':1})


# In[32]:


df_sm


# In[33]:


# split data inot train and test
# preparing dataset for model training
X = cv.fit_transform(corpus).toarray()


# In[34]:


X


# In[35]:


y = df_sm['target']


# In[36]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)


# In[37]:


from sklearn.naive_bayes import MultinomialNB


# In[38]:


model = MultinomialNB()


# In[39]:


model.fit(X_train,Y_train)


# In[40]:


y_pred = model.predict(X_test)


# In[41]:


y_pred


# In[42]:


model.score(X_test,y_test)

