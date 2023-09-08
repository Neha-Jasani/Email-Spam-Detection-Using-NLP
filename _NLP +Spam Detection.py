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


df_sm.head()


# In[21]:


df_sm.shape


# In[22]:


df_sm.describe()


# In[23]:


df_sm.info()


# # DAta preprocessing

# In[24]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[25]:


# instance of stemmer
ps = PorterStemmer()


# In[26]:


corpus = []


# In[27]:


nltk.download('stopwords')


# In[28]:


for i in range(0,len(df_sm)):
    review = re.sub('[^a-zA-Z]',' ',df_sm['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[29]:


print(corpus)


# In[30]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[31]:


# make an instance of vector if not none
# biult a vocabulary that only consider the top max_features(2500) ordered by term frequency
# across the corpus
tf_idf = TfidfVectorizer()


# In[32]:


X = df_sm['message']


# In[33]:


print(X)


# In[34]:


y = df_sm['label']


# In[35]:


print(y)


# In[36]:


X = tf_idf.fit_transform(X)


# In[37]:


print(X)


# In[38]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)


# In[39]:


from sklearn.naive_bayes import MultinomialNB


# In[40]:


model = MultinomialNB()


# In[41]:


model.fit(X_train,Y_train)


# In[42]:


y_pred = model.predict(X_test)


# In[43]:


y_pred


# In[44]:


model.score(X_test,y_test)

