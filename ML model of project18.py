# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 12:26:22 2021

@author: asus
"""


import pandas

df = pandas.read_csv("balanced_reviews.csv")

df.shape
df.columns.tolist()

df.head()

df['reviewText'].head()

df['reviewText'][0]

df['overall'].unique()

df['overall'].value_counts()

df.isnull().any(axis = 0)

df.dropna(inplace = True)


df['overall'].value_counts()

df['overall'] != 3

df  = df[df['overall'] != 3]

df['overall'].value_counts()


import numpy as np

df['Positivity']  = np.where(df['overall'] > 3, 1, 0)

# reviewText - feature
# positivity - labels

df['Positivity'].value_counts()


features = df['reviewText']
labels = df['Positivity']

from sklearn.model_selection import train_test_split


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)


#Tfidf
from sklearn.feature_extraction.text import TfidfVectorizer

#min_df = 5 means "ignore terms/words that appear in less
#than 5 documents

vect = TfidfVectorizer(min_df = 5).fit(features_train)

len(vect.get_feature_names())


features_train_vectorized = vect.transform(features_train)

#model building
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(features_train_vectorized,labels_train)

# 0 -> negative
# 1 -> positive

predictions = model.predict(vect.transform(features_test))

from sklearn.metrics import confusion_matrix

confusion_matrix(labels_test, predictions)


from sklearn.metrics import accuracy_score

accuracy_score(labels_test, predictions)


#Team A

import pickle

file  = open("pickle_model.pkl","wb")
pickle.dump(model, file)



#Team B
file = open("pickle_model.pkl",'rb')
recreated_model = pickle.load(file)

preds = recreated_model.predict(vect.transform(features_test))

from sklearn.metrics import accuracy_score

accuracy_score(labels_test, preds)


vocab_file = open('features.pkl','wb')
pickle.dump(vect.vocabulary_, vocab_file)



