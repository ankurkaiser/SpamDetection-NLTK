# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 12:41:51 2021

@author: Lenovo
"""
#importing libraries
import pandas as pd
import numpy as np

import re
import nltk
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


data=pd.read_csv(r'D:\Datasets and Projects\NLP\NLTK\smsspamcollection\SMSSpamCollection', sep='\t',names=["label", "message"])

data.head()

lemmatizer=WordNetLemmatizer()

#text cleaning and lemmatizing
corpus=[]
for i in range(len(data)):
    word=re.sub('[^a-zA-Z]',' ', data['message'][i])
    word=word.lower()
    word=word.split()
    word=[lemmatizer.lemmatize(review) for review in word if not review in stopwords.words('english')]
    word=' '.join(word)
    corpus.append(word)
    
#creating vectors of the datapoints    
tf = TfidfVectorizer()

X=tf.fit_transform(corpus).toarray()

y=pd.get_dummies(data['label'])
y=y.iloc[:,1].values


#train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10,shuffle=True)

#model creation

mb=MultinomialNB()

model= mb.fit(X_train,y_train)
prediction=model.predict(X_test)

#report generation
accuracy=accuracy_score(y_test, prediction)
classificationreport=classification_report(y_test,prediction)
confusionmatrix=confusion_matrix(y_test, prediction)

print("The classification report is : ", classification_report(y_test, prediction))
print("The accuracy score is :", accuracy_score(y_test, prediction))
print("The confusion matrix is :", confusion_matrix(y_test,prediction))




