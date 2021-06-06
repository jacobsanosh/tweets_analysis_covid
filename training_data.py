from os import confstr
from typing import final
import pandas as pd
import numpy as  np
from pandas.core.indexes.base import Index
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

#reading the csv file
temp_data = pd.read_csv('Covid_tweets_US.csv')
print(temp_data.head())

datas=temp_data.drop(['sl_no','created_at','place_full_name','place_full_name','place_type','retweet_count','Subjectivity'],axis=1)
datas=datas.dropna()
print(datas.head())

types = datas.Polarity.value_counts()
print(types)

#creting an row according to their polarity
datas['emotion']=datas.Polarity.map({'Neutral':0,'Positive':1,'Negative':2})
print(datas.head())

#setting datas
x=datas.text
y=datas.emotion

#printing therir shape
print(x.shape)
print(y.shape)

#here trainig our data
x_train,x_test,y_train,y_test=train_test_split(x ,y,test_size = 0.20, random_state=10)


print(x_train.isna().values.any())
print(x_test.isna().values.any())
print(y_train.isna().values.any())
print(y_test.isna().values.any())

print(x_train.shape)
print(x_test.shape)
print("\n\n",y_train.shape)
print(y_test.shape)


#creating unique words from my data set
un_vect = CountVectorizer()
un_vect.fit(x_train)

# un_vect.fit_transform(x_train.values.astype('U'))
# print(un_vect.vocabulary_)

#removing stopwords
un_vect =CountVectorizer(stop_words='english')

#taking the full data 
my_vectorizer = un_vect.fit(x)




# print(un_vect.vocabulary_)
#converting the data into numeric format that machine learning format
x_train_transformed = un_vect.transform(x_train)
x_test_transformed = un_vect.transform(x_test)
print(x_train_transformed.shape)
print(x_test_transformed.shape)

mnb =MultinomialNB()

my_naive_model = mnb.fit(x_train_transformed,y_train)
y_predicted_class =mnb.predict(x_test_transformed)

print(metrics.accuracy_score(y_test,y_predicted_class))
#creating an confusion metrix
confusion= metrics.confusion_matrix(y_test,y_predicted_class)

a1 = confusion[0,0]
a2 = confusion[0,1]
a3 = confusion[0,2]

b1 = confusion[1,0]
b2 = confusion[1,1]
b3 = confusion[1,2]

c1 = confusion[2,0]
c2 = confusion[2,1]
c3 = confusion[2,2]

sensitivity=a1/float(a1+a2+a3)
specificity=b2/float(b1+b2+b3)
precision=a1/float(a1+b1+c1)
print("sensitivity",sensitivity)
print("specificity",specificity)
print("precision",precision)
# print("f1 score = ",metrics.f1_score(y_test, y_predicted_class))

final_model =my_naive_model
joblib.dump(final_model,'my_naive_model.joblib')
joblib.dump(my_vectorizer,'my_vectorizer.joblib')

