import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import joblib

st.title("Tweet detection")


my_naive_model=joblib.load('my_naive_model.joblib')
my_vectorizer = joblib.load('my_vectorizer.joblib')
user_input=st.text_area('Please enter the text here',height=200)

vect_text = my_vectorizer.transform([user_input])

predicted=''
def textanalysis(user_input):
    prediction=my_naive_model.predict(user_input)
    if prediction==0:
        predicted='Neutral'
    elif prediction==1:
        predicted='Positive'
    else:
        predicted='Negative'
    return predicted
if st.button('check'):
    st.write("you entered :",textanalysis(vect_text)," message")

    # https://covidtexttweetanalysis.herokuapp.com/








