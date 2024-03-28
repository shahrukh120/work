import streamlit as st 
import pickle
import nltk
import string 
from nltk.stem.porter import PorterStemmer    #importing for stemming
ps=PorterStemmer()
from nltk.corpus import stopwords# these are unecessory words
stopwords.words('english')
rounded_button_style = """
    <style>
        .rounded-button {
            border-radius: 12px;
            padding: 10px 20px;
            background-color: white;
            color: black;
            font-weight: bold;
            border: 2px solid #007bff;
            transition: background-color 0.3s, color 0.3s;
        }
        .rounded-button:hover {
            background-color: #0056b3;
            border-color: #0056b3;
            cursor: pointer;
        }
    </style>
    """


st.write(rounded_button_style, unsafe_allow_html=True)

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text) # to make list words
    y=[]
    for i in text:   # to remove special words like %% & etc
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string .punctuation: # to remove punctuation and stopwords
            y.append(i)
    text=y[:]
    y.clear()    
    
    for i in text:
        y.append(ps.stem(i))   #to stem each word
    return " ".join(y)  # return string 


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))


st.title("Email/SMS Spam Classifier")
input_sms=st.text_area("Enter Message")

#

if st.button("Predict", key="rounded_button"):
      

      transformed_sms=transform_text(input_sms)
      vectorized_sms=tfidf.transform([transformed_sms])

      result=model.predict(vectorized_sms)[0]

#diplay

      if result==1:
           
           
           st.header("Spam")
      else:
           st.header("Not Spam")
  
  






