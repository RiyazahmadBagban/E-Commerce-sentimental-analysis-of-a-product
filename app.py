import streamlit as st
import re
import string
from pickle import load
from textblob import TextBlob
from PIL import Image

import nltk
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer 
ls=WordNetLemmatizer()
nltk.download('stopwords')
sw=nltk.corpus.stopwords.words('english')
sw.remove('not')

st.markdown("<h1 style='text-align: center; color: greeen;'>Sentiment Reviews</h1>", unsafe_allow_html=True)
image = Image.open('Boat.jpeg')
col1, col2, col3 = st.columns(3)
with col1:
    st.write('')
with col2:
    st.image(image,width=300)
with col3:
    st.write('')
st.markdown("<h1 style='text-align: centre; color: black; font-size: 25px';>- P125 Team 2</h1>", unsafe_allow_html=True)


def no_punc(text):
    docs = []
    text=text.lower()
    text=re.sub('[^A-Za-z]+', ' ',text)
    text = text.split()
    text=[ls.lemmatize(word) for word in text if word not in sw]
    text=' '.join(text)
    docs.append(text)
    return docs



def predict(text):
    vector=load(open('vectorizer.pkl','rb'))
    classify=load(open('model.pkl','rb'))
    clean_text=no_punc(text)
    vector_clean_text=vector.transform(clean_text)
    vector_array=vector_clean_text.toarray()
    prediction=classify.predict(vector_array)
    return prediction

def main():
    st.markdown("<h2 style='text-align: center; color: green;'>Please enter your review</h2>", unsafe_allow_html=True)
    input_msg = st.text_input("")
    prediction = predict(input_msg)


    if(input_msg):
        st.subheader('Prediction')
        if prediction == 2:
            st.write("<h1 style='text-align: right; font-size: 30px; color: green;'>Positive 😍</h1>", unsafe_allow_html=True) 
        elif prediction == 0:
            st.write("<h1 style='text-align: right; font-size: 30px; color: red;'>Negative 😱</h1>", unsafe_allow_html=True)
        else:
            st.write("<h1 style='text-align: right; font-size: 30px; color: orange;'>Neutral 🤐</h1>", unsafe_allow_html=True)
            
        review_pol=''.join(input_msg)
        pol_score=TextBlob(review_pol)
        polarity=round(pol_score.sentiment.polarity,4)
        st.subheader('The polarity of the review is: {}'.format(polarity))


if __name__ == '__main__' :
    main()