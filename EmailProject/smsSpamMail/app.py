import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):

    # 1 → Ensure string
    if not isinstance(text, str):
        text = str(text)

    # 2 → Lowercase
    text = text.lower()

    # 3 → Tokenization
    text = nltk.word_tokenize(text)

    # 4 → Remove special chars
    y = []
    for i in text:
        if i.isalnum():        # keep only alphanumeric
            y.append(i)

    # 5 → Remove stopwords + punctuation
    y2 = []
    for i in y:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y2.append(i)

    # 6 → Stemming
    y3 = []
    for i in y2:
        y3.append(ps.stem(i))

    return " ".join(y3)

with open('vectorizer.pkl','rb') as file:
    tfidf = pickle.load(file)
with open('model.pkl', 'rb') as vector_file:
    model = pickle.load(vector_file)


st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('predict'):
    # preprocess
    transform_sms = transform_text(input_sms)
    # vectorize
    vector_input = tfidf.transform([transform_sms])
    # predict
    result = model.predict(vector_input)[0]
    # display
    if result == 1:
        st.header("Spam")
    else:
        st.header("not spam")


