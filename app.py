import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from joblib import dump, load

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import re

def remove_tags(text):
    TAG_RE = re.compile(r'<[^>]+>')
    return TAG_RE.sub('', text)

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def text_process(text):
    text = re.sub(r"http\S+","", text)
    text = remove_tags(text)
    text = decontracted(text)
    text = re.sub("\S*\d\S*", "", text).strip()
    text = re.sub('[^A-Za-z]+', ' ', text)
    text = text.lower()
    return text

def word_count(text):
    text = text.split()
    return len(text)

def remove_stopwords(text):
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    clean_text = ''
    for w in text.split():
        if w not in stop_words:
            clean_text = clean_text + " " + w
    return clean_text.strip()

def text_stemming(text):
    sbs = SnowballStemmer(language='english')
    words = word_tokenize(text)
    new_text = ''
    for w in words:
        new_text = new_text + " " + sbs.stem(w)
    return new_text.strip()


st.text('Adesh Kumar(@gangwaradi)\n17.09.2021')
@st.cache(allow_output_mutation=True)
def load_data():
    scaler_fn = load('scaler.pkl')
    model_fn = load('model.pkl')
    tfidf_fn = load('tfidf_vectorizer.pkl')
    return scaler_fn, model_fn, tfidf_fn

scaler, model, tfidf_vec = load_data()

st.title('Compare the rading level complexity between two Texts')
text_1 = st.text_input("Text 1")
text_2 = st.text_input("Text 2")
if st.button('Submmit'):
    data = pd.DataFrame(columns = ['excerpt'])
    data['excerpt'] = [text_1,text_2]

    data["word_count"] = data["excerpt"].apply(lambda x: word_count(x))
    data["excerpt_len"] = data["excerpt"].apply(lambda x: len(x))
    data['ratio'] = np.divide(data['word_count'], data['excerpt_len'])
    data["process_excerpt"] = data["excerpt"].apply(lambda x: text_process(x))
    data["TWC"] = data["process_excerpt"].apply(lambda x: word_count(x))
    data["process_excerpt"] = data["process_excerpt"].apply(lambda x: remove_stopwords(x))
    data["process_excerpt"] = data["process_excerpt"].apply(lambda x: text_stemming(x))
    data["PWC"] =data["process_excerpt"].apply(lambda x: word_count(x))
    data["SWC"] = data["TWC"] - data["PWC"]
    data['TW_PW_ratio'] = np.divide(data['TWC'],data['PWC'])

    data_excerpt_Tfidf = tfidf_vec.transform(data['process_excerpt'].values)
    data_scale = scaler.transform(data[['word_count','excerpt_len','ratio','TWC','PWC','SWC','TW_PW_ratio']].values)

    data_tr = hstack([data_excerpt_Tfidf, data_scale]).tocsr()

    y_pred = model.predict(data_tr)
    y_min = -3.676268
    y_max = 1.711390
    y_pred = (y_pred - y_min)/(y_max - y_min)
    st.write('**Text 1: **' + text_1)
    st.write('**Text 2: **' + text_2)
    if (y_pred[0] > y_pred[1]):
        st.write('**Text 2 is ' + str(int((y_pred[0]-y_pred[1])*100)) + '% more complex than Text 1**')
    elif (y_pred[1] > y_pred[0]):
        st.write('**Text 1 is ' + str(int((y_pred[1]-y_pred[0])*100)) + '% more complex than Text 2**')
    else:
        st.write('**Text 1 and Text 2 are same**')
        
