# -*- coding: utf-8 -*-
#import libraries
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from flask import Flask, render_template,request,url_for
from flask_bootstrap import Bootstrap 
from collections import Counter
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd  
import re     
import nltk
import json
import numpy
from nltk.corpus import stopwords
#training sets train0 and trainSample are for testing, to get precise answer
#train is actual dataset
train = pd.read_csv('train2.csv',encoding='utf-8')
#train = pd.read_csv('train0.csv',encoding='utf-8')

app = Flask(__name__)
Bootstrap(app)
#preprocessing of input data, because inserted data is raw and needs cleaning...
def text_to_words( raw_text ):
    letters_only = re.sub("[^а-яА-Я]", " ", raw_text) 
    words = letters_only.lower()
    stops = set(stopwords.words("russian"))             
    meaningful_words = [w for w in words if not w in stops]  
    return( " ".join( meaningful_words ))
def preprocess_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
    text = re.sub('@[^\s]+', 'USER', text)
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', '', text)
    text = re.sub(' +', '', text)
    return text.strip()

#we refer to web page index.html
@app.route('/')
def index():
    return render_template('index.html')

#analyse() function does synthesis of input data 
@app.route('/analyse',methods=['POST'])
def analyse():
    if request.method == 'POST':
        clean_train_texts = train["text"]
        vectorizer = CountVectorizer(analyzer = "word") 
        train_data_features = vectorizer.fit_transform(clean_train_texts)
        train_data_features = train_data_features.toarray()
        vocab = vectorizer.get_feature_names()
        forest = RandomForestClassifier() 
        forest = forest.fit( train_data_features, train["sentiment"] )
        rawtext = request.form['rawtext']
        clean_test_texts = rawtext 
        data = preprocess_text(text_to_words(clean_test_texts))
        clean_test_text = [data]

        test_data_features = vectorizer.transform(clean_test_text)
        test_data_features = test_data_features.toarray()
        result = forest.predict(test_data_features)
        received = rawtext

        


    return render_template('index.html',received_text = received,summary=result)



if __name__ == '__main__':
    app.run(debug=True)