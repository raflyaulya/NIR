from flask import Flask, render_template, request, url_for 
# import string
# import re
!pip install bs4 pandas as pd
!pip numpy 
!pip matplotlib
!pip wordcloud 
!pip io
!pip base64
!pip seaborn 
!piprequests
!pip nltk
!pip nltk.corpus
!pip textblob 

from bs4 import BeautifulSoup
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import io
import base64
import seaborn as sn
import requests

import nltk
# nltk.download()
from nltk.corpus import stopwords
from textblob import TextBlob
# from file_analyze import airline_func           # change ur analysis library

app = Flask(__name__)


# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('stopwords')

def airline_func(airline_name):
    formatted_name = '-'.join(airline_name.split())
    base_url = f"https://www.airlinequality.com/airline-reviews/{formatted_name}"         # output_name
    pages = 10
    page_size = 100

    reviews = []

    for i in range(1, pages + 1):
        # print(f"Scraping page {i}")
        url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"
        response = requests.get(url)

        # Parse content
        content = response.content
        parsed_content = BeautifulSoup(content, 'html.parser')
        for para in parsed_content.find_all("div", {"class": "text_content"}):
            reviews.append(para.get_text())
        

    df = pd.DataFrame()
    df["reviews"] = reviews

    # set index
    df.index = range(1, len(df)+1)

    unnecessary_statement1 = '✅ Trip Verified | '
    unnecessary_statement2 = 'Not Verified | '
    unnecessary_word3 = '✅ Verified Review | '

    df.reviews = df.reviews.apply(lambda x: x.replace(unnecessary_statement1, ''))
    df.reviews = df.reviews.apply(lambda x: x.replace(unnecessary_statement2, ''))
    df.reviews = df.reviews.apply(lambda x: x.replace(unnecessary_word3, ''))


    # ==============             DATA ANALYSIS       =================================
    
    def analyze_sentiment(reviews):
        tokens = nltk.word_tokenize(reviews)
            
        tagged_tokens = nltk.pos_tag(tokens)

        lemmatized_words = []    
        lemmatizer = nltk.WordNetLemmatizer()
        for word, tag in tagged_tokens:
            tag = tag[0].lower() if tag[0].lower() in ['a', 'r', 'n', 'v'] else 'n'
            lemmatized_words.append(lemmatizer.lemmatize(word, tag))    

        cust_stopwords = nltk.corpus.stopwords.words('english')
        clean_words = [word for word in lemmatized_words if word.lower() not in cust_stopwords]

        clean_text = ' '.join(clean_words)

        blob = TextBlob(clean_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity > 0:
            sentiment = 'Positive'
        elif polarity < 0:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return sentiment, polarity, subjectivity

    df[['sentiment', 'polarity', 'subjectivity']] = df['reviews'].apply(analyze_sentiment).apply(pd.Series)


    # print('Please select which sentiment reviews you want to display')
    # review_option = input('positive / negative / neutral?\n')
    positive_option = 'Positive'        #===============================

    if positive_option == 'Positive':
        positive_reviews = df[df['sentiment'] == 'Positive']
    result_positive = np.random.choice(positive_reviews['reviews'])

    negative_review = 'Negative'        #===============================

    if negative_review == 'Negative':
        negative_reviews = df[df['sentiment'] == 'Negative']
    result_negative = np.random.choice(negative_reviews['reviews'])


    plots =[]
    
    # BAR CHART
    bar_sentiment_counts = df.groupby('sentiment')['reviews'].count()
    sn.barplot(x=bar_sentiment_counts.index, y=bar_sentiment_counts.values)
    plt.title('Sentiment Analysis Results')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')


    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    image_png = buffer.getvalue()
    plot = base64.b64encode(image_png).decode()
    plots.append(plot)
    buffer.close()

    plt.clf()


    # PIE CHART
    pie_sentiment_counts = df.groupby('sentiment')['reviews'].count()
    plt.pie(pie_sentiment_counts.values, labels=pie_sentiment_counts.index, autopct='%1.2f%%')
    plt.title('Sentiment Analysis Results')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    image_png = buffer.getvalue()
    plot = base64.b64encode(image_png).decode()
    plots.append(plot)
    buffer.close()
    
    plt.clf()

    return plots, result_positive, result_negative


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/finalResult', methods=['POST'])
def finalResult():
    if request.method == 'POST':
        res_airlineName = request.form['airlineName']
        plots, result_positive, result_negative = airline_func(res_airlineName)  
        return render_template('finalResult.html',plots=plots, 
                               result_positive=result_positive, result_negative = result_negative)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0') #debug = TRUE
    # d:/ИПМКН/Семестр 5/НИР/NIR/app.py