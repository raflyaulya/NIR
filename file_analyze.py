import requests
from bs4 import BeautifulSoup
import random
import pandas as pd


def airline_func(airline_name):
    formatted_name = '-'.join(airline_name.split())
    # return formatted_name
    

                                        # UNTUK MASUKIN NAMA AIRLINES NYA, NTAR DI WEBSITE HARUS DI PERINGATIN DULU, 
                                        # SUPAYA NULIS NAMA AIRLINESNYA GAK SALAH 
                                        # CONTOH: air france BETUL,, airfrance SALAH 
                                        # CONTOH: air asia SALAH,, airasia BENAR

# airline_name = input('input the airline\'s name:')
# output_name = airline_func(airline_name)
# print(output_name)  # Output: british-airways


# base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
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


    randcol = random.choice(df['reviews'])
    
    return randcol   # print out RANDOM REVIEWS 


# airline_name = input('enter the name of airlines: ')
# res = airline_func(airline_name)
# print(res)

    # DATA ANALYSIS
    # def analyze_sentiment(reviews):
    #     tokens = nltk.word_tokenize(reviews)
        
    #     tagged_tokens = nltk.pos_tag(tokens)

    #     lemmatized_words = []
    #     lemmatizer = nltk.WordNetLemmatizer()
    #     for word, tag in tagged_tokens:
    #         tag = tag[0].lower() if tag[0].lower() in ['a', 'r', 'n', 'v'] else 'n'
    #         lemmatized_words.append(lemmatizer.lemmatize(word, tag))

    #     stopwords = nltk.corpus.stopwords.words('english')
    #     clean_words = [word for word in lemmatized_words if word.lower() not in stopwords]

    #     clean_text = ' '.join(clean_words)

    #     blob = TextBlob(clean_text)
    #     polarity = blob.sentiment.polarity
    #     subjectivity = blob.sentiment.subjectivity

    #     if polarity > 0:
    #         sentiment = 'Positive'
    #     elif polarity < 0:
    #         sentiment = 'Negative'
    #     else:
    #         sentiment = 'Neutral'

    #     return sentiment, polarity, subjectivity

    # df[['sentiment', 'polarity', 'subjectivity']] = df['reviews'].apply(analyze_sentiment).apply(pd.Series)


    # print('Please select which sentiment reviews you want to display')
    # review_option = input('positive / negative / neutral?\n')
    # review_option = 'Positive'

    # # if review_option == 'Positive':
    # positive_reviews = df[df['sentiment'] == 'Positive']
    # random_review = np.random.choice(positive_reviews['reviews'])
    
    # return random_review
                                         
    # if review_option == 'Positive':
    #     positive_reviews = df[df['sentiment'] == 'Positive']
    #     random_review = np.random.choice(positive_reviews['reviews'])
    # if review_option == 'Negative':
    #     negative_reviews = df[df['sentiment'] == 'Negative']
    #     random_review = np.random.choice(negative_reviews['reviews'])
    # if review_option == 'Neutral':
    #     neutral_reviews = df[df['sentiment'] == 'Neutral']
    #     random_review = np.random.choice(neutral_reviews['reviews'])
        
    # print(random_review)



#  ==================    POSITIVE    ==================
# ==========              ANALYZING DATA         =====================
# print('Please select which sentiment reviews you want to analyze')
# review_option = 'Positive'

# sentiment_reviews = df[df['sentiment'] == review_option]
# sentiment_reviews = sentiment_reviews['reviews'].tolist()
# sentiment_reviews = ' '.join(sentiment_reviews)
# text = sentiment_reviews
# # text
# text = text.lower()
# text = re.sub(r'\d+','', text)

# text = text.translate(str.maketrans('', '', string.punctuation))

# tokens = [word for word in text.split()]
# # nltk.download()
# clean_tokens = tokens[:]
# for token in tokens:
#     if token in stopwords.words('english'):
#         clean_tokens.remove(token)

# freq = nltk.FreqDist(clean_tokens)
# for key, val in freq.items():
#     print(str(key), ':', str(val))


# ==============   VISUALISASI DATA     =============
# =================================================================
# freq.plot(30)       

# lis = []
# freq = nltk.FreqDist(clean_tokens)
# for key in freq.items():
#     lis.append(str(key[0]))

# print(lis)
# print('==================\n')

# stopwords= set(STOPWORDS)
# wordcloud = WordCloud(width=800, height=800,
#                       background_color='white', stopwords=stopwords,
#                       max_words=50,
#                       min_font_size=10).generate(' '.join(lis))

# plt.figure(figsize= (8,8), facecolor=None)
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.tight_layout(pad=0)

# plt.show()

# total_positive = df['sentiment'] == 'Positive'
# total_negative = df['sentiment'] == 'Negative'
# total_neutral = df['sentiment'] == 'Neutral'

# print('total positive sentiments:', total_positive.sum())
# print('total negative sentiments:', total_negative.sum())
# print('total neutral sentiments:', total_neutral.sum())