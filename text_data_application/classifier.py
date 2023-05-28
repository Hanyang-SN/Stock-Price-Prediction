from datetime import date
import re

# import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

from crawler import crawler
from preprocessor import preprocesser

# 최초 실행 시
# nltk.download('all')

search_keyword = "삼성전자"
search_date = date(2018, 1, 1)
search_num = 50

# 데이터 수집
df = crawler(search_keyword, search_date, search_num)

# 데이터 전처리
preprocesser(df)

# TODO: 저장된 데이터의 마지막 번호 뒷 부분 부터 크롤링 시작하기

score_list, title_containing_keyword_list = [], []

# 각각의 감성 점수
sia = SentimentIntensityAnalyzer()
for [en, ko] in zip(df["title-en"], df["title-ko"]):
    sia_score = sia.polarity_scores(en)['compound']
    if sia_score != 0:
        score_list.append(sia_score)
        title_containing_keyword_list.append(ko.find(search_keyword) != -1)

new_df = pd.DataFrame({"score" : score_list, "title containing keyword" : title_containing_keyword_list})
new_df.to_csv(f"./text_data_application/data/{search_keyword}-{search_date}-{search_num}")
