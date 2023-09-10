from datetime import date, timedelta
import os

# import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

from crawler import crawler
from preprocessor import preprocesser
from g_variable import *

# 최초 실행 시
# nltk.download('all')

def show_processing():
    print("■", end="")
    
    
def classifier(search_keyword, search_date, search_num):
    
    # 이미 local에 있으면 
    if not os.path.isdir("./text_data_application"):
        os.mkdir("./text_data_application")
    
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)
    
    if os.path.isfile(f"{DATA_DIR}/{search_keyword}-{search_date}-{search_num}"):
        return
    
    # 데이터 수집
    df = crawler(search_keyword, search_date, search_num)
    show_processing()
    
    # 데이터 전처리
    df = preprocesser(df)
    show_processing()

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
    new_df.to_csv(f"{DATA_DIR}/{search_keyword}-{search_date}-{search_num}")
    show_processing()


# 날짜 별로 자료 수집하기 위함
while not os.path.isfile(f"{DATA_DIR}/{KEYWORD}-{END_DATE}-50"):
    try:
        for delta in range(DIFF_DAYS.days):
            curr_date = START_DATE + timedelta(days=delta)
            classifier(KEYWORD, curr_date, 50)
    except:
        pass