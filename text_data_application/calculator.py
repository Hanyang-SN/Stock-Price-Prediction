from datetime import date, timedelta
import pandas as pd
import os

from g_variable import *



def calculate_score(keyword, data_num):
    
    new_df = pd.DataFrame({"data" :[], "score" : []})
    
    for delta in range(DIFF_DAYS.days):
        curr_date = START_DATE + timedelta(days=delta)
        
        file_name = f"{DATA_DIR}/{keyword}-{curr_date}-{data_num}"
        
        # 아직 데이터 수집이 완료되지 않은 경우
        if not os.path.isfile(file_name):
            print("There is no file:", file_name)
            break
        
        # 데이터가 있는 경우
        df = pd.read_csv(file_name)
        total_senti_score = 0
        weight = 2
        num_of_score = len(df)
        for i in range(num_of_score):
            if df.iloc[i]['title containing keyword']:
                total_senti_score += (weight * df.iloc[i]['score']) / num_of_score
            else:
                total_senti_score += df.iloc[i]['score'] / num_of_score
        
        new_df.loc[delta + 1] = [curr_date, total_senti_score]
    
    print(new_df)
    
    if not os.path.isdir("./text_data_application"):
        os.mkdir("./text_data_application")
    
    if not os.path.isdir(SCORE_DIR):
        os.mkdir(SCORE_DIR)
        
    new_df.to_csv(f"{SCORE_DIR}/{keyword}")

calculate_score(KEYWORD, 50)