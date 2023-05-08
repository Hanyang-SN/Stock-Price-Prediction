# 2023-04-17일자의 코스피 데이터를 한국거래소에서 csv파일로 다운로드하여,
# 시가총액 기준 1~50위 주식을 종목코드, 종목명, 시가총액 정보가 담긴 파일을 만듬

import pandas as pd

# csv 파일 읽기
df = pd.read_csv("data_20230417.csv", encoding="UTF-8")

# 종목코드, 시가총액 열만 선택
df = df[["종목코드", "종목명", "시가총액"]]

# 시가총액 기준으로 내림차순 정렬
df = df.sort_values(by="시가총액", ascending=False)

# 상위 50개 데이터만 선택
df = df.head(50)

# 새로운 csv 파일 생성
df.to_csv("data.csv", index=False, encoding="cp949")
