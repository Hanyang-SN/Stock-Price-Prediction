# 각 text들을 csv 형태로 만들어 준다.
import pandas as pd

def print_divider():
    print("--------------------------------------------------\n")

df = pd.read_excel("./data.xlsx")
# 중간에 다 비어있는 col이 있어서 다 결측치 취급됨.
# df.dropna(inplace=True)


# 키워드만 가져오기
keywords = df['키워드']
keywords.dropna(inplace=True)

keywords.to_csv("./keywords.csv")