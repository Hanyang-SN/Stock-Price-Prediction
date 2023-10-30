import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pykrx as stock
from sklearn.preprocessing import MinMaxScaler
from pykrx import stock
from pykrx import bond


min_max = MinMaxScaler() #[[nan],[nan],[0.34]] 형태로 저장됨

# # finance_code_list = "KB금융	105560 신한지주	055550 하나금융지주	086790 메리츠금융지주	138040 기업은행	024110 미래에셋증권	006800 NH투자증권	005940 삼성증권	016360".split()

# 분석할 주식 정보 담긴 csv 파일 읽어오기
df = pd.DataFrame({'code': ["105560","055550","086790","138040","024110","006800","005940","016360"],
                   'name':["KB금융","신한지주","하나금융지주","메리츠금융지주","기업은행","미래에셋증권","NH투자증권","삼성증권"]})

print(df)

# index - 날짜 | col - 시가, 고가, 저가, 종가, 거래량, 거래대금, 등락률
stock_list = []

# 기술지표 데이터
RSI_list = []

# ohlcv 데이터 입력
# : 특정일 코스피 종목들의 OHLCV(주식시세) O: 시가 (Open) H: 고가 (High) L: 저가 (Low) C: 종가 (Close) V: 거래량 (Volume)

for i in df["code"]:
    stock_list.append((stock.get_market_ohlcv("20230910", "20231030", str(i))))
    # 특정 기간의 주식 OHLCV데이터를 불러와 stock_list에 append한다.

# 기술지표 계산

# 1) RSI 계산 (14일부터) 값 범위 - 0~100

# 스무딩 과정 구현하기 위해 rolling() 함수 사용.
# 일반적으로 RSI 지수는 14일을 기준으로 계산.
# 따라서 14일 전 RSI는 NaN상태
print(stock_list[0][-20:])

def calculate_rsi(dataframe, window_size=14):
    # 전일 대비 수익률을 계산
    diff = dataframe["종가"].diff(1)
    # diff = diff.dropna()  # NaN 값을 제거.

    # RSI 계산
    up = diff.copy()
    down = diff.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    U = up.rolling(window=window_size).mean()
    D = abs(down.rolling(window=window_size).mean())
    rsi = 100.0 - (100.0 / (1.0 + (U / D)))

    # 스무딩 적용
    rsi = rsi.rolling(window=5).mean()

    # 기존 DataFrame에 "rsi" 열 추가
    dataframe["rsi"] = rsi

    rsi_dataframe = pd.DataFrame({'rsi': rsi})

    return rsi_dataframe



# 2) MACD 계산 -> macd - signal 을 반환
# 26일 부터

def calculate_macd(dataframe, n_fast=12, n_slow=26, smoothing=9):
    # prices = prices["종가"]  # .diff(1)
    
    prices = dataframe["종가"].diff(1)

    # 단기, 장기 이동평균 계산
    ema_fast = prices.ewm(span=n_fast, min_periods=n_fast).mean()
    ema_slow = prices.ewm(span=n_slow, min_periods=n_slow).mean()
    # MACD 계산
    macd = ema_fast - ema_slow
    # MACD 신호선 계산
    signal = macd.ewm(span=smoothing, min_periods=smoothing).mean()
    # MACD 히스토그램 계산
    macd_diff = macd - signal
    dataframe["macd"] = macd_diff
    macd_df = pd.DataFrame({'macd': macd_diff})
    return macd_df


# 3 ) 거래량 (volume) => 계산 필요 없음
def calculate_volume(dataframe):
    vol=dataframe["거래량"]
    vol_df = pd.DataFrame({'volume': vol})
    return vol_df


# 볼린저 밴드 -> 주가의 이동평균선, 상단 밴드와 하단 밴드로 구성되어 과매수, 과매도 상태를 파악할 수 있게 하는 지표
# (상단밴드-종가), (종가-하단밴드) => 이렇게 데이터 맞췄음.

def calculate_bollinger_bands(dataframe, window=20, num_std=2):
    df = dataframe.copy()
    df['MA'] = df['종가'].rolling(window=window).mean()  # 이동평균 계산
    df['std'] = df['종가'].rolling(window=window).std()  # 주가의 표준편차 계산
    df['Upper Band'] = df['MA'] + num_std * df['std']  # 상단 밴드 계산
    df['Lower Band'] = df['MA'] - num_std * df['std']  # 하단 밴드 계산

    df['upper band diff'] = df['Upper Band'] - df['종가']
    df['lower band diff'] = df['종가'] - df['Lower Band']
    boll_df = df[['upper band diff','lower band diff']]

    return boll_df


# 스토캐스틱

def calculate_stochastic(data, window=14):
    df = data.copy()
    # 이동 최고가(Highest High) 계산
    df['Highest High'] = df['고가'].rolling(window=window).max()
    # 이동 최저가(Lowest Low) 계산
    df['Lowest Low'] = df['저가'].rolling(window=window).min()
    # Fast %K 계산
    df['K'] = (df['종가'] - df['Lowest Low']) / (df['Highest High'] - df['Lowest Low']) * 100
    # Slow %D 계산
    df['D'] = df['K'].rolling(window=3).mean()
    stochastic=df[['K','D']]
    return stochastic

# cci
def calculate_cci(data, window=20):
    tp = (data['고가'] + data['저가'] + data['종가']) / 3  # Typical Price 계산
    tp_mean = tp.rolling(window=window).mean()  # 이동 평균 계산
    tp_std = tp.rolling(window=window).std()  # 이동 표준편차 계산
    cci = (tp - tp_mean) / (0.015 * tp_std)  # CCI 계산
    cci_df = pd.DataFrame(cci, columns=['CCI'])  # 새로운 데이터프레임 생성
    return cci_df


#csv파일로 저장

for i in range(8):
    rsi = calculate_rsi(stock_list[i])
    macd = calculate_macd(stock_list[i])
    bollinger_bands = calculate_bollinger_bands(stock_list[i])
    cci = calculate_cci(stock_list[i])
    stochastic = calculate_stochastic(stock_list[i])
    volume = calculate_volume(stock_list[i])
    combined_df = pd.concat([rsi, macd, bollinger_bands,cci ,stochastic, volume], axis=1)
    combined_df.to_csv(df.iloc[i,1]+'_technical_index_data.csv', index=True)  # index 열을 저장하지 않으려면 index=False로 설정


