import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pykrx as stock
from pykrx import stock
from pykrx import bond


# 분석할 주식 정보 담긴 csv 파일 읽어오기
df = pd.read_csv(
    "data.csv", encoding="cp949", dtype={"code": str, "amount": float}
)

# index - 날짜 | col - 시가, 고가, 저가, 종가, 거래량, 거래대금, 등락률
stock_list = []

# 기술지표 데이터
RSI_list = []

# ohlcv 데이터 입력
# : 특정일 코스피 종목들의 OHLCV(주식시세) O: 시가 (Open) H: 고가 (High) L: 저가 (Low) C: 종가 (Close) V: 거래량 (Volume)

for i in df["code"]:
    stock_list.append((stock.get_market_ohlcv("20180101", "20230101", str(i))))
    # 특정 기간의 주식 OHLCV데이터를 불러와 stock_list에 append한다.
    if True: 
        break


# 기관합계, 기타법인, 개인, 외국인합계 데이터 입력
investor=pd.DataFrame()

for i in df["code"]:
    investor.apply
    investor.append((stock.get_market_trading_value_by_date("20180101", "20230101", str(i))))
    if True:
        break

print(investor[0])

ki=[]

for j in range(0,investor.size):
    i=investor[j]
    for i in 4:
        i.loc[i['기관합계']>0,'기관합계']=1

# df.loc[df['Name'] == 'John', 'Age'] = 26 

investor['기관합계']=ki

print(investor[0])

# 기술지표 계산

# 1) RSI 계산 (14일부터)
# 값 범위 - 0~100

# 스무딩 과정 구현하기 위해 rolling() 함수 사용.
# 일반적으로 RSI 지수는 14일을 기준으로 계산.
# 따라서 14일 전 RSI는 NaN상태


def calculate_rsi(dataframe, window_size=14):
    # 전일 대비 수익률을 계산
    diff = dataframe["종가"].diff(1)
    diff = diff.dropna()  # NaN 값을 제거.

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
    df.rename(columns={"date": "rsi"}, inplace=True)

    # 결과 반환
    return rsi


rsi = calculate_rsi(stock_list[0])
# plt.plot(rsi)
# plt.show()


# plt.plot(rsi, xlim=(2018, 2019))
# # plt.plot(rsi)
# plt.show()


# 2) MACD 계산
# 26일 부터

def calculate_macd(prices, n_fast=12, n_slow=26, smoothing=9):
    prices = prices["종가"]  # .diff(1)

    # 단기, 장기 이동평균 계산
    ema_fast = prices.ewm(span=n_fast, min_periods=n_fast).mean()
    ema_slow = prices.ewm(span=n_slow, min_periods=n_slow).mean()

    # MACD 계산
    macd = ema_fast - ema_slow

    # MACD 신호선 계산
    signal = macd.ewm(span=smoothing, min_periods=smoothing).mean()

    # MACD 히스토그램 계산
    macd_diff = macd - signal

    return macd, signal
    #return macd_diff


# Test And Print macd
# print("@@ print MACD : ")
# print(calculate_macd(stock_list[0]))

# 여러 선 시각화- plot은 꺾은선그래프

# 시각화
macd, signal = calculate_macd(stock_list[0])

# plt.plot(macd,color='#e35f62')
# plt.plot(signal, color='forestgreen')
# plt.show()

# 3 ) 거래량 (volume) => 계산 필요 없음
def calculate_volume(dataframe):
    vol=dataframe["거래량"]
    vol.columns = ['volume']
    return vol

# 시각화
volume=calculate_volume(stock_list[0])
# print(volume.columns[0])

# 특정 범위의 데이터프레임 선택
start_date = '2022-12-01'
end_date = '2022-12-29'
df_range = volume.loc[start_date:end_date]

df_range.index = df_range.index.strftime('%Y-%m-%d') #이 코드를 추가해주지 않으면 x축의 데이터에 날짜 + 00:00:00이 추가되는데 왜 그런거?

# 막대 그래프 그리기
# df_range.plot(kind='bar', figsize=(10, 6))
# plt.title('2022년 12월 거래량')
# plt.xticks(rotation=45)
# plt.show()


# 4 ) ATR 계산, 14일 기준
# 왜틀..?

def calculate_atr(prices, period=20):
    # True Range 계산
    prices.dropna(inplace=True)
    prices["high-low"] = abs(prices["고가"] - prices["저가"])
    prices["high-pc"] = abs(prices["고가"] - prices["종가"].shift(1))  # shift(1) -> 현재 행을 기준으로 한 행 위로!! 이동시킴
    prices["low-pc"] = abs(prices["저가"] - prices["종가"].shift(1))
    TR = prices[["high-low", "high-pc", "low-pc"]].max(axis=1)

    # ATR 계산
    ATR = TR.rolling(period).mean()
    
    # 결과 반환
    # return pd.DataFrame({"ATR": ATR})
    return ATR


# print("@@ print ATR :")
# print(calculate_atr(stock_list[0]))

atr=calculate_atr(stock_list[0])
print(atr)

plt.plot(atr)
plt.show()


# 볼린저 밴드

def calculate_bollinger_bands(data, window=20, num_std=2):
    df = data.copy()
    boll=pd.DataFrame()
    boll['MA'] = df['종가'].rolling(window=window).mean()  # 이동평균 계산
    boll['std'] = df['종가'].rolling(window=window).std()  # 주가의 표준편차 계산
    boll['Upper Band'] = boll['MA'] + num_std * boll['std']  # 상단 밴드 계산
    boll['Lower Band'] = boll['MA'] - num_std * boll['std']  # 하단 밴드 계산
    return boll

df_bollinger=calculate_bollinger_bands(stock_list[0])
print(df_bollinger)

def plot_bollinger_bands(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['MA'], label='Moving Average')
    plt.plot(data.index, data['Upper Band'], label='Upper Band')
    plt.plot(data.index, data['Lower Band'], label='Lower Band')
    plt.fill_between(data.index, data['Upper Band'], data['Lower Band'], alpha=0.2)  # 상단 밴드와 하단 밴드 사이를 채움
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Bollinger Bands')
    plt.legend()
    plt.show()

plot_bollinger_bands(df_bollinger)


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

# 계산 및 시각화
df_stochastic = calculate_stochastic(stock_list[0])
print(df_stochastic)

def plot_stochastic(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['K'], label='K')
    plt.plot(data.index, data['D'], label='D')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Stochastic')
    plt.legend()
    plt.show()

plot_stochastic(df_stochastic)

# cci

def calculate_cci(data, window=20):
    tp = (data['고가'] + data['저가'] + data['종가']) / 3  # Typical Price 계산
    tp_mean = tp.rolling(window=window).mean()  # 이동 평균 계산
    tp_std = tp.rolling(window=window).std()  # 이동 표준편차 계산
    cci = (tp - tp_mean) / (0.015 * tp_std)  # CCI 계산
    cci_df = pd.DataFrame(cci, columns=['CCI'])  # 새로운 데이터프레임 생성
    return cci_df

# CCI 계산
df_cci = calculate_cci(stock_list[0])

def plot_cci(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['CCI'], label='CCI')
    plt.xlabel('Date')
    plt.ylabel('CCI')
    plt.title('Commodity Channel Index (CCI)')
    plt.legend()
    plt.show()

# CCI 데이터를 시각화
plot_cci(df_cci)