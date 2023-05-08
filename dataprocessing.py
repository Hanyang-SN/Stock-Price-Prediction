import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pykrx as stock
from pykrx import stock
from pykrx import bond


# 분석할 주식 정보 담긴 csv 파일 읽어오기
df = pd.read_csv("data.csv", encoding="UTF-8", dtype={"code": str, "amount": float})

"""
시가:날짜
고가:날짜

이렇게 데이터가 구성되어있음"""


# 이거는 내가 추가해준거임 -> 날짜, 시가, 고가, 저가, 종가, 거래량, 거래대금, 등락률
stock_list = []

# 기술지표 데이터
RSI_list = []

# ohlcv : 특정일 코스피 종목들의 OHLCV(주식시세) O: 시가 (Open) H: 고가 (High) L: 저가 (Low) C: 종가 (Close) V: 거래량 (Volume)

for i in df["code"]:
    stock_list.append((stock.get_market_ohlcv("20180101", "20230101", str(i))))
    # 특정 기간의 주식 OHLCV데이터를 불러와 stock_list에 append한다.
    if True:
        break

print(stock_list)

# 기술지표 계산

# 1) RSI 계산 (14일부터)


# 스무딩 과정 구현하기 위해 rolling() 함수 사용.
# 일반적으로 RSI 지수는 14일을 기준으로 계산한다. 따라서 14일 전 RSI는 NaN상태


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
    # rsi = rsi.rolling(window=5).mean()
    # df.rename(columns={"date": "rsi"}, inplace=True)

    # 결과 반환
    return rsi


rsi = calculate_rsi(stock_list[0])
print("@@ print RSI : ")
print(rsi)

plt.plot(rsi)
plt.show()

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
    histogram = macd - signal

    return histogram


# Test And Print macd
# print("@@ print MACD : ")
# print(calculate_macd(stock_list[0]))


# 3 ) 거래량 (volume) => 계산 필요 없음
def calculate_volume(dataframe):
    return dataframe["거래량"]


# 4 ) ATR 계산


def calculate_atr(prices, period=14):
    # True Range 계산
    prices["high-low"] = prices["고가"] - prices["저가"]
    prices["high-pc"] = abs(prices["고가"] - prices["종가"].shift(-1))  # 전일 종가이므로 -1 으로 시프트
    prices["low-pc"] = abs(prices["저가"] - prices["종가"].shift(-1))
    TR = prices[["high-low", "high-pc", "low-pc"]].max(axis=1)

    # ATR 계산
    ATR = TR.rolling(period).mean()

    # 결과 반환
    # return pd.DataFrame({"ATR": ATR})
    return ATR


atr = calculate_atr(stock_list[0])
plt.plot(atr)
plt.show()
