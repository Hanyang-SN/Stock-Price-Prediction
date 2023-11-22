# -*- coding: utf-8 -*-
"""SPP_ensemble_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TuQ-ts3CjIqE4nTXcnAbTAqtgOPIBxsm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import unicodedata
import math
import torch, gc
import torch.optim as optim
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from pykrx import stock
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from keras.layers import Dense, LSTM
from keras.models import Sequential, load_model

"""구글 드라이브에 마운트
드라이브에 저장된 데이터를 불러옴 (로컬에서 추가할 수도 있지만 런타임 해제되면 없어짐..ㅠㅠ)

data폴더 내에 news_data, technical_data폴더가 위치

폴더 경로 : /content/gdrive/MyDrive/data
"""

## "KB금융	105560 신한지주	055550 하나금융지주	086790 메리츠금융지주	138040 기업은행	024110 미래에셋증권	006800 NH투자증권	005940 삼성증권	016360".split()

"""# 1. 데이터 다운로드 및 전처리

## 2) 데이터 불러오기 (5년 치, 10년 치, 50년 치)

### (1) 8개 종목 선택

KB금융	105560 신한지주	055550 하나금융지주	086790 메리츠금융지주	138040 기업은행	024110 미래에셋증권	006800 NH투자증권	005940 삼성증권	016360
"""

# Make code dictionary.
finance_code_dict = dict()
finance_code_list = "KB금융	105560 신한지주	055550 하나금융지주	086790 메리츠금융지주	138040 기업은행	024110 미래에셋증권	006800 NH투자증권	005940 삼성증권	016360".split()
for i in range(8):
  finance_code_dict[finance_code_list[2*i]] = finance_code_list[2*i + 1]

print(finance_code_dict)

"""{'KB금융': '105560', '신한지주': '055550', '하나금융지주': '086790', '메리츠금융지주': '138040', '기업은행': '024110', '미래에셋증권': '006800', 'NH투자증권': '005940', '삼성증권': '016360'}

### (2) 데이터 가져오기 함수 정의 (5y, 10y)
"""

def get_5y_10y(ticker_name):
  ticker_code = finance_code_dict[ticker_name]
  return stock.get_market_ohlcv("20180101", "20221231", ticker_code), stock.get_market_ohlcv("20130101", "20221231", ticker_code)

"""### (3) 데이터 그리기 함수 정의"""

import matplotlib
import matplotlib.pyplot as plt

def draw_graph_10y(ticker_name):

  _, df = get_5y_10y(ticker_name)

  # 1 line, 3 graphs

  # graph 1
  plt.subplot(3, 1, 1)
  series = df['종가']
  plt.title(f"{ticker_name} time series")
  plt.spring()
  plt.plot(series)

  # graph 2
  plt.subplot(3, 1, 2)
  plt.title(f"{ticker_name} difference, time series")
  series_diff = series - series.shift(1)
  plt.plot(series_diff)

  # graph 3
  plt.subplot(3, 1, 3)
  plt.title(f"{ticker_name} difference, histogram")
  plt.hist(series_diff)

  plt.tight_layout()

  plt.show()

"""### (4) train_data, test_data 얻는 함수"""

# 데이터 기간 설정하는 부분
start_date_train = "20210101"
end_date_train = "20230101"
start_date_test = "20230101"
end_date_test = "20231109"

def get_10y_data(ticker_name): #train, test데이터 따로 df으로 -> 통합하여 리턴하도록 변경함
  ticker_code = finance_code_dict[ticker_name]
  selected_columns = ['종가']  # 포함하려는 열 이름 리스트
  df = stock.get_market_ohlcv(start_date_train, end_date_test, ticker_code)
  df = df.astype('float32')
  return pd.DataFrame(df[selected_columns])

df = pd.DataFrame(columns=["date"] + list(finance_code_dict.keys()))
df.set_index('date', inplace=True)
for ticker_name in finance_code_dict:
  df[ticker_name] = stock.get_market_ohlcv(start_date_train, end_date_test, finance_code_dict[ticker_name])['종가']
df.tail()

print(df.iloc[-2:])
print(df.iloc[-1] - df.iloc[-2])

int_list = df.iloc[-2] - df.iloc[-1]
['상승' if i < 0 else '하락' for i in df.iloc[-2] - df.iloc[-1]]

df3 = pd.DataFrame(columns=["date"] + list(finance_code_dict.keys()))
df3.set_index('date', inplace=True)

for i in range(5):
  df3.loc[f"{i}일 전 결과"] = ['상승' if j < 0 else '하락' for j in df.iloc[-(i-2)] - df.iloc[-(i-1)]]

['상승' if df3[ticker_name].value_counts()[0] <=
          (df3[ticker_name].value_counts()[1] if len(df3[ticker_name].value_counts()) >1 else 100)
          else '하락' for ticker_name in finance_code_dict]
# df3

df2 = pd.DataFrame(columns=["date"] + list(finance_code_dict.keys()))
df2.set_index('date', inplace=True)




df2.loc["11/09 예측"] = ['하락', '하락', '상승', '하락', '하락', '하락', '하락', '상승']
df2.loc['11/09 결과'] = ['상승' if i < 0 else '하락' for i in df.iloc[-2] - df.iloc[-1]]

df2.loc["11/09 5일 보팅"] = ['하락', '하락', '하락', '하락', '하락', '하락', '하락', '하락']

df2.loc["11/10 예측"] = ['하락', '상승', '하락', '상승', '하락', '상승', '하락', '하락']
df2.loc['11/10 5일 보팅'] = ['하락', '하락', '하락', '하락', '하락', '하락', '하락', '하락']






df2

print(f"- 앙상블 모델: {sum(df2.iloc[0] == df2.iloc[1]) / 8 * 100}%")
print(f"- 5일 보팅: {sum(df2.iloc[1] == df2.iloc[2]) / 8 * 100}%")

"""### 주식 미개장일의 뉴스 데이터는 익일(개장날)에 반영 -> 미개장 기간 동안의 평균값을 반영해주는 함수."""

def calculate_sentimental_avg(code, name, df_news_data):
  # 주식 미개장일의 뉴스 데이터는 익일(개장날)에 반영 -> 평균값을 반영하도록 구현하는 함수.
  # 종가 데이터를 불러옴 -> train, test 나눠줘야함.
  total_df = stock.get_market_ohlcv(start_date_train, end_date_test, code)
  total_df = total_df['종가']
  total_df = pd.DataFrame(total_df)
  total_df['date'] = total_df.index

  # 데이터프레임 A: 뉴스 데이터
  df_a = df_news_data
  df_a['Date'] = pd.to_datetime(df_a['date'])
  df_a.set_index('Date', inplace=True)
  print(df_a)

  # 데이터프레임 B: 주가 데이터
  price_data = total_df
  df_b = pd.DataFrame(price_data)
  df_b['Date'] = pd.to_datetime(df_b['date'])
  df_b.set_index('Date', inplace=True)
  print(df_b)

  # 데이터프레임 A와 데이터프레임 B를 병합 (외부 조인)
  merged_df = pd.merge(df_a, df_b, how='outer', left_index=True, right_index=True)

  # NaN 값 (주말 뉴스)을 처리하여 다음 개장일의 뉴스 데이터에 반영
  merged_df[name].fillna(method='ffill', inplace=True)

  # 평균 계산하여 NaN 값을 대체
  merged_df['종가'] = merged_df['종가'].fillna(0)  # NaN 값을 0으로 설정
  next_open_day = merged_df.index[merged_df['종가'] != 0][0]  # 다음 개장일 찾기

  # next_open_day=next_open_day+ timedelta(days=1)
  avg_news_score = merged_df.loc[merged_df.index <= next_open_day, name].mean()
  merged_df.loc[merged_df.index == next_open_day, name] = avg_news_score
  merged_df = merged_df.drop(columns=['date_y','종가'])

  return merged_df

"""# 2.Dataset 윈도우"""

from torch.utils.data import DataLoader, Dataset
class windowDataset(Dataset):
  # data_stream     : input_window, output_window 크기에 따라 쪼개질 데이터
  # input_window    : 인풋 기간
  # output_window   : 아웃풋 기간
  # stride          :
    def __init__(self, data_stream, input_window, output_window, n_features=3, stride=5):
        # data_stream의 행 개수를 구한다.
        L = data_stream.shape[0]
        # stride에 따라 샘플 개수를 구한다.
        num_samples = (L - input_window - output_window) // stride + 1

        # [window 크기 * sample 개수] 크기의, 0으로 채워진 배열을 만든다.
        X = np.zeros([input_window, num_samples, n_features])
        Y = np.zeros([output_window, num_samples])

        # np.arange(num_samples): range(num_samples) 와 같음
        for i in np.arange(num_samples):
            # 1) X:   input_window 만큼 자르기 (stride * i ~)
            start_x = stride * i
            X[:,i] = data_stream[start_x:start_x + input_window]
            # 2) Y:   output_window 만큼 자르기 (stride * i + input_window ~)
            start_y = start_x + input_window
            Y[:,i] = data_stream[start_y:start_y + output_window]['종가']


        # shape       : [window 크기, sample 개수]
        X = X.reshape(X.shape[0], X.shape[1], n_features).transpose((1,0,2))
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1,0,2))
        X=X.astype('float32')
        Y=Y.astype('float32')
        self.x = X
        self.y = Y

        self.len = len(X)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
    def __len__(self):
        return self.len

"""# 3.Transformer 모델 정의"""

class TFModel(nn.Module):

# iw/ow:      input window, output window
# d_model:    인풋 개수
# nlayers:    인코더 부분의 인코더 개수
# nhead:      multihead attention 개수

    def __init__(self, iw: int, ow: int, d_model: int, nhead: int, nlayers: int, dropout=0.5, n_features=3):
        super(TFModel, self).__init__()

        # TransformerEncoderLayer 인스턴스 생성 ) 1개 인코더, 인풋 사이즈가 d_model이고 attention 개수는 nhead
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)

        # stacked 인코더, nlayers 만큼 쌓여있다.
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 인풋 차원 변환. 1차원 -> d_model//2차워 -> d_model차원
        self.encoder = nn.Sequential(
            # nn.Linear(1, d_model//2),
            nn.Linear(n_features, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )

        # 차원 변환. d_model -> d_model//2 -> 1
        self.linear =  nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )

        # 차원 변환. iw -> ow
        self.linear2 = nn.Sequential(
            nn.Linear(iw, (iw+ow)//2),
            nn.ReLU(),
            nn.Linear((iw+ow)//2, ow)
        )

    def generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, srcmask):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src.transpose(0,1), srcmask).transpose(0,1)
        output = self.linear(output)[:,:,0]
        output = self.linear2(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def gen_attention_mask(x):
    mask = torch.eq(x, 0)
    return mask

"""# 4.학습

- 입출력 윈도우 사이즈
- Learning Rate
- Model
  - layer
  - dropout
  - multihead attention 개수
- Cost Function
- Optimizer

# **lstm, svm 정의**

### **LSTM**
"""

def lstm_fit(): # lstm 데이터 생성 및 모델 구성, 학습하는 함수.

  # 데이터 시퀀스
  sequence_length = INPUT_WINDOW

  lstm_X = []
  lstm_y = []
  lstm_test_X = []
  lstm_test_y = []

  for i in range(len(train_X) - sequence_length):
      lstm_X.append(train_X[i:i+sequence_length])
      lstm_y.append([train_y[i+sequence_length]])

  for i in range(len(test_X) - sequence_length):
      lstm_test_X.append(test_X[i:i+sequence_length])
      lstm_test_y.append([test_y[i+sequence_length]])

  lstm_X = np.array(lstm_X)
  lstm_y = np.array(lstm_y)
  lstm_test_X = np.array(lstm_test_X)
  lstm_test_y = np.array(lstm_test_y)

  # LSTM 모델 구성
  lstm_model = Sequential()
  lstm_model.add(LSTM(50, input_shape=(lstm_X.shape[1], lstm_X.shape[2]))) # 21,3
  lstm_model.add(Dense(1))
  lstm_model.compile(optimizer='adam', loss='mean_squared_error')

  # 모델 학습
  lstm_model.fit(lstm_X, lstm_y, epochs=100, batch_size=32, verbose=0)

  # 모델 실행
  loss = lstm_model.evaluate(lstm_test_X, lstm_test_y)
  print(f'Test Loss: {loss}')

  return lstm_model, lstm_test_X, lstm_test_y

"""## **SVM**

svm 모델 생성하고 학습하는 함수

linear kernel을 사용할 때 정확도가 소폭 상승함.

*   항목 추가
*   항목 추가

**분류 문제를 위한 상승, 하락 이진분류 라벨을 담은 배열을 만들어주는 함수.**
"""

def svm_binary_classify(train_data_y):
  # 전일 대비 상승(1) 또는 하락(0) 여부를 담을 배열 초기화
  price_movement = np.zeros(len(train_data_y), dtype=int)

  # 주가 데이터를 기반으로 전일 대비 상승 또는 하락 여부를 계산
  for i in range(1, len(train_data_y)):
      if train_data_y[i] > train_data_y[i - 1]:
          price_movement[i] = 1  # 상승
      elif train_data_y[i] < train_data_y[i - 1]:
          price_movement[i] = 0  # 하락

  return price_movement

def svm_fit():
  train_y_binary = svm_binary_classify(train_y)

  test_y_binary = svm_binary_classify(test_y)
  #  SVM 모델 생성 및 학습
  svm_model = SVC(kernel='linear', C=1)
  svm_model.fit(train_X[:-1], train_y_binary[1:])

  return svm_model, test_y_binary

"""## **LSTM, SVM 예측 및 정확도 측정**

**lstm 정확도 측정 함수**
"""

def calculate_accuracy(predicted_prices, actual_prices):
    if len(actual_prices) != len(predicted_prices):
        print(len(actual_prices),len(predicted_prices))
        raise ValueError("데이터의 길이가 일치해야 합니다.")

    correct_predictions = 0

    for i in range(len(actual_prices) - 1):
        if (predicted_prices[i+1] - actual_prices[i]) * (actual_prices[i+1] - actual_prices[i]) >= 0:
            correct_predictions += 1

    accuracy = (correct_predictions / (len(actual_prices) - 1)) * 100  # 정확도 계산 (마지막 데이터는 다음 데이터가 없어서 제외)

    return accuracy

"""# **✅ 실험 부분**

## **Data setting**
"""

from scipy.stats import zscore

# 선택한 열의 값들을 z-score 정규화 적용하여 return 하는 함수

def to_z_score(_data):
  # selected_column = ['종가']
  # selected_column_values = _data[selected_column]
  numeric_cols = _data.select_dtypes(include='number').columns
  _data[numeric_cols] = _data[numeric_cols].apply(zscore)
  return _data

"""아래 코드로 돌리려 했는데.. 구글 드라이브에서 파일 불러오는 중 오류 발생 ㅠㅠ(error num 107) => 파일 업로드 코드로 대체 실행

### **구글드라이브에서 파일 불러오기**
"""

# for key, value in finance_code_dict.items():
#     key2=key.encode('utf-8').decode('utf-8')
#     key=unicodedata.normalize('NFC', key2)
#     # 뉴스 파일 불러옴
#     for filename in files_news:
#         filename=filename.encode('utf-8').decode('utf-8')
#         filename=unicodedata.normalize('NFC',filename)
#         if filename.startswith(key):
#             file_path = os.path.join(directory_path_news, filename)
#             try:
#                 df_news_data = pd.read_csv(file_path)
#             except FileNotFoundError:
#                 print(f"File '{filename}' not found.")
#             except Exception as e:
#                 print(f"An error occurred while opening '{filename}': {str(e)}")
#     # 기술지표 파일 불러옴
#     for filename in files_tech:
#         filename=filename.encode('utf-8').decode('utf-8')
#         filename=unicodedata.normalize('NFC',filename)
#         if filename.startswith(key):
#             file_path = os.path.join(directory_path_tech, filename)
#             try:
#                 df_tech_data = pd.read_csv(file_path)
#             except FileNotFoundError:
#                 print(f"File '{filename}' not found.")
#             except Exception as e:
#                 print(f"An error occurred while opening '{filename}': {str(e)}")

# print(df_news_data)
# print(df_tech_data)

"""# **파일 업로드**"""

# 디렉토리 경로 설정
directory_path_news = '/content/Data/news_data'
directory_path_tech = '/content/Data/technical_data'

# 디렉토리 내의 파일 목록 가져오기
files_news = os.listdir(directory_path_news)
print(files_news)
files_tech = os.listdir(directory_path_tech)

# # 디렉토리 경로 설정
# directory_path_news = '/content/gdrive/MyDrive/data/news_data'
# directory_path_tech = '/content/gdrive/MyDrive/data/technical_data'

# # 디렉토리 내의 파일 목록 가져오기
# files_news = os.listdir(directory_path_news)
# files_tech = os.listdir(directory_path_tech)

"""# **모델 실행 함수**

### **데이터 결합하여 리턴하는 함수 -> 일단은 안씀**
"""

# 기술지표 파일과 감성분석 파일을 종가 데이터와 결합하여 리턴하는 함수.
def return_data(df_news_data, df_tech_data, stock_name, stock_code):
    # 종가 데이터를 불러옴
    price_data = get_10y_data(stock_name)
    price_data['date'] = price_data.index
    # print(price_data)

    # 감성분석 파일 - 미 개장일 누적 데이터의 평균을 익일(개장일)에 반영하도록 하는 함수.
    df_news_data = calculate_sentimental_avg(stock_code, stock_name, df_news_data)
    df_news_data = df_news_data.rename(columns={'date_x': 'date'})
    df_news_data = df_news_data.rename(columns={stock_name: 'news_score'})
    # print(df_news_data)

    # tech data 열 이름 변경 (날짜를 date로)
    df_tech_data = df_tech_data.rename(columns={'날짜': 'date'})

    # 종가 + 감성분석 + 기술지표 통합
    price_data['date'] = pd.to_datetime(price_data['date'])
    df_news_data['date'] = pd.to_datetime(df_news_data['date'])
    df_tech_data['date'] = pd.to_datetime(df_tech_data['date'])
    print("ok")
    result_df = pd.merge(pd.merge(price_data, df_news_data, on='date', how='inner'), df_tech_data, on='date', how='inner')

    # 기술지표 계산 이슈 반영
    end_row = 35 # 35일치 데이터를 삭제해 줌(macd 계산 이슈)
    result_df = result_df.drop(result_df.index[0:end_row])
    result_df = result_df.set_index('date')

    # z-정규화 적용
    result_df = to_z_score(result_df)

    # train_data, test_data로 나누어주기
    # 특정 날짜를 기준으로 데이터프레임 분할
    date_string = start_date_test
    datetime_obj = datetime.strptime(date_string, "%Y%m%d")
    formatted_date = datetime_obj.strftime("%Y-%m-%d")
    split_date = formatted_date
    print(split_date)

    train_data = result_df[result_df.index <= split_date]  # split_date 이하의 데이터
    test_data = result_df[result_df.index > split_date]   # split_date 이후의 데이터
    print(train_data)
    return train_data, test_data

"""## **종목별로 파일 읽어와서 실행**"""

train_db = []
test_db = []

"""✅ 아래 코드 설명(코드가 김)

*   주식 목록을 불러와,
*   각 주식에 해당하는 뉴스 파일과 기술지표 파일을 폴더에서 찾고, 읽어와 df를 만들어 준다.
* 종가 데이터는 price_data, 뉴스 데이터는 df_news_data, 기술지표는 df_tech_data에 저장됨
* 데이터 통합
* z 정규화 진행
* train_df, test_db에 저장한다.

train_db, test_db에 8종목의 실험할 df가 담기게 됨.
"""

for key, value in finance_code_dict.items():
    key2=key.encode('utf-8').decode('utf-8')
    key=unicodedata.normalize('NFC', key2)
    # 뉴스 파일 불러와 df_news_data에 저장
    for filename in files_news:
        filename=filename.encode('utf-8').decode('utf-8')
        filename=unicodedata.normalize('NFC',filename)
        if filename.startswith(key):
            file_path = os.path.join(directory_path_news, filename)
            try:
                df_news_data = pd.read_csv(file_path)
            except FileNotFoundError:
                print(f"news : File '{filename}' not found.")
            except Exception as e:
                print(f"An error occurred while opening '{filename}': {str(e)}")
    # 기술 지표 파일 불러와 df_tech_data에 저장
    for filename in files_tech:
        filename=filename.encode('utf-8').decode('utf-8')
        filename=unicodedata.normalize('NFC',filename)
        if filename.startswith(key):
            file_path = os.path.join(directory_path_tech, filename)
            try:
                df_tech_data = pd.read_csv(file_path)
            except FileNotFoundError:
                print(f"tec : File '{filename}' not found.")
            except Exception as e:
                print(f"An error occurred while opening '{filename}': {str(e)}")

    # run함수 실행하여 모델 실행하기
    # 종가 데이터를 불러옴
    stock_name=key
    stock_code=value
    price_data = get_10y_data(stock_name)
    price_data['date'] = price_data.index

    # 감성분석 - 미 개장일 누적 데이터의 평균을 익일(개장일)에 반영하도록 하는 함수.
    # df_news_data = calculate_sentimental_avg(stock_code, stock_name, df_news_data)
    # df_news_data = df_news_data.rename(columns={'date_x': 'date'})
    # df_news_data = df_news_data.rename(columns={stock_name: 'news_score'})

    # tech data 열 이름 변경 (날짜를 date로)
    # date열 이름 맞추기
    # df_tech_data = df_tech_data.rename(columns={'날짜': 'date'})
    # price_data['date'] = pd.to_datetime(price_data['date'])
    # # df_news_data['date'] = pd.to_datetime(df_news_data['date'])
    # df_tech_data['date'] = pd.to_datetime(df_tech_data['date'])
    # # df_tech_data = df_tech_data[['rsi','volume','date','CCI','macd']]
    # df_tech_data = df_tech_data[['rsi','volume','date']]
    # print(df_tech_data)

    # 데이터프레임 합치는 부분!✅

    #종가+감성+기술
    # result_df = pd.merge(pd.merge(price_data, df_news_data, on='date', how='inner'), df_tech_data, on='date', how='inner')
    #종가+감성
    # result_df = pd.merge(price_data, df_tech_data, on='date', how='inner')

    # result_df = pd.merge(price_data, df_news_data, on='date', how='inner')

    #종가만..
    result_df = price_data

    # 기술지표 계산 이슈 반영
    end_row = 35 # 35일치 데이터를 삭제해 줌(macd 계산 이슈로 35일치가 NAN임.)
    result_df = result_df.drop(result_df.index[0:end_row])
    result_df = result_df.set_index('date')

    # z-정규화 적용
    result_df = to_z_score(result_df)

    # train_data, test_data로 나누어주기
    # 특정 날짜를 기준으로 데이터프레임 분할시킨다.
    date_string = start_date_test
    datetime_obj = datetime.strptime(date_string, "%Y%m%d")
    formatted_date = datetime_obj.strftime("%Y-%m-%d")
    split_date = formatted_date

    train_data = result_df[result_df.index <= split_date]  # split_date 이하의 데이터
    test_data = result_df[result_df.index > split_date]   # split_date 이후의 데이터
    train_db.append(train_data)
    test_db.append(test_data)

    # train_data, test_data = run(df_news_data, df_tech_data, key, value)

"""# **✅종목 선택**"""

index = 7
train_data = train_db[index]
test_data = test_db[index]

N_FEATURES = train_data.shape[1]

N_FEATURES

"""# **✅모델 부분**

# TF model

ir = 1e-04
batch size = 32
"""

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

test_data

test_data

# @title Hyper-parameter
INPUT_WINDOW = 7
OUTPUT_WINDOW = 7

BATCH_SIZE= 32
lr = 1e-4 # 학습률을 적당히 설정하는 게 중요함.

train_dataset = windowDataset(train_data, input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW, stride=1, n_features=N_FEATURES)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)     # 64 = 2^6, 512 = 2^9
test_dataset = windowDataset(test_data, input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW, stride=1, n_features=N_FEATURES)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)     # 64 = 2^6, 512 = 2^9

tf_model = TFModel(iw=INPUT_WINDOW, ow=OUTPUT_WINDOW, d_model=512, nhead=8, nlayers=4, dropout=0.1, n_features=N_FEATURES).to(device)
criterion = nn.MSELoss()                                            # MSEloss(): ow 각 요소들의 합
optimizer = torch.optim.Adam(tf_model.parameters(), lr=lr)

train_losses = []  # 각 에포크의 훈련 손실 값을 저장할 리스트

# @title TF Train

gc.collect()
torch.cuda.empty_cache()

# for tqdm
from tqdm import tqdm

# for trainig mode
epoch = 50
tf_model.train()
progress = tqdm(range(epoch))

# for drawing loss per epoch.
max_non_improvement = 10  # 일정 기간동안 개선되지 않을 때 학습을 종료하기 위한 조건
best_loss = float('inf')  # 최적의 손실 값을 추적하기 위한 변수
no_improvement_count = 0  # 개선되지 않은 에포크 카운트

for i in progress:
  batchloss = 0.0
  for (inputs, outputs) in train_loader:
    # inputs.shape: [batch_size, iw, 1] -> 1 말구 num_of features
    # outputs.shape: [batch_size, ow, 1]
    # Initialize grad
    optimizer.zero_grad()                                           # zero_grad()로 Torch.Tensor.grad 초기화. 초기화하지 않으면 다음 루프 backward() 시에 간섭함.
    # 모델에 사용할 마스크 생성
    # Forward propagation with masking
    src_mask = tf_model.generate_square_subsequent_mask(inputs.shape[1]).to(device)

    result = tf_model(inputs.float().to(device), src_mask)             # forward

    # Backward propagation
    loss = criterion(result, outputs[:,:,0].float().to(device))     # ?? 64개 중 하나만 loss를 담네?
    # print(f"[result]\n{result}\n\n[output[:,:,0]]\n{outputs[:,:,0]}\n\n[outputs]\n{outputs}")
    loss.backward()                                                 # backward
    optimizer.step()
    batchloss += loss

  print()
  progress.set_description(f"loss: {batchloss.cpu().item() / len(train_loader):0.6f}")

  # 훈련 손실 값 저장
  train_losses.append(batchloss.cpu().item() / len(train_loader))

  # 조기 종료 검사 및 학습 곡선 그리기
  if batchloss < best_loss:
    best_loss = batchloss
    no_improvement_count = 0
  else:
    no_improvement_count += 1

  if no_improvement_count >= max_non_improvement:
    print(f"Early stopping due to no improvement for {max_non_improvement} epochs.")
    break

progress.close()

# 학습 곡선 그리기
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

"""## **Test**"""

# set evaluation mode
tf_model.eval()

# Initialize correct & total
correct = 0
total = 0

#트렌스포머 모델의 이진분류 결과를 저장해주는 배열
transformer_result_binary = []
true_ = []

# 기울기 계산을 방지하기 위해 torch.no_grad() 블록 안에서 평가
with torch.no_grad():
  for (inputs, outputs) in tqdm(test_loader, desc="Evaluating"):
    # Forward propagation with masking
    src_mask = tf_model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
    result = tf_model(inputs.float().to(device), src_mask)

    # 상승/하강 예측
    predicted_changes = torch.sign(result[:, -1] - inputs[:, -1, 0].to(device))             # 마지막 예측 값 - 마지막 입력 값
    true_changes = torch.sign(outputs[:, -1, 0].to(device) - inputs[:, -1, 0].to(device))  # 실제 마지막 값 - 마지막 입력 값
    transformer_result_binary.append(predicted_changes.cpu()) #gpu에서 cpu로 데이터를 옮겨줘야 함..

    true_.append(true_changes.cpu()) #gpu에서 cpu로 데이터를 옮겨줘야 함..

    # 예측이 맞는 경우
    correct += (predicted_changes == true_changes).sum().item()
    total += inputs.size(0)

  progress.set_description(f"current accuracy: {correct/total:0.6f}")

# 정확도 계산
accuracy = correct / total
print(f"\nTF model) Directional Accuracy: {accuracy * 100:.6f}%")

transformer_result_binary

"""index = 0 (kb) 마지막 30일치의 등락결과 [ 1 -1  1  1  1 -1 -1  1  1  1 -1 -1 -1 -1 -1  1  1 -1  1  1 -1 -1  1 -1
 -1 -1  1 -1 -1 -1] tf....어떻게 데이터가 저장된거지..?
"""

true_

"""# **LSTM**"""

#lstm 데이터 세팅
INPUT_WINDOW = 7
OUTPUT_WINDOW = 7

train_X = train_data.to_numpy()
test_X = test_data.to_numpy()

train_y = train_X[:,0]
test_y = test_X[:,0]

#트렌스포머 WINDOW사이즈를 고려하여 데이터를 맞춰준다.

cut_size = INPUT_WINDOW - 1
train_X=train_X[cut_size:,]
train_y=train_y[cut_size:,]
test_X=test_X[cut_size:,]
test_y=test_y[cut_size:,]

# 하위 모델 생성
lstm_model, lstm_test_X, lstm_test_y =  lstm_fit()  # LSTM 모델을 구성하고 학습한 모델

# 하위 모델 예측
lstm_model.save('lstm_model.h5')

lstm_predictions = lstm_model.predict(lstm_test_X)
lstm_predictions = lstm_predictions.reshape(lstm_predictions.shape[0])
lstm_test_y = lstm_test_y.reshape(lstm_test_y.shape[0])


# lstm 예측 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(lstm_predictions, label='Predicted')
plt.plot(lstm_test_y, label='True')
plt.title('lstm 예측 결과')
plt.legend()
plt.show()

"""## **SVM**"""

# 하위 모델 생성
svm_model, test_y_binary = svm_fit()

# 모델 예측
svm_predictions = svm_model.predict(test_X[:-1])

lstm_accuracy = calculate_accuracy(lstm_test_y, lstm_predictions)
svm_accuracy = accuracy_score(test_y_binary[1:], svm_predictions)
svm_accuracy = svm_accuracy * 100

print(f"lstm 정확도: {lstm_accuracy:.6f}%")
print(f"svm 정확도: {svm_accuracy:.6f}%")

"""# **Hard Voting 진행**"""

# transformer_result_binary

def replace_to_binary(arr):
    result = np.where(arr[1:] > arr[:-1], 1, -1)
    return result

def svm_to_binary(arr):
    result = np.where(arr > 0, 1, -1)
    return result

# 트렌스포머 결과 텐서 배열을 일반 1차원 배열으로 변경
list_binary = [tensor.tolist() for tensor in transformer_result_binary]
one_dimensional_array = [element for row in list_binary for element in row]
transformer_predictions = one_dimensional_array

svm_result = svm_to_binary(svm_predictions)
lstm_predictions = replace_to_binary(lstm_predictions)

svm_predictions = svm_result[INPUT_WINDOW - 1 : ]

len(transformer_predictions)

len(svm_predictions)

len(lstm_predictions)

test_y_ = replace_to_binary(test_y)
test_y_ = test_y_[INPUT_WINDOW - 1:]

test_y_ = np.insert(test_y_, 0, 1)
lstm_predictions = np.insert(lstm_predictions, 0, 1)

def transform_data(data_list):
    transformed_data = []
    for value in data_list:
        if value == -1.0:
            transformed_data.append(-1)
        elif value == 1.0:
            transformed_data.append(1)
        else:
            # 다른 경우에 대한 처리를 원하면 여기에 추가
            transformed_data.append(value)
    return transformed_data

def majority_vote(predictions):
    # 세 개의 예측 값을 비교하여 다수결로 의사결정
    result = []
    print(len(predictions[0]))
    print(len(predictions[1]))
    print(len(predictions[2]))

    for i in range(len(predictions[0])):
        votes = [predictions[j][i] for j in range(len(predictions))]
        majority = max(set(votes), key=votes.count)
        result.append(majority)
    return result

# 다수결로 의사결정
transformer_predictions = transform_data(transformer_predictions)
numpy_trans = np.array(transformer_predictions, dtype=int)

combined_predictions = [ lstm_predictions, numpy_trans, svm_predictions]

final_predictions = majority_vote(combined_predictions)

"""# **✅결과(test)**"""

def calculate_accuracy_for_voting(predictions, true_labels):
    # 예측값과 실제값을 비교하여 정확도 계산
    correct_count = sum(1 for p, t in zip(predictions, true_labels) if p == t)
    total_count = len(true_labels)
    accuracy = correct_count / total_count
    return accuracy

accuracy = accuracy_score(test_y_[1:], final_predictions)
accuracy = accuracy * 100
print(f'앙상블 모델 정확도 : {accuracy:.6f}%')

print("lstm")
print(lstm_predictions[-10:])
print("TF")
print(numpy_trans[-10:])
print("svm")
print(svm_predictions[-10:])
print("다수결 예측 결과:", final_predictions[-10:])
print("real val")
print(test_y_[-30:])

"""# **다음 날 예측**

lstm_accuracy = calculate_accuracy(lstm_test_y, lstm_predictions)
"""

test_for_next_day = test_db[index][-7:]
test_for_next_day

# 내일의 값을 예측하는 코드
last_date = test_for_next_day.index[-1]
print("기준 날짜 : ")
print(last_date) #하루 뒤를 print함 (주말일 수 있음.)
test_for_next_day = test_db[index][-7:]

print("이전 7일 data")
print(test_for_next_day)

# lstm 결과
svm_pre_next = svm_model.predict(test_for_next_day[-1:])
svm_f_result = 1 if svm_pre_next == 1 else -1
print("svm_predictions")
print(svm_f_result)

last_row_data = test_for_next_day.iloc[-1].values
last_arr = lstm_test_X[-1:]

# 두 배열을 이어붙이기 (축 0 방향으로)
concatenated_array = np.append(last_arr, last_row_data)
c_array = concatenated_array[1:]
c_array = c_array.reshape(1, 7, 1)
lstm_pre_next = lstm_model.predict(c_array)

#가장 최근의 종가
cur_price = c_array[0][-1][0]
#예측값
lstm_result = lstm_pre_next[0][0]
#lstm 결과
lstm_f_result = 1 if cur_price < lstm_result else -1
print("lstm_predictions")
print(lstm_f_result)

test_for_next_day = test_db[index][-14:]
test_for_next_day

"""## **TF**"""

test_dataset = windowDataset(test_for_next_day, input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW, stride=1, n_features=N_FEATURES)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)     # 64 = 2^6, 512 = 2^9

# set evaluation mode
tf_model.eval()


# 기울기 계산을 방지하기 위해 torch.no_grad() 블록 안에서 평가
with torch.no_grad():
  for (inputs, outputs) in tqdm(test_loader, desc="Evaluating"):
    # Forward propagation with masking
    src_mask = tf_model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
    result = tf_model(inputs.float().to(device), src_mask)

    # 상승/하강 예측
    predicted_changes = torch.sign(result[:, -1] - inputs[:, -1, 0].to(device))             # 마지막 예측 값 - 마지막 입력 값
    true_changes = torch.sign(outputs[:, -1, 0].to(device) - inputs[:, -1, 0].to(device))  # 실제 마지막 값 - 마지막 입력 값
    transformer_result_binary.append(predicted_changes.cpu()) #gpu에서 cpu로 데이터를 옮겨줘야 함..

    print(predicted_changes)

"""# **베이스라인**"""

# 1) 하루 전의 결과와 같음

result = test_db[index][-1:]['종가'].values - test_db[index][-2:-1]['종가'].values
result

print('[', end = "")

for index in range(0,8):
  result = test_db[index][-1:]['종가'].values - test_db[index][-2:-1]['종가'].values
  if index != 7:
    print('상승' if result > 0 else '하락', end = ", ")
  else:
    print('상승' if result > 0 else '하락', end = "")

print(']', end = "")

# 2) 14일 동안의 등/락 voting

print('[', end = "")

for index in range(0,8):
  # 후
  latest_close = test_db[index].iloc[-14:]['종가']

  # 전
  past_14_days = test_db[index].iloc[-15:-1]['종가']

  # 등락 계산
  diff = latest_close.values - past_14_days.values

  # 양수 및 음수 등락 개수 세기
  positive_changes = len(diff[diff > 0])
  negative_changes = len(diff[diff < 0])

  # 결과 평가
  if positive_changes >= negative_changes:
      print('상승', end = ", ")
  else:
      print('하락', end = ", ")

print(']')

"""다수결 실험 결과

*   {tf, lstm, svm} = 앙상블 모델 정확도 : 58.823529%

*   {lstm, svm} = 앙상블 모델 정확도 : 79.044118%


*   {tf, lstm} = 앙상블 모델 정확도 : 73.774510%



*   {tf, svm} = 앙상블 모델 정확도 : 52.450980%

* 비고) 두 모델을 보팅할 경우, 두 모델의 예측값이 다를 때, (상승 + 하락) -> (상승)으로 예측 됨

11/02 : [하락, 상승, 상승, 상승, 상승, 상승, 상승, 상승]
"""



"""# **model export**"""

import torch
from torch.nn import Transformer
import torch

# 모델 파일 경로
PATH = "transformer_model.pth"

# 모델 불러오기
model = torch.load(PATH, map_location=torch.device("cpu"))  # 모델 파일을 불러옵니다.

# # 필요하면 GPU로 이동
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

model

PATH = "tf_model2.pth"

torch.save(tf_model.state_dict(), PATH)

from joblib import dump, load

"""svm 모델 저장"""

dump(svm_model, 'svm_model.joblib')

"""11/03 5일보팅 : [하락, 하락, 하락, 하락, 하락, 상승, 하락, 하락, ]

"""