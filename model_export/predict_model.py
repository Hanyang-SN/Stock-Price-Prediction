

from joblib import load
from pykrx import stock
import datetime
from joblib import load
import numpy as np
import datetime
import pandas as pd
from keras.models import load_model
import torch
import sklearn

# 28일 전 날짜 계산
end_date = datetime.datetime.now() - datetime.timedelta(days=1)
start_date = end_date - datetime.timedelta(days=30)

# 종목 코드 설정
# <!-- /*## "KB금융 105560 신한지주 055550 하나금융지주 086790 메리츠금융지주 138040 기업은행 024110 미래에셋증권 006800 NH투자증권 005940 삼성증권 016360".split()*/ -->


import pandas as pd

# 주식 종목 코드와 이름 데이터
stock_data = {
    "종목코드": ["105560", "055550", "086790", "138040", "024110", "006800", "005940", "016360"],
    "종목명": ["KB금융", "신한지주", "하나금융지주", "메리츠금융지주", "기업은행", "미래에셋증권", "NH투자증권", "삼성증권"]
}

# DataFrame 생성
stock_df = pd.DataFrame(stock_data)

print(stock_df)

# tfmodel class..

import torch.nn as nn

class TFModel(nn.Module):

# iw/ow:      input window, output window
# d_model:    인풋 개수
# nlayers:    인코더 부분의 인코더 개수
# nhead:      multihead attention 개수

    def __init__(self, iw: int, ow: int, d_model: int, nhead: int, nlayers: int, dropout=0.5, n_features=1):
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

import math

# 참고) array([[-0.87532395],
      #  [-0.94049   ],

# 예측 결과를 담을 데이터프레임 생성
predictions_df = pd.DataFrame(columns=['종목명'])

# 각 종목별로 모델 예측 실행
for index, row in stock_df.iterrows():
    stock_name = row['종목명']
    # print(stock_name)

    # 주식 데이터 불러오기
    df = stock.get_market_ohlcv_by_date(start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"), stock_code)

    # 종가만을 추출하여 배열에 저장
    closing_prices_ = df['종가'].values
    closing_prices = closing_prices_[len(closing_prices_) - 14 : ]
    # print(len(closing_prices))
    reshaped_data = closing_prices.reshape(-1, 1)
    # print(reshaped_data)

    # 여기서 X_test를 어떻게 구성할지에 따라 모델 예측을 진행
    X_test = closing_prices

    # SVM 모델 불러오기 및 예측
    # svm의 경우 상승(1) 하락(0)으로 예측됨.
    svm_test = [[X_test[-1]]]
    loaded_svm_model = load('svm_model.joblib')
    svm_predictions = loaded_svm_model.predict(svm_test)
    # print("svm")
    if(svm_predictions==0) :
      svm_predictions = -1
    # print(svm_predictions)


    # LSTM 모델 불러오기 및 예측
    loaded_lstm_model = load_model('lstm_model.h5')
    lstm_predictions = loaded_lstm_model.predict(reshaped_data[-14:])
    # print("lstm")
    lstm_result = int(lstm_predictions[-1] - lstm_predictions[-2])

    if(lstm_result==0) :
      lstm_result = -1

    # print(lstm_result)

    # Transformer 모델 불러오기
    PATH = "tf_model2.pth"

    # # tf은 model_dict를 export해야함

    # 모델 초기화하고 argu ('iw', 'ow', 'd_model', 'nhead', and 'nlayers')
    INPUT_WINDOW = 7 # 내가 모델 저장을 7,7로 했음 ㅠㅠ
    OUTPUT_WINDOW = 7

    # 모델 초기화 및 가중치 불러오기
    model = TFModel(iw=INPUT_WINDOW, ow=OUTPUT_WINDOW, d_model=512, nhead=8, nlayers=4)
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model.eval()  # 모델을 평가(evaluation) 모드로 설정합니다.

    # 예측에 사용할 데이터 준비
    # 예측에 사용할 데이터를 PyTorch Tensor로 변환하여 준비합니다.
    # 예시: input_data = torch.tensor([[your_input_data]])

    inputs = torch.tensor(reshaped_data[:-7])
    print(inputs)

    # 모델을 사용하여 예측 수행
    with torch.no_grad():
        src_mask = model.generate_square_subsequent_mask(inputs.shape[0])
        result = model(inputs.float(), src_mask)
        predictions = model(inputs.float(), src_mask)

    # 결과 확인
    tf_predictions = predictions[0][-1] - predictions[0][-2]

    if(tf_predictions > 0 ) : tf_result = 1
    else :tf_result = 0

    # print(tf_result)

    # 각 모델의 예측 결과를 데이터프레임에 추가
    predictions_df = predictions_df.append({
        '종목명': stock_name,
        'SVM': svm_predictions,
        'LSTM': lstm_result,
        'Transformer': tf_result,
    }, ignore_index=True)

# 결과 데이터프레임 출력
print(predictions_df)


# /*        종목명  SVM  LSTM  Transformer
# 0     KB금융 -1.0  -1.0          1.0
# 1     신한지주 -1.0  -1.0          1.0
# 2   하나금융지주 -1.0  -1.0          1.0
# 3  메리츠금융지주 -1.0  -1.0          1.0
# 4     기업은행 -1.0  -1.0          1.0
# 5   미래에셋증권 -1.0  -1.0          1.0
# 6   NH투자증권 -1.0  -1.0          1.0
# 7     삼성증권 -1.0  -1.0          1.0
