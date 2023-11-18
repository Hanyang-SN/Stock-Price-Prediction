from joblib import load
import numpy as np

# svm 모델 불러오기
loaded_svm_model = load('svm_model.joblib')

# 예측에 사용할 데이터 준비

X_test = np.array([[1],[1]])

# 모델 예측
predictions = loaded_svm_model.predict(X_test)

from keras.models import load_model

# lstm 모델 불러오기
loaded_lstm_model = load_model('lstm_model.h5')

# 예측에 사용할 데이터 준비
# 예를 들어, X_test는 예측에 사용할 테스트 데이터일 수 있습니다.

# 모델 예측
predictions = loaded_lstm_model.predict(X_test)

import torch
from torch.nn import Transformer
import torch

# tf 모델 파일 경로
PATH = "transformer_model.pth"

# 모델 불러오기
tf_model = torch.load(PATH, map_location=torch.device("cpu"))  # 모델 파일을 불러옵니다.

