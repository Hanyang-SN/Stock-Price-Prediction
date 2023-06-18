# train data    : 2018-01-01 ~ 2022-12-31 주가 데이터를 다운로드
# test data     : 2023-01-01 ~ 2023-05-31 주가 데이터를 다운로드

import os
from pykrx import stock

STOCK_DATA_DIR = "./prediction/stock_data"
TRAIN_DIR = f"{STOCK_DATA_DIR}/train"
TEST_DIR = f"{STOCK_DATA_DIR}/test"


def make_train_data_file(file_name, ticker):
    
    train_file_name = f"{TRAIN_DIR}/{file_name}"
    
    # stock 데이터 파일 유무 확인
    if os.path.isfile(train_file_name):
        print("File already exists. File name : ", train_file_name)
        return
    
    # 데이터 수집
    df = stock.get_market_ohlcv("20180101", "20230101", ticker)

    df = df[["시가", "종가", "등락률"]]
    df.to_csv(train_file_name)
    print("New file is made. File name : ", train_file_name)
    
    
def make_test_data_file(file_name, ticker):
    
    test_file_name = f"{TEST_DIR}/{file_name}"
    
    #  stock 데이터 파일 유무 확인
    if os.path.isfile(test_file_name):
        print("File already exists. File name : ", test_file_name)
        return

    # 데이터 수집
    df = stock.get_market_ohlcv("20230101", "20230601", ticker)

    df = df[["시가", "종가", "등락률"]]
    df.to_csv(test_file_name)
    print("New file is made. File name : ", test_file_name)


def get_stock_price_data(ticker_str : str):
    
    # stock 데이터 디렉토리 생성
    if not os.path.isdir(STOCK_DATA_DIR):
        os.mkdir(STOCK_DATA_DIR)
    
    if not os.path.isdir(TRAIN_DIR):
        os.mkdir(TRAIN_DIR)
        
    if not os.path.isdir(TEST_DIR):
        os.mkdir(TEST_DIR)
    
    file_name = f"{ticker_str}.csv"
    
    make_train_data_file(file_name, ticker_str)
    make_test_data_file(file_name, ticker_str)


tmp_ticker = '105560'
get_stock_price_data(tmp_ticker)
