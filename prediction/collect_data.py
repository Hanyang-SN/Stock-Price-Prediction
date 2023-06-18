import os
from pykrx import stock

STOCK_DATA_DIR = "./stock_data"

def get_stock_price_data(ticker_str : str):
    
    # stock 데이터 디렉토리 생성
    if not os.path.isdir(STOCK_DATA_DIR):
        os.mkdir(STOCK_DATA_DIR)

    file_name = f"{STOCK_DATA_DIR}/{ticker_str}-{stock.get_market_ticker_name(ticker_str)}.csv"
        
    # stock 데이터 파일 유무 확인
    if os.path.isfile(file_name):
        print("File already exists.")
        return
    
    # 데이터 수집
    df = stock.get_market_ohlcv("20180101", "20230101", ticker_str)

    df = df[["시가", "종가", "등락률"]]
    df.to_csv(file_name)
    print("New file is made. File name : ", file_name)

tmp_ticker = '105560'
get_stock_price_data(tmp_ticker)
