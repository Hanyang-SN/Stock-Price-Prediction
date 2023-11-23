import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import psql_connector


finance_code_dict = dict()
finance_code_list = "KB금융	105560 신한지주	055550 하나금융지주	086790 메리츠금융지주	138040 기업은행	024110 미래에셋증권	006800 NH투자증권	005940 삼성증권	016360".split()
for i in range(8):
  finance_code_dict[finance_code_list[2*i]] = finance_code_list[2*i + 1]

def get_100_close(ticker_name): #train, test데이터 따로 df으로 -> 통합하여 리턴하도록 변경함
  ticker_code = finance_code_dicåt[ticker_name]
  selected_columns = ['종가']  # 포함하려는 열 이름 리스트
  df = stock.get_market_ohlcv(datetime.today() - timedelta(days=101), datetime.today() - timedelta(days=1), ticker_code)
  df = df.astype('float32')
  return pd.DataFrame(df[selected_columns])


# === Flask ===
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
	# html 스트링
	return "<h1> html test </h1> \
			<p> content </p>"

@app.route("/data")
def data():
	df = pd.DataFrame({"A":[1,4,7], "B":[2,5,8], "C":[3,6,9]})

	# json 형식의 데이터
	return df.to_json()


@app.route("/data/kb")
def kb():
	return get_100_close('KB금융').to_json()


@app.route("/data/shinhan")
def shinhan():
	return get_100_close('신한지주').to_json()


@app.route("/data/hana")
def hana():
	return get_100_close('하나금융지주').to_json()


@app.route("/data/meritz")
def meritz():
	return get_100_close('메리츠금융지주').to_json()


@app.route("/data/ibk")
def ibk():
	return get_100_close('기업은행').to_json()



@app.route("/data/miraesec")
def miraesec():
	return get_100_close('미래에셋증권').to_json()


@app.route("/data/nhsec")
def nhsec():
	return get_100_close('NH투자증권').to_json()


@app.route("/data/samsungsec")
def samsungsec():
	return get_100_close('삼성증권').to_json()



# === connector ===
db_connector = psql_connector.CRUD()


# === Flask ===
app = Flask(__name__)
CORS(app)


# === put ===
@app.route('/result/<table_name>', methods=['POST'])
def put_result(table_name):
	# 요청의 JSON 본문에서 데이터 추출
	json = request.json
	# 쿼리로 변환
	data = f"'{json['date']}', {json['LSTM']}, {json['SVM']}, {json['TF']}"

	if not db_connector.read_db(table=table_name, colum="*"):
		db_connector.insert_db(table=table_name, colum=",".join(json.keys()), data=data)
		
	return f"message from put_result", 200


@app.route('/result/<table_name>', methods=['GET'])
def get_result(table_name):   
	data = db_connector.read_db(table=table_name, colum="*")
	# print(data)
	# print(type(data), len(data))

	if not data:
		return {}
	else:
		json = {"date" : str(data[0][0]), "lstm": data[0][1], "svm" : data[0][2], "tf": data[0][3]}
		return json


if __name__ == "__main__":
	app.run(host='0.0.0.0', port=1235, debug=True)
