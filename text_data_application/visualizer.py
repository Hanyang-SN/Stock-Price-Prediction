from datetime import date

import pandas as pd
import matplotlib.pyplot as plt
import os

from g_variable import *

last_date = date(2018, 3, 19)

def visualize_score():
	# CSV 파일을 읽어 데이터프레임으로 변환합니다.
	dataframe = pd.read_csv(f"{SCORE_DIR}/{KEYWORD}")

	# 데이터프레임의 구조와 일부 데이터를 확인합니다.
	print(dataframe.head())

	# 시각화를 위한 데이터를 선택합니다.
	x_column = 'date'
	y_column = 'score'
	
	
	# 선 그래프를 그립니다.
	plt.plot(dataframe[x_column], dataframe[y_column])
	plt.xlabel(x_column)
	plt.ylabel(y_column)
	plt.title('[score - date]')
	plt.show()
 
 
def visualize_price():
	file_name =f"{SCORE_DIR}/stock-price-{KEYWORD}.csv"
	if not os.path.isfile(file_name):
		print("There is no file :", file_name)
		return

     # CSV 파일을 읽어 데이터프레임으로 변환합니다.
	dataframe = pd.read_csv(file_name, encoding='iso-8859-1')

	# 데이터프레임의 구조와 일부 데이터를 확인합니다.
	print(dataframe.head())

	# 시각화를 위한 데이터를 선택합니다.
	x_column = 0
	df_y_column = []
	
	for i in range(len(dataframe)):
		df_y_column.append(dataframe.iloc[i, 3] * dataframe.iloc[i, 4])
	
	# 선 그래프를 그립니다.
	plt.plot(dataframe.iloc[:, x_column], df_y_column)
	plt.xlabel("date")
	plt.ylabel("delta")
	plt.title('[delta price - date]')
	plt.show()
 

visualize_score()
visualize_price()