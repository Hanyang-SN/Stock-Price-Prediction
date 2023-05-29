from datetime import date

# TODO: 추후에 타겟 종목 설정하여 분석할 것
KEYWORD = "삼성전자"

START_DATE, END_DATE = date(2018, 1, 1), date(2023, 1, 1)
DIFF_DAYS = END_DATE - START_DATE


DATA_DIR = "./text_data_application/data"
SCORE_DIR = "./text_data_application/score"