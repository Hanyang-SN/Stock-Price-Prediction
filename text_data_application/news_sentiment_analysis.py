'''

1. 50개 기사 헤드라인 크롤링
2. 번역
3. 점수 계상
4. 취합.
'''


# for crawl_headline()
import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
import pandas as pd

# for translate_headline()
from googletrans import Translator

# for get_sentiment_score()
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# for logging
LOGGING = True	# 로그 포맷: f"level:{} function:{} content:{}  "
# debug(on demand)-> info -> warn -> error -> fatal

# for saving
import os
SCORE_DIR = "./news_sentiment_score"
START_DATE, END_DATE = date(2013, 1, 1), date(2023, 1, 1)			# 10년 데이터 수집

# 모델을 call하면 이전 정보에 이어서 계속적으로 실행
class NewsSentimentAnalysis:
	
	def __init__(self, ticker_name, news_keyword_list=[], news_num=30):
		self.ticker_name = ticker_name
		self.news_num = news_num
		self.news_keyword_list = news_keyword_list if news_keyword_list else [ticker_name]		# news_keyword_list가 전달되지 않으면 ticker_name 만 검색함.
		# self.news_date: date = None
		
		self.translator = Translator()
		self.sia = SentimentIntensityAnalyzer()
		self.file_name = f"{SCORE_DIR}/{ticker_name}.csv"

		if not os.path.isfile(self.file_name):
			self.sentiment_df = pd.DataFrame(columns=["date", self.ticker_name])
			self.sentiment_df.set_index('date', inplace=True)
			self.news_date = START_DATE
		else:
			self.sentiment_df = pd.read_csv(self.file_name)
			self.sentiment_df.set_index('date', inplace=True)
			self.news_date = date.fromisoformat(self.sentiment_df.index[-1])

		self.__print_log(level="INFO", function=f"__init__",\
				   content=f"객체 생성됨.\n\
				   ticker name: {ticker_name}\n\
				   news_date(시작 일자): {self.news_date}")

	def __crawl_headline(self) -> pd.DataFrame:

		def __wrapped():
			# def crawler(search_keyword: str, search_date: date, num_of_articles):
			search_date = self.news_date

			headline_list = []
			for i in range(self.news_num // 10):
				for search_keyword in self.news_keyword_list:
					url = f"https://search.naver.com/search.naver?where=news&query={search_keyword}&sm=tab_opt&sort=0&photo=0&field=0&pd=3&ds={search_date.year}.{search_date.month:02}.{search_date.day:02}&de={search_date.year}.{search_date.month:02}.{search_date.day:02}&docid=&related=0&mynews=0&office_type=0&office_section_code=&news_office_checked=&nso=so%3Ar%2Cp%3Afrom{search_date.year}{search_date.month:02}{search_date.day:02}to{search_date.year}{search_date.month:02}{search_date.day:02}&is_sug_officeid=0&office_category=0&service_area=0&start={i * 10 + 1}"

					# naver 서버에서 html을 줌
					resp = requests.get(url)
					if resp.status_code != 200:
						self.__print_log_(level="ERROR", function="__crawl_headline", content=f"request to NAVER failure")
						return

					# html paser가 soup를 만듬 
					soup = BeautifulSoup(resp.text, "html.parser")
					links = soup.select(".news_tit")


					for link in links:
						headline = link.text				# 태그 안의 텍스트 요소를 가져온다.
						headline_list.append(headline)
			
			result = pd.DataFrame({'title-ko' : headline_list})
			self.__print_log(level="INFO", function="__crawl_headline", content=f"크롤링 완료\n결과는 아래와 같다.\n{result}")	
			return result

		result = None
		while result is None:
			result = __wrapped()
		return result
					
	def __translate_headline(self, news_headline: pd.DataFrame):
		try:
			# 데이터 프레임에 열 추가
			news_headline["title-en"] = news_headline["title-ko"].apply(lambda x: self.translator.translate(str(x), dest='en', src='ko').text)
		except TypeError as e:
			print("----------------------------------")
			print("Exception Occured:", e)
			print("----------------------------------")
		
		self.__print_log(level="INFO", function="__translate_headline", content=f"번역 결과\n{news_headline}")
	
	def __get_sentiment_score(self, news_headline):
		# 데이터 프레임에 열 추가
		news_headline["score"] = news_headline["title-en"].apply(lambda x: self.sia.polarity_scores(x)['compound'])
		self.__print_log(level="INFO", function="__get_sentiment_score", content=f"감성 분석 결과\n{news_headline}")
	
	def __get_integrated_score(self, news_headline) -> int:
		
		integerated_score = sum(news_headline["score"]) / self.news_num
		self.__print_log(level="INFO", function="__get_integrated_score", content=f"감성 점수 취합\n{integerated_score}")
		return integerated_score
	
	def __analysis_one_day(self) -> int:
		news_headline = self.__crawl_headline()	
		self.__translate_headline(news_headline)        # 오래 걸린다. 참고 ㅎ
		self.__get_sentiment_score(news_headline)
		return self.__get_integrated_score(news_headline)
		
	def __backup_as_file(self):
	 	# Dir이 로컬에 없으면 생성		
		if not os.path.isdir(SCORE_DIR):
			os.mkdir(SCORE_DIR)
			self.__print_log(level="INFO", function="__backup_as_file", content=f"폴더 생성: {SCORE_DIR}")
		
		# File이 로컬에 있으면 삭제하고 새로 저장.
		if os.path.isfile(self.file_name):
			os.remove(self.file_name)
			self.__print_log(level="INFO", function="__backup_as_file", content=f"파일 삭제: {self.file_name}")

		self.sentiment_df.to_csv(self.file_name)
		self.__print_log(level="INFO", function="__backup_as_file", content=f"파일  생성: {self.file_name}\n파일 내용: {self.sentiment_df}")

	def __print_log(self, level="", function="", content=""):
		if not LOGGING:
			return
		print(f"============level:{level}============\
				\nfunc:\t\t{function}\
				\ncontent:\t{content}")
		print()


	def __call__(self):

		# self.news_date = date(2013, 1, 1)
		while self.news_date < END_DATE:
			self.sentiment_df.loc[self.news_date] = [self.__analysis_one_day()]
			# self.sentiment_df.loc[self.news_date] = [0]
			self.__print_log(level="INFO", function="__call",\
					content=f"{self.news_date}일자 감성 분석 완료\n감성 분석 결과:\n{self.sentiment_df.tail()}")

			if self.news_date.day == 1:
				self.__backup_as_file()
			self.news_date += timedelta(days=1)

		self.__backup_as_file()


news_module = NewsSentimentAnalysis('KB금융')
news_module()