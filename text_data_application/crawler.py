import requests
from bs4 import BeautifulSoup
from datetime import date,timedelta
import pandas as pd

def crawler(search_keyword: str, search_date: date, num_of_articles):
	search_date = search_date.strftime("%Y.%m.%d")
    
	title_list = []
	for i in range(num_of_articles // 10):
		url = f"https://search.naver.com/search.naver?where=news&sm=tab_pge&query={search_keyword}&sort=0&photo=0&field=0&pd=3&ds={search_date}&de={search_date}&cluster_rank=166&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:r,p:from20180101to20180101,a:all&start={10 * i + 1}"
		
		# naver 서버에서 html을 줌
		resp = requests.get(url)
		if resp.status_code != 200:
			print("status_code: ", resp.status_code)
			exit()
		
		# html paser가 soup를 만듬 
		soup = BeautifulSoup(resp.text, "html.parser")
		
		links = soup.select(".news_tit")
		for link in links:
			title = link.text	# 태그 안의 텍스트 요소를 가져온다.
			title_list.append(title)

	# df = pd.DataFrame({"title" : title_list})
	# df.to_csv(f"./text_data_application/data/{search_keyword}-{search_date}-{num_of_articles}")
 
	return pd.DataFrame({"title-ko" : title_list})