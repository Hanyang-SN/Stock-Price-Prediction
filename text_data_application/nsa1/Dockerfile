FROM python:3.7.16

# 디렉토리 만들기 및 워킹 디렉토리 설정
RUN mkdir -p /news_sentiment_analysis

WORKDIR  /news_sentiment_analysis

COPY . .


RUN pip install -r ./requirements.txt

CMD ["python", "news_sentiment_analysis2.py"]

