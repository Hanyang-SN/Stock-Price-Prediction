FROM python:3.7.16

# 디렉토리 만들기 및 워킹 디렉토리 설정
WORKDIR  /nsa2

COPY . /nsa2

RUN pip install -r ./requirements.txt

CMD ["python", "news_sentiment_analysis.py"]

