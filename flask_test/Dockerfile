FROM python:3.7.16

# 디렉토리 생성
WORKDIR /flask_test

# 현재 디렉토리(.)를 container의 /flask_test에 복사
COPY . /flask_test

# 필요한 패키지 설치
RUN pip install -r ./requirements.txt

# 컨테이나 시작 시 실행될 명령어 지정
CMD ["python", "./flask_test.py"]

