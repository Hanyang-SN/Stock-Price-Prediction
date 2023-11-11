#!/bin/bash


# terminal
docker pull postgres

docker run -p 5432:5432 --name test-postgres \ 
-e POSTGRES_PASSWORD=123456 \
-e TZ=Asia/Seoul\
-d postgres:latest

docker exec -it test-postgres bash


# root 어쩌고 ~~ 컨테이너 배쉬로 전환됨
psql -U postgres


# postgres=# 으로 postgreSQL로 접속됨
CREATE DATABASE testdb;                 # `;` 없으면 실행 안 됨 주의
\l                                      # Retrieve databases
