#!/bin/bash

mkdir log
while true; do
    nohup python3 ./news_sentiment_analysis.py > log/news_sentiment_anaysis.log &
    PID=$!

    wait $PID

    echo "Restarting program now ..."
    
    sleep 2
done

