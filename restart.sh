#!/bin/bash


while true; do
    python3 ./text_data_application/news_sentiment_analysis.py
    echo "Program exited with status $?."
    echo "REstarting program now ..."
    sleep 2
done

