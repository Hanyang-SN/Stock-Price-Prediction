@echo off
:run_program
python news_sentiment_analysis.py
echo Program exited. Restarting...
timeout /t 2 /nobreak
goto run_program
