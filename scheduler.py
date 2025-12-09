# scheduler.py
import schedule
import time
from src.predict import predict_for_ticker
from src.config import STOCKS

def job():
    print("[scheduler] running scheduled predictions...")
    for t in STOCKS:
        try:
            predict_for_ticker(t)
        except Exception as e:
            print("scheduler: error for", t, e)

if __name__ == "__main__":
    job()  # run immediately
    schedule.every(15).minutes.do(job)
    print("Scheduler started; running job every 15 minutes.")
    while True:
        schedule.run_pending()
        time.sleep(1)
