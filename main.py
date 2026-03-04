import datetime
import time
import dotenv
import os
from collect import CollectResults
from sender import Sender

dotenv.load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME")

while True:
    print("Iniciando processo...")

    print("Coletando dados...")
    collect_data = CollectResults(years=[datetime.datetime.now().year])
    collect_data.process_years()

    print("Enviado dados...")
    sender_data = Sender(bucket_name=BUCKET_NAME, bucket_folder="f1/results/")
    sender_data.process_folder("data/")

    print("Iteração finalizada.")
    time.sleep(60 * 60 * 6)
