import os
import logging
import time
import json
import threading
import sys
import requests

REQUEST_COUNT = 1
DELAY = 0.01
SERVER = "istio"
SERVER = "mine"
ip = "35.233.179.102:80" if SERVER == "istio" else "34.82.114.109:5000"
# ip = "35.233.179.102:80" if SERVER == "istio" else "10.108.0.195:5000"
log_file = "logs.log" if SERVER == "istio" else "logs_mine.log"
logging.basicConfig(filename=log_file, filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.info(f"Started Application test using DELAY of {DELAY} and {REQUEST_COUNT} of requests")
def synthesize(text, callid, index):
    url = f'http://{ip}/{callid}/synthesize'
    headers = {
        'accept': 'application/octet-stream',
        'Content-Type': 'application/json',
        'CallID': callid
    }
    data = {
        "text": text,
        "voice": "final_Either_way_weve-23_09_04__17-51-24.mp4",
        "rate": 1.1
    }
    s = time.time()
    done = False
    while not done:
        logging.info(f"Sent Request {index}")
        response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
        if response.status_code == 200:
            start = False
            file_start = True
            string = ""
            file = open(f'result/{index}.wav', 'wb')
            header = open(f"bark/assets/header.raw", 'rb').read()
            file.write(header)
            for chunk in response.iter_content(chunk_size=None):
                if start:
                    logging.info(f"Received: {index} {callid} {time.time() - s}")
                    start = False
                    string += chunk.decode('ascii')
                    logging.info(string)
                else:
                    if file_start:
                        logging.info(f"First Chunk Received {index} {callid} {time.time() - s}")
                        file_start = False
                    file.write(chunk)
            file.close()
            logging.info(f"Finished: {index} {time.time() - s} {string}")
            done = True
        elif response.status_code == 400:
            string = ""
            for chunk in response.iter_content(chunk_size=1024):
                string += chunk.decode('ascii')
                logging.info(string)
            logging.info(f"Failed {index}, Retrying {string}")
        else:
            logging.info(f"Error returned {response.status_code}")
            done = True


if __name__ == '__main__':
    args = [
        ("Meanwhile, your operational costs in terms of customer service could go down because the AI will handle "
         "a good chunk of initial queries. How does that sound from a growth perspective?", "CA0"),
    ]
    for i in range(REQUEST_COUNT - 1):
        args.append(("Okay, beautiful. Did it answer most of your questions, or did you have a few lingering questions that maybe "
         "you or your wife wanted to ask?", f"CA{i + 1}"))
    threads = []
    for i, (text, callid) in enumerate(args):
        threads.append(threading.Thread(target=synthesize, args=(text, callid, i)))

    for thread in threads:
        thread.start()
        # os.system("kubectl get pods")
        time.sleep(DELAY)
    # os.system("kubectl get pods")
    for thread in threads:
        thread.join()
