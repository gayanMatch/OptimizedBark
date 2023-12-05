import os
import uuid
import time
import json
import threading

import requests

ip = "34.168.167.28"
def synthesize(text, callid, index):
    url = f'http://{ip}:80/CA123/synthesize'
    headers = {
        'accept': 'application/octet-stream',
        'Content-Type': 'application/json',
        # 'CallID': callid
    }
    data = {
        "text": text,
        "voice": "final_Either_way_weve-23_09_04__17-51-24.mp4",
        "rate": 1.1
    }
    # cookies = {
    #     'uid': cookie
    # }
    s = time.time()
    done = False
    while not done:
        response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
        if response.status_code == 200:
            start = True
            string = ""
            # header_file = open(f'bark/assets/header.raw', 'rb')
            # header = header_file.read()
            # header_file.close()
            # file = open(f'result/{index}.wav', 'wb')
            # file.write(header)
            for chunk in response.iter_content():
                string += chunk.decode('ascii')
                # file.write(chunk)
                if start:
                    print("Received:", index, callid, time.time() - s)
                    start = False
            # file.close()
            print("Finished:", index, time.time() - s, string)
            done = True
        elif response.status_code == 400:
            print(f"Failed {index}, Retrying")
        else:
            print("Error")
            done = True


if __name__ == '__main__':
    args = [
        ("Meanwhile, your operational costs in terms of customer service could go down because the AI will handle "
         "a good chunk of initial queries. How does that sound from a growth perspective?", "CA0"),
    ]
    for i in range(2):
        args.append(("Okay, beautiful. Did it answer most of your questions, or did you have a few lingering questions that maybe "
         "you or your wife wanted to ask?", f"CA{i + 100}"))
    threads = []
    for i, (text, callid) in enumerate(args):
        threads.append(threading.Thread(target=synthesize, args=(text, callid, i)))

    for thread in threads:
        thread.start()
        # os.system("kubectl get pods")
        # time.sleep(1)
    # os.system("kubectl get pods")
    for thread in threads:
        thread.join()
