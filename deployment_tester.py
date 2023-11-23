import uuid
import time
import json
import threading

import requests

ip = "34.29.199.202"
def synthesize(text, index):
    url = f'http://{ip}:80/CA123/synthesize'
    headers = {'accept': 'application/octet-stream', 'Content-Type': 'application/json'}
    data = {
        "text": text,
        "voice": "final_Either_way_weve-23_09_04__17-51-24.mp4"
    }
    # cookies = {
    #     'uid': cookie
    # }
    s = time.time()
    response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
    if response.status_code == 200:
        s = time.time()
        start = True
        for chunk in response.iter_content():
            if start:
                print("Received:", index, time.time() - s)
                start = False
        print("Finished:", index, time.time() - s)


if __name__ == '__main__':
    args = [
        ("Okay, no problem. Let me walk you through exactly what we do. I can see how you're positioned.", "CA123"),
        ("Meanwhile, your operational costs in terms of customer service could go down because the AI will handle "
         "a good chunk of initial queries. How does that sound from a growth perspective?", "CA123"),
        ("Okay, beautiful. Did it answer most of your questions, or did you have a few lingering questions that maybe "
         "you or your wife wanted to ask?", "CA124"),
    ]
    threads = []
    for i, (text, cookie) in enumerate(args):
        threads.append(threading.Thread(target=synthesize, args=(text, i)))

    for thread in threads:
        thread.start()
        time.sleep(0.1)

    for thread in threads:
        thread.join()
