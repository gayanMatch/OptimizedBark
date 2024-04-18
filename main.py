import logging
import os
import re
import threading

import redis

redis_url = os.environ.get(
    "redis_url",
    'redis://default:eb7199cbf0f54bf5bb084f7f1d594692@fly-bark-queries.upstash.io:6379'
)
# Establish connections to Redis for both publishing results and subscribing to incoming tasks
r = redis.Redis.from_url(redis_url)
FLY_MACHINE_ID = os.environ.get("FLY_MACHINE_ID", '1111111111111111111')
r.set(f'migs_{FLY_MACHINE_ID}', 0)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
os.system("nvidia-smi -i 0")
os.system("nvidia-smi -i 0 -mig 1")
logging.info("###############################################################################################")
os.system("nvidia-smi mig -cgi 14,14,14 -C")
os.system("nvidia-smi -L > gpus.txt")
os.system("nvidia-smi -i 0")
data = open('gpus.txt').read()
logging.info("###############################################################################################")
logging.info(f"GPUs:\n{str(data)}")
logging.info("###############################################################################################")
gpu_ids = re.findall(r'.*UUID: MIG-(.*)\)', data)
threads = []
for gpu_id in gpu_ids:
    threads.append(threading.Thread(target=os.system, args=(f'CUDA_VISIBLE_DEVICES=MIG-{gpu_id} python3 worker.py',)))

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()
