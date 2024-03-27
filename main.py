import threading
import os
import re
import logging
import redis

redis_url = os.environ.get("redis_url", 'redis://default:eb7199cbf0f54bf5bb084f7f1d594692@fly-bark-queries.upstash.io:6379')
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
gpu_ids = re.findall('.*UUID: MIG-(.*)\)', data)
# os.system('source /home/pythonuser/project-venv/bin/activate && CUDA_VISIBLE_DEVICES=MIG-3b45d462-0ecd-5323-8c62-8ea8c6a2941f python3 "import torch,time;torch.zeros(100).cuda();time.sleep(10)"')
threads = []
for gpu_id in gpu_ids:
    threads.append(threading.Thread(target=os.system, args=(f'CUDA_VISIBLE_DEVICES=MIG-{gpu_id} python3 worker.py',)))

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()
