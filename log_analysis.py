import os
import shutil
import json
import pandas as pd

log_data_list = []
if os.path.exists('temp'):
    shutil.rmtree('temp')
os.mkdir('temp')
os.system('kubectl get pods > temp/pods.txt')
pod_results = open('temp/pods.txt')
pod_records = pod_results.readlines()[1:]
pod_results.close()
for pod_record in pod_records:
    pod_name = pod_record.split()[0]
    os.system(f'kubectl logs {pod_name} -c istio-proxy > logs/log-{pod_name}.log')
    filepath = f"logs/log-{pod_name}.log"
    file = open(filepath)
    lines = file.readlines()
    file.close()

    for line in lines:
        line = line.strip()
        try:
            log_data = json.loads(line)
            log_data_list.append(log_data)
        except json.JSONDecodeError:
            continue
logs = pd.DataFrame(log_data_list)
logs.to_csv("logs.csv")
