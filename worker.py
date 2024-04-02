import redis
import json
import os
import base64
from datetime import datetime, timezone
import logging
# from pymongo import MongoClient
import time  # Used for simulating long-running or streaming predictions
from bark.SynthesizeThread import SynthesizeThread
from bucket_utils import download_voice

DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "hey_james_reliable_1_small_coarse_fix")
FLY_MACHINE_ID = os.environ.get("FLY_MACHINE_ID", '1111111111111111111')
synthesize_thread = SynthesizeThread(DEFAULT_VOICE)
synthesize_thread.start()

redis_url = os.environ.get(
    "redis_url",
    'redis://default:eb7199cbf0f54bf5bb084f7f1d594692@fly-bark-queries.upstash.io:6379'
)
# mongo_uri = os.environ.get(
#     "mongo_uri",
#     "mongodb+srv://ginger:P%40ssw0rd131181@bark-log.1fit2mh.mongodb.net/?retryWrites=true&w=majority&appName=bark-log"
# )
# Establish connections to Redis for both publishing results and subscribing to incoming tasks
r = redis.Redis.from_url(redis_url)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
# client = MongoClient(mongo_uri)
# collection = client["bark"]["queries"]
r.setnx('stop_marked_gpu', '')
# r = redis.Redis(
#   host='localhost',  # Changed to localhost
#   port=6379,
#   password=''  # Likely no password if you're just testing locally
# )


def handle_predictions():
    while synthesize_thread.is_busy():
        time.sleep(1)
    r.incr(f'migs_{FLY_MACHINE_ID}')
    while True:
        # Block until a message is received; 'ml_requests' is the list with prediction tasks
        if r.get('stop_marked_gpu').decode('utf-8') == FLY_MACHINE_ID:
            break
        res = r.brpop("ml_requests", 1)
        if res is None:
            continue

        _, request_data = res
        r.decr(f'migs_{FLY_MACHINE_ID}')
        # Decode and load the request data (contains 'request_id' and 'features')
        request = json.loads(request_data)
        request_id = request["request_id"]
        text = request["text"]
        voice = request["voice"]
        rate = request["rate"]
        request_time = request["request_time"]
        process_start_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
        s = time.time()
        if voice + ".npz" not in os.listdir("bark/assets/prompts"):
            download_voice('tts-voices-npz', voice, 'bark/assets/prompts')
        # This is a simplistic approach; consider batching or other optimizations for your actual use case
        stream = synthesize_thread.add_request(text, voice, rate)

        # Simulate streaming data; in a real scenario, this loop could be replaced with actual streaming logic
        first_byte_time = -1
        for result in stream:
            # Publish the intermediate result to the channel named after 'request_id'
            encoded_result = base64.b64encode(result).decode('utf-8')
            if first_byte_time == -1:
                first_byte_time = time.time() - s
            r.publish(request_id, json.dumps({"prediction": encoded_result}))
        finish_time = time.time() - s
        # Signal completion of the streaming predictions
        r.publish(request_id, json.dumps({"complete": True}))

        # logging
        logging.info(
            f"Request Time: {request_time}, Process Start Time: {process_start_time}, Text: {text}, Voice: {voice}, "
            f"Rate: {rate}, First Byte Generation Time: {first_byte_time}, Process Finish Time: {finish_time}"
        )
        # collection.insert_one(
        #     {
        #         "request_time": request_time,
        #         "process_start_time": process_start_time,
        #         "text": text,
        #         "voice": voice,
        #         "rate": rate,
        #         'first_byte_time': first_byte_time,
        #         'finish_time': finish_time
        #     }
        # )
        r.incr(f'migs_{FLY_MACHINE_ID}')
        r.decr('active_requests')


if __name__ == "__main__":
    print("Starting consumer...")
    handle_predictions()
