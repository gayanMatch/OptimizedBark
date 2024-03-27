import redis
import json
import os
import base64
import time  # Used for simulating long-running or streaming predictions
from bark.SynthesizeThread import SynthesizeThread
from bucket_utils import download_voice

DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "hey_james_reliable_1_small_coarse_fix")
FLY_MACHINE_ID = os.environ.get("FLY_MACHINE_ID", '1111111111111111111')
synthesize_thread = SynthesizeThread(DEFAULT_VOICE)
synthesize_thread.start()

redis_url = 'redis://default:eb7199cbf0f54bf5bb084f7f1d594692@fly-bark-queries.upstash.io:6379'
# Establish connections to Redis for both publishing results and subscribing to incoming tasks
r = redis.Redis.from_url(redis_url)
r.setnx('stop_marked_gpu', '')
# r = redis.Redis(
#   host='localhost',  # Changed to localhost
#   port=6379,
#   password=''  # Likely no password if you're just testing locally
# )


def handle_predictions():
    while True:
        # Block until a message is received; 'ml_requests' is the list with prediction tasks
        if r.get('stop_marked_gpu').decode('utf-8') == FLY_MACHINE_ID:
            break
        res = r.brpop("ml_requests", 1)
        if res is None:
            continue

        _, request_data = res
        # Decode and load the request data (contains 'request_id' and 'features')
        request = json.loads(request_data)
        request_id = request["request_id"]
        text = request["text"]
        voice = request["voice"]
        rate = request["rate"]
        request_time = request["request_time"]

        if voice + ".npz" not in os.listdir("bark/assets/prompts"):
            download_voice('tts-voices-npz', voice, 'bark/assets/prompts')
        # This is a simplistic approach; consider batching or other optimizations for your actual use case
        stream = synthesize_thread.add_request(text, voice, rate)

        # Simulate streaming data; in a real scenario, this loop could be replaced with actual streaming logic
        # first_byte_time = -1
        for result in stream:
            # Publish the intermediate result to the channel named after 'request_id'
            encoded_result = base64.b64encode(result).decode('utf-8')
            # if first_byte_time == -1:
            #     first_byte_time = time.time()
            r.publish(request_id, json.dumps({"prediction": encoded_result}))
        # finish_time = time.time()
        # Signal completion of the streaming predictions
        r.publish(request_id, json.dumps({"complete": True}))
        # r.lpush(
        #     "bark-query-logs",
        #     json.dumps(
        #         {
        #             "text": text,
        #             "voice": voice,
        #             "rate": rate,
        #             "request_time": request_time,
        #             'first_byte_time': first_byte_time,
        #             'finish_time': finish_time
        #         }
        #     )
        # )


if __name__ == "__main__":
    print("Starting consumer...")
    handle_predictions()
