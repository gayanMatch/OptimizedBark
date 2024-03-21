import redis
import json
import os
import base64
import time  # Used for simulating long-running or streaming predictions
from bark.SynthesizeThread import SynthesizeThread

DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "hey_james_reliable_1_small_coarse_fix")
synthesize_thread = SynthesizeThread(DEFAULT_VOICE)
synthesize_thread.start()

redis_url = 'redis://default:eb7199cbf0f54bf5bb084f7f1d594692@fly-bark-queries.upstash.io:6379'
# Establish connections to Redis for both publishing results and subscribing to incoming tasks
r = redis.Redis.from_url(redis_url)
# r = redis.Redis(
#   host='localhost',  # Changed to localhost
#   port=6379,
#   password=''  # Likely no password if you're just testing locally
# )


def handle_predictions():
    while True:
        # Block until a message is received; 'ml_requests' is the list with prediction tasks
        _, request_data = r.brpop("ml_requests")

        # Decode and load the request data (contains 'request_id' and 'features')
        request = json.loads(request_data)
        request_id = request["request_id"]
        text = request["text"]
        voice = request["voice"]
        rate = request["rate"]

        # This is a simplistic approach; consider batching or other optimizations for your actual use case
        stream = synthesize_thread.add_request(text, voice, rate)

        # Simulate streaming data; in a real scenario, this loop could be replaced with actual streaming logic
        for result in stream:
            # Publish the intermediate result to the channel named after 'request_id'
            encoded_result = base64.b64encode(result).decode('utf-8')
            r.publish(request_id, json.dumps({"prediction": encoded_result}))

        # Signal completion of the streaming predictions
        r.publish(request_id, json.dumps({"complete": True}))


if __name__ == "__main__":
    print("Starting consumer...")
    handle_predictions()
