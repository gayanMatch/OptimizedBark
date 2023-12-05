import os
import shutil
import time
import numpy as np
import sys
import uuid
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
from bark.SynthesizeThread import SynthesizeThread

DEFAULT_VOICE = 'test'
CALL_INDEX = 0
free_threads = []
synthesize_thread = SynthesizeThread(DEFAULT_VOICE)
synthesize_thread.start()

import logging

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

# Initialize Flask.
app = FastAPI()
is_process_available = True

@app.post('/{call_id}/synthesize')
async def synthesize(call_id: str, request: Request):
    global is_process_available
    if not is_process_available:
        def stream_results():
            yield os.getenv('MY_POD_NAME')
        return StreamingResponse(stream_results(), status_code=400)
    is_process_available = False
    global CALL_INDEX
    # call_id = "CA123" if port == 5000 else "CA124"
    call_id = f"CA{CALL_INDEX}"
    data = await request.json()
    text = data.pop("text")
    voice = data.pop("voice")
    rate = data.pop("rate") if "rate" in data.keys() else 1.0
    rate -= 0.1
    stream = synthesize_thread.add_request(text, voice, rate)
    async def stream_results():
        async for out in stream:
            # yield out
            yield os.getenv('MY_POD_NAME')
        global is_process_available
        is_process_available = True
    return StreamingResponse(stream_results(), media_type="application/octet-stream")


@app.get('/process_available')
def get_readiness():
    global is_process_available
    if is_process_available:
        print("Ready", os.getenv('MY_POD_NAME'))
        return Response(status_code=200)
    else:
        print("Busy", os.getenv('MY_POD_NAME'))
        return Response(status_code=401)


# launch a Tornado server with HTTPServer.
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="debug",
    )
