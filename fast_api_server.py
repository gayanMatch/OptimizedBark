import os
import shutil
import time
import numpy as np
import sys
import uuid
import uvicorn
# Tornado web server
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from bark.SynthesizeThread import SynthesizeThread

DEFAULT_VOICE = 'test'
CALL_INDEX = 0
free_threads = []
synthesize_thread = SynthesizeThread(DEFAULT_VOICE)
synthesize_thread.start()

import logging

logging.basicConfig(filename='logs.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.info("===================== Server Started ======================================")
# Initialize Flask.
app = FastAPI()


@app.post('/{call_id}/synthesize')
async def synthesize(call_id: str, request: Request):
    global CALL_INDEX
    # call_id = "CA123" if port == 5000 else "CA124"
    call_id = f"CA{CALL_INDEX}"
    data = await request.json()
    text = data.pop("text")
    voice = data.pop("voice").replace('.npz', '')
    rate = data.pop("rate") if "rate" in data.keys() else 1.0
    logging.info(f"Request Info for {call_id} - Text: {text}, Voice: {voice}, Rate: {rate}")
    # rate -= 0.1
    stream = synthesize_thread.add_request(text, voice, rate)
    async def stream_results():
        index = 0
        async for out in stream:
            logging.info(f"Response Info for {call_id} - Sent {index}")
            index += 1
            yield out
    return StreamingResponse(stream_results(), media_type="application/octet-stream")

# launch a Tornado server with HTTPServer.
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="debug",
    )
