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


@app.post('/{call_id}/synthesize')
async def synthesize(call_id: str, request: Request):
    global CALL_INDEX
    # call_id = "CA123" if port == 5000 else "CA124"
    call_id = f"CA{CALL_INDEX}"
    data = await request.json()
    text = data.pop("text")
    voice = data.pop("voice")
    stream = synthesize_thread.add_request(text, voice)
    async def stream_results():
        async for out in stream:
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
