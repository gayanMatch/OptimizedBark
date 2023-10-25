import os
import shutil
import time
import numpy as np
import sys
import uuid
# Tornado web server
from flask import Flask, render_template, Response, send_file, send_from_directory, request, jsonify, redirect, stream_with_context, flash
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from bark.SynthesizeThread import SynthesizeThread

DEFAULT_VOICE = 'test'
CALL_INDEX = 0
free_threads = []
synthesize_thread = SynthesizeThread(DEFAULT_VOICE)
synthesize_thread.start()


# Initialize Flask.
app = Flask(__name__)
@app.route('/<call_id>/synthesize', methods=["POST"])
def synthesize(call_id):
    global CALL_INDEX
    # call_id = "CA123" if port == 5000 else "CA124"
    call_id = f"CA{CALL_INDEX}"
    data = request.get_json()
    text = data['text']
    voice = data['voice']
    synthesize_thread.voice = voice.replace('.npz', '')
    directory_path = f'bark/static/{call_id}'
    print("#" * 50)
    # print("Previous Synthesis Finished:", len(os.listdir(directory_path)) == 0)
    print(text)
    print("#" * 50)
    synthesize_thread.synthesize_queue.append((text, f"bark/static/{call_id}"))
    while not os.path.exists(f'{directory_path}/audio_0.raw'):
        time.sleep(0.01)
    url_root = request.url_root.replace(str(port), '4000')
    CALL_INDEX += 1
    return redirect(f"{url_root}{call_id}/play")

# launch a Tornado server with HTTPServer.
if __name__ == "__main__":
    port = 5000 if len(sys.argv) < 2 else int(sys.argv[1])
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(port, address='0.0.0.0')
    IOLoop.instance().start()
    # app.run(host='0.0.0.0', port=80, debug=True)
