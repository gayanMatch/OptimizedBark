import os
import glob
import sys
import shutil
import time
import numpy as np
from flask import Flask, render_template, Response, send_file, send_from_directory, request, jsonify, redirect, stream_with_context, flash
import sys
from werkzeug.utils import secure_filename
import uuid
# import librosa
# Tornado web server
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from bark.SynthesizeThread import SynthesizeThread
# Debug logger
import logging

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)
DEFAULT_VOICE = 'test'
CALL_INDEX = 0
free_threads = []
synthesize_thread = SynthesizeThread(DEFAULT_VOICE)
synthesize_thread.start()


# Initialize Flask.
app = Flask(__name__)
UPLOAD_FOLDER = 'bark/assets/prompts'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'supersecretkey'
# Route to render GUI
@app.route('/', methods=['GET', 'POST'])
def show_entries():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            synthesize_thread.synthesize_queue.append((filename.replace('.npz', ''), True))
            while synthesize_thread.isWorking:
                time.sleep(0.01)
            flash('File uploaded successfully')
            return redirect(request.url)

    uploaded_files = list(glob.glob(f"{app.config['UPLOAD_FOLDER']}/*.npz")) + list(glob.glob(f"{app.config['UPLOAD_FOLDER']}/v2/*.npz"))
    uploaded_files.sort()
    uploaded_files = [os.path.relpath(file, app.config['UPLOAD_FOLDER']) for file in uploaded_files]
    selected_file = DEFAULT_VOICE
    if request.args.get('selected_file'):
        selected_file = request.args.get('selected_file')
    synthesize_thread.voice = selected_file.replace('.npz', '')
    print(synthesize_thread.voice)
    return render_template('index.html', uploaded_files=uploaded_files, selected_file=selected_file)

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

# @app.route('/<call_id>/start')
# def create_call(call_id):
#     synthesize_thread = free_threads.pop()
#     synthesize_thread.directory = f"bark/static/{call_id}"
#     os.makedirs(f"bark/static/{call_id}", exist_ok=True)
#     thread_dict[call_id] = synthesize_thread
#     return "Success"
#
# @app.route('/<call_id>/end')
# def finish_call(call_id):
#     synthesize_thread = thread_dict[call_id]
#     shutil.rmtree(synthesize_thread.directory)
#     free_threads.append(synthesize_thread)
#     return "Success"


    # def generate():
    #     with open(f'{directory_path}/{filename}', "rb") as fwav:
    #         data = fwav.read(1024)
    #         while data:
    #             yield data
    #             data = fwav.read(1024)
    #
    # if os.path.exists(f'{directory_path}/{filename}'):
    #     return Response(generate(), mimetype="audio/wav")
    # else:
    #     return send_from_directory('static', filename)

def stream_file(file_name, chunk_size=1024):
    with open(file_name, 'rb') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk

# @app.route('/stream_file')
# def serve_sse():
#     def event_stream():
#         i = 0
#         chunk_size = 1024
#         directory_path = 'bark/static'
#         while True:
#             path = f'{directory_path}/audio_{i}.wav'
#             print(path)
#             if os.path.exists(path):
#                 i += 1
#                 for chunk in stream_file(path):
#                     yield 'data: %s\n\n' % chunk
#             elif synthesize_thread.isWorking:
#                 while not os.path.exists(path):
#                     time.sleep(0.01)
#             else:
#                 i += 1
#                 break
#     return Response(event_stream(), mimetype="text/event-stream")


# launch a Tornado server with HTTPServer.
if __name__ == "__main__":
    port = 5000 if len(sys.argv) < 2 else int(sys.argv[1])
    http_server = HTTPServer(WSGIContainer(app))
    logging.debug("Started Server, Kindly visit http://0.0.0.0:" + str(port))
    http_server.listen(port, address='0.0.0.0')
    IOLoop.instance().start()
    # app.run(host='0.0.0.0', port=80, debug=True)
