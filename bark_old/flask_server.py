import os
import glob
import sys
import shutil
import time
import numpy as np
from flask import Flask, render_template, Response, send_file, send_from_directory, request, jsonify, redirect, stream_with_context, flash
import sys
from werkzeug.utils import secure_filename
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
DEFAULT_VOICE = 'en_fiery.npz'
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
            flash('File uploaded successfully')
            return redirect(request.url)

    uploaded_files = list(glob.glob(f"{app.config['UPLOAD_FOLDER']}/*.npz")) + list(glob.glob(f"{app.config['UPLOAD_FOLDER']}/v2/*.npz"))
    uploaded_files.sort()
    uploaded_files = [os.path.relpath(file, app.config['UPLOAD_FOLDER']) for file in uploaded_files]
    selected_file = DEFAULT_VOICE
    if request.args.get('selected_file'):
        selected_file = request.args.get('selected_file')
    synthesize_thread.voice = selected_file[:-4]
    print(synthesize_thread.voice)
    return render_template('index.html', uploaded_files=uploaded_files, selected_file=selected_file)


# Route to synthesize
@app.route('/synthesize', methods=["POST"])
def synthesize():
    text = request.form['text']
    print(text)
    directory_path = 'bark/static'
    shutil.rmtree(directory_path)
    os.mkdir(directory_path)
    synthesize_thread.synthesize_queue.append(text)
    while not os.path.exists(f'{directory_path}/audio_0.mp3'):
        time.sleep(0.01)
    return redirect("http://138.2.225.7:4000/file")


@app.route('/file')
def file_stream():
    def generate():
        with open("bark/audio.mp3", "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):  # 4096 bytes chunk size
                print('xx')
                yield chunk
                time.sleep(0.1)
    return Response(generate(), mimetype='audio/mpeg')


@app.route('/audio/<path:filename>')
def serve_audio(filename):
    server_directory = 'static'
    directory_path = 'bark/static'
    while not os.path.exists(f'{directory_path}/{filename}') and synthesize_thread.isWorking:
        time.sleep(0.01)
    return send_from_directory(server_directory, filename)
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
    port = 5000
    http_server = HTTPServer(WSGIContainer(app))
    logging.debug("Started Server, Kindly visit http://0.0.0.0:" + str(port))
    http_server.listen(port, address='0.0.0.0')
    IOLoop.instance().start()
    # app.run(host='0.0.0.0', port=80, debug=True)
