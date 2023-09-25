import time
import os
from flask import Flask, Response

app = Flask(__name__)

@app.route('/<call_id>/play')
def file_stream(call_id):
    def event_stream():
        i = 0
        chunk_size = 1024
        directory_path = f'static/{call_id}'
        # print(directory_path)
        while True:
            path = f'{directory_path}/audio_{i}.wav'
            print(path)
            if os.path.exists(path):
                time.sleep(0.015)
                i += 1
                with open(path, "rb") as f:
                    print(path)
                    f.read(92)
                    for chunk in iter(lambda: f.read(chunk_size), b""):  # 2048 bytes chunk size
                        print(chunk[:20])
                        yield chunk
                        time.sleep(0.015)
            elif not os.path.exists(f'{directory_path}/finish.lock'):
                time.sleep(0.03)
            else:
                i += 1
                break
        print("Finished")
    return Response(event_stream(), mimetype='application/octet-stream')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)