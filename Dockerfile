ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt
ARG BASE_TAG=24.02-py3

FROM ${BASE_IMAGE}:${BASE_TAG} as fastapi_bark
LABEL authors="ginger"

WORKDIR /app

COPY . .

COPY --from=trt_bark /app/models models
RUN pip install nvidia-pyindex && \
    pip install -r requirements.txt --no-cache && \
    python3 -c "import nltk;nltk.download('punkt')" && \
    python3 -c "from vocos import Vocos;Vocos.from_pretrained('charactr/vocos-encodec-24khz')"
EXPOSE 5000

CMD ["python3", "fast_api_server.py"]