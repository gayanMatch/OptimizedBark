ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt
ARG BASE_TAG=23.10-py3

FROM ${BASE_IMAGE}:${BASE_TAG} as fastapi_bark
LABEL authors="ginger"

WORKDIR /app

COPY . .

RUN mkdir models
RUN mkdir models/bark_large
RUN mkdir models/bark_large/trt-engine
RUN mkdir models/bark_coarse
RUN mkdir models/bark_coarse/trt-engine

COPY --from=trt_bark /app/models/bark_large/trt-engine /app/models/bark_large/trt-engine/
COPY --from=trt_bark /app/models/bark_coarse/trt-engine /app/models/bark_coarse/trt-engine/

RUN mkdir bark/static

RUN pip install nvidia-pyindex
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python3", "fast_api_server.py"]