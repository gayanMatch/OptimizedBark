ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt
ARG BASE_TAG=23.10-py3

FROM ${BASE_IMAGE}:${BASE_TAG} as fastapi_bark
LABEL authors="ginger"

WORKDIR /app

COPY . .

#RUN mkdir models
#RUN mkdir models/bark_large
RUN mkdir models/bark_large/trt-engine
RUN mkdir models/bark_large/pytorch
RUN mkdir models/bark_coarse
RUN mkdir models/bark_coarse/trt-engine
RUN mkdir models/bark_coarse/pytorch
RUN mkdir models/bark_fine_large
RUN mkdir models/bark_fine_large/pytorch

COPY --from=trt_bark /app/models/bark_large/trt-engine /app/models/bark_large/trt-engine/
COPY --from=trt_bark /app/models/bark_coarse/trt-engine /app/models/bark_coarse/trt-engine/

#COPY ./models/bark_large/pytorch /app/models/bark_large/pytorch/
#COPY ./models/bark_coarse/pytorch /app/models/bark_coarse/pytorch/
#COPY ./models/bark_fine_large/pytorch /app/models/bark_fine_large/pytorch/

RUN mkdir bark/static

RUN pip install nvidia-pyindex
RUN pip install -r requirements.txt

RUN python3 -c "import nltk;nltk.download('punkt')"
RUN python3 -c "from vocos import Vocos;Vocos.from_pretrained('charactr/vocos-encodec-24khz')"
EXPOSE 5000

CMD ["python3", "fast_api_server.py"]