export TAG=1.2
docker build -t fastapi-tts:$TAG .
docker tag fastapi-tts:$TAG gcr.io/air-pulumi/tts:$TAG
docker push gcr.io/air-pulumi/tts:$TAG