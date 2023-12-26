export TAG=1.1.test.1225
docker build -t fastapi-tts:$TAG .
docker tag fastapi-tts:$TAG gcr.io/air-pulumi/tts:$TAG
docker push gcr.io/air-pulumi/tts:$TAG