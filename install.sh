gcloud container clusters get-credentials tts --zone=us-central1-c
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml
istioctl install --set profile=default
istioctl verify-install
kubectl label namespace default istio-injection=enabled
kubectl apply -f kube_config/deploy.yaml
kubectl apply -f kube_config/service.yaml
kubectl apply -f kube_config/gateway.yaml
kubectl apply -f kube_config/virtual_service.yaml