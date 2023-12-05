gcloud auth login
gcloud config set project air-pulumi
gcloud container clusters create-auto tts-cluster --region=us-west1 --release-channel=rapid --cluster-version=1.28.3-gke.1203000 --workload-policies=allow-net-admin --project=air-pulumi
gcloud container clusters get-credentials tts-cluster --region=us-west1
#gcloud container clusters get-credentials tts-cluster --region=us-central1
#kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml
istioctl install --set profile=default
#istioctl verify-install
kubectl label namespace default istio-injection=enabled
kubectl apply -f kube_config/deploy.yaml
kubectl apply -f kube_config/service.yaml
kubectl apply -f kube_config/gateway.yaml
kubectl apply -f kube_config/virtual_service.yaml