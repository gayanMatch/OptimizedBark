kubectl delete namespace istio-system
kubectl label namespace default istio-injection-
kubectl delete -f kube_config/virtual_service.yaml
kubectl delete -f kube_config/gateway.yaml
kubectl delete -f kube_config/service.yaml
kubectl delete -f kube_config/deploy.yaml
