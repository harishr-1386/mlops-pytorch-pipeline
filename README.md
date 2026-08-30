# mlops-pytorch-pipeline

End-to-end MLOps pipeline for CIFAR-10 image classification using PyTorch, Docker, and Kubernetes.
Built for DA5402W Assignment 2 (IIT Madras Online M.Tech in AI).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Developer Machine                    │
│                                                             │
│  src/  ──► docker build ──► mlops-train:v1                  │
│                         ──► mlops-serve:v1                  │
└───────────────────────────────┬─────────────────────────────┘
                                │  kubectl apply
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                   Kubernetes Cluster (ml-training ns)       │
│                                                             │
│  ConfigMap ──────────────────────────────────┐              │
│                                              ▼              │
│  PVC: data-pvc ──► Job: cifar10-training ──► PVC: ckpt-pvc  │
│                         (ResNet-18 train)         │         │
│                                                   ▼         │
│                         Deployment: model-serving (x2)      │
│                         FastAPI /predict, /health           │
│                              │                              │
│                         Service: ClusterIP :80              │
│                              │                              │
│                         HPA: 2–6 replicas (CPU 70%)         │
└─────────────────────────────────────────────────────────────┘
```

| Item | Link |
|---|---|
| GitHub Repository | https://github.com/harishr-1386/mlops-pytorch-pipeline |
| Final PR (K8s deployment + validation screenshots) | https://github.com/harishr-1386/mlops-pytorch-pipeline/pull/5 |


### Pull Request History

| PR | Branch | Description | Week |
|---|---|---|---|
| #1 | feature/project-structure | Repo skeleton, CI workflow, requirements, config | Week 1 |
| #2 | feature/pytorch-model | ResNet-18 model, dataset, training loop, FastAPI serving | Week 1 |
| #3 | feature/docker-containerization | Multi-stage training Dockerfile, slim serving Dockerfile | Week 2 |
| #5 | feature/k8s-deployment | Kubernetes manifests, end-to-end validation screenshots | Week 2 |
## Project Structure

```
mlops-pytorch-pipeline/
├── .github/workflows/ci.yml   # Lint + test on every push
├── configs/
│   └── training_config.yaml   # Hyperparameters (mirrored in K8s ConfigMap)
├── docker/
│   ├── Dockerfile.train       # Multi-stage training image
│   └── Dockerfile.serve       # Slim serving image (FastAPI)
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── pvc.yaml               # data-pvc + checkpoints-pvc
│   ├── training-job.yaml      # Batch training Job (GPU-enabled)
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
├── requirements/
│   ├── train.txt              # Pinned training deps
│   └── serve.txt              # Pinned inference-only deps
├── src/
│   ├── model.py               # ResNet-18 adapted for 32x32 CIFAR-10
│   ├── dataset.py             # CIFAR-10 loaders + augmentation
│   ├── train.py               # Training loop with LR scheduler + early stopping
│   └── serve.py               # FastAPI inference server
└── tests/
    └── test_model.py
```

## Prerequisites

- Python 3.11+
- Docker Desktop or Docker Engine
- `kubectl` CLI
- Minikube (or kind / cloud cluster)

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/<harishr-1386>/mlops-pytorch-pipeline.git
cd mlops-pytorch-pipeline
```

### 2. Local training (without Docker)

```bash
pip install -r requirements/train.txt
python src/train.py
```

### 3. Docker: build and run training

```bash
# Build
docker build -f docker/Dockerfile.train -t mlops-train:v1 .

# Run (data downloaded on first run, checkpoint written to ./checkpoints/)
mkdir -p data checkpoints
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1
```

### 4. Docker: build and run serving

```bash
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

docker run --rm -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1
```

Test the endpoints:

```bash
curl http://localhost:8080/health

curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"
```

### 5. Kubernetes deployment

```bash
# Start Minikube
minikube start --driver=docker

# Load local images into Minikube (no registry required)
minikube image load mlops-train:v1
minikube image load mlops-serve:v1

# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/training-job.yaml

# Watch training
kubectl logs -f job/cifar10-training -n ml-training

# Once training completes, deploy serving
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

# Verify
kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training

# Port-forward and test
kubectl port-forward svc/model-serving 8080:80 -n ml-training &
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

> **Note on GPU Job:** `training-job.yaml` includes a `nodeSelector` and `toleration` for NVIDIA GPU nodes.
> On CPU-only Minikube, remove the `nodeSelector`, `tolerations`, and `nvidia.com/gpu` resource lines before applying.

## Git Workflow

```
main
 └── develop
      ├── feature/project-structure   (PR 1)
      ├── feature/pytorch-model       (PR 2)
      ├── feature/docker-containerization (PR 3)
      └── feature/k8s-deployment      (PR 5)
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `docs:`, `chore:`, `test:`

## Model Details

| Item | Value |
|---|---|
| Architecture | ResNet-18 (CIFAR-adapted stem) |
| Dataset | CIFAR-10 (50k train / 10k val) |
| Optimizer | Adam (lr=0.001) |
| LR Scheduler | CosineAnnealingLR (T_max=10) |
| Augmentation | RandomFlip, RandomCrop, ColorJitter, RandomErasing |
| Early Stopping | Patience=3 on val accuracy |

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```
