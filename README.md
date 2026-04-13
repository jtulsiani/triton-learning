# Triton Learning Pipeline

## Folder Structure

```
.
├── docker-compose.yml
├── requirements.txt
├── README.md
├── model-provider
│   ├── Dockerfile
│   └── app.py
├── client-app
│   ├── Dockerfile
│   └── client.py
└── model_repository
    └── resnet50
        ├── config.pbtxt
        └── 1
            └── model.onnx  # generated at runtime by model-provider
```

## Run

```bash
docker compose up --build
```

The `client-app` container will print:
- top-1 class and latency from raw PyTorch (`model-provider`)
- top-1 class and latency from Triton (`triton-server`)
