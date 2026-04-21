# Triton Learning Pipeline

Simple setup to learn about Triton Inference Server. Running using docker compose will:
- start up model-provider, which will download Resnet50 weights and set it up with PyTorch. It will then export the ONNX file for the Triton server, and also serve the model directly with FastAPI from PyTorch.
- start up triton-server, which will serve the exported ONNX model.
- start the client-app, which will make the same request to both servers, measure latency, and print the returned image classification and latency in the terminal.

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
