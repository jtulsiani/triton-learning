import os
import time

import numpy as np
import requests
import tritonclient.http as httpclient

MODEL_PROVIDER_URL = os.getenv("MODEL_PROVIDER_URL", "http://model-provider:8000")
TRITON_URL = os.getenv("TRITON_URL", "triton-server:8000")


def wait_for_http(url: str, timeout_seconds: int = 180) -> None:
    start = time.perf_counter()
    while True:
        print(f"Checking {url}")
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass

        if time.perf_counter() - start > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for {url}")
        time.sleep(2)


def run_model_provider_inference(tensor: np.ndarray) -> tuple[int, float]:
    payload = {"tensor": tensor.tolist()}
    start = time.perf_counter()
    response = requests.post(f"{MODEL_PROVIDER_URL}/infer", json=payload, timeout=60)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response.raise_for_status()
    top1 = int(response.json()["top1_class"])
    return top1, elapsed_ms


def run_triton_inference(tensor: np.ndarray) -> tuple[int, float]:
    client = httpclient.InferenceServerClient(url=TRITON_URL)
    infer_input = httpclient.InferInput("input", list(tensor.shape), "FP32")
    infer_input.set_data_from_numpy(tensor)
    requested_output = httpclient.InferRequestedOutput("output")

    start = time.perf_counter()
    result = client.infer(
        model_name="resnet50",
        inputs=[infer_input],
        outputs=[requested_output],
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logits = result.as_numpy("output")
    top1 = int(np.argmax(logits, axis=1)[0])
    return top1, elapsed_ms


def main() -> None:
    print(f"Waiting for model-provider")
    wait_for_http(f"{MODEL_PROVIDER_URL}/health")
    print(f"Waiting for triton-server")
    wait_for_http(f"http://{TRITON_URL}/v2/health/ready")

    dummy_tensor = np.ones((1, 3, 224, 224), dtype=np.float32)

    provider_top1, provider_ms = run_model_provider_inference(dummy_tensor)
    triton_top1, triton_ms = run_triton_inference(dummy_tensor)

    print(f"Model Provider: top1_class={provider_top1}, latency_ms={provider_ms:.2f}")
    print(f"Triton Server:  top1_class={triton_top1}, latency_ms={triton_ms:.2f}")


if __name__ == "__main__":
    main()
