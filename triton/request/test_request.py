import time

import matplotlib.pyplot as plt
import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image
from torchvision import transforms


def plot(src, cmap=None, title=None, size=(10, 10)):
    plt.rcParams["figure.figsize"] = size


def batch_preprocessing(img, transform) -> np.ndarray:

    img = transform(img).unsqueeze(0).numpy()
    return img


def main():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize((224, 224)),
        ]
    )

    triton_url = "172.17.0.2:8001"

    client = grpcclient.InferenceServerClient(url=triton_url, verbose=True)

    img = Image.open("bean.jpg").convert("RGB")

    batch = batch_preprocessing(img, transform)

    inputs = []
    inputs.append(grpcclient.InferInput("input", batch.shape, "FP32"))
    inputs[0].set_data_from_numpy(batch)

    outputs = []
    outputs.append(grpcclient.InferRequestedOutput("output"))

    st_time = time.time()

    results = client.infer("resnet18_onnx", inputs, outputs=outputs)

    finish_time = time.time()

    print(f"Request took {round(finish_time - st_time, 3)}")

    ans = {0: "Dark", 1: "Green", 2: "Light", 3: "Medium"}

    print(ans[results.as_numpy("output").argmax(1)[0]])


if __name__ == "__main__":
    main()
