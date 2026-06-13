import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large

CHECKPOINT_PATH = "model/checkpoints/mobilenetv3_finetune_partb_exp2_extended_best.pth"
ONNX_PATH = "model/onnx/food101_mobilenetv3.onnx"

NUM_CLASSES = 101


def build_model():
    model = mobilenet_v3_large(weights=None)

    in_features = model.classifier[0].in_features

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES),
    )

    return model


def main():
    model = build_model()

    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
    )

    print(f"Exported ONNX model to {ONNX_PATH}")


if __name__ == "__main__":
    main()