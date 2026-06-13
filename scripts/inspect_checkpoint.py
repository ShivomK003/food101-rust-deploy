import torch

CHECKPOINT_PATH = "model/checkpoints/mobilenetv3_finetune_partb_exp2_extended_best.pth"

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

print("Type:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("\nKeys:")
    for key in checkpoint.keys():
        print("-", key)