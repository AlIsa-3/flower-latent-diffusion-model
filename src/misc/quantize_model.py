import torch
import argparse
import pathlib


def quantize_fp16(model, save_path):
    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fp16_dict = {
        key: value.half() if value.is_floating_point() else value
        for key, value in model.items()
    }
    torch.save(fp16_dict, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="The original model path")
    parser.add_argument("output_path", help="The destination for the fp16 model")

    args = parser.parse_args()

    model = torch.load(args.model_path, map_location="cpu", weights_only=True)
    quantize_fp16(model, args.output_path)
