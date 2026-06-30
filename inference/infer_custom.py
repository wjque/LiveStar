import argparse
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


DEFAULT_PROMPT = (
    "You are an expert in real-time streaming video description. "
    "I will provide video frames sequentially, and you need to comprehend "
    "each frame's content in real-time while dynamically generating concise descriptions. "
    "Use transitional phrases to maintain textual coherence and avoid repeating already described content.\n"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run LiveStar inference with weights stored in a custom directory.")
    parser.add_argument(
        "--weights-dir",
        default="/data1/LiveStar_8B",
        help="Directory containing model-*.safetensors. Default: /data1/LiveStar_8B",
    )
    parser.add_argument(
        "--model-code-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory containing LiveStar model code, tokenizer files, and model.safetensors.index.json.",
    )
    parser.add_argument(
        "--video",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "videos" / "HPtIGhOsViM.mp4"),
        help="Input video path.",
    )
    parser.add_argument("--num-frames", type=int, default=1, help="Number of frames to sample from the video.")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Sampling FPS used when selecting frames.")
    parser.add_argument("--input-size", type=int, default=448, help="Image size expected by the visual encoder.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated tokens.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt prefix for video frame descriptions.")
    return parser.parse_args()


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")


def make_runtime_model_dir(model_code_dir, weights_dir):
    model_code_dir = Path(model_code_dir).resolve()
    weights_dir = Path(weights_dir).resolve()

    require_file(model_code_dir / "model.safetensors.index.json")
    require_file(model_code_dir / "config.json")
    require_file(model_code_dir / "tokenizer.model")

    weight_files = sorted(weights_dir.glob("model-*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No model-*.safetensors files found in {weights_dir}")

    tmp = tempfile.TemporaryDirectory(prefix="livestar_model_")
    runtime_dir = Path(tmp.name)

    for item in model_code_dir.iterdir():
        if item.name.startswith("model-") and item.suffix == ".safetensors":
            continue
        target = runtime_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, symlinks=True)
        else:
            target.symlink_to(item)

    for item in weight_files:
        (runtime_dir / item.name).symlink_to(item)

    return tmp, runtime_dir


def build_transform(input_size):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_video_frames(video_path, input_size, num_frames, sample_fps):
    video_path = Path(video_path).resolve()
    require_file(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or sample_fps
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / sample_fps)))
    indices = list(range(0, max(frame_count, 1), step))[:num_frames]

    transform = build_transform(input_size)
    pixel_values = []
    num_patches_list = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame).convert("RGB")
        pixel_values.append(transform(image))
        num_patches_list.append(1)

    cap.release()

    if not pixel_values:
        raise RuntimeError(f"No frames could be read from video: {video_path}")

    return torch.stack(pixel_values), num_patches_list


def main():
    args = parse_args()
    tmp, runtime_model_dir = make_runtime_model_dir(args.model_code_dir, args.weights_dir)

    with tmp:
        print(f"runtime_model_dir: {runtime_model_dir}")
        print(f"weights_dir: {Path(args.weights_dir).resolve()}")
        print(f"video: {Path(args.video).resolve()}")
        print(f"cuda_available: {torch.cuda.is_available()}")

        tokenizer = AutoTokenizer.from_pretrained(runtime_model_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained(runtime_model_dir, trust_remote_code=True).half().cuda().to(torch.bfloat16)
        model.eval()

        pixel_values, num_patches_list = load_video_frames(
            args.video,
            input_size=args.input_size,
            num_frames=args.num_frames,
            sample_fps=args.sample_fps,
        )
        pixel_values = pixel_values.to(torch.bfloat16).to(model.device)

        video_frame_prompt = "".join([f"Frame-{i + 1}: <image>\n" for i in range(len(num_patches_list))])
        question = args.prompt + video_frame_prompt
        generation_config = {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "num_beams": 1,
            "repetition_penalty": 1.05,
        }

        with torch.no_grad():
            output, _, _ = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )

        print("OUTPUT_BEGIN")
        print(output)
        print("OUTPUT_END")
        if torch.cuda.is_available():
            print(f"max_memory_allocated_gb: {torch.cuda.max_memory_allocated() / 1024**3:.2f}")


if __name__ == "__main__":
    main()
