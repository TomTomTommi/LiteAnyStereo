import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.append("core")

from core.liteanystereo import LiteAnyStereo
from core.utils.utils import InputPadder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(device: torch.device) -> LiteAnyStereo:
    model = LiteAnyStereo()
    model.to(device)
    model.eval()
    return model


def build_inputs(height: int, width: int, device: torch.device, pad: bool):
    left = torch.randint(0, 256, (1, 3, height, width), dtype=torch.float32, device=device)
    right = torch.randint(0, 256, (1, 3, height, width), dtype=torch.float32, device=device)

    original_size = (height, width)
    padded_size = original_size
    if pad:
        padder = InputPadder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)
        padded_size = tuple(left.shape[-2:])

    return left, right, original_size, padded_size


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed_stage(fn, device: torch.device):
    sync_if_needed(device)
    start = time.perf_counter()
    output = fn()
    sync_if_needed(device)
    elapsed = time.perf_counter() - start
    return output, elapsed


@torch.no_grad()
def benchmark(model, left, right, warmup, total, max_disp, use_amp):
    timings = []
    amp_enabled = use_amp and left.device.type == "cuda"
    amp_dtype = torch.float16

    for idx in range(total):
        if left.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            _ = model(left, right, max_disp=max_disp, test_mode=True)
        if left.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)

    measured = np.array(timings[warmup:], dtype=np.float64)
    return {
        "mean_ms": float(measured.mean() * 1000.0),
        "median_ms": float(np.median(measured) * 1000.0),
        "fps": float(1.0 / measured.mean()),
        "count": int(len(measured)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark LiteAnyStereo inference speed.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=1248)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_pad", action="store_true", help="disable InputPadder(divis_by=32)")
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="disable CUDA autocast during benchmarking",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count())
        torch.set_num_interop_threads(1)
    else:
        torch.backends.cudnn.benchmark = True

    if args.total <= args.warmup:
        raise ValueError(f"--total ({args.total}) must be larger than --warmup ({args.warmup}).")
    model = build_model(device)
    model = torch.compile(model, mode="reduce-overhead")

    left, right, original_size, padded_size = build_inputs(
        args.height, args.width, device, pad=not args.no_pad
    )

    logging.info("Checkpoint: none (random initialization)")
    logging.info("Device: %s", device)
    logging.info("Params: %.2f M", count_parameters(model) / 1e6)
    logging.info("Input size: %sx%s", original_size[0], original_size[1])
    logging.info("Padded size: %sx%s", padded_size[0], padded_size[1])
    logging.info(
        "Warmup: %d, total: %d, max_disp: %d, amp: %s",
        args.warmup,
        args.total,
        args.max_disp,
        str(not args.no_amp and device.type == "cuda").lower(),
    )
    stats = benchmark(
        model=model,
        left=left,
        right=right,
        warmup=args.warmup,
        total=args.total,
        max_disp=args.max_disp,
        use_amp=not args.no_amp,
    )

    logging.info(
        "Average runtime: %.2f ms | Median runtime: %.2f ms | FPS: %.2f | Measured iters: %d",
        stats["mean_ms"],
        stats["median_ms"],
        stats["fps"],
        stats["count"],
    )
