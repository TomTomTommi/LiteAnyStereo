import argparse

import torch
import torch.nn as nn

from core.models import build_model, load_model_weights

import torch.nn.functional as F
import core.liteanystereov2 as liteanystereov2
import core.liteanystereov2_H as liteanystereov2_H
import core.submodule as submodule

# F.unfold is not exportable to ONNX; replace it with an equivalent identity conv3x3.
def _context_upsample(depth_low, up_weights):
    b, c, h, w = depth_low.shape

    w3 = torch.eye(9, dtype=depth_low.dtype, device=depth_low.device).reshape(9, 1, 3, 3)
    depth_unfold = F.conv2d(depth_low.reshape(b,c,h,w), w3, padding=1).reshape(b,-1,h,w)
    depth_unfold = F.interpolate(depth_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)

    depth = torch.sum(depth_unfold*up_weights, dim=1, keepdim=True)

    return depth


# .unfold is not exportable to ONNX; replace it with an equivalent stack of slices.
def _build_correlation_volume(left_feature, right_feature, max_disp):
    B, C, H, W = left_feature.shape

    left_volume = left_feature.unsqueeze(2).expand(B, C, max_disp, H, W)
    padded_right = F.pad(right_feature, (max_disp - 1, 0, 0, 0))
    unfolded_right = torch.stack([padded_right[:, :, :, i:i+W] for i in range(max_disp)], dim=3)
    right_volume = torch.flip(unfolded_right, [3]).permute(0, 1, 3, 2, 4)

    cost_volume = (left_volume * right_volume).mean(dim=1)
    return cost_volume.contiguous()


# .unfold is not exportable to ONNX; replace it with an equivalent stack of slices.
def _build_gwc_volume_fast(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups

    ref_volume = refimg_fea.unsqueeze(2).expand(B, C, maxdisp, H, W)
    padded_target = F.pad(targetimg_fea, (maxdisp - 1, 0, 0, 0))
    unfolded_target = torch.stack([padded_target[:, :, :, i:i+W] for i in range(maxdisp)], dim=3)
    target_volume = torch.flip(unfolded_target, [3]).permute(0, 1, 3, 2, 4)

    ref_volume = ref_volume.view(B, num_groups, channels_per_group, maxdisp, H, W)
    target_volume = target_volume.view(B, num_groups, channels_per_group, maxdisp, H, W)
    volume = (ref_volume * target_volume).mean(dim=2)
    return volume.contiguous()


class Wrapper(nn.Module):
    """Bakes in max_disp/test_mode so the exported graph only takes (left, right)."""
    def __init__(self, model, max_disp):
        super().__init__()
        self.model = model
        self.max_disp = max_disp

    def forward(self, left, right):
        return self.model(left, right, max_disp=self.max_disp, test_mode=True)


# Export checkpoint to ONNX file. 
# - Fix uncompatable functions.
# - Export with fixed input shape.
def export(version, model_size, restore_ckpt, width, height, max_disp, output_name):
    # Replace functions that not compatable with ONNX export with an equivalent fixed functions.
    submodule.context_upsample = _context_upsample
    liteanystereov2.context_upsample = _context_upsample
    liteanystereov2_H.context_upsample = _context_upsample
    liteanystereov2.build_correlation_volume = _build_correlation_volume
    submodule.build_gwc_volume_fast = _build_gwc_volume_fast
    liteanystereov2_H.build_gwc_volume_fast = _build_gwc_volume_fast

    # Load model
    model = build_model(version, model_size=model_size, max_disp=max_disp)
    checkpoint = torch.load(restore_ckpt, map_location="cpu")
    load_model_weights(model, checkpoint, strict=True)
    model = Wrapper(model.eval(), max_disp=max_disp).eval()

    left = torch.randint(0, 256, (1, 3, height, width), dtype=torch.float32)
    right = torch.randint(0, 256, (1, 3, height, width), dtype=torch.float32)
    
    torch.onnx.export(
        model, (left, right), output_name,
        input_names=["left", "right"], output_names=["disparity"],
        opset_version=18,
        dynamo=False
    )
    print("Saved", output_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Export LiteAnyStereo checkpoint to ONNX.")
    parser.add_argument("--version", default="las2", help="model version, e.g. las1 or las2")
    parser.add_argument("--model_size", default="m", help="LAS2 model size: s, m, l, or h")
    parser.add_argument("--restore_ckpt", default="./checkpoints/LAS2_M.pth", help="path to .pth checkpoint file")
    parser.add_argument("--width", type=int, default=1248, help="export input width, per single image (not left+right combined)")
    parser.add_argument("--height", type=int, default=384, help="export input height")
    parser.add_argument("--max_disp", type=int, default=192, help="maximum disparity used by the model")
    parser.add_argument("--output_name", default="liteanystereo.onnx", help="output ONNX file path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export(version=args.version,
           model_size=args.model_size,
           restore_ckpt=args.restore_ckpt,
           width=args.width,
           height=args.height,
           max_disp=args.max_disp,
           output_name=args.output_name)