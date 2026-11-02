from __future__ import annotations
import ast
import json
import torch
import argparse
from torch import nn
from pathlib import Path
from scipy.io import loadmat

# Defaults from fairseq_signals.models.ecg_transformer and models.transformer.
DEFAULTS = {
    "conv_feature_layers": "[(256, 2, 2)] * 4",
    "conv_bias": False,
    "extractor_mode": "default",
    "in_d": 12,
    "encoder_embed_dim": 768,
    "encoder_ffn_embed_dim": 3072,
    "encoder_attention_heads": 12,
    "encoder_layers": 12,
    "layer_norm_first": False,
    "conv_pos": 128,
    "conv_pos_groups": 16,
}
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# Order from ECG-Agent/src/medrax/tools/classification.py.
CLASS_LABELS = [
    '1AVB', '2AVB', '3AVB', 'ABQRS', 'AFIB', 'AFLT', 'ALMI', 'AMI', 'ANEUR', 'ASMI',
    'BIGU', 'CLBBB', 'CRBBB', 'DIG', 'EL', 'HVOLT', 'ILBBB', 'ILMI', 'IMI', 'INJAL',
    'INJAS', 'INJIL', 'INJIN', 'INJLA', 'INVT', 'IPLMI', 'IPMI', 'IRBBB', 'ISCAL',
    'ISCAN', 'ISCAS', 'ISCIL', 'ISCIN', 'ISCLA', 'ISC_', 'IVCD', 'LAFB', 'LAO/LAE',
    'LMI', 'LNGQT', 'LOWT', 'LPFB', 'LPR', 'LVH', 'LVOLT', 'NDT', 'NORM', 'NST_',
    'NT_', 'PAC', 'PACE', 'PMI', 'PRC(S)', 'PSVT', 'PVC', 'QWAVE', 'RAO/RAE', 'RVH',
    'SARRH', 'SBRAD', 'SEHYP', 'SR', 'STACH', 'STD_', 'STE_', 'SVARR', 'SVTAC',
    'TAB_', 'TRIGU', 'VCLVH', 'WPW'
]
PROBABILITY_THRESHOLD = 0.5


def conv_layers(expression: str) -> list[tuple[int, int, int]]:
    """Read the training config's list or repeated list without executing Python."""
    node = ast.parse(expression, mode="eval").body
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        layers = ast.literal_eval(node.left) * ast.literal_eval(node.right)
    else:
        layers = ast.literal_eval(node)
    if not layers or any(
        len(layer) != 3 or any(type(value) is not int or value <= 0 for value in layer)
        for layer in layers
    ):
        raise ValueError("Expected positive (channels, kernel, stride) convolution triples")
    return layers


class ECGModel(nn.Module):
    """Training architecture expressed with standard PyTorch inference layers."""

    def __init__(self, cfg: dict, num_labels: int) -> None:
        super().__init__()
        if cfg["extractor_mode"] != "default" or cfg["in_d"] != 12:
            raise ValueError("This classifier requires the default extractor and all 12 leads")
        layers = conv_layers(cfg["conv_feature_layers"])
        self.min_samples = 1
        for _, kernel, stride in reversed(layers):
            self.min_samples = (self.min_samples - 1) * stride + kernel

        self.features = nn.ModuleList()
        channels = cfg["in_d"]
        for index, (width, kernel, stride) in enumerate(layers):
            block = [nn.Conv1d(channels, width, kernel, stride=stride, bias=cfg["conv_bias"]),
                     nn.Dropout(0.0)]
            if index == 0:
                block.append(nn.GroupNorm(width, width))
            block.append(nn.GELU())
            self.features.append(nn.Sequential(*block))
            channels = width

        dim = cfg["encoder_embed_dim"]
        self.feature_norm = nn.LayerNorm(channels)
        self.feature_proj = nn.Linear(channels, dim) if channels != dim else nn.Identity()
        self.position = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(dim, dim, cfg["conv_pos"], padding=cfg["conv_pos"] // 2,
                      groups=cfg["conv_pos_groups"]), dim=2)
        self.encoder_norm = nn.LayerNorm(dim)
        self.norm_first = cfg["layer_norm_first"]
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                dim, cfg["encoder_attention_heads"], cfg["encoder_ffn_embed_dim"],
                dropout=0.0, activation="gelu", norm_first=self.norm_first)
            for _ in range(cfg["encoder_layers"])
        ])
        self.proj = nn.Linear(dim, num_labels)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        x = source
        for block in self.features:
            x = block(x)
        x = self.feature_proj(self.feature_norm(x.transpose(1, 2)))
        position = self.position(x.transpose(1, 2))[..., :x.shape[1]]
        x = x + nn.functional.gelu(position.transpose(1, 2))
        if not self.norm_first:
            x = self.encoder_norm(x)
        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(0, 1)
        if self.norm_first:
            x = self.encoder_norm(x)
        # Match the original classifier's per-channel nonzero-count pooling.
        return self.proj(x.sum(dim=1) / (x != 0).sum(dim=1))


def convert_weights(state: dict, layers: int) -> dict:
    """Map fairseq parameter names and combine Q/K/V for PyTorch attention."""
    prefixes = {
        "encoder.feature_extractor.conv_layers.": "features.",
        "encoder.layer_norm.": "feature_norm.",
        "encoder.post_extract_proj.": "feature_proj.",
        "encoder.conv_pos.pos_conv.0.": "position.",
        "encoder.encoder.layer_norm.": "encoder_norm.",
        "encoder.encoder.layers.": "layers.",
    }
    weights = {}
    for key, value in state.items():
        for old, new in prefixes.items():
            if key.startswith(old):
                key = new + key[len(old):]
                break
        for old, new in (("self_attn_layer_norm", "norm1"), ("final_layer_norm", "norm2"),
                         ("fc1", "linear1"), ("fc2", "linear2")):
            key = key.replace(f".{old}.", f".{new}.")
        weights[key] = value
    for index in range(layers):
        prefix = f"layers.{index}.self_attn."
        for kind in ("weight", "bias"):
            weights[f"{prefix}in_proj_{kind}"] = torch.cat([
                weights.pop(f"{prefix}{part}_proj.{kind}") for part in ("q", "k", "v")
            ])
    return weights


class Classifier:
    def __init__(self, checkpoint: Path, device: str | None = None) -> None:
        state = torch.load(checkpoint.expanduser(), map_location="cpu", weights_only=False)
        cfg = state.get("cfg")
        model_cfg = DEFAULTS | cfg["model"]
        num_labels = state["model"]["proj.weight"].shape[0]
        if num_labels != len(CLASS_LABELS):
            raise ValueError(
                f"ECG-Agent requires {len(CLASS_LABELS)} classifier outputs; "
                f"checkpoint has {num_labels}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ECGModel(model_cfg, num_labels)
        self.model.load_state_dict(convert_weights(state["model"], model_cfg["encoder_layers"]), strict=True)
        self.model.to(self.device).eval().requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, ecg_path: str | Path) -> dict[str, float]:
        data = loadmat(ecg_path)
        rate = float(torch.as_tensor(data.get("curr_sample_rate", 500)).item())
        if rate != 500:
            raise ValueError("ECG-Agent's classifier requires 500 Hz ECGs")
        signal = torch.as_tensor(data["feats"].copy(), dtype=torch.float32)
        if signal.ndim != 2 or signal.shape[0] != 12:
            raise ValueError(f"Expected (12, samples) in lead order {LEADS}")
        if signal.shape[1] < self.model.min_samples or not torch.isfinite(signal).all():
            raise ValueError("ECG is too short or contains nonfinite values")
        probabilities = self.model(signal.unsqueeze(0).to(self.device)).sigmoid()[0]
        if not torch.isfinite(probabilities).all():
            raise ValueError("Classifier produced nonfinite probabilities")
        return {label: round(probability, 4)
                for label, probability in zip(CLASS_LABELS, probabilities.cpu().tolist())}

    def evaluate(self, ecg_path: str | Path) -> list[str]:
        """Match the classifier output presented to ECG-Agent's dialogue model."""
        outputs = self(ecg_path)
        findings = sorted(
            ((label, probability) for label, probability in outputs.items()
             if probability > PROBABILITY_THRESHOLD),
            key=lambda item: item[1], reverse=True)
        return [f"{label} ({probability:.2%})" for label, probability in findings] or [
            f"None > {PROBABILITY_THRESHOLD:.0%}"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", help="PyTorch device (default: CUDA when available, else CPU)")
    parser.add_argument("ecg_path", type=Path)
    args = parser.parse_args()
    classifier = Classifier(args.checkpoint, args.device)
    print(json.dumps(classifier.evaluate(args.ecg_path), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
