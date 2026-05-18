from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .board_encoding import EXTRA_FEATURES_DIM, encode_fen, pretty_label_from_eval, validate_fen
from .dataset import adjust_extra_features, inverse_target_to_cp
from .model import create_model
from .train import split_model_output
from .utils import BUCKET_LABELS, apply_bucket_logit_bias, get_device, normalize_state_dict_keys


def _load_for_prediction(model_path: str | Path, device: torch.device):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train first with: uv run python -m src.train")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})
    model_name = str(config.get("model_name", "cnn"))
    if model_name == "cnn" and "model_width" not in config:
        model_name = "legacy_cnn"
    model = create_model(
        model_name=model_name,
        extra_features_dim=int(config.get("extra_features_dim", EXTRA_FEATURES_DIM)),
        width=int(config.get("model_width", 192)),
        depth=int(config.get("model_depth", 8)),
        dropout=float(config.get("dropout", 0.15)),
        head_hidden=int(config.get("head_hidden", 512)),
        num_buckets=int(config.get("num_buckets", 0)),
    ).to(device)
    model.load_state_dict(normalize_state_dict_keys(checkpoint["model_state_dict"], model.state_dict()), strict=False)
    model.eval()
    return model, config


class ChessEvaluationPredictor:
    def __init__(self, model_path: str | Path = "models/best_model.pt", device: str = "auto"):
        self.model_path = Path(model_path)
        self.device = get_device(device)
        self.model, self.config = _load_for_prediction(self.model_path, self.device)
        self.eval_clip = float(self.config.get("eval_clip", 1500.0))
        self.target_transform = str(self.config.get("target_transform", "linear"))
        self.target_scale = float(self.config.get("target_scale", 600.0))
        self.extra_features_dim = int(self.config.get("extra_features_dim", EXTRA_FEATURES_DIM))

    def predict(self, fen: str) -> dict[str, object]:
        if not validate_fen(fen):
            raise ValueError(f"Invalid FEN: {fen}")
        board_np, extras_np = encode_fen(fen)
        extras_np = adjust_extra_features(extras_np, self.extra_features_dim)
        board = torch.from_numpy(board_np).unsqueeze(0).to(self.device)
        extras = torch.from_numpy(extras_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(board, extras)
            pred, bucket_logits = split_model_output(output)
            pred_norm = float(pred.item())
        cp = float(
            inverse_target_to_cp(
                np.array([pred_norm], dtype=np.float32),
                self.target_transform,
                self.eval_clip,
                self.target_scale,
            )[0]
        )
        label = pretty_label_from_eval(cp)
        bucket_probs = None
        if bucket_logits is not None:
            logits_np = bucket_logits.squeeze(0).detach().cpu().numpy().reshape(1, 3)
            adjusted_logits = apply_bucket_logit_bias(logits_np, self.config.get("bucket_logit_bias"))
            probs = torch.softmax(torch.from_numpy(adjusted_logits), dim=-1).squeeze(0).numpy()
            labels = BUCKET_LABELS
            label = labels[int(np.argmax(probs))]
            bucket_probs = {name: float(prob) for name, prob in zip(labels, probs)}
        return {
            "predicted_cp": cp,
            "centipawns": cp,
            "normalized_prediction": pred_norm,
            "normalized": pred_norm,
            "raw_model_output": pred_norm,
            "label": label,
            "regression_label": pretty_label_from_eval(cp),
            "bucket_probabilities": bucket_probs,
            "explanation": "Positive means White is better. Negative means Black is better.",
        }


def predict_fen(fen: str, model_path: str = "models/best_model.pt", device: str = "auto") -> dict[str, object]:
    return ChessEvaluationPredictor(model_path=model_path, device=device).predict(fen)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a static chess position evaluation from FEN.")
    parser.add_argument("--fen", required=True)
    parser.add_argument("--model", default="models/best_model.pt")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        result = predict_fen(args.fen, args.model, args.device)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Predicted evaluation: {result['centipawns']:.1f} centipawns")
    print(f"Label: {result['label']}")
    print(result["explanation"])


if __name__ == "__main__":
    main()
