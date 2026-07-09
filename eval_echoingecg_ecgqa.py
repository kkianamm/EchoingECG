import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.signal import resample_poly
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_qa(text):
    try:
        obj = json.loads(text)
    except Exception:
        obj = ast.literal_eval(text)

    qtype = obj[0]
    question = obj[1]
    answers = obj[2]
    return qtype, question, answers


def get_candidates(qtype, question):
    q = question.strip()

    if qtype == "single-verify":
        return ["yes", "no"]

    if qtype == "single-choose":
        # Example:
        # "Which form-related symptom does this ECG show, ST Depression or high amplitude T-waves?"
        m = re.search(r",\s*(.+?)(?:, including|, excluding|\?)", q, flags=re.I)
        if not m:
            return None

        part = m.group(1).strip()
        cands = [x.strip(" .?") for x in re.split(r"\s+or\s+", part) if x.strip()]
        if "none" not in [c.lower() for c in cands]:
            cands.append("none")
        return cands

    return None


def resolve_ecg_path(ecg_path, dataset_dir, ecg_root):
    raw = str(ecg_path).replace("\\", "/")
    raw = raw[2:] if raw.startswith("./") else raw

    candidates = [
        Path(raw),
        Path(dataset_dir) / raw,
        Path(ecg_root) / raw,
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def load_and_prepare_ecg(path, scaler, src_fs=250, target_fs=100, target_len=1000):
    x = np.load(path).astype(np.float32)

    # Accept either (12, T) or (T, 12)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D ECG array, got shape {x.shape}")

    if x.shape[0] != 12 and x.shape[1] == 12:
        x = x.T

    if x.shape[0] != 12:
        raise ValueError(f"Expected 12 leads, got shape {x.shape}")

    if src_fs != target_fs:
        x = resample_poly(x, target_fs, src_fs, axis=1).astype(np.float32)

    t = x.shape[1]

    if t > target_len:
        start = (t - target_len) // 2
        x = x[:, start:start + target_len]
    elif t < target_len:
        pad_left = (target_len - t) // 2
        pad_right = target_len - t - pad_left
        x = np.pad(x, ((0, 0), (pad_left, pad_right)), mode="constant")

    center = scaler.mean_.astype(np.float32)
    scale = np.maximum(scaler.scale_.astype(np.float32), 1e-8)
    x = (x - center[:, None]) / scale[:, None]

    return torch.from_numpy(x).unsqueeze(0)


@torch.no_grad()
def encode_ecg(model, ecg, device):
    ecg = ecg.to(device)
    out = model({"ecg": ecg})["ecg"]["mean"]
    return F.normalize(out, dim=-1)


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device):
    toks = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    toks = {k: v.to(device) for k, v in toks.items()}
    out = model({
        "text": toks["input_ids"],
        "attention_mask": toks["attention_mask"],
    })["text"]["mean"]
    return F.normalize(out, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", default=".")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--ecg_root", required=True)
    parser.add_argument("--split_file", default="data/fold1_test-00000-of-00001.parquet")
    parser.add_argument("--weights", default="weights/echoingecg.pt")
    parser.add_argument("--scaler", default="weights/ecg_scaler.pkl")
    parser.add_argument("--src_fs", type=int, default=250)
    parser.add_argument("--max_rows", type=int, default=1000)
    parser.add_argument("--out_csv", default="echoingecg_ecgqa_results.csv")
    parser.add_argument("--out_json", default="echoingecg_ecgqa_metrics.json")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    sys.path.insert(0, str(repo_dir))

    from src.model.echoingecg_model import EchoingECG

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(repo_dir / "src/configs/model.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)

    model = EchoingECG(model_cfg)
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    scaler = joblib.load(args.scaler)
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

    parquet_path = Path(args.dataset_dir) / args.split_file
    df = pd.read_parquet(parquet_path)

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows)

    rows = []
    total = 0
    correct = 0
    skipped = 0
    missing_ecg = 0

    by_type = {}

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            qtype, question, answers = parse_qa(row["text"])
            candidates = get_candidates(qtype, question)

            if candidates is None:
                skipped += 1
                continue

            gold = str(answers[0]).strip()

            ecg_file = resolve_ecg_path(row["ecg_path"], args.dataset_dir, args.ecg_root)
            if ecg_file is None:
                missing_ecg += 1
                continue

            ecg = load_and_prepare_ecg(
                ecg_file,
                scaler=scaler,
                src_fs=args.src_fs,
                target_fs=100,
                target_len=1000,
            )

            ecg_emb = encode_ecg(model, ecg, device)

            prompts = [
                f"Question: {question} Answer: {cand}."
                for cand in candidates
            ]
            text_emb = encode_texts(model, tokenizer, prompts, device)

            scores = (ecg_emb @ text_emb.T).squeeze(0)
            pred_idx = int(torch.argmax(scores).item())
            pred = candidates[pred_idx]

            is_correct = pred.lower() == gold.lower()
            total += 1
            correct += int(is_correct)

            by_type.setdefault(qtype, {"total": 0, "correct": 0})
            by_type[qtype]["total"] += 1
            by_type[qtype]["correct"] += int(is_correct)

            rows.append({
                "ecg_path": row["ecg_path"],
                "qtype": qtype,
                "question": question,
                "gold": gold,
                "prediction": pred,
                "correct": is_correct,
                "candidates": json.dumps(candidates),
                "scores": json.dumps([float(s) for s in scores.detach().cpu().tolist()]),
            })

        except Exception as e:
            skipped += 1
            rows.append({
                "ecg_path": row.get("ecg_path", ""),
                "qtype": "",
                "question": "",
                "gold": "",
                "prediction": "",
                "correct": False,
                "candidates": "",
                "scores": "",
                "error": str(e),
            })

    metrics = {
        "evaluated": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "skipped": skipped,
        "missing_ecg_files": missing_ecg,
        "by_type": {
            k: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": v["correct"] / v["total"] if v["total"] else None,
            }
            for k, v in by_type.items()
        },
    }

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved results to: {args.out_csv}")
    print(f"Saved metrics to: {args.out_json}")


if __name__ == "__main__":
    main()
