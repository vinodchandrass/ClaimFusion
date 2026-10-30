import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODELS = {
    "frozen": "S7_NLI_Model",
    "weighted": "CPU_Domain_Adaptation_Output/checkpoint_weighted",
    "unweighted": "CPU_Domain_Adaptation_Output/checkpoint_unweighted",
}
LENGTHS = (192, 512)
KEYS = ["claim_id", "doc_id", "doc_rank", "sentence_index"]
LABELS = ("contradiction", "entailment", "neutral")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"D:\DataLineage"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    root = args.root
    source = root / "CrossDomain_Fusion_Output" / "adapted_scifact_candidates_DEV_ONLY.csv"
    output = root / "S118_Matched_Inference_Output"
    output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source, low_memory=False)
    required = KEYS + ["claim", "sentence"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if len(df) != 4500 or df.claim_id.nunique() != 300:
        raise ValueError("Unexpected candidate dimensions")
    if df[KEYS].isna().any().any() or df.duplicated(KEYS).any():
        raise ValueError("Invalid candidate keys")
    if df[["claim", "sentence"]].isna().any().any():
        raise ValueError("Missing input text")

    for _, group in df.groupby("claim_id"):
        if len(group) != 15 or group.doc_id.nunique() != 5:
            raise ValueError("Unexpected retrieval structure")

    input_hash = sha256(source)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    device = torch.device(args.device)
    manifest = {
        "input_sha256": input_hash,
        "rows": len(df),
        "conditions": {},
    }

    for model_name, relative_path in MODELS.items():
        directory = root / relative_path
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        label_map = {
            int(k): str(v).lower()
            for k, v in config["id2label"].items()
        }
        if set(label_map.values()) != set(LABELS):
            raise ValueError(f"Unexpected labels: {model_name}")

        weights = directory / "model.safetensors"
        weight_hash = sha256(weights)
        tokenizer_hash = sha256(directory / "tokenizer.json")

        tokenizer = AutoTokenizer.from_pretrained(
            str(directory), local_files_only=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(directory),
            local_files_only=True,
            use_safetensors=True,
        ).to(device).eval()

        for max_length in LENGTHS:
            name = f"{model_name}_L{max_length}"
            final = output / f"{name}.csv"
            partial = output / f"{name}.partial.csv"
            state = output / f"{name}.state.json"

            expected = {
                "input_sha256": input_hash,
                "weight_sha256": weight_hash,
                "tokenizer_sha256": tokenizer_hash,
                "max_length": max_length,
                "model": model_name,
            }

            if final.exists():
                old = json.loads(state.read_text()) if state.exists() else {}
                saved = pd.read_csv(final)
                if any(old.get(k) != v for k, v in expected.items()):
                    raise ValueError(f"Provenance mismatch: {name}")
                if len(saved) != len(df) or not saved[KEYS].equals(df[KEYS]):
                    raise ValueError(f"Alignment mismatch: {name}")
                print("SKIP completed", name, flush=True)
                manifest["conditions"][name] = {
                    "status": "complete",
                    "sha256": sha256(final),
                }
                continue

            done = 0
            if partial.exists():
                if not state.exists():
                    raise ValueError(f"Missing resume state: {name}")
                old = json.loads(state.read_text())
                if any(old.get(k) != v for k, v in expected.items()):
                    raise ValueError(f"Resume mismatch: {name}")
                previous = pd.read_csv(partial)
                done = len(previous)
                if done > len(df) or not previous[KEYS].equals(
                    df.iloc[:done][KEYS].reset_index(drop=True)
                ):
                    raise ValueError(f"Resume alignment failed: {name}")

            state.write_text(json.dumps({
                **expected, "status": "running"
            }, indent=2))

            print(f"RUN {name}: {done}/4500", flush=True)
            start = time.time()

            for i in range(done, len(df), args.batch_size):
                batch = df.iloc[i:i + args.batch_size]

                tokens = tokenizer(
                    batch["sentence"].astype(str).tolist(),
                    batch["claim"].astype(str).tolist(),
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                tokens = {k: v.to(device) for k, v in tokens.items()}

                with torch.inference_mode():
                    logits = model(**tokens).logits
                    probabilities = torch.softmax(
                        logits.float(), dim=-1
                    ).cpu().numpy()

                records = batch[KEYS].copy()
                for index, label in label_map.items():
                    records[f"p_{label}"] = probabilities[:, index]

                records.to_csv(
                    partial,
                    mode="a",
                    header=(i == 0),
                    index=False,
                    float_format="%.10g",
                )

                if (i // args.batch_size) % 25 == 0:
                    print(
                        name, min(i + len(batch), len(df)),
                        "/ 4500", flush=True
                    )

            completed = pd.read_csv(partial)
            if len(completed) != len(df) or not completed[KEYS].equals(df[KEYS]):
                raise ValueError(f"Final alignment failed: {name}")

            probs = completed[
                [f"p_{label}" for label in LABELS]
            ].to_numpy(dtype=float)

            if not np.isfinite(probs).all() or not np.allclose(
                probs.sum(axis=1), 1.0, atol=2e-5
            ):
                raise ValueError(f"Invalid probabilities: {name}")

            os.replace(partial, final)
            state.write_text(json.dumps({
                **expected,
                "status": "complete",
                "output_sha256": sha256(final),
                "elapsed_seconds": round(time.time() - start, 2),
            }, indent=2))

            manifest["conditions"][name] = {
                "status": "complete",
                "sha256": sha256(final),
            }
            (output / "manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )
            print("DONE", name, flush=True)

        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print("ALL SIX CONDITIONS COMPLETE", flush=True)

if __name__ == "__main__":
    main()

