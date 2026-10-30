from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

root = Path(r"D:\DataLineage")
source = root / "CrossDomain_Fusion_Output" / "adapted_scifact_candidates_DEV_ONLY.csv"
out = root / "S121_Quick_Validation"
out.mkdir(exist_ok=True)

df = pd.read_csv(source).head(100).copy()

models = {
    "frozen": (
        root / "S7_NLI_Model",
        ["p_contradiction", "p_entailment", "p_neutral"]
    ),
    "weighted": (
        root / "CPU_Domain_Adaptation_Output/checkpoint_weighted",
        ["adapted_p_contradiction", "adapted_p_entailment", "adapted_p_neutral"]
    )
}

for name, (path, historical_columns) in models.items():
    print(f"\nTesting {name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(path), local_files_only=True
    ).eval()

    labels = {
        int(k): str(v).lower()
        for k, v in model.config.id2label.items()
    }

    for length in [192, 512]:
        predictions = []

        for start in range(0, len(df), 8):
            batch = df.iloc[start:start + 8]

            tokens = tokenizer(
                batch["sentence"].astype(str).tolist(),
                batch["claim"].astype(str).tolist(),
                padding=True,
                truncation=True,
                max_length=length,
                return_tensors="pt"
            )

            with torch.inference_mode():
                probabilities = torch.softmax(
                    model(**tokens).logits.float(), dim=-1
                ).numpy()

            predictions.append(probabilities)

        predictions = np.concatenate(predictions)

        result = df[[
            "claim_id", "doc_id", "doc_rank", "sentence_index"
        ]].copy()

        for index, label in labels.items():
            result[f"new_p_{label}"] = predictions[:, index]

        historical = df[historical_columns].to_numpy(dtype=float)
        ordered = np.column_stack([
            result[f"new_p_{label}"].to_numpy()
            for label in ["contradiction", "entailment", "neutral"]
        ])

        mae = np.abs(ordered - historical).mean()
        max_difference = np.abs(ordered - historical).max()
        close_rows = np.all(
            np.isclose(ordered, historical, atol=1e-4, rtol=0),
            axis=1
        ).sum()

        print(
            f"{name} length={length}: "
            f"MAE={mae:.6f}, "
            f"maximum_difference={max_difference:.6f}, "
            f"matching_rows={close_rows}/100",
            flush=True
        )

        result.to_csv(
            out / f"{name}_L{length}_100rows.csv",
            index=False
        )

print("\nS121 QUICK VALIDATION COMPLETE")
