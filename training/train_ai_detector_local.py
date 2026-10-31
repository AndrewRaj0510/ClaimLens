"""Train the AI-image detector on your own machine (CPU is enough), when
Colab has no GPU to give. Same steps and code as train_ai_detector.ipynb.

    pip install -r training/requirements-ai-detector.txt
    python training/train_ai_detector_local.py            # ~4.2 GB download

CLIP is frozen, so the only heavy step is one forward pass per image (~10
images/s on a laptop CPU). Tiny-GenImage's shards are each an even mix of all
generators (checked: every train shard has 143 fakes per generator and 1,000
reals), so a subset of shards is a fair sample: the defaults take 7 of the 14
train shards (~1,000 SD v1.5 fakes) and 2 of the 4 validation shards (250
fakes per generator).

Outputs, in data/models/ai-detector/ (git-ignored), which is what the
guardrails container mounts: clip_visual.onnx, probe.json, report.json.
Downloads and extracted features are cached under data/genimage-tiny/, so a
re-run skips finished work.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

import ai_detector as ad

ROOT = Path(__file__).resolve().parent.parent
REVISION = "89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69"  # pinned dataset commit
CLIP_MODEL = "openai/clip-vit-base-patch16"


def download(data_dir, train_shards, val_shards):
    from huggingface_hub import hf_hub_download

    names = [f"data/train-{i:05d}-of-00014.parquet" for i in range(train_shards)]
    names += [f"data/validation-{i:05d}-of-00004.parquet" for i in range(val_shards)]
    paths = []
    for name in names:
        print(f"shard {name}", flush=True)
        paths.append(
            hf_hub_download(
                ad.TINY_GENIMAGE,
                name,
                repo_type="dataset",
                revision=REVISION,
                local_dir=data_dir / "parquet",
            )
        )
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--train-shards", type=int, default=7, help="of 14")
    p.add_argument("--val-shards", type=int, default=2, help="of 4")
    p.add_argument("--n-train", type=int, default=1000, help="images per class")
    p.add_argument("--n-test", type=int, default=250, help="per class per generator")
    p.add_argument("--train-generator", default="sd15")
    p.add_argument("--max-fpr", type=float, default=0.01)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-shortcut-experiment", action="store_true")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "genimage-tiny")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "models" / "ai-detector")
    a = p.parse_args()
    started = time.time()

    # 1. Data, laid out like GenImage.
    images_dir = a.data / "images"
    if not images_dir.exists():
        parquet = download(a.data, a.train_shards, a.val_shards)
        print("writing images ...", flush=True)
        counts = ad.export_tiny_genimage(parquet, images_dir)
        print(json.dumps(counts), flush=True)
    gens = ad.find_generators(images_dir)
    train_gen = next(g for g in gens if a.train_generator in g.lower())
    test_gens = [g for g in gens if "val" in gens[g]]
    train = {
        label: ad.sample(
            ad.list_images(gens[train_gen]["train"][label]), a.n_train, a.seed
        )
        for label in ("ai", "nature")
    }
    test = {
        g: {
            label: ad.sample(ad.list_images(gens[g]["val"][label]), a.n_test, a.seed)
            for label in ("ai", "nature")
        }
        for g in test_gens
    }
    formats = ad.format_report(train)
    print(
        "train on", train_gen, {k: len(v) for k, v in train.items()}, "formats", formats
    )

    # 2. CLIP features (cached per set, keyed by the file list and settings).
    import torch
    from transformers import CLIPVisionModelWithProjection

    torch.set_num_threads(max(1, torch.get_num_threads()))
    clip = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL).eval()

    @torch.no_grad()
    def encode(x):
        return clip(pixel_values=torch.from_numpy(x)).image_embeds.float().numpy()

    cache = a.data / "features"
    cache.mkdir(parents=True, exist_ok=True)

    def features(name, paths, seed, aligned=True):
        f = cache / f"{name}-{'aligned' if aligned else 'raw'}-{len(paths)}-{seed}.npy"
        if f.exists():
            return np.load(f)
        t = time.time()
        x = ad.extract(paths, encode, batch_size=a.batch, seed=seed, aligned=aligned)
        np.save(f, x)
        print(
            f"  {name}: {len(paths)} images, {len(paths) / (time.time() - t):.1f}/s",
            flush=True,
        )
        return x

    X_ai = features(f"train-{train_gen}-ai", train["ai"], a.seed)
    X_real = features(f"train-{train_gen}-nature", train["nature"], a.seed + 1)
    F_test = {
        g: (
            features(f"val-{g}-ai", t["ai"], a.seed + 2),
            features(f"val-{g}-nature", t["nature"], a.seed + 3),
        )
        for g, t in test.items()
    }

    # 3. Probe, threshold at the target false-positive rate (on out-of-fold
    # scores of every training real), evaluation.
    X = np.r_[X_ai, X_real]
    y = np.r_[np.ones(len(X_ai)), np.zeros(len(X_real))]
    clf = ad.fit_probe(X, y, seed=a.seed)
    w, b = clf.coef_[0], float(clf.intercept_[0])
    oof = ad.out_of_fold_scores(X, y, clf.C_[0], seed=a.seed)
    threshold = ad.pick_threshold(oof[y == 0], a.max_fpr)
    results = {"cv_" + train_gen: ad.metrics(oof[y == 1], oof[y == 0], threshold)}
    for g, (fa, fr) in F_test.items():
        results[g] = ad.metrics(
            ad.probe_scores(fa, w, b), ad.probe_scores(fr, w, b), threshold
        )
    unseen = [m for g, m in results.items() if g in F_test and g != train_gen]
    summary = {
        k: round(float(np.mean([m[k] for m in unseen])), 4)
        for k in ("auc", "accuracy", "tpr", "fpr")
    }
    print(f"\nC = {clf.C_[0]}, threshold {threshold:.4f} (target FPR {a.max_fpr})")
    print(f"{'set':22s} {'AUC':>6s} {'acc':>6s} {'TPR':>6s} {'FPR':>6s}")
    for g, m in results.items():
        print(
            f"{g:22s} {m['auc']:6.3f} {m['accuracy']:6.3f} {m['tpr']:6.3f} {m['fpr']:6.3f}"
        )
    print("mean over unseen generators:", summary, flush=True)

    # 4. Optional: what the JPEG shortcut hides.
    shortcut = {}
    if not a.no_shortcut_experiment:
        n = min(500, len(train["ai"]), len(train["nature"]))
        Xs = np.r_[
            features(f"train-{train_gen}-ai", train["ai"][:n], a.seed, aligned=False),
            features(
                f"train-{train_gen}-nature", train["nature"][:n], a.seed, aligned=False
            ),
        ]
        ys = np.r_[np.ones(n), np.zeros(n)]
        raw = ad.fit_probe(Xs, ys, seed=a.seed)
        rw, rb = raw.coef_[0], float(raw.intercept_[0])
        raw_oof = ad.out_of_fold_scores(Xs, ys, raw.C_[0], seed=a.seed)
        raw_thr = ad.pick_threshold(raw_oof[ys == 0], a.max_fpr)
        print("\ntrained on original files (PNG fakes, JPEG reals), tested aligned:")
        for g, (fa, fr) in F_test.items():
            shortcut[g] = ad.metrics(
                ad.probe_scores(fa, rw, rb), ad.probe_scores(fr, rw, rb), raw_thr
            )
            m = shortcut[g]
            print(
                f"{g:22s} {m['auc']:6.3f} {m['accuracy']:6.3f} {m['tpr']:6.3f} {m['fpr']:6.3f}"
            )

    # 5. Export the encoder to ONNX and check it against PyTorch.
    import onnxruntime as ort

    class Visual(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            return self.model(pixel_values=pixel_values).image_embeds

    a.out.mkdir(parents=True, exist_ok=True)
    onnx_path = a.out / "clip_visual.onnx"
    visual = Visual(clip)
    torch.onnx.export(
        visual,
        (torch.zeros(1, 3, 224, 224),),
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    x = ad.load_batch(
        test[test_gens[0]]["ai"][:4] + test[test_gens[0]]["nature"][:4],
        random.Random(a.seed),
    )
    with torch.no_grad():
        ref = visual(torch.from_numpy(x)).numpy()
    got = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(
        None, {"pixel_values": x}
    )[0]

    def unit(f):
        return f / np.linalg.norm(f, axis=1, keepdims=True)

    diff = float(np.abs(unit(ref) - unit(got)).max())
    print(f"\nONNX vs PyTorch: max feature diff {diff:.2e}")
    assert diff < 1e-3, "ONNX export doesn't match PyTorch"

    # 6. Save.
    report = {
        "per_set": results,
        "mean_unseen": summary,
        "shortcut_unaligned_training": shortcut,
        "train_generator": train_gen,
        "n_train_per_class": len(train["ai"]),
        "n_test_per_class": a.n_test,
        "train_formats": formats,
        "max_fpr": a.max_fpr,
        "dataset": f"{ad.TINY_GENIMAGE}@{REVISION[:8]}, {a.train_shards}/14 train + {a.val_shards}/4 val shards",
        "onnx_max_feature_diff": diff,
        "minutes": round((time.time() - started) / 60, 1),
    }
    ad.save_probe(
        a.out / "probe.json",
        ad.probe_json(
            w,
            b,
            threshold,
            report,
            model_name=f"{CLIP_MODEL} + logistic probe (Tiny-GenImage {train_gen})",
            output_name="image_embeds",
            notes="Images JPEG-aligned before features; threshold set for the target FPR on out-of-fold scores of the training reals.",
        ),
    )
    (a.out / "report.json").write_text(json.dumps(report, indent=1))
    print(
        f"saved to {a.out} ({onnx_path.stat().st_size / 1e6:.0f} MB) in {report['minutes']} min"
    )


if __name__ == "__main__":
    main()
