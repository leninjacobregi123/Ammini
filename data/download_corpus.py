"""Stream Malayalam pretraining text from verified public sources and write
it to sharded .txt files under data/raw/.

Sources (verified to exist on the HF Hub before writing this script):
  - wikimedia/wikipedia, config "20231101.ml"          (encyclopedic Malayalam)
  - ai4bharat/sangraha, config "verified", split "mal"  (large web-text corpus,
    ~12GB across 37 parquet files)
  - ai4bharat/IndicCorpV2, config "indiccorp_v2", split "mal_Mlym" (large
    monolingual news/web corpus, AI4Bharat's successor to IndicCorp v1)
  All three are streamed, so nothing is downloaded in full; use
  --max-mb-per-source to bound how much we actually pull.

  Deliberately left out: oscar-corpus/OSCAR-2301 (Malayalam subset exists but
  the dataset is gated behind manual approval on the HF Hub -- not worth the
  friction for an automated pipeline) and rajeshradhakrishnan/malayalam_news
  (uses a legacy loading script requiring trust_remote_code=True, i.e. it runs
  arbitrary code from the dataset repo -- skipped for a ~100MB corpus that
  doesn't move the needle next to the three sources above).

Usage:
    python data/download_corpus.py --out-dir data/raw --max-mb-per-source 500
"""
import argparse
from pathlib import Path

from datasets import load_dataset

SOURCES = [
    dict(name="wikipedia", hf_id="wikimedia/wikipedia", hf_config="20231101.ml",
         split="train", text_field="text"),
    dict(name="sangraha", hf_id="ai4bharat/sangraha", hf_config="verified",
         split="mal", text_field="text"),
    dict(name="indiccorp", hf_id="ai4bharat/IndicCorpV2", hf_config="indiccorp_v2",
         split="mal_Mlym", text_field="text"),
]


def stream_source(hf_id, hf_config, split, text_field, out_dir, max_mb, shard_mb=50):
    out_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = max_mb * 1024 * 1024
    shard_bytes = shard_mb * 1024 * 1024

    ds = load_dataset(hf_id, hf_config, split=split, streaming=True)

    written_total = 0
    shard_idx = 0
    shard_written = 0
    fh = None

    def open_shard():
        nonlocal fh, shard_idx, shard_written
        if fh is not None:
            fh.close()
        path = out_dir / f"{hf_id.replace('/', '__')}_{shard_idx:04d}.txt"
        fh = open(path, "w", encoding="utf-8")
        shard_written = 0
        return path

    open_shard()
    try:
        for example in ds:
            text = (example.get(text_field) or "").strip()
            if not text:
                continue
            text += "\n\n"
            encoded_len = len(text.encode("utf-8"))

            if shard_written + encoded_len > shard_bytes:
                shard_idx += 1
                open_shard()

            fh.write(text)
            shard_written += encoded_len
            written_total += encoded_len

            if written_total >= max_bytes:
                break
    finally:
        if fh is not None:
            fh.close()

    print(f"[{hf_id}] wrote {written_total / (1024 * 1024):.1f} MB across {shard_idx + 1} shard(s) -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/raw")
    ap.add_argument("--max-mb-per-source", type=float, default=500,
                     help="cap per source, in MB, to keep disk usage bounded. Raise this "
                          "on Shannon once the pipeline is proven; sangraha alone has ~12GB available.")
    ap.add_argument("--sources", nargs="+", choices=[s["name"] for s in SOURCES],
                     default=[s["name"] for s in SOURCES])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    for src in SOURCES:
        if src["name"] not in args.sources:
            continue
        print(f"Streaming {src['hf_id']} (config={src['hf_config']}, split={src['split']}) ...")
        stream_source(
            hf_id=src["hf_id"], hf_config=src["hf_config"], split=src["split"],
            text_field=src["text_field"], out_dir=out_dir, max_mb=args.max_mb_per_source,
        )


if __name__ == "__main__":
    main()
