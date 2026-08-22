#!/usr/bin/env python3
"""拒识三组对照：不做分离 / 一次分离 / 多次（级联）分离。

对每个样本同时打三种 score，各自扫 thr；中间轨 wav 写入
  VE_OUT/sep_streams/d1/{split}/{uid}/*.wav
  VE_OUT/sep_streams/d2/{split}/{uid}/*.wav

用法:
  python scripts/compare_sep_reject.py \\
    --samples $VE_OUT/manifest/samples.jsonl \\
    --out-dir /root/autodl-tmp/ve_gate_cmp \\
    --save-sep-wavs --limit 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from audio_io import load_audio
from calibrate_presence import stratified_limit, sweep_thresholds
from paths import (
    default_cohort_dir,
    default_eres2net_dir,
    default_spk_chs_dir,
    default_test_cohort_dir,
    default_ve_out,
    ensure_dir,
    normalize_presence_label,
    setup_sys_path,
)
from presence_encoder import create_presence_encoder
from presence_gate import PresenceGate, try_create_onnx_separator

setup_sys_path()

DEPTHS = (0, 1, 2)
DEPTH_LABEL = {
    0: "no_sep",
    1: "sep_once",
    2: "sep_multi",
}


def load_samples(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Presence 拒识：无分离/一次/多次 对照")
    p.add_argument("--samples", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--presence-backend", default="eres2netv2")
    p.add_argument("--eres-dir", type=Path, default=None)
    p.add_argument("--spk-chs-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save-sep-wavs", action="store_true", default=True)
    p.add_argument("--no-save-sep-wavs", action="store_true")
    p.add_argument("--select-by", default="contest", choices=("contest", "frr"))
    p.add_argument("--target-frr", type=float, default=0.02)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--depths",
        default="0,1,2",
        help="逗号分隔，默认 0,1,2",
    )
    p.add_argument("--cohort-dir", type=Path, default=None)
    p.add_argument("--test-cohort-dir", type=Path, default=None)
    p.add_argument("--enroll-znorm", action="store_true")
    p.add_argument("--test-znorm", action="store_true")
    p.add_argument("--asnorm", action="store_true")
    p.add_argument("--cohort-per-spk", type=int, default=2)
    p.add_argument("--cohort-max-files", type=int, default=400)
    p.add_argument("--test-cohort-max-files", type=int, default=500)
    p.add_argument("--cohort-seed", type=int, default=0)
    p.add_argument("--znorm-eps", type=float, default=1e-3)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    save_wavs = bool(args.save_sep_wavs) and not bool(args.no_save_sep_wavs)
    depths = tuple(int(x.strip()) for x in args.depths.split(",") if x.strip())
    for d in depths:
        if d < 0:
            raise SystemExit(f"非法 depth={d}")

    ve_out = (args.out_dir or (default_ve_out().parent / "ve_gate_cmp")).resolve()
    ensure_dir(ve_out)
    samples_path = args.samples
    if samples_path is None:
        for cand in (
            ve_out / "manifest" / "samples.jsonl",
            default_ve_out() / "manifest" / "samples.jsonl",
            Path("/root/autodl-tmp/ve_ps4/manifest/samples.jsonl"),
            Path("/root/autodl-tmp/ve/manifest/samples.jsonl"),
        ):
            if cand.is_file():
                samples_path = cand
                break
        if samples_path is None:
            raise SystemExit("找不到 samples.jsonl，请 --samples 或先 build_manifest")
    samples_path = Path(samples_path).resolve()
    samples = load_samples(samples_path)
    if args.limit and args.limit > 0:
        samples = stratified_limit(samples, int(args.limit))

    n_pos = sum(
        1 for r in samples
        if normalize_presence_label(r.get("label"), split=r.get("split")) == "present"
    )
    n_neg = sum(
        1 for r in samples
        if normalize_presence_label(r.get("label"), split=r.get("split")) == "absent"
    )
    print(f"[INFO] out={ve_out}")
    print(
        f"[INFO] samples={len(samples)} (pos≈{n_pos} neg≈{n_neg}) "
        f"depths={depths} save_sep_wavs={save_wavs}"
    )
    if n_neg == 0:
        raise SystemExit(
            "当前子集没有 neg 样本，无法评估 RR/FAR。"
            "请不要用「只截断前 N 条」；已改为分层抽样，请更新脚本后重跑，"
            "或 LIMIT=0 全量。"
        )

    enc = create_presence_encoder(
        args.presence_backend,
        eres_dir=args.eres_dir or default_eres2net_dir(),
        resnet_dir=args.spk_chs_dir or default_spk_chs_dir(),
        device=args.device,
    )
    need_sep = any(d >= 1 for d in depths)
    sep = try_create_onnx_separator(peak=0.95, device=args.device) if need_sep else None
    if need_sep and sep is None:
        raise SystemExit("需要 MossFormer：./download_moss_onnx.sh + extract/scripts/mossformer2_onnx.py")

    # 归一化模式
    if args.asnorm or (
        (args.enroll_znorm or args.cohort_dir)
        and (args.test_znorm or args.test_cohort_dir)
    ):
        norm_mode = "asnorm"
    elif args.test_znorm or args.test_cohort_dir is not None:
        norm_mode = "test_znorm"
    elif args.enroll_znorm or args.cohort_dir is not None:
        norm_mode = "enroll_znorm"
    else:
        norm_mode = "raw"

    score_norm = None
    if norm_mode != "raw":
        from cohort_znorm import build_score_normalizer

        score_norm = build_score_normalizer(
            enc,
            mode=norm_mode,  # type: ignore[arg-type]
            enroll_dir=(
                Path(args.cohort_dir) if args.cohort_dir else default_cohort_dir()
            )
            if norm_mode in ("enroll_znorm", "asnorm")
            else None,
            test_dir=(
                Path(args.test_cohort_dir)
                if args.test_cohort_dir
                else default_test_cohort_dir()
            )
            if norm_mode in ("test_znorm", "asnorm")
            else None,
            enroll_per_spk=int(args.cohort_per_spk),
            enroll_max_files=int(args.cohort_max_files),
            test_max_files=int(args.test_cohort_max_files),
            seed=int(args.cohort_seed),
            eps=float(args.znorm_eps),
        )

    # 共用一个 Gate，改 sep_depth；enroll 嵌入只算一次
    gate = PresenceGate(
        enc,
        thr=0.0,
        use_sep=need_sep,
        separator=sep,
        sep_depth=max(depths) if need_sep else 0,
        score_normalizer=score_norm,
    )

    scored: dict[int, list[tuple[str, float]]] = {d: [] for d in depths}
    details: dict[int, list[dict[str, Any]]] = {d: [] for d in depths}

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    t0 = time.time()
    iterator = (
        tqdm(samples, desc="compare_sep", unit="utt", mininterval=0.5)
        if tqdm is not None
        else samples
    )
    for i, it in enumerate(iterator):
        enroll, sr = load_audio(it["enroll_wav"])
        cmd, _ = load_audio(it["cmd_wav"])
        uid = it["uid"]
        split = str(it.get("split", "x"))
        # 预热 enroll 缓存
        gate.cache_enroll(uid, enroll, sr)
        for d in depths:
            gate.sep_depth = d
            gate.use_sep = d >= 1
            if d >= 1 and gate.separator is None:
                raise SystemExit("sep missing")
            save_dir = None
            if save_wavs and d >= 1:
                save_dir = ve_out / "sep_streams" / f"d{d}" / split / uid
            pr = gate.score(
                enroll, cmd, enroll_key=uid, sr=sr, save_dir=save_dir
            )
            lab = normalize_presence_label(it.get("label"), split=split)
            scored[d].append((lab, pr.score))
            details[d].append(
                {
                    "uid": uid,
                    "label": lab,
                    "split": split,
                    "arm": DEPTH_LABEL.get(d, f"d{d}"),
                    **pr.to_dict(),
                }
            )
        if (i + 1) % 200 == 0 or (i + 1) == len(samples):
            msg = f"[INFO] {i + 1}/{len(samples)}"
            if tqdm is not None:
                tqdm.write(msg)
            else:
                print(msg, flush=True)

    summary_rows: list[dict[str, Any]] = []
    compare: dict[str, Any] = {
        "n_samples": len(samples),
        "n_present": n_pos,
        "n_absent": n_neg,
        "depths": list(depths),
        "presence_backend": enc.name,
        "save_sep_wavs": save_wavs,
        "score_norm": norm_mode,
        "elapsed_sec": round(time.time() - t0, 2),
        "arms": {},
    }
    if score_norm is not None:
        compare["norm_meta"] = score_norm.to_meta()

    for d in depths:
        label = DEPTH_LABEL.get(d, f"d{d}")
        arm_dir = ensure_dir(ve_out / "reports" / f"presence_calib_{label}")
        cal = sweep_thresholds(
            scored[d], target_frr=args.target_frr, select_by=args.select_by
        )
        cal["sep_depth"] = d
        cal["arm"] = label
        cal["presence_backend"] = enc.name
        (arm_dir / "calibration.json").write_text(
            json.dumps(cal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with (arm_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
            for row in details[d]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rec = cal["recommended"]
        thr_obj = {
            "presence_thr": rec["thr"],
            "frr": rec["frr"],
            "far": rec["far"],
            "rr": rec["rr"],
            "cer": rec["cer"],
            "contest_score": rec["contest_score"],
            "sep_depth": d,
            "arm": label,
            "backend": enc.name,
            "select_by": args.select_by,
            "score_norm": compare["score_norm"],
            "metric": "0.5*RR + 0.5*(1-CER) presence-only",
        }
        (arm_dir / "recommended_thr.json").write_text(
            json.dumps(thr_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        compare["arms"][label] = thr_obj
        summary_rows.append(
            {
                "arm": label,
                "sep_depth": d,
                "thr": rec["thr"],
                "rr": round(rec["rr"], 4),
                "frr": round(rec["frr"], 4),
                "far": round(rec["far"], 4),
                "contest_score": round(rec["contest_score"], 4),
                "mean_score_present": round(cal["mean_score_present"], 4),
                "mean_score_absent": round(cal["mean_score_absent"] or 0.0, 4),
            }
        )

    # 选 contest 最高臂
    best = max(summary_rows, key=lambda r: (r["contest_score"], -r["frr"]))
    compare["best_arm"] = best
    compare["table"] = summary_rows

    reports = ensure_dir(ve_out / "reports")
    (reports / "sep_reject_compare.json").write_text(
        json.dumps(compare, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# 拒识对照：无分离 / 一次分离 / 多次分离",
        "",
        f"- n={len(samples)} backend=`{enc.name}` score_norm=`{compare['score_norm']}` "
        f"save_sep_wavs={save_wavs}",
        f"- 中间音: `{ve_out}/sep_streams/d{{1,2}}/{{split}}/{{uid}}/`",
        "",
        "| arm | depth | thr | RR | FRR | FAR | contest |",
        "|-----|-------|-----|----|-----|-----|---------|",
    ]
    for r in summary_rows:
        md.append(
            f"| {r['arm']} | {r['sep_depth']} | {r['thr']} | {r['rr']} | "
            f"{r['frr']} | {r['far']} | **{r['contest_score']}** |"
        )
    md += [
        "",
        f"**推荐臂**: `{best['arm']}` (depth={best['sep_depth']}, "
        f"contest={best['contest_score']}, thr={best['thr']})",
        "",
    ]
    if score_norm is not None:
        md.append(
            f"策略：score_norm=`{norm_mode}`；raw=`max_k sim`；`<thr` → absent。"
        )
    else:
        md.append(
            "策略：`score=max_k sim(enroll,stream_k)`；`<thr` → `speaker_absent`。"
        )
    md.append("depth2：对 d1 各轨再分离，取全部中间轨 max。")
    (reports / "sep_reject_compare.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(compare, ensure_ascii=False, indent=2))
    print(f"[OK] {reports / 'sep_reject_compare.md'}")
    if save_wavs:
        print(f"[OK] sep wavs → {ve_out / 'sep_streams'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
