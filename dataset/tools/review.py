#!/usr/bin/env python3
"""
Fast native-review tool for generated/collected conversation pairs.

Shows one pair at a time; you press:
    a  accept as-is
    f  fix   (edit the user and/or assistant text, Enter keeps current)
    r  reject
    s  skip  (decide later)
    q  quit  (progress is saved; rerun to continue)

Resumable: each pair gets a stable id; reviewed ids are remembered, so you and your
friends can stop/restart and split the work. Accepted+fixed go to a training-ready file.

Usage:
    python dataset/tools/review.py --in aigenerateddataset/cs_pairs.jsonl --reviewer jihed
    python dataset/tools/review.py --in dataset/collected/native_pairs.jsonl --reviewer ali
"""
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

USER_KEYS = ("instruction", "prompt", "user", "customer")
ASST_KEYS = ("output", "response", "assistant", "agent")


def get(d, keys):
    for k in keys:
        if d.get(k):
            return d[k]
    return ""


def pair_id(rec) -> str:
    return hashlib.sha1((get(rec, USER_KEYS) + "||" + get(rec, ASST_KEYS)).encode()).hexdigest()[:12]


def load_jsonl(p: Path):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def append_jsonl(p: Path, rec):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="aigenerateddataset/cs_pairs.jsonl")
    ap.add_argument("--reviewer", required=True, help="your name (logged on each decision)")
    ap.add_argument("--out-dir", default="dataset/reviewed")
    args = ap.parse_args()

    inp = (ROOT / args.inp) if not Path(args.inp).is_absolute() else Path(args.inp)
    if not inp.exists():
        print(f"Input not found: {inp}\n(Generate pairs first, e.g. "
              f"python aigenerateddataset/generate_cs_dataset.py --target 50)")
        return
    out_dir = (ROOT / args.out_dir)
    accepted_f = out_dir / "accepted.jsonl"
    rejected_f = out_dir / "rejected.jsonl"
    progress_f = out_dir / "progress.json"

    progress = json.loads(progress_f.read_text()) if progress_f.exists() else {"reviewed": {}}
    reviewed = progress["reviewed"]

    records = load_jsonl(inp)
    todo = [r for r in records if pair_id(r) not in reviewed]
    print(f"\n{len(records)} pairs total | {len(reviewed)} already reviewed | "
          f"{len(todo)} left. Reviewer: {args.reviewer}")
    print("Keys: [a]ccept  [f]ix  [r]eject  [s]kip  [q]uit\n" + "-" * 60)

    counts = {"accept": 0, "fix": 0, "reject": 0, "skip": 0}
    for n, rec in enumerate(todo, 1):
        pid = pair_id(rec)
        u, a = get(rec, USER_KEYS), get(rec, ASST_KEYS)
        cat = rec.get("category", ""); topic = rec.get("topic", "")
        print(f"\n[{n}/{len(todo)}] id={pid}  {cat}/{topic}")
        print(f"  USER : {u}")
        print(f"  ASST : {a}")
        try:
            choice = input("  > [a/f/r/s/q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "q"

        if choice == "q":
            print("\nsaved. rerun to continue.")
            break
        elif choice == "s":
            counts["skip"] += 1
            continue
        elif choice == "r":
            append_jsonl(rejected_f, {**rec, "reviewer": args.reviewer})
            reviewed[pid] = {"by": args.reviewer, "v": "reject"}
            counts["reject"] += 1
        elif choice == "f":
            nu = input(f"  fix USER [{u}]: ").strip() or u
            na = input(f"  fix ASST [{a}]: ").strip() or a
            out = {**rec, "instruction": nu, "output": na,
                   "reviewer": args.reviewer, "review": "fixed", "human_reviewed": True}
            append_jsonl(accepted_f, out)
            reviewed[pid] = {"by": args.reviewer, "v": "fix"}
            counts["fix"] += 1
        else:  # default accept
            out = {**rec, "instruction": u, "output": a,
                   "reviewer": args.reviewer, "review": "accepted", "human_reviewed": True}
            append_jsonl(accepted_f, out)
            reviewed[pid] = {"by": args.reviewer, "v": "accept"}
            counts["accept"] += 1

        progress_f.parent.mkdir(parents=True, exist_ok=True)
        progress_f.write_text(json.dumps(progress, ensure_ascii=False))

    kept = counts["accept"] + counts["fix"]
    done = sum(counts.values())
    print("\n" + "=" * 60)
    print(f"this session: {counts}")
    if done:
        print(f"accept-rate (accept+fix / decided): {kept}/{done - counts['skip']} = "
              f"{kept/max(1, done-counts['skip']):.0%}")
    print(f"accepted file -> {accepted_f.relative_to(ROOT)} "
          f"({len(load_jsonl(accepted_f)) if accepted_f.exists() else 0} total)")
    print("GATE: if accept-rate < 80%, fix the generator prompt before scaling up.")


if __name__ == "__main__":
    main()
