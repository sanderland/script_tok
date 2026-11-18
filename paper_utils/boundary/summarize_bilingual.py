"""Collect completed local bilingual runs into a compact, versionable report."""

import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--vocab", type=int, default=34685, help="Exclude small integration-test vocabularies")
    budget.add_argument("--learned-vocab", type=int, help="Compare equal learned budgets, excluding fixed atomics")
    args = parser.parse_args()
    rows = []
    for path in sorted((ROOT / "results/boundary_bilingual").glob("*/*.analysis.json")):
        result = json.loads(path.read_text())
        training = json.loads(path.with_name(result["arm"] + ".train.json").read_text())
        analysis = result["analysis"]
        vocabulary = analysis["vocabulary"]
        if args.learned_vocab is not None:
            if vocabulary["learned_tokens"] != args.learned_vocab:
                continue
        elif vocabulary["total_tokens"] != args.vocab:
            continue
        evaluation = analysis["evaluation"]
        rss_path = path.parent / "rss_highwater.json"
        rss = json.loads(rss_path.read_text())["stages"].get(f"train:{result['arm']}") if rss_path.exists() else None
        rows.append({"corpus": result["corpus"], "arm": result["arm"], "trainer": training["trainer"],
                     "vocab": vocabulary["total_tokens"], "atomic": vocabulary["atomic_tokens"],
                     "learned_categories": vocabulary["categories"],
                     "duplicates": vocabulary["duplicates"], "evaluation": evaluation,
                     "evaluation_provenance": result["evaluation"],
                     "training": training, "training_memory": rss})
    if not rows:
        raise SystemExit("No completed bilingual analyses")
    order = {arm: i for i, arm in enumerate(("plain", "bnd_w", "bnd_w_caps", "bnd_wp", "bnd_wp_caps", "bnd_wpd", "bnd_wpd_caps"))}
    rows.sort(key=lambda row: (row["corpus"], row["vocab"], row["trainer"], order.get(row["arm"], 99)))
    out = ROOT / "paper_utils/boundary/paper/generated"
    suffix = f"_learned{args.learned_vocab}" if args.learned_vocab is not None else (f"_v{args.vocab}" if args.vocab != 34685 else "")
    (out / f"bilingual_results{suffix}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    lines = ["# English/Korean shared-vocabulary ablation", "",
             "CPU BPE/MinGram ablation using prefixes of the local FineWeb quick caches. The 20M-character runs are exploratory pilots. "
             "All schemes share the same text within a mixture. Budgets and vocabulary sizes are recorded below. "
             "Evaluation uses fixed prefixes of the cached English and Korean Goldfish slices. "
             "Compression counts characters after the tokenizer's normalization. "
             "Mixture ratios count characters, not documents or UTF-8 bytes.", "",
             "Allocation counts below cover learned pieces only. Latin/Hangul are script labels, not exclusive language ownership. "
             "Duplicate surplus counts use n−1 per equivalent group, retaining boundary markers. "
             "Space/case columns overlap and must not be added. These counts do not measure all possible positional redundancy. "
             "Space surplus includes whitespace-only pairs, such as newline versus space+newline; per-script counts are in the JSON. "
             "Individual SCRIPT atomics may be undecodable fragments, so usage in that category does not imply failed text round trips.", "",
             "| Train chars | Trainer | Scheme | Latin | Hangul | Other scripts | Nonlexical | Atomics | English chars/token | Korean chars/token |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        budgets = row["training"]["sample"]["budgets"]
        cats, ev = row["learned_categories"], row["evaluation"]
        percent = round(100 * budgets["en"] / sum(budgets.values()))
        lines.append(f"| {sum(budgets.values()):,} ({percent}% en) | {row['trainer']} | {row['arm']} | {cats['latin']:,} | {cats['hangul']:,} | "
                     f"{cats['other_script']:,} | {cats['nonlexical']:,} | {row['atomic']:,} | "
                     f"{ev['en']['chars_per_token']:.4f} | {ev['ko']['chars_per_token']:.4f} |")
    lines += ["", "## Paired scheme comparisons", ""]
    groups = {}
    for row in rows:
        comparison_budget = args.learned_vocab if args.learned_vocab is not None else row["vocab"]
        groups.setdefault((row["corpus"], comparison_budget, row["trainer"]), {})[row["arm"]] = row
    for (corpus, vocab, trainer), arms in groups.items():
        budget_label = "learned vocabulary" if args.learned_vocab is not None else "total vocabulary"
        lines.append(f"### {corpus}, {trainer}, {budget_label} {vocab:,}")
        lines.append("")
        if "plain" in arms:
            base = arms["plain"]
            for arm, row in arms.items():
                if arm == "plain":
                    continue
                changes = {lang: 100 * (row["evaluation"][lang]["chars_per_token"] / base["evaluation"][lang]["chars_per_token"] - 1)
                           for lang in ("en", "ko")}
                lines.append(f"- `{arm}` versus plain: English compression {changes['en']:+.3f}%, Korean {changes['ko']:+.3f}%. Higher is better.")
                delta = {script: row["learned_categories"][script] - base["learned_categories"][script] for script in ("latin", "hangul", "other_script", "nonlexical")}
                lines.append(f"  Allocation versus plain: {delta}.")
        for scope in ("bnd_w", "bnd_wp", "bnd_wpd"):
            if scope not in arms or scope + "_caps" not in arms:
                continue
            before, after = arms[scope], arms[scope + "_caps"]
            delta = {script: after["learned_categories"][script] - before["learned_categories"][script]
                     for script in ("latin", "hangul")}
            change = 100 * (after["evaluation"]["ko"]["chars_per_token"] / before["evaluation"]["ko"]["chars_per_token"] - 1)
            lines.append(f"- Adding case codes to `{scope}` changes Latin allocation by {delta['latin']:+,} learned slots and Hangul by {delta['hangul']:+,}; Korean compression changes {change:+.3f}%.")
        lines.append("")
    lines += ["## MinGram versus BPE allocation", ""]
    matched = {}
    for row in rows:
        matched.setdefault((row["corpus"], row["vocab"], row["arm"]), {})[row["trainer"]] = row
    for (corpus, vocab, arm), trainers in matched.items():
        if "bpe" not in trainers or "mingram" not in trainers:
            continue
        before, after = trainers["bpe"], trainers["mingram"]
        if before["training"]["sample"] != after["training"]["sample"] or before["evaluation_provenance"] != after["evaluation_provenance"]:
            raise ValueError(f"Mismatched inputs for trainer comparison: {corpus} {arm}")
        delta = {script: after["learned_categories"][script] - before["learned_categories"][script]
                 for script in ("latin", "hangul", "other_script", "nonlexical")}
        lines.append(f"- {corpus}, {vocab:,} tokens, `{arm}`: MinGram minus BPE learned slots {delta}.")
    failures = sum(ev["roundtrip_failures"] for row in rows for ev in row["evaluation"].values())
    lines += ["", f"Round-trip failures across completed runs: {failures}.", "",
              "This experiment measures tokenizer allocation and compression. It does not test LM quality, "
              "establish generality across training samples, or isolate English-only changes: boundary markers affect both scripts. "
              "The case-code contrast is more specific to cased text, but Korean documents can contain Latin text.", "",
              f"Full counts, usage frequencies, sample hashes, vocabulary hashes, and timings are in `bilingual_results{suffix}.json`.", ""]
    (out / f"bilingual_report{suffix}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
