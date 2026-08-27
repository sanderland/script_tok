"""Download tokenizer.json for every std-1B tokeval run; externals from their own repos."""
import csv
import json
import os
from huggingface_hub import hf_hub_download

SP = os.environ.get("TOKEVAL_WORKDIR", ".")
EXTERNAL = {"swiss-ai/Apertus-70B-2509": "swiss-ai/Apertus-70B-2509",
            "NousResearch/Meta-Llama-3-8B": "NousResearch/Meta-Llama-3-8B"}

rows = list(csv.DictReader(open(f"{SP}/tokeval/ablation_grid.csv")))
panel = [r for r in rows if 'tokeval' in r['works'].split(';') and r['training_regime'] == 'std-1B' and r['val_bpb']]
out, missing = {}, []
for r in panel:
    run, slug = r['run'], r['tokenizer_slug']
    try:
        if r['tok_redistributable'] == 'True':
            p = hf_hub_download('cmeister/tokenizer-lm-ablations', f'models/{run}/tokenizer.json',
                                local_dir=f'{SP}/tokeval')
        elif slug in EXTERNAL:
            p = hf_hub_download(EXTERNAL[slug], 'tokenizer.json', local_dir=f'{SP}/tokeval/external/{slug.replace("/","_")}')
        else:
            missing.append(run)
            continue
        out[run] = {"path": p, "slug": slug, "val_bpb": float(r['val_bpb']),
                    "flores_mean_bpb": float(r['flores_mean_bpb'] or 0)}
    except Exception as e:
        missing.append(f"{run}: {type(e).__name__} {str(e)[:60]}")
json.dump(out, open(f"{SP}/panel_tokenizers.json", "w"), indent=1)
print(f"got {len(out)} tokenizers; missing: {missing}")
