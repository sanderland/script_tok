"""Multilingual, all-scheme loss attribution on identical raw-text spans.

Counts are optional. Source bytes and all model losses are partitioned exactly,
including structural tokens. Documents not round-tripping in any tokenizer are
excluded for every model and audited. This evaluates common document prefixes,
not nanochat's original tokenizer-dependent packing/cropping protocol.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import regex
import torch

import paper_utils.boundary.downstream.boundary_tokenizer  # registration
import paper_utils.boundary.legacy_extcaps_pretokenizer  # published English caps
from script_bpe.tokenizers.load import load_tokenizer

VERSION = 1
VARIANTS = ('plain', 'bnd_w', 'bnd_wpd', 'bnd_wpd_caps')
UNITS = regex.compile(r' ?[\p{L}\p{M}]+| ?\p{N}+| ?[^\p{L}\p{M}\p{N}\s]+|\s+')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, obj):
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def source_offsets(tok, text, ids):
    """Trace the decoder, assigning structural token costs to their source unit."""
    pt = tok.pretokenizer
    atoms, owners = [], []
    for owner, tid in enumerate(ids):
        seq = tok.tokens[tid].atomic_tokens
        atoms.extend(seq)
        owners.extend([owner]*len(seq))
    chunk_ends, cursor = set(), 0
    flat_chunks = []
    for chunk in pt.pretokenize(text):
        flat_chunks.extend(chunk)
        cursor += len(chunk)
        chunk_ends.add(cursor-1)
    assert flat_chunks == atoms, 'Atomic encoding differs from tokenization'
    starts = np.full(len(ids), len(text), dtype=np.int64)
    ends = np.zeros(len(ids), dtype=np.int64)
    marker = getattr(pt, 'marker_token_id', None)
    codes = getattr(pt, 'caps_code_ids', set())
    digits = getattr(pt, 'digit_token_ids', set())
    pending, initial, pos, i = None, True, 0, 0
    pieces = []

    def touch(owner, a, b):
        starts[owner] = min(starts[owner], a)
        ends[owner] = max(ends[owner], b)

    def anchor(owner, point):
        point = min(max(0, point), len(text)-1)
        touch(owner, point, point+1)

    while i < len(atoms):
        a = atoms[i]
        if a in codes:
            pending, initial = a, True
            anchor(owners[i], pos)
            i += 1
            continue
        if a == marker:
            closing = i in chunk_ends
            anchor(owners[i], pos-1 if closing else pos)
            if closing:
                pending, initial = None, True
            j = i+1
            between = []
            while j < len(atoms) and atoms[j] in codes:
                between.append(j)
                j += 1
            if j < len(atoms) and atoms[j] == marker:
                pieces.append(' ')
                touch(owners[j], pos, pos+1)
                pos += 1
                anchor(owners[j], pos)
                for k in between:
                    anchor(owners[k], pos)
                    pending, initial = atoms[k], True
                i = j+1
            else:
                i += 1
            continue
        b = atoms[i+1] if i+1 < len(atoms) else None
        if a in digits:
            piece, step = pt.atomic_tokens[a], 1
        else:
            if (a,b) not in pt.detokenize_map:
                raise ValueError(f'Invalid atomic pair {(a,b)}')
            piece, step = pt.detokenize_map[a,b], 2
        if pending == getattr(pt, 'caps_token_id', None) and pending is not None:
            piece = piece.upper()
        elif pending is not None and initial:
            piece = piece[:1].upper()+piece[1:]
        initial = False
        for k in range(i,i+step):
            touch(owners[k], pos, pos+len(piece))
        pos += len(piece)
        pieces.append(piece)
        i += step
    assert ''.join(pieces) == text, 'Alignment decoder disagrees with source'
    assert (starts < ends).all() and (ends <= len(text)).all()
    return starts, ends


def classification(text):
    core = text.strip()
    if not core:
        return 'whitespace', 'uncased'
    if regex.fullmatch(r'[\p{L}\p{M}]+', core):
        kind = 'word'
    elif regex.fullmatch(r'\p{N}+', core):
        kind = 'number'
    elif regex.fullmatch(r'[^\p{L}\p{M}\p{N}\s]+', core):
        kind = 'punctuation_or_symbol'
    else:
        kind = 'mixed'
    case = ('uncased' if core.lower() == core.upper() else 'lower' if core.islower()
            else 'upper' if core.isupper() else 'title' if core.istitle() else 'mixed')
    return kind, case


def non_roundtripping_variants(toks, encodings, text):
    """Reject the document for all schemes if any decoder changes the source."""
    return [v for v,t in toks.items() if t.decode(encodings[v]) != text]


def aligned_rows(text, encodings, toks, counts, plain_forms=None):
    offsets = {v:source_offsets(toks[v],text,ids) for v,ids in encodings.items()}
    # Cut only at lexical-unit boundaries crossed by no token in any scheme.
    coverage = np.zeros(len(text)+2, dtype=np.int64)
    for starts,ends in offsets.values():
        for a,b in zip(starts,ends):
            if b > a+1:
                coverage[a+1] += 1
                coverage[b] -= 1
    blocked = np.cumsum(coverage)
    matches = list(UNITS.finditer(text))
    assert ''.join(m.group() for m in matches) == text
    cuts = [0]+[m.end() for m in matches if blocked[m.end()] == 0]
    assert cuts[-1] == len(text)
    rows = [{'start':a,'end':b,'bytes':len(text[a:b].encode('utf-8')),
             'kind':classification(text[a:b])[0], 'case':classification(text[a:b])[1],
             'slices':{}} for a,b in zip(cuts,cuts[1:])]
    for v,(starts,ends) in offsets.items():
        group = np.searchsorted(cuts, starts, side='right')-1
        assert np.all(group[1:] >= group[:-1]), 'Non-monotonic token grouping'
        for k,row in enumerate(rows):
            hits = np.flatnonzero(group == k)
            assert len(hits) and ends[hits].max() <= row['end']
            row['slices'][v] = [int(hits[0]), int(hits[-1])+1]
        assert rows[0]['slices'][v][0] == 0 and rows[-1]['slices'][v][1] == len(encodings[v])
    if plain_forms is None:
        plain_forms = {toks['plain'].pretokenizer.decode(t.atomic_tokens):tid for tid,t in toks['plain'].tokens.items()}
    for row in rows:
        a,b = row['slices']['plain']
        row['plain_tokens'] = b-a
        row['plain_whole_word'] = row['kind'] == 'word' and b-a == 1
        if counts and row['kind'] == 'word':
            core = text[row['start']:row['end']].strip()
            raw = text[row['start']:row['end']]
            # Whole-word baseline token frequency. Fragmented words stay explicitly unknown.
            if b-a == 1:
                tid = encodings['plain'][a]
                row['frequency'] = counts.get(tid)
                other = core if raw.startswith(' ') else ' '+core
                oid = plain_forms.get(other)
                if oid in counts and row['frequency'] is not None:
                    row['other_frequency'] = counts[oid]
    assert sum(r['bytes'] for r in rows) == len(text.encode('utf-8'))
    return rows


def load_counts(args, lang, tok, tokenizer_path):
    roots = [args.counts_root, Path('paper_utils/boundary/paper/generated/token_counts')]
    for root in roots:
        p = root/f'token_counts_{lang}_plain_mingram.json.gz'
        if p.is_file():
            d = json.load(gzip.open(p,'rt'))
            if d['tokenizer_sha256'] != sha(tokenizer_path) or d['ids'] != sorted(tok.tokens):
                raise ValueError(f'Counts/tokenizer mismatch: {p}')
            return dict(zip(d['ids'],d['counts'])),sha(p)
    if lang == 'en':
        p = Path('results/magikarp/training_vocabulary_counts.npz')
        if p.is_file():
            values = np.load(p)['counts'][0]
            assert len(values) == len(tok.tokens)
            return dict(zip(sorted(tok.tokens), map(int,values))),sha(p)
    print(f'{lang}: no verified plain counts found; frequency groups will be omitted.',flush=True)
    return {},None


def folder(args,lang,v):
    return args.models_root/lang/f'boundary-markers-{lang}-d12-{v}-mingram'


def download(args,lang,v,patterns):
    if not args.download_missing:
        raise FileNotFoundError(f'Missing model/tokenizer files in {folder(args,lang,v)}; use --download-missing to fetch published files')
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=f'cmeister/boundary-markers-{lang}-d12-{v}-mingram',
                      local_dir=str(folder(args,lang,v)),allow_patterns=patterns)


def prepare(args,lang,validation,out):
    toks, paths, remaps = {}, {}, {}
    for v in args.variants:
        candidates = list((folder(args,lang,v)/'tokenizer').glob('*.json.gz'))
        if not candidates:
            download(args,lang,v,['tokenizer/*.json.gz'])
            candidates = list((folder(args,lang,v)/'tokenizer').glob('*.json.gz'))
        if len(candidates) != 1:
            raise ValueError(f'Expected one tokenizer for {lang}/{v}')
        paths[v]=candidates[0]
        toks[v]=load_tokenizer(str(paths[v]))
        remaps[v]={tid:i for i,tid in enumerate(sorted(toks[v].tokens))}
    counts,count_hash=load_counts(args,lang,toks['plain'],paths['plain'])
    plain_forms = {toks['plain'].pretokenizer.decode(t.atomic_tokens):tid for tid,t in toks['plain'].tokens.items()}
    identity={'version':VERSION,'language':lang,'variants':args.variants,'docs':args.docs,
              'max_tokens':args.max_tokens,'max_chars':args.max_chars,
              'validation_sha256':sha(validation),'validation_path':str(validation.resolve()),
              'tokenizer_sha256':{v:sha(p) for v,p in paths.items()},'counts_sha256':count_hash}
    plan=out/'plan.json'
    if plan.is_file():
        saved=json.loads(plan.read_text())
        if saved['identity'] != identity:
            raise ValueError(f'Cache identity changed in {out}; choose a new --output-root')
        return saved
    docs,rejected=[],[]
    source_index=-1
    for batch in pq.ParquetFile(validation).iter_batches(batch_size=128,columns=['text']):
        for original in batch.column('text').to_pylist():
            source_index+=1
            text=original[:args.max_chars]
            while text:
                ids={v:t.encode(text) for v,t in toks.items()}
                longest=max(map(len,ids.values()))
                if longest > args.max_tokens:
                    text=text[:max(0,int(len(text)*args.max_tokens/longest*.95))]
                    continue
                if len(text)<len(original) and not text[-1:].isspace():
                    end=max((i+1 for i,c in enumerate(text) if c.isspace()),default=0)
                    if end != len(text):
                        text=text[:end]
                        continue
                break
            if not text or min(map(len,ids.values())) < 2:
                rejected.append({'source_document':source_index,'reason':'empty_or_too_short_prefix'})
                continue
            failed=non_roundtripping_variants(toks,ids,text)
            if failed:
                rejected.append({'source_document':source_index,'reason':'non_roundtrip','variants':failed})
                continue
            rows=aligned_rows(text,ids,toks,counts,plain_forms)
            docs.append({'source_document':source_index,'text':text,'bytes':len(text.encode('utf-8')),
                         'ids':{v:[remaps[v][j] for j in ids[v]] for v in ids},
                         'bos':{v:len(t.tokens) for v,t in toks.items()},'rows':rows})
            if len(docs)%100==0:
                print(lang,'prepared',len(docs),'excluded',len(rejected),flush=True)
            if len(docs)>=args.docs:break
        if len(docs)>=args.docs:break
    if not docs:raise ValueError('No eligible documents')
    saved={'identity':identity,'examined':source_index+1,'excluded':rejected,'documents':docs}
    dump(plan,saved)
    dump(out/'prepare_audit.json',{'identity':identity,'accepted':len(docs),
                                 'examined':source_index+1,'excluded':rejected})
    return saved


def score(args,lang,plan,out):
    from nanochat.gpt import GPT,GPTConfig
    docs=plan['documents']
    plan_hash=sha(out/'plan.json')
    for seed in args.seeds:
        for v in args.variants:
            base=folder(args,lang,v)/f'seed{seed}'
            weights=base/'model_002553.pt'; meta=base/'meta_002553.json'
            if not weights.is_file() or not meta.is_file():
                download(args,lang,v,[f'seed{seed}/model_002553.pt',f'seed{seed}/meta_002553.json'])
            ident={'plan_sha256':plan_hash,'checkpoint_sha256':sha(weights),
                   'meta_sha256':sha(meta),'dtype':os.environ.get('NANOCHAT_DTYPE','float32'),
                   'nanochat_commit':subprocess.check_output(['git','-C',str(args.nanochat_root),'rev-parse','HEAD'],text=True).strip()}
            dest=out/f'{v}_seed{seed}.npz'; provenance=dest.with_suffix('.json')
            if dest.exists() and provenance.exists():
                if json.loads(provenance.read_text()) != ident:raise ValueError(f'Stale loss cache: {dest}')
                print('Using',dest,flush=True);continue
            model=GPT(GPTConfig(**json.loads(meta.read_text())['model_config']))
            model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True,mmap=True))
            model=model.to(args.device).eval()
            loss={};t0=time.monotonic()
            with torch.inference_mode():
                for i,d in enumerate(docs):
                    ids=d['ids'][v]
                    x=torch.tensor([[d['bos'][v]]+ids[:-1]],device=args.device)
                    y=torch.tensor([ids],device=args.device)
                    z=model(x,y,loss_reduction='none').float().cpu().numpy()
                    assert z.shape==(len(ids),) and np.isfinite(z).all()
                    loss[str(i)]=z
                    if (i+1)%100==0:print(lang,v,seed,i+1,round(time.monotonic()-t0,1),'seconds',flush=True)
            tmp=dest.with_name(dest.stem+'.tmp.npz')
            np.savez_compressed(tmp,**loss);tmp.replace(dest);dump(provenance,ident)
            del model
            if args.device=='mps':torch.mps.empty_cache()


def report(args,lang,plan,out):
    docs=plan['documents'];nbytes=sum(d['bytes'] for d in docs)
    result={'language':lang,'documents':len(docs),'source_bytes':nbytes,
            'excluded_documents':len(plan['excluded']),'identity':plan['identity'],
            'positive_gain':'plain minus boundary BPB','seeds':{}}
    for seed in args.seeds:
        cache={v:np.load(out/f'{v}_seed{seed}.npz') for v in args.variants}
        totals={v:sum(float(cache[v][str(i)].sum(dtype=np.float64)) for i in range(len(docs)))/math.log(2) for v in cache}
        comparisons={}
        for v in args.variants[1:]:
            groups={};accounted=np.zeros(2)
            for i,d in enumerate(docs):
                # Decompress each document once, not once per source span.
                arrays={a:cache[a][str(i)] for a in ('plain',v)}
                for row in d['rows']:
                    p,b=[float(arrays[a][slice(*row['slices'][a])].sum(dtype=np.float64))/math.log(2) for a in ('plain',v)]
                    accounted+=[p,b]
                    delta=(row['slices'][v][1]-row['slices'][v][0])-row['plain_tokens']
                    segmentation='fewer' if delta<0 else 'more' if delta>0 else 'equal'
                    keys=['all',f'kind/{row["kind"]}',f'case/{row["case"]}',f'segmentation/{segmentation}',
                          f'kind_segmentation/{row["kind"]}/{segmentation}',
                          'baseline/whole_word' if row['plain_whole_word'] else 'baseline/other']
                    f=row.get('frequency')
                    if f is not None:
                        band='zero' if f==0 else '<1k' if f<1000 else '1k-10k' if f<10000 else '10k-100k' if f<100000 else '100k+'
                        keys += [f'frequency/{band}',f'frequency_segmentation/{band}/{segmentation}']
                        other=row.get('other_frequency')
                        if other is not None:
                            split='weaker' if f<other else 'stronger' if f>other else 'equal'
                            imbalance=max(f,other)/max(1,min(f,other))
                            keys += [f'duplicate/{split}', f'imbalance/{"<2" if imbalance<2 else "2-10" if imbalance<10 else "10+"}']
                    for key in keys:
                        g=groups.setdefault(key,{'spans':0,'bytes':0,'plain_bits':0.,'boundary_bits':0.})
                        g['spans']+=1;g['bytes']+=row['bytes'];g['plain_bits']+=p;g['boundary_bits']+=b
            assert np.allclose(accounted,[totals['plain'],totals[v]],rtol=1e-9), 'Loss conservation failed'
            assert groups['all']['bytes']==nbytes
            for g in groups.values():
                g['gain_bpb']=(g['plain_bits']-g['boundary_bits'])/g['bytes']
                g['contribution_to_total_gain_bpb']=(g['plain_bits']-g['boundary_bits'])/nbytes
            comparisons[v]={'gain_bpb':(totals['plain']-totals[v])/nbytes,'groups':groups}
        result['seeds'][str(seed)]={'bpb':{v:t/nbytes for v,t in totals.items()},'comparisons':comparisons}
        for f in cache.values():f.close()
    dump(out/'summary.json',result)
    print(out/'summary.json',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--languages',default='en',help='Comma-separated en,ko,ru')
    p.add_argument('--variants',default=','.join(VARIANTS))
    p.add_argument('--validation',action='append',default=[],help='LANG=/path/to/validation.parquet; repeat per language')
    p.add_argument('--models-root',type=Path,default=Path('models/boundary-marker-lms'))
    p.add_argument('--counts-root',type=Path,default=Path('results/magikarp/pr15_counts'))
    p.add_argument('--nanochat-root',type=Path,default=Path('results/magikarp/nanochat-source'))
    p.add_argument('--output-root',type=Path,default=Path('results/magikarp/aligned_spans'))
    p.add_argument('--docs',type=int,default=128)
    p.add_argument('--max-tokens',type=int,default=2048)
    p.add_argument('--max-chars',type=int,default=10000)
    p.add_argument('--seeds',default='0,1,2')
    p.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--report-only',action='store_true')
    p.add_argument('--download-missing',action='store_true')
    args=p.parse_args()
    args.languages=args.languages.split(',');args.variants=args.variants.split(',');args.seeds=list(map(int,args.seeds.split(',')))
    if args.variants[0]!='plain' or any(v not in VARIANTS for v in args.variants) or len(set(args.variants))!=len(args.variants):p.error('Variants must be unique and start with plain')
    if args.docs<1 or not 2<=args.max_tokens<=2048:p.error('Positive docs and max-tokens in [2,2048] required')
    validations=dict(s.split('=',1) for s in args.validation)
    defaults={'en':'results/magikarp/climbmix/shard_06542.parquet',
        'ko':'/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/fineweb-2_0_1-quality_10-filterrobots/data/output/kor_Hang/000_00003.parquet',
        'ru':'/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/fineweb-2_0_1-quality_10-filterrobots/data/output/rus_Cyrl/000_00010.parquet'}
    for lang in args.languages:
        if lang not in defaults:p.error(f'Unknown language: {lang}')
        validations[lang]=Path(validations.get(lang,defaults[lang]))
        if not validations[lang].is_file():p.error(f'Missing {lang} validation: {validations[lang]}; use --validation {lang}=/path/to/shard.parquet')
    os.environ.setdefault('NANOCHAT_DTYPE','float32')
    torch.set_num_threads(4)
    sys.path.insert(0,str(args.nanochat_root.resolve()))
    for lang in args.languages:
        out=args.output_root/f'{lang}_{args.docs}_{args.max_tokens}'
        out.mkdir(parents=True,exist_ok=True)
        plan=prepare(args,lang,validations[lang],out)
        print(lang,len(plan['documents']),'accepted;',len(plan['excluded']),'excluded',flush=True)
        if args.prepare_only:continue
        if not args.report_only:score(args,lang,plan,out)
        report(args,lang,plan,out)


if __name__=='__main__':main()
