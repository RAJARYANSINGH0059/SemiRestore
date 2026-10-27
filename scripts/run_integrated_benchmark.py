#!/usr/bin/env python3
"""Run the original baselines and the locked NOVA-Loc hybrid on bundled test pairs.

The bundled 60 pairs remain holdout. A second deterministic experiment translates only
the search image by up to +/-50 px and updates the exact synthetic ground truth.
Temporary translated images are deleted after inference; per-sample records retain
source ids, shifts, file hashes, predictions, and errors.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,tempfile,time,sys
from pathlib import Path
import cv2,numpy as np
cv2.setNumThreads(1)  # avoid OpenCV over-subscription in this repeated benchmark
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
try:
    from baseline import baseline_fixed,baseline_multiscale,baseline_center_aware
    from nova_inference import locate
except ImportError:
    from src.baseline import baseline_fixed,baseline_multiscale,baseline_center_aware
    from src.nova_inference import locate

METHODS=["B1_fixed_ncc","B2_multiscale_ncc","B3_center_aware_ncc","NOVA_hybrid_roi_gate"]

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def load_rows(root:Path):
    rows=[]
    for arch in ['DRAM','FinFET']:
        gt=root/'results'/'baseline'/arch/'ground_truth.csv'
        if not gt.exists():raise FileNotFoundError(gt)
        for r in csv.DictReader(gt.open(newline='')):
            q=dict(r)
            for key in ['reference','search']:
                p=Path(q[key]);q[key]=str((root/p).resolve() if not p.is_absolute() else p)
                if not Path(q[key]).is_file():raise FileNotFoundError(q[key])
            rows.append(q)
    return rows

def one_pair(r,mode,dx,dy,source_id,source_search_hash,run_search_path):
    results={}; timings={}
    t=time.perf_counter(); score,pred,inner=baseline_fixed(r['reference'],run_search_path);timings['B1_fixed_ncc']=(time.perf_counter()-t)*1000;results['B1_fixed_ncc']=(pred[0],pred[1],score,None)
    t=time.perf_counter(); score,pred,inner,ratio=baseline_multiscale(r['reference'],run_search_path);timings['B2_multiscale_ncc']=(time.perf_counter()-t)*1000;results['B2_multiscale_ncc']=(pred[0],pred[1],score,ratio)
    t=time.perf_counter(); score,pred,inner,ratio,nc=baseline_center_aware(r['reference'],run_search_path);timings['B3_center_aware_ncc']=(time.perf_counter()-t)*1000;results['B3_center_aware_ncc']=(pred[0],pred[1],score,ratio)
    t=time.perf_counter(); z=locate(r['reference'],run_search_path);timings['NOVA_hybrid_roi_gate']=(time.perf_counter()-t)*1000;results['NOVA_hybrid_roi_gate']=(z.x,z.y,z.score,z.chosen_scale_ratio)
    gt_x=float(r['x'])+dx;gt_y=float(r['y'])+dy
    processed_hash=sha256(Path(run_search_path))
    records=[]
    for method in METHODS:
        px,py,score,ratio=results[method];err=float(np.hypot(px-gt_x,py-gt_y))
        records.append({'split':mode,'id':source_id,'architecture':r['architecture'],'reference_path':Path(r['reference']).name,
          'source_search_path':Path(r['search']).name,'source_search_sha256':source_search_hash,
          'processed_search_sha256':processed_hash, 'shift_dx_px':dx,'shift_dy_px':dy,
          'gt_x':gt_x,'gt_y':gt_y,'method':method,'pred_x':float(px),'pred_y':float(py),
          'localization_error_px':err,'score':float(score),'chosen_scale_ratio':ratio,
          'runtime_ms':float(timings[method]),
          'gate_status':z.status if method=='NOVA_hybrid_roi_gate' else '',
          'center_prior_used':z.used_center_prior if method=='NOVA_hybrid_roi_gate' else '',
          'center_score_gap':z.score_gap_from_best if method=='NOVA_hybrid_roi_gate' else ''})
    return records

def summarize(records):
    out={}
    for split in sorted({r['split'] for r in records}):
      out[split]={}
      rr=[r for r in records if r['split']==split]
      for method in METHODS:
        for arch in ['ALL','DRAM','FinFET']:
          v=[r for r in rr if r['method']==method and (arch=='ALL' or r['architecture']==arch)]
          if not v:continue
          e=np.array([r['localization_error_px'] for r in v],dtype=float);lat=np.array([r['runtime_ms'] for r in v],dtype=float)
          out[split].setdefault(method,{})[arch]={'n':len(e),'accuracy_1px':float(np.mean(e<=1)), 'accuracy_2px':float(np.mean(e<=2)),
            'accuracy_5px':float(np.mean(e<=5)), 'accuracy_10px':float(np.mean(e<=10)), 'mean_error_px':float(e.mean()),
            'median_error_px':float(np.median(e)),'p95_error_px':float(np.percentile(e,95)), 'max_error_px':float(e.max()),
            'mean_latency_ms':float(lat.mean()),'p95_latency_ms':float(np.percentile(lat,95))}
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--out-dir',type=Path,default=None);p.add_argument('--shift-seed',type=int,default=930091);p.add_argument('--max-shift',type=int,default=50);p.add_argument('--start-index',type=int,default=0);p.add_argument('--limit',type=int,default=20);args=p.parse_args()
    root=args.root.resolve();out=(args.out_dir or root/'results'/'integrated_benchmark').resolve();out.mkdir(parents=True,exist_ok=True)
    rows=load_rows(root);rng=np.random.default_rng(args.shift_seed);offsets=[(int(rng.integers(-args.max_shift,args.max_shift+1)),int(rng.integers(-args.max_shift,args.max_shift+1))) for _ in rows];records=[];t0=time.perf_counter()
    print(f'Loaded {len(rows)} holdout pairs ({sum(r["architecture"]=="DRAM" for r in rows)} DRAM, {sum(r["architecture"]=="FinFET" for r in rows)} FinFET)',flush=True)
    stop=min(len(rows),args.start_index+args.limit)
    if args.start_index<0 or args.start_index>=len(rows): raise ValueError('start-index must be within the dataset')
    subset=rows[args.start_index:stop]
    for local_i,r in enumerate(subset,1):
      i=args.start_index+local_i
      source_hash=sha256(Path(r['search']))
      records.extend(one_pair(r,'holdout_centered',0,0,r['id'],source_hash,r['search']))
      img=cv2.imread(r['search'],cv2.IMREAD_GRAYSCALE)
      if img is None:raise FileNotFoundError(r['search'])
      dx,dy=offsets[i-1]
      M=np.float32([[1,0,dx],[0,1,dy]])
      shifted=cv2.warpAffine(img,M,(img.shape[1],img.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
      tmp=out/'_temporary_shifted_inputs';tmp.mkdir(exist_ok=True);sp=tmp/f"{r['id']}_dx{dx:+d}_dy{dy:+d}.png"
      if not cv2.imwrite(str(sp),shifted):raise IOError(sp)
      records.extend(one_pair(r,'holdout_translated',dx,dy,r['id'],source_hash, str(sp)))
      try:sp.unlink()
      except OSError:pass
      if local_i%5==0 or local_i==len(subset):print(f'Holdout pair {i}/{len(rows)} (chunk {local_i}/{len(subset)}): centered and shifted runs complete',flush=True)
    temp=out/'_temporary_shifted_inputs'
    try:temp.rmdir()
    except OSError:pass
    fields=list(dict.fromkeys(k for r in records for k in r))
    csv_path=out/'integrated_per_sample_results.csv'
    with csv_path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    summary=summarize(records);(out/'integrated_metrics_summary.json').write_text(json.dumps(summary,indent=2))
    md=['# NOVA-Loc integrated benchmark results','',f'- Bundled holdout: {len(subset)} pairs in this chunk; full dataset has 60 synthetic pairs (30 DRAM + 30 FinFET).',f'- Drift stress: same holdout images transformed with deterministic translations up to +/-{args.max_shift} px (seed {args.shift_seed}); ground truth is shifted accordingly.', '- These are synthetic generator measurements, not real wafer or official competition results.','- Model-based candidate rankers were tested separately on validation; current candidate-only ML configuration did not improve over the classical method and is not used in the default inference entrypoint.','', '## Results by split and method','']
    for split in ['holdout_centered','holdout_translated']:
      md += [f'### {split.replace("_"," ").title()}','', '| Method | Arch | n | Acc ≤1 px | Acc ≤5 px | Mean error (px) | P95 error (px) | Mean latency (ms) |','|---|---|---:|---:|---:|---:|---:|---:|']
      for method in METHODS:
       for arch in ['ALL','DRAM','FinFET']:
        if arch not in summary[split][method]: continue
        d=summary[split][method][arch]
        md.append(f'| {method} | {arch} | {d["n"]} | {d["accuracy_1px"]:.1%} | {d["accuracy_5px"]:.1%} | {d["mean_error_px"]:.2f} | {d["p95_error_px"]:.2f} | {d["mean_latency_ms"]:.1f} |')
      md.append('')
    md += ['## Reproducibility and interpretation','',f'- Total runtime for all four methods, both splits: {time.perf_counter()-t0:.2f} s.', '- In the centered holdout, the center prior is source-aligned because the supplied generator places the target at (500,500). This is an explicit benchmark prior and may inflate centered accuracy.', '- The translation stress changes the target location while retaining the source visual motif; degradation here is a warning that center-prior matching is not a complete drift solution.', '- The ML ranker experiments use separate generated training/validation seeds and are documented in the ablation folder. Their poor results are retained as negative evidence, not hidden.', '- The dataset contains no real SEM acquisitions or physical scale calibration; pixel errors cannot be converted to nanometres.','']
    (out/'INTEGRATED_BENCHMARK_REPORT.md').write_text('\n'.join(md))
    print('Wrote:',csv_path);print('Wrote:',out/'integrated_metrics_summary.json');print('Wrote:',out/'INTEGRATED_BENCHMARK_REPORT.md')
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()