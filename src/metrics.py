#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from baseline import baseline_fixed, baseline_multiscale, baseline_center_aware


def collect(gt: Path, mode: str):
    rows=list(csv.DictReader(gt.open()))
    out=[]
    for r in rows:
        if mode=='fixed':
            s,p,dt=baseline_fixed(r['reference'],r['search']); extra={}
        elif mode=='multiscale':
            s,p,dt,ratio=baseline_multiscale(r['reference'],r['search']); extra={'ratio':ratio}
        else:
            s,p,dt,ratio,n=baseline_center_aware(r['reference'],r['search']); extra={'ratio':ratio,'eligible':n}
        e=float(np.hypot(p[0]-float(r['x']),p[1]-float(r['y'])))
        out.append({'id':r['id'],'architecture':r['architecture'],'error_px':e,'score':s,'latency_ms':dt*1000,**extra})
    return out

p=argparse.ArgumentParser(); p.add_argument('--gt',type=Path,required=True); p.add_argument('--mode',choices=['fixed','multiscale','center'],default='center'); p.add_argument('--out',type=Path,required=True)
a=p.parse_args(); rows=collect(a.gt,a.mode)
summary={'n':len(rows),'accuracy_1px':float(np.mean([r['error_px']<=1 for r in rows])),'accuracy_2px':float(np.mean([r['error_px']<=2 for r in rows])),'accuracy_5px':float(np.mean([r['error_px']<=5 for r in rows])),'accuracy_10px':float(np.mean([r['error_px']<=10 for r in rows])),'mean_error_px':float(np.mean([r['error_px'] for r in rows])),'median_error_px':float(np.median([r['error_px'] for r in rows])),'p95_error_px':float(np.percentile([r['error_px'] for r in rows],95)),'mean_latency_ms':float(np.mean([r['latency_ms'] for r in rows])),'p95_latency_ms':float(np.percentile([r['latency_ms'] for r in rows],95))}
a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps({'summary':summary,'per_sample':rows},indent=2)); print(json.dumps(summary,indent=2))