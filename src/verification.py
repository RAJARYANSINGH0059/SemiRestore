#!/usr/bin/env python3
"""Candidate-level NOVA-Loc verifier for the supplied synthetic benchmark.

Augments the supplied baseline. Does not claim matched context-ring similarity:
the supplied inference contract contains a cropped reference patch, not its surrounding context.
"""
from __future__ import annotations
import csv, json, time, argparse
from pathlib import Path
from typing import Any
import cv2, numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
try:
    from .baseline import load_gray, baseline_center_aware
    from .nova_inference import locate
except ImportError:
    from baseline import load_gray, baseline_center_aware
    from nova_inference import locate

RATIOS = np.linspace(0.085, 0.115, 7)
FEATURE_NAMES = ["ncc", "gradient_corr", "periodicity_cos", "edge_density_delta", "contrast_delta", "score_gap", "ratio"]

def corr(a: np.ndarray, b: np.ndarray) -> float:
    a=np.asarray(a,np.float32).ravel(); b=np.asarray(b,np.float32).ravel()
    a=a-float(a.mean()); b=b-float(b.mean()); den=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/den) if den>1e-9 else 0.0

def periodicity_cos(a: np.ndarray,b: np.ndarray,size:int=32)->float:
    def spec(im):
        z=cv2.resize(im,(size,size),interpolation=cv2.INTER_AREA).astype(np.float32); z-=z.mean()
        win=np.outer(np.hanning(size),np.hanning(size)).astype(np.float32)
        s=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(z*win)))).astype(np.float32)
        yy,xx=np.ogrid[:size,:size]; s[(xx-size//2)**2+(yy-size//2)**2<=9]=0
        return s.ravel()
    sa,sb=spec(a),spec(b); den=float(np.linalg.norm(sa)*np.linalg.norm(sb))
    return float(np.dot(sa,sb)/den) if den>1e-9 else 0.0

def get_candidates(reference_path,search_path,ratios=RATIOS,peaks_per_scale=16,roi_radius=75,roi_peaks=10):
    ref=load_gray(reference_path); search=load_gray(search_path); H,W=search.shape
    center=((W-1)/2.0,(H-1)/2.0); raw=[]
    for ratio in ratios:
        w=max(8,int(round(ref.shape[1]*float(ratio)))); h=max(8,int(round(ref.shape[0]*float(ratio))))
        if h>=H or w>=W: continue
        templ=cv2.resize(ref,(w,h),interpolation=cv2.INTER_AREA)
        res=cv2.matchTemplate(search,templ,cv2.TM_CCOEFF_NORMED); flat=res.ravel()
        def top_peaks(sub, offset_x, offset_y, count, min_dist2=36):
            f=sub.ravel(); nprobe=min(max(count*80,300),f.size)
            ix=np.argpartition(f,-nprobe)[-nprobe:]; ix=ix[np.argsort(f[ix])[::-1]]
            chosen=[]
            for j in ix:
                yy,xx=np.unravel_index(int(j),sub.shape); xx=int(xx+offset_x);yy=int(yy+offset_y)
                if any((xx-px)**2+(yy-py)**2<min_dist2 for px,py,_ in chosen): continue
                chosen.append((xx,yy,float(res[yy,xx])))
                if len(chosen)>=count: break
            return chosen
        global_peaks=top_peaks(res,0,0,peaks_per_scale)
        # Explicit bounded drift ROI, centered on expected site but allowing displacement.
        # This is candidate generation only: every candidate is still ranked from image evidence.
        xlo=max(0,int(round(center[0]-roi_radius-(w-1)/2))); xhi=min(res.shape[1],int(round(center[0]+roi_radius-(w-1)/2))+1)
        ylo=max(0,int(round(center[1]-roi_radius-(h-1)/2))); yhi=min(res.shape[0],int(round(center[1]+roi_radius-(h-1)/2))+1)
        roi=res[ylo:yhi,xlo:xhi]
        local_peaks=top_peaks(roi,xlo,ylo,roi_peaks) if roi.size else []
        cx0=int(round(center[0]-(w-1)/2)); cy0=int(round(center[1]-(h-1)/2))
        cx0=max(0,min(W-w,cx0)); cy0=max(0,min(H-h,cy0))
        center_peak=(cx0,cy0,float(res[cy0,cx0]))
        chosen=[(xx,yy,sc,'roi') for xx,yy,sc in local_peaks]+[(xx,yy,sc,'global') for xx,yy,sc in global_peaks]+[(center_peak[0],center_peak[1],center_peak[2],'center')]
        seen=set(); max_score=float(np.max(res))
        for xx,yy,score,source in chosen:
            if (xx,yy,source) in seen: continue
            seen.add((xx,yy,source)); patch=search[yy:yy+h,xx:xx+w]
            gx=cv2.Sobel(templ,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(templ,cv2.CV_32F,0,1,ksize=3)
            px=cv2.Sobel(patch,cv2.CV_32F,1,0,ksize=3); py=cv2.Sobel(patch,cv2.CV_32F,0,1,ksize=3)
            rg=cv2.magnitude(gx,gy); pg=cv2.magnitude(px,py)
            edge_ref=float(np.mean(rg>np.percentile(rg,70))); edge_patch=float(np.mean(pg>np.percentile(pg,70)))
            x=xx+(w-1)/2; y=yy+(h-1)/2
            raw.append({"x":float(x),"y":float(y),"x0":xx,"y0":yy,"w":w,"h":h,"ratio":float(ratio),"ncc":score,
                "gradient_corr":corr(rg,pg),"periodicity_cos":periodicity_cos(templ,patch),
                "edge_density_delta":abs(edge_ref-edge_patch),"contrast_delta":abs(float(np.std(templ))-float(np.std(patch))),
                "score_gap":max_score-score,"center_distance":float(np.hypot(x-center[0],y-center[1])),
                "center_prior_candidate":source=='center',"candidate_source":source})
    # Reserve local-ROI candidates before global peaks; otherwise high-scoring far-away
    # repetitions can evict every plausible drift candidate from the final Top-K pool.
    prio={'roi':0,'center':1,'global':2}
    raw.sort(key=lambda c:(prio.get(c['candidate_source'],9),-c['ncc'],c['center_distance'],c['ratio'],c['y'],c['x']))
    dedup=[]
    for c in raw:
        if any((c['x']-d['x'])**2+(c['y']-d['y'])**2<16 for d in dedup): continue
        dedup.append(c)
        if len(dedup)>=160: break
    if not any(c['center_prior_candidate'] for c in dedup): dedup.append(min(raw,key=lambda c:(c['center_distance'],-c['ncc'])))
    best=max(c['ncc'] for c in dedup)
    for c in dedup: c['score_gap']=best-c['ncc']
    return dedup

def percentile_rank(values):
    values=np.asarray(values); order=np.argsort(values,kind="mergesort"); ranks=np.empty(len(values),dtype=float)
    ranks[order]=np.linspace(0.0,1.0,len(values)) if len(values)>1 else 1.0
    return ranks

def choose_ranked(cands,scores,tie_margin=0.035):
    vals=np.asarray(scores,dtype=float); top=float(vals.max()); idxs=np.flatnonzero(vals>=top-tie_margin)
    idx=min(idxs,key=lambda i:(cands[int(i)]["center_distance"],-float(vals[int(i)]),-cands[int(i)]["ncc"]))
    return cands[int(idx)],top

def features(cands): return np.asarray([[c[k] for k in FEATURE_NAMES] for c in cands],dtype=np.float32)

def train_ranker(gt_csvs,model_path,seed=20261027):
    Xs=[]; ys=[]; pairs=0; candidate_recall=0
    for gt_csv in gt_csvs:
        with Path(gt_csv).open(newline="") as f: rows=list(csv.DictReader(f))
        for r in rows:
            cands=get_candidates(r["reference"],r["search"]); gx,gy=float(r["x"]),float(r["y"])
            lab=np.asarray([int(np.hypot(c["x"]-gx,c["y"]-gy)<=5.0) for c in cands],dtype=np.int8)
            candidate_recall+=int(lab.any()); pairs+=1; Xs.append(features(cands)); ys.append(lab)
    X=np.vstack(Xs); y=np.concatenate(ys)
    if len(np.unique(y))<2: raise RuntimeError("Ranker training data must contain positive and negative candidate labels")
    clf=RandomForestClassifier(n_estimators=160,max_depth=10,min_samples_leaf=3,class_weight="balanced_subsample",random_state=seed,n_jobs=-1,max_features=0.85)
    clf.fit(X,y); model_path=Path(model_path); model_path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({"model":clf,"feature_names":FEATURE_NAMES,"seed":seed,"training_pairs":pairs,"candidate_recall_at_100":candidate_recall/pairs},model_path)
    return {"training_pairs":pairs,"training_candidates":int(len(y)),"positive_candidate_rate":float(y.mean()),"candidate_recall_at_100":candidate_recall/pairs,
            "feature_importances":dict(zip(FEATURE_NAMES,clf.feature_importances_.round(6).tolist())),"model_path":str(model_path)}

def _argmax_candidate(cands, scores):
    values=np.asarray(scores,dtype=float)
    # np.argmax returns first maximum; candidates are sorted deterministically upstream.
    return cands[int(np.argmax(values))]

def predict_pair(r,model=None):
    cands=get_candidates(r["reference"],r["search"])
    ncc=np.asarray([c["ncc"] for c in cands]); gr=np.asarray([c["gradient_corr"] for c in cands]); per=np.asarray([c["periodicity_cos"] for c in cands])
    nr=percentile_rank(ncc); grr=percentile_rank(gr); pr=percentile_rank(per)
    # P1: gradient evidence influences ranking, but does not replace the raw NCC response.
    p1=.78*nr+.22*grr
    # P2: add frequency-domain periodicity agreement.
    p2=.65*nr+.22*grr+.13*pr
    # P3: ambiguity-gated verifier. Keep candidates within a raw-NCC margin, then use
    # structural evidence; use center-nearest only as the final tie resolver.
    eligible=np.flatnonzero(ncc >= float(ncc.max())-.06)
    p3=.62*nr+.24*grr+.14*pr
    local_scores=p3[eligible]; top=float(local_scores.max())
    near=eligible[local_scores >= top-.005]
    idx=min(near,key=lambda j:(cands[int(j)]["center_distance"],-float(p3[int(j)]),-float(ncc[int(j)])))
    out={"P1_gradient":_argmax_candidate(cands,p1),"P2_gradient_periodicity":_argmax_candidate(cands,p2),"P3_ambiguity_gate":cands[int(idx)]}
    conf={}
    if model is not None:
        probs=model["model"].predict_proba(features(cands))[:,list(model["model"].classes_).index(1)]
        out["P4_ml_ranker"]=_argmax_candidate(cands,probs)
        p5=.60*percentile_rank(probs)+.25*nr+.15*grr
        out["P5_ml_plus_structure"]=_argmax_candidate(cands,p5)
        so=np.sort(probs)
        conf={"ml_top_probability":float(so[-1]),"ml_probability_margin":float(so[-1]-so[-2]) if len(so)>1 else 1.0}
    return cands,{k:(float(v["x"]),float(v["y"])) for k,v in out.items()},conf

def score_dataset(gt_csv,model,out_dir,split):
    with Path(gt_csv).open(newline="") as f: rows=list(csv.DictReader(f))
    methods=["B3_center_aware","G1_center_candidate_gate","P1_gradient","P2_gradient_periodicity","P3_ambiguity_gate"]
    if model is not None: methods += ["P4_ml_ranker","P5_ml_plus_structure"]
    details=[]
    for i,r in enumerate(rows,1):
        t=time.perf_counter(); _,b3,_,_,_=baseline_center_aware(r["reference"],r["search"]); gate=locate(r["reference"],r["search"]); cands,preds,conf=predict_pair(r,model); elapsed=(time.perf_counter()-t)*1000
        gx,gy=float(r["x"]),float(r["y"]); row={"split":split,"id":r.get("id",f"{split}_{i}"),"architecture":r.get("architecture","unknown"),"gt_x":gx,"gt_y":gy,
            "candidate_count":len(cands),"gate_score_gap":gate.score_gap_from_best,"gate_used_center_prior":gate.used_center_prior,"gate_status":gate.status,"gate_runtime_ms":gate.runtime_ms,"candidate_recall_at_5px":int(any(np.hypot(c["x"]-gx,c["y"]-gy)<=5 for c in cands)),"candidate_recall_at_10px":int(any(np.hypot(c["x"]-gx,c["y"]-gy)<=10 for c in cands)),"end_to_end_all_methods_ms":elapsed,**conf}
        all_preds={"B3_center_aware":(float(b3[0]),float(b3[1])),"G1_center_candidate_gate":(float(gate.x),float(gate.y)),**preds}
        for name,(px,py) in all_preds.items(): row[f"{name}_x"]=px;row[f"{name}_y"]=py;row[f"{name}_error_px"]=float(np.hypot(px-gx,py-gy))
        details.append(row)
        if i%10==0 or i==len(rows): print(f"[{split}] {i}/{len(rows)} pairs processed",flush=True)
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True); fields=list(dict.fromkeys(k for d in details for k in d))
    with (out_dir/f"{split}_per_sample.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(details)
    summary={"split":split,"n":len(details),"candidate_recall_at_5px":float(np.mean([r["candidate_recall_at_5px"] for r in details])),"candidate_recall_at_10px":float(np.mean([r["candidate_recall_at_10px"] for r in details])),"methods":{}}
    for method in methods:
        e=np.asarray([r[f"{method}_error_px"] for r in details]); m={"accuracy_1px":float(np.mean(e<=1)),"accuracy_2px":float(np.mean(e<=2)),"accuracy_5px":float(np.mean(e<=5)),"accuracy_10px":float(np.mean(e<=10)),"mean_error_px":float(np.mean(e)),"median_error_px":float(np.median(e)),"p95_error_px":float(np.percentile(e,95)),"max_error_px":float(np.max(e))}
        for arch in sorted({r["architecture"] for r in details}):
            ea=np.asarray([r[f"{method}_error_px"] for r in details if r["architecture"]==arch]);m[arch]={"n":len(ea),"accuracy_5px":float(np.mean(ea<=5)),"mean_error_px":float(np.mean(ea))}
        summary["methods"][method]=m
    lat=[r["end_to_end_all_methods_ms"] for r in details];summary["runtime"]={"mean_end_to_end_all_methods_ms":float(np.mean(lat)),"p95_end_to_end_all_methods_ms":float(np.percentile(lat,95))}
    (out_dir/f"{split}_summary.json").write_text(json.dumps(summary,indent=2));return details,summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--gt",type=Path,required=True);p.add_argument("--model",type=Path,default=None);p.add_argument("--out-dir",type=Path,required=True);p.add_argument("--split",default="test");a=p.parse_args();model=joblib.load(a.model) if a.model else None
    _,summary=score_dataset(a.gt,model,a.out_dir,a.split);print(json.dumps(summary,indent=2))
if __name__=="__main__": main()