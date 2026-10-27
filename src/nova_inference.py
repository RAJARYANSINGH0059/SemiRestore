#!/usr/bin/env python3
"""NOVA-Loc bounded-ROI multiscale inference with a late center ambiguity gate.

The method searches a configurable center-relative drift ROI at seven template scales.
It compares the strongest candidate in that ROI with the candidate explicitly scored
at the nominal center. The center is selected only when its NCC score is within a
validation-selected score window of the best ROI candidate; otherwise the best ROI
candidate wins. This is calibrated to the supplied synthetic generator and is not
claimed to generalize to real SEM images without separate validation.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import cv2
import numpy as np
try:
    from .baseline import load_gray
except ImportError:
    from baseline import load_gray

@dataclass
class LocalizationResult:
    x: float
    y: float
    score: float
    best_roi_score: float
    score_gap_from_best: float
    chosen_scale_ratio: float
    candidate_count: int
    eligible_center_gate: bool
    used_center_prior: bool
    roi_radius_px: int
    score_window: float
    runtime_ms: float
    status: str
    def to_dict(self): return asdict(self)


def locate(reference_path: str | Path, search_path: str | Path,
           ratios: np.ndarray | None = None, roi_radius: int = 50,
           score_window: float = 0.05) -> LocalizationResult:
    """Localize from explicit image paths using a bounded multiscale ROI.

    Defaults (ROI radius 50 px; NCC window 0.05) were selected on a separate
    validation set generated with independent seeds. The score window is in normalized
    correlation units, not a probability. Center prior is never used unconditionally.
    """
    if not isinstance(roi_radius,int) or roi_radius < 0: raise ValueError("roi_radius must be a non-negative integer")
    if not (0 <= score_window <= 1): raise ValueError("score_window must be in [0,1]")
    ref=load_gray(reference_path); search=load_gray(search_path)
    if ref.size==0 or search.size==0: raise ValueError("reference/search image is empty")
    H,W=search.shape
    if ref.shape[0]<8 or ref.shape[1]<8: raise ValueError("reference dimensions must be at least 8x8")
    if ref.shape[0]>=H or ref.shape[1]>=W: raise ValueError("reference dimensions must be smaller than search dimensions")
    if ratios is None: ratios=np.linspace(0.085,0.115,7)
    if len(ratios)==0: raise ValueError("ratios cannot be empty")
    center=((W-1)/2.0,(H-1)/2.0); best_roi=(-1.0,None,None); best_center=(-1.0,None,None); n_candidates=0
    t0=time.perf_counter()
    for ratio in sorted(float(v) for v in ratios):
        if ratio<=0: raise ValueError("scale ratios must be positive")
        w=max(8,int(round(ref.shape[1]*ratio)));h=max(8,int(round(ref.shape[0]*ratio)))
        if h>=H or w>=W: continue
        templ=cv2.resize(ref,(w,h),interpolation=cv2.INTER_AREA)
        res=cv2.matchTemplate(search,templ,cv2.TM_CCOEFF_NORMED)
        # Search candidate CENTER coordinates within center +/- ROI radius.
        xlo=max(0,int(np.ceil(center[0]-roi_radius-(w-1)/2.0)))
        xhi=min(res.shape[1],int(np.floor(center[0]+roi_radius-(w-1)/2.0))+1)
        ylo=max(0,int(np.ceil(center[1]-roi_radius-(h-1)/2.0)))
        yhi=min(res.shape[0],int(np.floor(center[1]+roi_radius-(h-1)/2.0))+1)
        roi=res[ylo:yhi,xlo:xhi]
        if not roi.size: continue
        _,score,_,loc=cv2.minMaxLoc(roi)
        cand=(float(score),(float(loc[0]+xlo+(w-1)/2.0),float(loc[1]+ylo+(h-1)/2.0)),ratio)
        if cand[0]>best_roi[0]: best_roi=cand
        # Score the nominal center explicitly at this scale.
        cx0=int(round(center[0]-(w-1)/2.0));cy0=int(round(center[1]-(h-1)/2.0))
        cx0=max(0,min(W-w,cx0));cy0=max(0,min(H-h,cy0))
        cc=(float(res[cy0,cx0]),(float(cx0+(w-1)/2.0),float(cy0+(h-1)/2.0)),ratio)
        if cc[0]>best_center[0]: best_center=cc
        n_candidates+=2
    if best_roi[1] is None or best_center[1] is None: raise ValueError("No candidates; check search dimensions and scale ratios")
    gap=float(best_roi[0]-best_center[0])
    use_center=bool(gap <= score_window)
    chosen=best_center if use_center else best_roi
    status=("center_prior_resolved_ambiguity" if use_center and gap>1e-6 else
            "best_visual_candidate" if not use_center else "center_candidate_is_best")
    return LocalizationResult(x=chosen[1][0],y=chosen[1][1],score=float(chosen[0]),best_roi_score=float(best_roi[0]),
       score_gap_from_best=max(0.0,gap),chosen_scale_ratio=float(chosen[2]),candidate_count=n_candidates,
       eligible_center_gate=use_center,used_center_prior=use_center,roi_radius_px=roi_radius,score_window=score_window,
       runtime_ms=(time.perf_counter()-t0)*1000,status=status)


def locate_cli(reference_path: str | Path, search_path: str | Path) -> tuple[float,float]:
    result=locate(reference_path,search_path)
    return result.x,result.y
