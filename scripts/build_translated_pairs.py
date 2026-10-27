#!/usr/bin/env python3
"""Create reproducible synthetic drift variants from a generated-pair CSV.

Translations are applied to the search image only and exact known offsets are added
to the ground-truth center. These are synthetic drift experiments, not real SEM data.
"""
import argparse,csv
from pathlib import Path
import cv2,numpy as np

def build(input_csv:Path, output_dir:Path, seed:int=20261027, max_shift:int=50):
    output_dir.mkdir(parents=True,exist_ok=True); search_dir=output_dir/'search_shifted';search_dir.mkdir(exist_ok=True)
    rows=list(csv.DictReader(input_csv.open(newline='')))
    rng=np.random.default_rng(seed); out=[]
    for i,row in enumerate(rows):
        img=cv2.imread(row['search'],cv2.IMREAD_GRAYSCALE)
        if img is None: raise FileNotFoundError(row['search'])
        dx=int(rng.integers(-max_shift,max_shift+1));dy=int(rng.integers(-max_shift,max_shift+1))
        M=np.float32([[1,0,dx],[0,1,dy]])
        shifted=cv2.warpAffine(img,M,(img.shape[1],img.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        name=f"{row.get('id',i)}_dx{dx:+d}_dy{dy:+d}.png"; target=search_dir/name
        if not cv2.imwrite(str(target),shifted): raise IOError(f'Could not write {target}')
        new=dict(row);new['search']=str(target.resolve());new['x']=f"{float(row['x'])+dx:.3f}";new['y']=f"{float(row['y'])+dy:.3f}";new['dx']=str(dx);new['dy']=str(dy);out.append(new)
    fields=list(dict.fromkeys(k for r in out for k in r))
    outcsv=output_dir/'ground_truth_translated.csv'
    with outcsv.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
    print(f"Created {len(out)} translated pairs -> {outcsv}; seed={seed}; max_shift={max_shift}px")
    return outcsv
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-csv',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--seed',type=int,default=20261027);p.add_argument('--max-shift',type=int,default=50);a=p.parse_args();build(a.input_csv,a.output_dir,a.seed,a.max_shift)