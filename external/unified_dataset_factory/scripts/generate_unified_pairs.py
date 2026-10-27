from pathlib import Path
import argparse, json, random
import numpy as np
from PIL import Image, ImageFilter, ImageOps

EXTS={".png",".jpg",".jpeg",".tif",".tiff",".bmp",".webp"}

def list_images(root):
    return [p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in EXTS]

def read_gray(path):
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("L"),
                       dtype=np.float32)/255.0

def resize_np(a,h,w):
    im=Image.fromarray(np.clip(a*255,0,255).astype(np.uint8))
    return np.asarray(im.resize((w,h),Image.Resampling.LANCZOS),
                      dtype=np.float32)/255.0

def fit(a,h,w): return resize_np(a,h,w)

def random_crop(a,h,w,rng):
    H,W=a.shape
    if H<h or W<w: return fit(a,h,w)
    y=int(rng.integers(0,H-h+1)); x=int(rng.integers(0,W-w+1))
    return a[y:y+h,x:x+w].copy()

def tone_match(src,dst):
    sm,ss=float(src.mean()),float(src.std()+1e-6)
    dm,ds=float(dst.mean()),float(dst.std()+1e-6)
    return np.clip((dst-dm)/ds*ss+sm,0,1)

def edge_brighten(a,amount):
    if amount<=0:return a
    gx=np.zeros_like(a); gy=np.zeros_like(a)
    gx[:,1:-1]=0.5*(a[:,2:]-a[:,:-2])
    gy[1:-1,:]=0.5*(a[2:,:]-a[:-2,:])
    return a+amount*np.sqrt(gx*gx+gy*gy)

def degrade(a,rng,noise_sigma,shot_strength,blur,edge,gamma,contrast):
    out=edge_brighten(a,edge)
    if blur>0:
        im=Image.fromarray(np.clip(out*255,0,255).astype(np.uint8))
        out=np.asarray(im.filter(ImageFilter.GaussianBlur(blur)),dtype=np.float32)/255.0
    out+=rng.normal(0,noise_sigma,out.shape).astype(np.float32)
    lam=np.clip(out,0,1)*shot_strength
    out=rng.poisson(lam).astype(np.float32)/max(shot_strength,1)
    out=(out-0.5)*contrast+0.5
    out=np.power(np.clip(out,0,1),max(gamma,1e-3))
    return np.clip(out,0,1)

def difficulty(rng):
    cfgs={
        "easy":(0.002,0.006,0,0.5,0,0.02,0.95,1.05),
        "medium":(0.004,0.012,0,1,0.01,0.05,0.90,1.10),
        "hard":(0.008,0.025,0.5,1.5,0.02,0.08,0.82,1.18),
        "extreme":(0.015,0.040,1,2.5,0.03,0.12,0.75,1.25)
    }
    name=str(rng.choice(list(cfgs))); n0,n1,b0,b1,e0,e1,c0,c1=cfgs[name]
    return {
        "difficulty":name,
        "ref_noise_sigma":float(rng.uniform(n0,n1)),
        "search_noise_sigma":float(rng.uniform(n0,n1)),
        "ref_blur_sigma":float(rng.uniform(b0,b1)),
        "search_blur_sigma":float(rng.uniform(b0,b1)),
        "ref_edge_amount":float(rng.uniform(e0,e1)),
        "search_edge_amount":float(rng.uniform(e0,e1)),
        "ref_contrast":float(rng.uniform(c0,c1)),
        "search_contrast":float(rng.uniform(c0,c1)),
        "ref_gamma":float(rng.uniform(.92,1.08)),
        "search_gamma":float(rng.uniform(.90,1.10)),
        "ref_shot_strength":int(rng.integers(300,1800)),
        "search_shot_strength":int(rng.integers(200,1400))
    }

def make_pair(src, ref_size, search_size, scale, rng):
    target_src=max(int(round(ref_size*scale)),8)
    ctx=random_crop(src,max(target_src*2, min(src.shape)),max(target_src*2,min(src.shape)),rng)
    ch,cw=ctx.shape
    y0=int(rng.integers(0,max(1,ch-target_src+1)))
    x0=int(rng.integers(0,max(1,cw-target_src+1)))
    patch=ctx[y0:y0+target_src,x0:x0+target_src].copy()
    if patch.shape!=(target_src,target_src):
        patch=fit(patch,target_src,target_src)

    reference=resize_np(patch,ref_size,ref_size)
    search=fit(ctx,search_size,search_size)

    ts=max(3,int(round(ref_size/scale)))
    small=resize_np(patch,ts,ts)
    margin=max(2,ts//2+1)
    cx=int(rng.integers(margin,search_size-margin))
    cy=int(rng.integers(margin,search_size-margin))
    sx=cx-ts//2; sy=cy-ts//2
    ex,ey=sx+ts,sy+ts
    region=search[sy:ey,sx:ex]
    alpha=np.ones((ts,ts),np.float32)
    if ts>=8:
        yy,xx=np.mgrid[0:ts,0:ts]
        d=np.minimum.reduce([xx,yy,ts-1-xx,ts-1-yy])
        alpha=np.clip(d/2,0,1).astype(np.float32)
    search[sy:ey,sx:ex]=alpha*small+(1-alpha)*region
    return reference,search,cx,cy

def save(a,p):
    p.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.clip(a*255,0,255).astype(np.uint8)).save(p)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-root",default="data/sources")
    ap.add_argument("--out",default="data/generated")
    ap.add_argument("--pairs",type=int,default=5000)
    ap.add_argument("--reference-size",type=int,default=64)
    ap.add_argument("--search-size",type=int,default=1000)
    ap.add_argument("--scale",type=float,default=10)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--architecture",choices=["mixed","dram","finfet"],default="mixed")
    a=ap.parse_args()
    rng=np.random.default_rng(a.seed); py=random.Random(a.seed)
    root=Path(a.source_root); out=Path(a.out)
    ims=list_images(root)
    if not ims: raise SystemExit(f"No images under {root}")
    groups={}
    for p in ims: groups.setdefault(p.relative_to(root).parts[0],[]).append(p)
    names=sorted(groups)
    out.mkdir(parents=True,exist_ok=True)
    with (out/"metadata.jsonl").open("w",encoding="utf-8") as mf:
        generated=0
        while generated<a.pairs:
            sname=py.choice(names); aname=py.choice(names)
            sp=py.choice(groups[sname]); apath=py.choice(groups[aname])
            seed=int(rng.integers(0,2**31-1)); lr=np.random.default_rng(seed)
            try:
                src=read_gray(sp); app=read_gray(apath)
                ref,search,cx,cy=make_pair(src,a.reference_size,a.search_size,a.scale,lr)
                if lr.random()<0.70: ref=tone_match(app,ref)
                if lr.random()<0.85: search=tone_match(app,search)
                d=difficulty(lr)
                ref=degrade(ref,lr,d["ref_noise_sigma"],d["ref_shot_strength"],d["ref_blur_sigma"],d["ref_edge_amount"],d["ref_gamma"],d["ref_contrast"])
                search=degrade(search,lr,d["search_noise_sigma"],d["search_shot_strength"],d["search_blur_sigma"],d["search_edge_amount"],d["search_gamma"],d["search_contrast"])
                stem=f"{generated:07d}"
                save(ref,out/"reference"/f"{stem}.png"); save(search,out/"search"/f"{stem}.png")
                arch=a.architecture if a.architecture!="mixed" else py.choice(["dram","finfet"])
                rec={"id":stem,"reference":f"reference/{stem}.png","search":f"search/{stem}.png",
                     "x":cx,"y":cy,"architecture":arch,"scale":a.scale,
                     "structure_source":str(sp),"appearance_source":str(apath),"seed":seed,**d}
                mf.write(json.dumps(rec)+"\n")
                generated+=1
            except Exception as e:
                print(f"[WARN] skipped {sp}: {e}")
    print(f"Generated {generated} pairs in {out}")

if __name__=="__main__":
    main()