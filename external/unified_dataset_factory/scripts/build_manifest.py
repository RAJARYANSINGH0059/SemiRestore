from pathlib import Path
import argparse, csv, hashlib

EXTS = {".png",".jpg",".jpeg",".tif",".tiff",".bmp",".webp"}

def sha256(path, chunk=1<<20):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while b:=f.read(chunk): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="data/sources")
    ap.add_argument("--out",default="data/source_manifest.csv")
    ap.add_argument("--hash",action="store_true")
    a=ap.parse_args()
    root=Path(a.root); rows=[]
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            rel=p.relative_to(root)
            rows.append({
                "source": rel.parts[0] if rel.parts else "unknown",
                "path": str(p),
                "bytes": p.stat().st_size,
                "sha256": sha256(p) if a.hash else ""
            })
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["source","path","bytes","sha256"])
        w.writeheader(); w.writerows(rows)
    print(f"Indexed {len(rows)} images -> {out}")

if __name__=="__main__":
    main()