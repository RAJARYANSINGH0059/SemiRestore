from pathlib import Path
import argparse, json
from collections import defaultdict

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--metadata",default="data/generated/metadata.jsonl")
    ap.add_argument("--out",default="data/generated/splits.json")
    a=ap.parse_args()
    rows=[json.loads(x) for x in Path(a.metadata).read_text().splitlines() if x.strip()]
    groups=defaultdict(list)
    for r in rows:
        # `structure_source` may be relative (`data/sources/refics/...`) or an
        # absolute Colab/Drive path. Group by the actual source folder *under*
        # `sources`, not by the shared prefix `data` or `content`.
        parts=list(Path(r["structure_source"]).as_posix().split("/"))
        if "sources" in parts and parts.index("sources") + 1 < len(parts):
            src=parts[parts.index("sources") + 1]
        else:
            src=parts[0] if parts else "unknown"
        groups[src].append(r["id"])
    names=sorted(groups)
    n=len(names)
    k=max(1,int(.6*n)); m=max(k+1,int(.8*n))
    train=names[:k]; val=names[k:m]; ood=names[m:]
    split={"train_sources":train,"val_sources":val,"ood_sources":ood,
           "train_ids":sum((groups[s] for s in train),[]),
           "val_ids":sum((groups[s] for s in val),[]),
           "ood_ids":sum((groups[s] for s in ood),[])}
    Path(a.out).write_text(json.dumps(split,indent=2))
    print("Wrote",a.out)
if __name__=="__main__": main()