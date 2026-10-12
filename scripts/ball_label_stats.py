"""ball_label_stats.py -- ball size, per-gap displacement and frame-edge distance from GSR labels.

Reads only Labels-GameState.json from each split zip (no extraction). Numbers back
docs/path_b/plans/BALL_TEMPORAL_PLAN.md section 1.

    python scripts/ball_label_stats.py ../SoccerNetGS/gamestate-2024/valid.zip ../SoccerNetGS/gamestate-2024/train.zip
"""
import json, sys, zipfile, numpy as np
from collections import defaultdict
GAPS=[1,2,3,5,10,25]
for zp in sys.argv[1:]:
    z=zipfile.ZipFile(zp); names=[n for n in z.namelist() if n.endswith('Labels-GameState.json')]
    W=[];H=[];disp={g:[] for g in GAPS}; frames=0; ballframes=0; multi=0; edge=[]; seqs=0
    for n in names:
        d=json.loads(z.read(n)); seqs+=1
        imgs={im['image_id']:i for i,im in enumerate(d['images'])}
        frames+=sum(1 for im in d['images'] if im.get('is_labeled',True))
        pos=defaultdict(list)
        for a in d['annotations']:
            if a.get('category_id')!=4: continue
            b=a['bbox_image']; f=imgs.get(a['image_id'])
            if f is None: continue
            pos[f].append((b['x_center'],b['y_center'])); W.append(b['w']);H.append(b['h'])
            cx,cy=b['x_center'],b['y_center']; edge.append(min(cx,1920-cx,cy,1080-cy))
        ballframes+=len(pos); multi+=sum(1 for v in pos.values() if len(v)>1)
        for f,v in pos.items():
            for g in GAPS:
                if f+g in pos and len(v)==1 and len(pos[f+g])==1:
                    (x0,y0),(x1,y1)=v[0],pos[f+g][0]; disp[g].append(max(abs(x1-x0),abs(y1-y0)))
    W=np.array(W);H=np.array(H);edge=np.array(edge);sz=np.maximum(W,H)
    print(f'== {zp}: {seqs} seqs, {frames} labelled frames, ball in {ballframes} ({ballframes/frames:.1%}), multi-ball frames {multi}')
    print(' ball max(w,h) px pcts 1/5/25/50/75/95/99:',np.percentile(sz,[1,5,25,50,75,95,99]).round(1))
    print(' share max(w,h)<10:',(sz<10).mean().round(3),' <20:',(sz<20).mean().round(3))
    print(' dist to frame edge px pcts 1/5/10/25/50:',np.percentile(edge,[1,5,10,25,50]).round(0),' <160px:',(edge<160).mean().round(3),' <320px:',(edge<320).mean().round(3))
    for g in GAPS:
        a=np.array(disp[g]); print(f' gap {g:2d}: n={len(a)} chebyshev disp px p50/90/95/99/99.9/max:',np.percentile(a,[50,90,95,99,99.9,100]).round(0))

# --- 640x640 window coverage per gap (shift-clamped into the frame), both splits pooled ---
print('== 640 window coverage, all zips pooled')
for zp in sys.argv[1:]:
    z=zipfile.ZipFile(zp)
    for n in [n for n in z.namelist() if n.endswith('Labels-GameState.json')]:
        d=json.loads(z.read(n)); idx={im['image_id']:i for i,im in enumerate(d['images'])}; pos=defaultdict(list)
        for a in d['annotations']:
            if a.get('category_id')==4 and a['image_id'] in idx:
                b=a['bbox_image']; pos[idx[a['image_id']]].append((b['x_center'],b['y_center'],max(b['w'],b['h'])))
        for f,v in pos.items():
            if len(v)!=1: continue
            for g in gaps:
                w=pos.get(f+g)
                if not w or len(w)!=1: continue
                (x0,y0,_),(x1,y1,s)=v[0],w[0]
                X=min(max(round(x0-320),0),1280); Y=min(max(round(y0-320),0),440)  # shift-clamped 640 window
                cov[g].append(X<=x1-s/2 and x1+s/2<=X+640 and Y<=y1-s/2 and y1+s/2<=Y+640)
for g in gaps: print(f'gap {g:2d}: ball fully inside 640 window {np.mean(cov[g]):.2%}  n={len(cov[g])}')
