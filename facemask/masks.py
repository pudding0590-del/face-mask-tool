"""Head tracking and face masks.

v6 (0.1.x, validated 2026-09-22/23): pose-based head tracking, whole-head ellipse.
v7 (0.2.0, 2026-09-23 evening, after the user's first look at six new clips):
  - tracks never continue across a scene cut (the pipeline passes the cut frames);
  - the mask is an ellipse a little smaller than the face, centred on the eyes/nose ("比人脸小一圈即可");
    frames where only the ears/back of the head locate the head get a somewhat larger ellipse, blended smoothly;
  - tracks that look like statues are not masked (see statue_evidence) - Buddha statues were being masked.

Input: per-frame lists of people, each {"keypoints": {name: [x, y, score]}} with COCO names
(nose, leftEye, rightEye, leftEar, rightEar, leftShoulder, rightShoulder).
Output: per-frame mask polygons plus per-track segments for the review report.

Modes:
  face  - draw only while nose/eyes are visible (lenient)
  smart - draw whenever the head is visible EXCEPT while no face keypoint (nose/eyes) is seen at all
  head  - draw whenever the head is visible, including from behind
"""
import math

import cv2
import numpy as np

HEAD = ["nose", "leftEye", "rightEye", "leftEar", "rightEar"]
MODES = {"smart": "除纯背面（推荐）", "head": "含背面", "face": "仅正侧脸"}
# Ellipse semi-axes and the shift towards the chin, in head widths (s).  "face": the eyes/nose locate the face;
# "head": only the ears / head outline do (turned-away views), so the ellipse is larger and centred on the head.
SHAPE = {"face_a": .44, "face_b": .50, "face_down": .02, "head_a": .60, "head_b": .66, "head_down": .10,
         "pad": .005,     # plus this share of the frame's short side on both axes, so small heads keep a few pixels of margin
         "small": .09,    # heads narrower than this share of the short side get relatively larger ellipses ...
         "small_gain": .8,  # ... up to (1 + small_gain) x for the tiniest (their keypoints are much less precise)
         "centre": "eyes",  # "eyes": a third of the way from the eyes to the nose; "mean": mean of visible eyes + nose
         "head_centre": "toward_face"}  # face not located: "toward_face" (ears pulled towards the eye/nose guesses) or "keypoints"


def _pt(k, n, thr):
    v = k.get(n)
    return None if v is None or v[2] < thr else np.array(v[:2], float)


def _dist(a, b):
    return None if a is None or b is None else float(np.linalg.norm(a - b))


def observe(person):
    """One person in one frame -> head centre, face centre, head width estimate, visibility signals."""
    k = person["keypoints"]
    conf = lambda n: k.get(n, [0, 0, 0])[2]
    nose, le, re, la, ra = [_pt(k, n, .3) for n in HEAD]
    have = [p for p in (nose, le, re, la, ra) if p is not None]
    if not have:
        return None
    # head-width cues; profile views only shrink ear-ear / eye-eye, so the max of several cues is taken
    est = [d * f for d, f in [(_dist(la, ra), 1.0), (_dist(le, re), 2.3), (_dist(la, nose), 1.0), (_dist(ra, nose), 1.0),
                              (_dist(la, le), 1.4), (_dist(la, re), 1.4), (_dist(ra, le), 1.4), (_dist(ra, re), 1.4)] if d]
    d = _dist(_pt(k, "leftShoulder", .3), _pt(k, "rightShoulder", .3))
    if not est and d:
        est.append(d * .4)
    if not est:
        return None
    face = [p for p in (nose, le, re) if p is not None]
    vis = max(conf("nose"), conf("leftEye"), conf("rightEye"))
    # mask centre: a third of the way from the eyes to the nose - the plain mean of the visible points drifts
    # half-way to the nose in profile (one eye + nose), leaving the eye near the ellipse's edge
    eyes = [p for p in (le, re) if p is not None]
    if eyes:
        e = np.mean(eyes, axis=0)
        mcenter = e + (nose - e) / 3 if nose is not None else e
    else:
        mcenter = nose
    # head placement when the face itself is not located: RTMO still puts its low-confidence eye/nose guesses on
    # the side the face is turned to, so pull the ear-based centre half-way towards them (a turned-away profile
    # otherwise gets its ellipse over the back of the head, with the eye at the rim - clip 11, 4.2 s)
    ears = [p for p in (la, ra) if p is not None]
    guesses = [np.array(k[n][:2], float) for n in ("nose", "leftEye", "rightEye") if n in k and k[n][2] > 0]
    hcenter = np.mean(have, axis=0)
    if ears and guesses and not face:
        hcenter = (np.mean(ears, axis=0) + np.mean(guesses, axis=0)) / 2
    o = dict(center=np.mean(have, axis=0), fcenter=np.mean(face, axis=0) if face else None, scale=max(est),
             vis=vis, hconf=max(conf(n) for n in HEAD), back=vis < .15, angle=None, person=person, mcenter=mcenter,
             hcenter=hcenter)
    for p, q, cn, cm, minlen in [(le, re, "leftEye", "rightEye", .25), (la, ra, "leftEar", "rightEar", .5)]:
        if p is not None and q is not None and min(conf(cn), conf(cm)) >= .5:
            v = q - p
            if v[0] < 0:
                v = -v
            if np.linalg.norm(v) >= minlen * o["scale"]:
                o["angle"] = math.degrees(math.atan2(v[1], v[0]))
                break
    return o


def track(people_per_frame, fps, cuts=()):
    """Greedy nearest-neighbour linking; gates use a robust per-track head size (median of the last second),
    so a bowed or turned head whose single-frame size estimate collapses does not break the track.
    A track never continues across a scene cut (frame numbers in `cuts` start a new shot)."""
    tracks = []
    maxgap = round(fps * 1.0)
    w = int(round(fps))
    cutset, shot, shot_of = set(cuts or ()), 0, []
    for n in range(len(people_per_frame)):
        shot += n in cutset
        shot_of.append(shot)
    for n, people in enumerate(people_per_frame):
        obs = [o for o in (observe(p) for p in people) if o]
        cands = []
        for i, o in enumerate(obs):
            for ti, t in enumerate(tracks):
                ln, lo = t[-1]
                if ln == n or n - ln > maxgap or shot_of[ln] != shot_of[n]:
                    continue
                ref = float(np.median([x["scale"] for _, x in t[-w:]]))
                u = max(o["scale"], ref)
                d = np.linalg.norm(o["center"] - lo["center"]) / u
                if d < 0.9 + 0.12 * (n - ln) and 0.33 < o["scale"] / ref < 3:
                    cands.append((d, i, ti))
        ui, ut = set(), set()
        for d, i, ti in sorted(cands):
            if i in ui or ti in ut:
                continue
            ui.add(i)
            ut.add(ti)
            tracks[ti].append((n, obs[i]))
        for i, o in enumerate(obs):
            if i not in ui:
                tracks.append([(n, o)])
    return tracks


def _smooth(x, win):
    x = np.asarray(x, float)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, np.ones(win) / win, mode="valid")[:len(x)]


def _medfilt(x, win):
    x = np.asarray(x, float)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.array([np.median(xp[i:i + win]) for i in range(len(x))])


def _fill_nan(x):
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    return x if ok.sum() == 0 else np.interp(np.arange(len(x)), np.flatnonzero(ok), x[ok])


def _dilate(mask, r):
    out = mask.copy()
    for i in np.flatnonzero(mask):
        out[max(0, i - r):i + r + 1] = True
    return out


def _runs(flags, offset=0):
    runs, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        if not f and start is not None:
            runs.append((start + offset, i - 1 + offset))
            start = None
    return runs


def ellipse_poly(cx, cy, a, b, ang):
    return cv2.ellipse2Poly((int(round(cx)), int(round(cy))), (max(2, int(round(a))), max(2, int(round(b)))),
                            int(round(ang)), 0, 360, 12).tolist()


def build_masks(people_per_frame, fps, mode="smart", cuts=(), shape=None, statue=None, frame_size=None):
    """Returns (frames, segments): frames[n] = {"frame": n, "bars": [...]}; segments = per-track review info.

    cuts   - frame numbers that start a new shot (tracks and smoothing never cross them)
    shape  - overrides for SHAPE (used by the evaluation scripts)
    statue - optional callable(track) -> evidence dict with a boolean "statue"; such tracks are not drawn
             but are reported in segments so the review notes can point at them.
    frame_size - (width, height), for the small fixed margin added to every ellipse"""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}")
    shape = {**SHAPE, **(shape or {})}
    pad_px = shape["pad"] * min(frame_size) if frame_size else 0.0
    small_px = shape["small"] * min(frame_size) if frame_size else 0.0
    N = len(people_per_frame)
    out = [{"frame": n, "bars": []} for n in range(N)]
    segments = []
    w5 = max(3, int(round(fps / 5)) | 1)
    w1s = int(round(fps)) | 1
    wq = int(round(fps * 3)) | 1
    pad = round(fps * .25)
    hi, lo, fill = (.4, .15, round(fps * 1.0)) if mode == "face" else (.5, .3, round(fps * .6))
    tracks = [t for t in track(people_per_frame, fps, cuts) if len(t) >= max(4, fps * .2) and max(o["hconf"] for _, o in t) >= .5]
    tracks.sort(key=len, reverse=True)   # longer tracks are drawn first and win when two masks overlap
    for tid, t in enumerate(tracks):
        n0, n1 = t[0][0], t[-1][0]
        L = n1 - n0 + 1
        idx = {n: o for n, o in t}
        ii = np.arange(L)
        seen = np.zeros(L, bool)
        seen[[n - n0 for n in idx]] = True

        def signal(o):
            if mode == "face":
                return o["vis"]
            if mode == "smart":
                return 0.0 if o["back"] else o["hconf"]
            return o["hconf"]

        sig = np.array([signal(idx[n]) if n in idx else np.nan for n in range(n0, n1 + 1)])
        sig = _medfilt(_fill_nan(sig), w5)
        on = np.zeros(L, bool)
        st = False
        for i in range(L):
            st = sig[i] >= (lo if st else hi)
            on[i] = st
        if on.sum() < max(4, fps * .2):
            continue
        evidence = statue(t) if statue else None
        if evidence and evidence.get("statue"):
            segments.append({"track": tid, "start": n0, "end": n1, "drawn": [], "seen_not_drawn": [],
                             "filled_gaps": [], "statue": evidence, "statue_spans": [(n0, n1)]})   # whole span: statues flicker
            continue
        # per frame: face centre (eyes/nose), head centre (any head keypoint), size, roll, and whether the face locates it
        fx, fy, hx, hy, sc, an, wf = (np.full(L, np.nan) for _ in range(7))
        for i, n in enumerate(range(n0, n1 + 1)):
            o = idx.get(n)
            if o is None:
                continue
            face_ok = o["fcenter"] is not None and o["vis"] >= .3
            if face_ok:
                fx[i], fy[i] = o["mcenter"] if shape["centre"] == "eyes" else o["fcenter"]
            hx[i], hy[i] = o["hcenter"] if shape["head_centre"] == "toward_face" else o["center"]
            wf[i] = 1.0 if face_ok else 0.0
            sc[i] = o["scale"]
            if o["angle"] is not None:
                an[i] = o["angle"]
        okf, okh = ~np.isnan(fx), ~np.isnan(hx)
        if mode == "face":
            ok = okf
            hx, hy, wf = fx.copy(), fy.copy(), np.where(okf, 1.0, np.nan)
        else:
            ok = okh
            if not okf.any():
                fx, fy = hx.copy(), hy.copy()
        if ok.sum() < 3:
            continue
        fx, fy = _smooth(_fill_nan(fx), w5), _smooth(_fill_nan(fy), w5)
        hx, hy = _smooth(_fill_nan(hx), w5), _smooth(_fill_nan(hy), w5)
        wf = np.clip(_smooth(_fill_nan(wf), w5 * 2 + 1), 0, 1)   # blend face/head placement over ~0.4 s, no jumps
        cx, cy = wf * fx + (1 - wf) * hx, wf * fy + (1 - wf) * hy
        ka = wf * shape["face_a"] + (1 - wf) * shape["head_a"]
        kb = wf * shape["face_b"] + (1 - wf) * shape["head_b"]
        kd = wf * shape["face_down"] + (1 - wf) * shape["head_down"]
        # head size: median kills single-frame spikes; foreshortening only shrinks the estimate, so use a rolling upper
        # envelope (+-1.5s, 90th pct) and never drop below 85% of the largest size seen within +-5s (95th pct, capped
        # at 1.3x the 90th pct of that window so a spike shorter than ~1s cannot inflate it)
        sc = _medfilt(_fill_nan(sc), w5)
        padq = wq // 2
        scp = np.pad(sc, (padq, padq), mode="edge")
        env = np.array([np.percentile(scp[i:i + wq], 90) for i in range(L)])
        wr = int(round(fps * 10)) | 1
        padr = wr // 2
        scr = np.pad(sc, (padr, padr), mode="edge")
        recent = np.array([min(np.percentile(scr[i:i + wr], 95), 1.3 * np.percentile(scr[i:i + wr], 90)) for i in range(L)])
        sc = _smooth(np.maximum(env, .85 * recent), w1s)
        an = np.clip(_smooth(_medfilt(_fill_nan(an), w5), w5), -35, 35) if (~np.isnan(an)).sum() >= 2 else np.zeros(L)
        draw = on.copy()
        last = None
        for i in range(L):
            if on[i]:
                if last is not None and 1 < i - last <= fill:
                    draw[last:i] = True
                last = i
        draw = _dilate(draw, pad)   # start a quarter second early and end a quarter second late
        first, lastok = ii[ok][0], ii[ok][-1]
        draw[:first] = False
        draw[lastok + 1:] = False   # never extrapolate past real observations
        for i in range(L):
            if not draw[i]:
                continue
            s = sc[i]
            grow = 1 + shape["small_gain"] * max(0.0, 1 - s / small_px) if small_px else 1.0
            a, b = ka[i] * s * grow + pad_px, kb[i] * s * grow + pad_px
            th = math.radians(an[i])
            ccx, ccy = cx[i] - math.sin(th) * kd[i] * s, cy[i] + math.cos(th) * kd[i] * s   # shift towards the chin
            out[n0 + i]["bars"].append({"track": tid, "polygon": ellipse_poly(ccx, ccy, a, b, an[i]),
                                        "ellipse": [float(ccx), float(ccy), float(a), float(b), float(an[i])],
                                        "face": [float(ccx - s / 2), float(ccy - s / 2), float(s), float(s)],
                                        "face_weight": round(float(wf[i]), 2), "origin": "pose_face", "mode": mode})
        segments.append({"track": tid, "start": n0, "end": n1, "drawn": _runs(draw, n0),
                         "seen_not_drawn": _runs(seen & ~draw, n0), "filled_gaps": _runs(draw & ~seen, n0), "statue": evidence})
    for row in out:   # one mask per head: a later (shorter-track) mask within 0.6 head-widths of a kept one is a duplicate
        keep = []
        for b in row["bars"]:
            c = np.asarray(b["ellipse"][:2])
            if any(np.linalg.norm(c - np.asarray(k["ellipse"][:2])) < .6 * max(b["face"][2], k["face"][2]) for k in keep):
                continue
            keep.append(b)
        row["bars"] = keep
    return out, segments
