"""Per-frame scene signals computed during the pose pass (no extra decoding):

- scene cuts: a large, isolated jump in a 32x32 grey thumbnail (tracks must not continue across a cut);
- motion: dense optical flow on a small grey frame, pooled to a coarse grid; for each head, how it moved
  against the ring of scene around it (a statue moves exactly like its niche; a live head does not);
- face colour: share of skin-coloured pixels in the centre of each face (a hue/saturation sector set on the
  Asian faces of the 26 test clips, 2026-09-23), and how colourful the footage is overall.

These feed masks.build_masks (cuts) and statue_evidence (statue-like tracks are not masked).
"""
import cv2
import numpy as np

# Skin colour as a sector around neutral grey in the (Cr, Cb) plane: skin is tinted towards red-yellow (hue
# between SKIN["hue"] degrees, measured as atan2(Cb-128, Cr-128)), more than grey stone and less than gold.
# Set on the Asian faces of the 26 test clips (2026-09-23): pale faces under cool light sit close to grey but on the
# red-yellow side; grey stone sits on the blue side; gilded statues have the skin hue but about twice the saturation.
SKIN = {"hue": (-80.0, 0.0), "chroma": (3.0, 40.0), "luma": (40, 235)}
CUT_MAD, CUT_RATIO = 25.0, 4.0


def _crop(frame, cx, cy, side, out=32):
    h, w = frame.shape[:2]
    half = max(2.0, side / 2)
    x0, y0, x1, y1 = int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return None
    pad = cv2.copyMakeBorder(frame[max(0, y0):min(h, y1), max(0, x0):min(w, x1)],
                             max(0, -y0), max(0, y1 - h), max(0, -x0), max(0, x1 - w), cv2.BORDER_CONSTANT)
    return cv2.resize(pad, (out, out), interpolation=cv2.INTER_AREA)


def skin_mask(ycrcb):
    """Boolean skin test for an (N, 3) array of Y, Cr, Cb values."""
    y, cr, cb = ycrcb[:, 0], ycrcb[:, 1] - 128.0, ycrcb[:, 2] - 128.0
    hue = np.degrees(np.arctan2(cb, cr))
    chroma = np.hypot(cr, cb)
    return ((y >= SKIN["luma"][0]) & (y <= SKIN["luma"][1]) & (hue >= SKIN["hue"][0]) & (hue <= SKIN["hue"][1])
            & (chroma >= SKIN["chroma"][0]) & (chroma <= SKIN["chroma"][1]))


def skin_share(crop32, inner=20):
    """Share of skin-coloured pixels in the centre of a 32x32 face crop."""
    o = (32 - inner) // 2
    c = cv2.cvtColor(np.ascontiguousarray(crop32[o:o + inner, o:o + inner]), cv2.COLOR_BGR2YCrCb).reshape(-1, 3).astype(float)
    return float(np.mean(skin_mask(c)))


def head_motion(pooled, cell, head, scale, camera_step):
    """How the pixels on a head moved between two frames, relative to the pixels in a ring around it
    (0.9-2 head widths).  A statue moves exactly like the niche/rock/wall around it; a live head does not.
    Falls back to the dominant camera motion when the ring is mostly outside the frame (close-ups)."""
    gh, gw = pooled.shape[:2]
    ys, xs = np.mgrid[0:gh, 0:gw]
    dist = np.hypot((xs + .5) * cell - head[0], (ys + .5) * cell - head[1]) / max(1.0, scale)
    on_head = pooled[dist <= .35]
    if len(on_head) == 0:
        near = np.unravel_index(np.argmin(dist), dist.shape)
        on_head = pooled[near][None]
    ring = pooled[(dist >= .9) & (dist <= 2.0)]
    if len(ring) >= 4:
        return on_head.mean(axis=0) - np.median(ring, axis=0)
    A = np.asarray(camera_step, float).reshape(2, 3)
    return on_head.mean(axis=0) - (A[:, :2] @ head + A[:, 2] - head)


def colourfulness(small_bgr):
    """Mean distance from grey in the Cr/Cb plane (0 for black-and-white footage)."""
    c = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2YCrCb).reshape(-1, 3).astype(float)
    return float(np.mean(np.hypot(c[:, 1] - 128, c[:, 2] - 128)))


def head_point(person):
    """Face centre (eyes/nose) or head centre, and a head-width estimate - same cues as masks.observe."""
    from .masks import observe
    o = observe(person)
    if not o:
        return None, None
    c = o["fcenter"] if o["fcenter"] is not None else o["center"]
    return c, o["scale"]


class SceneAnalyzer:
    def __init__(self, width, height):
        self.W, self.H = width, height
        sw = 270 if width < height else 480
        self.sh = int(round(height * sw / width)) // 8 * 8
        self.sw = sw // 8 * 8
        self.r = width / self.sw
        gy, gx = np.mgrid[4:self.sh:8, 4:self.sw:8]
        self.grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32) * self.r
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.prev_small = self.prev_thumb = None
        self.prev_people = []
        self.mad, self.affine, self.colour = [], [], []

    def add(self, frame, people):
        """Call once per decoded frame, in order, with the people found in it (annotated in place)."""
        small_bgr = cv2.resize(frame, (self.sw, self.sh), interpolation=cv2.INTER_AREA)
        small = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
        self.colour.append(colourfulness(small_bgr))
        thumb = cv2.resize(small, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.prev_small is None:
            self.mad.append(0.0)
            self.affine.append(np.array([[1, 0, 0], [0, 1, 0]], float))
        else:
            self.mad.append(float(np.abs(thumb - self.prev_thumb).mean()))
            f = self.dis.calc(self.prev_small, small, None)
            pooled = f.reshape(self.sh // 8, 8, self.sw // 8, 8, 2).mean(axis=(1, 3)) * self.r
            A, inl = cv2.estimateAffinePartial2D(self.grid, self.grid + pooled.reshape(-1, 2), method=cv2.RANSAC,
                                                 ransacReprojThreshold=.01 * self.W)
            self.affine.append(A if A is not None else np.array([[1, 0, 0], [0, 1, 0]], float))
            for p in self.prev_people:     # how each previous head moved into this frame, against its surroundings
                if p.get("_head") is not None:
                    v = head_motion(pooled, 8 * self.r, p["_head"], p["_scale"], self.affine[-1])
                    p["rel_next"] = [float(v[0]), float(v[1])]
        for p in people:
            c, s = head_point(p)
            p["_head"], p["_scale"] = c, s
            if c is not None:
                crop = _crop(frame, c[0], c[1] + .1 * s, .8 * s)
                p["skin"] = skin_share(crop) if crop is not None else None
        self.prev_small, self.prev_thumb, self.prev_people = small, thumb, people

    def cuts(self):
        """Frame numbers that start a new shot: a big thumbnail change that is an isolated spike (not a fast pan)."""
        mad = np.asarray(self.mad, float)
        out = []
        for n in range(1, len(mad)):
            nb = np.concatenate([mad[max(1, n - 5):n], mad[n + 1:n + 6]])
            base = float(np.median(nb)) if len(nb) else 0.0
            if mad[n] >= CUT_MAD and mad[n] >= CUT_RATIO * max(base, 1.0):
                out.append(n)
        return out


# ---- statue test (thresholds set on the 26 test clips, 2026-09-23; see 新片测试_2026-09-23/分析说明.md) ----
# Two kinds of statue were seen: grey/gilded faces with no skin colour at all (Lingyin stone and gold Buddhas),
# and weathered carvings whose stone has a skin-like tint but which the detector keeps losing (box score hovering
# at the 0.7 detection threshold, found in under half of the frames) and which never move against their niche.
# Real people in the 26 clips: any with no skin colour were back views with box >= 0.93 that moved; any with
# box <= 0.8 were tiny background figures (< 40 px heads).  The rule only looks at heads at least 4% of the
# frame's short side (smaller ones stay masked: their cues are unreliable).
STATUE = {"skin_share": .30,      # a face crop "shows skin" when this share of its centre is skin-coloured
          "min_face_frames": 5,   # need at least this many clear-face frames before calling anything a statue
          "min_head": .04,        # ... and a head at least this share of the frame's short side
          "box_max": .90,         # a statue never convinces the detector more than this (median box score)
          "no_skin": .10,         # kind 1: skin in at most this share of its clear-face frames ...
          "no_skin_motion": .05,  # ... and no more than this motion against its surroundings (or too few frames to tell)
          "flicker_box": .78,     # kind 2: detector barely sure (median box score) ...
          "flicker_density": .5,  # ... found in at most this share of the frames of its span ...
          "still_max": .03,       # ... and measured perfectly still (head widths per 0.5 s against its surroundings)
          "colour_min": 2.0}      # footage greyer than this (black-and-white) never gets the statue test


def relative_motion(track, fps, window=.5):
    """90th percentile over sliding 0.5 s windows of the head's net displacement against its surroundings,
    in head widths: ~0 for anything fixed to the scene (a statue in its niche), larger for a live head."""
    k = max(2, int(round(fps * window)))
    steps = {}
    for n, o in track:
        p = o["person"]
        if p.get("rel_next") is not None:
            steps[n] = np.asarray(p["rel_next"]) / max(1.0, p["_scale"])
    vals = []
    for n in sorted(steps):
        win = [steps[m] for m in range(n - k + 1, n + 1) if m in steps]
        if len(win) >= .6 * k:
            vals.append(float(np.linalg.norm(np.sum(win, axis=0))))
    return float(np.percentile(vals, 90)) if vals else None


def statue_evidence(track, fps, colour=None, min_side=None, params=None):
    """Decide whether a head track is a statue (or other lifelike figure) rather than a person.
    Deliberately one-sided - a real person must never be dropped - so every condition has to hold."""
    P = {**STATUE, **(params or {})}
    if colour is not None and colour < P["colour_min"]:
        return {"statue": False, "reason": "black-and-white footage"}
    scale = float(np.median([o["scale"] for _, o in track]))
    if min_side and scale < P["min_head"] * min_side:
        return {"statue": False, "reason": "head too small to judge"}
    faces = [o for _, o in track if o["vis"] >= .5 and o["person"].get("skin") is not None]
    if len(faces) < P["min_face_frames"]:
        return {"statue": False, "reason": "too few clear face frames"}
    skin = float(np.mean([o["person"]["skin"] >= P["skin_share"] for o in faces]))
    box = float(np.median([o["person"].get("box_score", 1.0) for _, o in track]))
    density = len(track) / (track[-1][0] - track[0][0] + 1)
    motion = relative_motion(track, fps)
    no_skin = skin <= P["no_skin"] and (motion is None or motion <= P["no_skin_motion"])
    flicker = box <= P["flicker_box"] and density <= P["flicker_density"] and motion is not None and motion <= P["still_max"]
    statue = box <= P["box_max"] and (no_skin or flicker)
    return {"statue": bool(statue), "kind": "no_skin" if statue and no_skin else ("flicker" if statue else None),
            "skin_frames": round(skin, 3), "box": round(box, 3), "density": round(density, 2),
            "motion": None if motion is None else round(motion, 3), "head_px": round(scale)}
