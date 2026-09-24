"""Logic self-test for masks + statue test with synthetic people (no video, no model): runs in a second.
- a moving person with skin is masked, with an ellipse about 0.9 head widths wide (0.2.0: smaller than the face);
- a still, skin-less, low-confidence "statue" is not masked and is reported as a statue;
- a head at the same place on both sides of a scene cut becomes two tracks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facemask import masks, scene  # noqa: E402

FPS, N, W, H = 30.0, 90, 1080, 1920


def person(x, y, s, box, skin, still):
    k = {"nose": [x, y + .15 * s, .95], "leftEye": [x + .22 * s, y, .95], "rightEye": [x - .22 * s, y, .95],
         "leftEar": [x + .5 * s, y + .05 * s, .9], "rightEar": [x - .5 * s, y + .05 * s, .9],
         "leftShoulder": [x + s, y + 1.3 * s, .9], "rightShoulder": [x - s, y + 1.3 * s, .9]}
    return {"keypoints": k, "box_score": box, "skin": skin, "_head": None, "_scale": s,
            "rel_next": [0.0, 0.0] if still else [.08 * s, .03 * s]}


people = []
for n in range(N):
    row = [person(300 + 3 * n, 600, 150, .96, 1.0, still=False)]          # a live person walking to the right
    if n % 3 == 0:                                                         # a statue the detector keeps losing
        row.append(person(800, 1200, 120, .72, 0.0, still=True))
    people.append(row)
for row in people:
    for p in row:
        p["_head"] = masks.observe(p)["fcenter"]

frames, segments = masks.build_masks(people, FPS, "smart", frame_size=(W, H),
                                     statue=lambda t: scene.statue_evidence(t, FPS, colour=20.0, min_side=min(W, H)))
live = [b for f in frames for b in f["bars"] if b["ellipse"][0] < 700]
statue_bars = [b for f in frames for b in f["bars"] if abs(b["ellipse"][0] - 800) < 100]
assert len(live) >= N - 2, f"live person masked in only {len(live)} of {N} frames"
assert not statue_bars, f"statue masked in {len(statue_bars)} frames"
assert any((s.get("statue") or {}).get("statue") for s in segments), "statue not reported"
width = 2 * live[len(live) // 2]["ellipse"][2] / 150
assert .8 <= width <= 1.05, f"ellipse width {width:.2f} head widths (expected ~0.9)"

same_place = [[person(500, 800, 150, .96, 1.0, still=False)] for _ in range(60)]
tracks = [t for t in masks.track(same_place, FPS, cuts=[30])]
assert len(tracks) == 2 and len(tracks[0]) == 30, [len(t) for t in tracks]
print(f"logic test ok: live person masked {len(live)}/{N} frames, ellipse {width:.2f} head widths, statue not masked, cut splits tracks")
