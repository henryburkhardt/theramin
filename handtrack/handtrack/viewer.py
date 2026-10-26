"""Optional viewer: skeletons, track IDs, handedness and live stats.

It never slows the tracker: it runs on the main (GUI) thread, takes the newest
(HandFrame, JPEG) pair whenever it's ready, decodes the JPEG itself on the CPU at
half resolution, and skips anything it can't keep up with.
"""
import os
import time

import cv2
import numpy as np

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
         (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
COLORS = {"Left": (255, 160, 0), "Right": (0, 140, 255)}        # BGR
TIPS = [4, 8, 12, 16, 20]


def ensure_display():
    """Over SSH, DISPLAY may be unset or lack XAUTHORITY; use the local GDM session.
    Returns True if a display looks usable."""
    if not os.environ.get("DISPLAY"):
        for d in (":1", ":0"):
            if os.path.exists(f"/tmp/.X11-unix/X{d[1:]}"):
                os.environ["DISPLAY"] = d
                break
    if os.environ.get("DISPLAY") and not os.environ.get("XAUTHORITY"):
        gdm = f"/run/user/{os.getuid()}/gdm/Xauthority"
        if os.path.exists(gdm):
            os.environ["XAUTHORITY"] = gdm
    return bool(os.environ.get("DISPLAY"))


def _draw_hand(img, h, mirror, show_raw):
    H, W = img.shape[:2]

    def pts(lm_norm):                       # normalized coords -> this image's pixels
        p = lm_norm[:, :2] * [W, H]
        if mirror:
            p[:, 0] = W - 1 - p[:, 0]
        return p.astype(int)

    col = COLORS.get(h.handedness, (200, 200, 200))
    if show_raw:
        pr = pts(h.landmarks_norm)
        for a, b in EDGES:
            cv2.line(img, tuple(pr[a]), tuple(pr[b]), (120, 120, 120), 1, cv2.LINE_AA)
    p = pts(h.landmarks_filtered)
    for a, b in EDGES:
        cv2.line(img, tuple(p[a]), tuple(p[b]), col, 2, cv2.LINE_AA)
    for i, q in enumerate(p):
        cv2.circle(img, tuple(q), 5 if i in TIPS else 3, (255, 255, 255) if i in TIPS else col, -1, cv2.LINE_AA)
    x, y = p[0]
    label = f"#{h.track_id} {h.handedness} {h.handedness_score:.2f}"
    cv2.putText(img, label, (x - 40, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, label, (x - 40, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)


def _stats_lines(st, viewer_fps):
    def ms(k):
        v = st.get(k, {})
        return f"{v.get('p50', float('nan')):5.2f}"
    return [
        f"camera {st['camera_fps']:5.1f} fps   processing {st['processing_fps']:5.1f} fps   viewer {viewer_fps:4.1f} fps",
        f"latency capture->landmarks p50 {ms('latency')} ms  p95 {st['latency']['p95']:5.2f} ms",
        f"usb {ms('transfer')}  decode {ms('decode')}  detect {ms('detect')} / deferred {ms('detect_deferred')} "
        f"({st['detector_ran_frac']*100:3.0f}%)",
        f"landmarks {ms('landmarks')}  post+filter {ms('post_filter')}  publish {ms('publish')}  (ms, p50)",
    ]


def run_viewer(service, mirror=True, show_raw=False, title="handtrack (q to quit)", max_seconds=None,
               snapshot=None, should_stop=lambda: False):
    """snapshot: path to save the rendered view to (key s, and once at exit if given)."""
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    t_end = time.monotonic() + max_seconds if max_seconds else None
    img = None
    was_visible = False                     # GTK maps the window a moment after imshow
    cv2.resizeWindow(title, 1280, 720)
    width = service.cfg.camera.width
    version, shown, t_last = 0, 0, time.monotonic()
    fps = 0.0
    while True:
        r = service.wait_newer(version, timeout=0.1)
        if r is not None:
            version, (hf, jpeg) = r
            if jpeg is not None:
                img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
            else:
                img = np.zeros((service.cfg.camera.height // 2, width // 2, 3), np.uint8)
            if mirror:
                img = cv2.flip(img, 1)
            for h in hf.hands:
                _draw_hand(img, h, mirror, show_raw)
            shown += 1
            now = time.monotonic()
            if now - t_last >= 1.0:
                fps, shown, t_last = shown / (now - t_last), 0, now
            y = 20
            for line in _stats_lines(service.stats(), fps):
                cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                y += 18
            cv2.imshow(title, img)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or should_stop() or (t_end and time.monotonic() > t_end):
            break
        if key == ord("s") and img is not None:
            path = snapshot or time.strftime("handtrack_%Y%m%d_%H%M%S.png")
            cv2.imwrite(path, img)
            print(f"saved {path}", flush=True)
        if key == ord("m"):
            mirror = not mirror
        if key == ord("r"):
            show_raw = not show_raw
        try:                                # window closed with the title-bar button?
            visible = cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            visible = False
        if visible:
            was_visible = True
        elif was_visible:
            break
    if snapshot and img is not None:
        cv2.imwrite(snapshot, img)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
