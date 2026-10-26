#!/usr/bin/env python3
"""Run the hand tracker on the live camera.

  python run.py                         # viewer window on the Jetson's display
  python run.py --headless              # no window; stats printed once per second
  python run.py --udp 127.0.0.1:9000    # also stream results as OSC over UDP
  python run.py --udp 192.168.1.20:9000 --udp-format json

Viewer keys: q / Esc quit, m toggle mirrored view, r toggle raw (unfiltered) skeleton,
s save a snapshot PNG.
"""
import argparse
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from handtrack import config as cfgmod  # noqa: E402


def print_stats(service):
    st = service.stats()
    hf = service.latest()
    hands = "" if hf is None else " ".join(f"#{h.track_id}{h.handedness[0]}" for h in hf.hands)

    def p(k, q="p50"):
        return st.get(k, {}).get(q, float("nan"))
    print(f"cam {st['camera_fps']:4.1f} fps | proc {st['processing_fps']:4.1f} fps | "
          f"latency p50 {p('latency'):5.2f} p95 {p('latency', 'p95'):5.2f} ms | "
          f"usb {p('transfer'):4.1f} dec {p('decode'):4.1f} det {p('detect'):4.1f}/def {p('detect_deferred'):4.1f} "
          f"({st['detector_ran_frac'] * 100:3.0f}%) lm {p('landmarks'):4.1f} post {p('post_filter'):4.2f} "
          f"pub {p('publish'):4.2f} ms | hands: {hands or '-'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="config file (default: config.yaml next to this script)")
    ap.add_argument("--headless", action="store_true", help="no window; print stats once per second")
    ap.add_argument("--udp", metavar="HOST:PORT", help="send results over UDP")
    ap.add_argument("--udp-format", choices=["osc", "json"], help="UDP payload format (default from config: osc)")
    ap.add_argument("--seconds", type=float, help="stop after this long (for tests)")
    ap.add_argument("--snapshot", metavar="PNG", help="viewer: save the rendered view here on exit / key s")
    args = ap.parse_args()

    overrides = {}
    if args.udp:
        overrides.setdefault("output", {})["udp"] = args.udp
    if args.udp_format:
        overrides.setdefault("output", {})["udp_format"] = args.udp_format
    cfg = cfgmod.load(args.config, overrides)

    from handtrack.tracker import TrackingService
    print("starting (first run builds TensorRT engines; that can take a few minutes)...", flush=True)
    service = TrackingService(cfg)
    sender = None
    if cfg.output.udp:
        from handtrack.output_udp import UdpSender, parse_hostport
        host, port = parse_hostport(cfg.output.udp)
        sender = UdpSender(host, port, cfg.output.udp_format)
        service.subscribe(sender)
        print(f"sending {cfg.output.udp_format.upper()} to udp://{host}:{port}", flush=True)
    service.start()

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    t_end = time.monotonic() + args.seconds if args.seconds else None
    try:
        use_viewer = not args.headless
        if use_viewer:
            from handtrack.viewer import ensure_display, run_viewer
            if not ensure_display():
                print("no display found (set DISPLAY or use --headless); running headless", flush=True)
                use_viewer = False
        if use_viewer:
            run_viewer(service, mirror=cfg.viewer.mirror, show_raw=cfg.viewer.show_raw,
                       max_seconds=args.seconds, snapshot=args.snapshot, should_stop=lambda: stop["flag"])
        else:
            next_print = time.monotonic() + 1
            while not stop["flag"] and (t_end is None or time.monotonic() < t_end):
                time.sleep(0.05)
                if service.error:
                    raise service.error
                if time.monotonic() >= next_print:
                    print_stats(service)
                    next_print += 1
    finally:
        service.stop()
        if sender:
            sender.close()
            print(f"udp: {sender.sent} datagrams sent, {sender.errors} errors")


if __name__ == "__main__":
    main()
