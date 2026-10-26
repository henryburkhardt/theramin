"""UDP output for a separate synth process. Off by default (run.py --udp HOST:PORT).

Format "osc" (default): one OSC bundle per processed frame, timetag "immediately":
  /handtrack/frame  i frame_id, h t_capture_ns, h t_publish_ns, i n_hands
  /handtrack/hand   i track_id, s handedness ("Left"/"Right"), f score,
                    63 f landmarks_filtered (x0 y0 z0 ... x20 y20 z20, normalized 0..1),
                    42 f velocity (vx0 vy0 ... , normalized units per second)
One /handtrack/hand message per hand; ~1.2 KB per bundle with two hands (fits one datagram).

Format "json": one compact JSON object per datagram:
  {"f": frame_id, "tc": t_capture_ns, "tp": t_publish_ns,
   "hands": [{"id": .., "h": "Left", "s": .., "lm": [63 floats], "v": [42 floats]}]}

Sending runs on its own thread with a newest-only slot, so a slow network never
delays tracking; if the sender falls behind, it skips to the latest frame.
"""
import json
import socket
import threading

from .capture import LatestFrame


def encode_osc(hf):
    from pythonosc.osc_bundle_builder import IMMEDIATELY, OscBundleBuilder
    from pythonosc.osc_message_builder import OscMessageBuilder
    b = OscBundleBuilder(IMMEDIATELY)
    m = OscMessageBuilder("/handtrack/frame")
    m.add_arg(int(hf.frame_id), "i")
    m.add_arg(int(hf.t_capture_ns), "h")
    m.add_arg(int(hf.t_publish_ns), "h")
    m.add_arg(len(hf.hands), "i")
    b.add_content(m.build())
    for h in hf.hands:
        m = OscMessageBuilder("/handtrack/hand")
        m.add_arg(int(h.track_id), "i")
        m.add_arg(h.handedness, "s")
        m.add_arg(float(h.score), "f")
        for v in h.landmarks_filtered.reshape(-1):
            m.add_arg(float(v), "f")
        for v in h.velocity.reshape(-1):
            m.add_arg(float(v), "f")
        b.add_content(m.build())
    return b.build().dgram


def encode_json(hf):
    return json.dumps({
        "f": hf.frame_id, "tc": hf.t_capture_ns, "tp": hf.t_publish_ns,
        "hands": [{"id": h.track_id, "h": h.handedness, "s": round(float(h.score), 3),
                   "lm": [round(float(v), 5) for v in h.landmarks_filtered.reshape(-1)],
                   "v": [round(float(v), 4) for v in h.velocity.reshape(-1)]} for h in hf.hands],
    }, separators=(",", ":")).encode()


class UdpSender:
    def __init__(self, host, port, fmt="osc"):
        self.addr = (host, int(port))
        self.encode = {"osc": encode_osc, "json": encode_json}[fmt]
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.slot = LatestFrame()
        self.sent = self.errors = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="udp-sender", daemon=True)
        self._thread.start()

    def __call__(self, hf):                  # subscribe() callback: just hand off
        self.slot.put(hf)

    def _run(self):
        version = 0
        while not self._stop.is_set():
            r = self.slot.wait_newer(version, timeout=0.2)
            if r is None:
                continue
            version, hf = r
            try:
                self.sock.sendto(self.encode(hf), self.addr)
                self.sent += 1
            except OSError:
                self.errors += 1

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)
        self.sock.close()


def parse_hostport(s):
    host, _, port = s.rpartition(":")
    return host or "127.0.0.1", int(port)
