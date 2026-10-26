#!/usr/bin/env bash
# Start the SuperCollider theremin headless, through PipeWire's JACK layer.
# Usage: ./run.sh [CONFIG]   (default config.yaml, e.g. ./run.sh config_chord.yaml)
# QUANTUM is the audio buffer: 128/48000 is ~2.7 ms; raise it (256/48000) if you hear crackles.
cd "$(dirname "$0")"
export QT_QPA_PLATFORM=offscreen     # sclang without a window, e.g. over SSH
exec env PIPEWIRE_QUANTUM="${QUANTUM:-128/48000}" pw-jack sclang theremin.scd "$@"
