// Hand-tracked theremin for SuperCollider.
//
// Listens for the hand tracker's OSC stream (see handtrack/handtrack/output_udp.py):
//   /handtrack/hand  i track_id, s handedness, f score,
//                    63 f filtered landmarks (x0 y0 z0 ... x20 y20 z20, normalized 0..1), 42 f velocity
//
// Run (two terminals):
//   ~/theramin/synth/run.sh
//   cd ~/theramin/handtrack && ~/handtrack-env/bin/python run.py --udp 127.0.0.1:9000
// Or open this file in scide and evaluate the block between the outer parentheses (Ctrl+Enter).
//
// Controls:
//   right index fingertip, left <-> right   pitch (moving toward your right goes up)
//   left hand height                        volume (left hand out of view = silent)
//   right thumb-index pinch                 brightness (open = bright)
//   right hand height                       delay level and feedback (up = wilder)
//   left thumb-index pinch                  vibrato (pinch closed = more vibrato)
//
// Settings live in config.yaml next to this file (restart after editing); run.sh config_chord.yaml
// uses another file, e.g. a major 7th chord that slides between semitone steps. camera_mirrored and the
// camera size are read from ../handtrack/config.yaml. While it runs you can still override from
// scide, e.g. ~snap = 1; ~debug = true;

(
// ---- settings: config.yaml here, plus the camera settings from the tracker's config ----
var dir = thisProcess.nowExecutingPath !? (_.dirname) ? "~/theramin/synth".standardizePath;
var cfgPath = thisProcess.argv.first ? "config.yaml";      // run.sh config_chord.yaml picks another
var cfg = (if(cfgPath[0] == $/) { cfgPath } { dir +/+ cfgPath }).parseYAMLFile;  // values arrive as strings
var track = (dir +/+ "../handtrack/config.yaml").parseYAMLFile;
var lo;
("config: " ++ cfgPath).postln;
~port = cfg["port"].asInteger;
~cameraMirrored = track["tracking"]["camera_mirrored"] == "true";
~aspect = track["camera"]["width"].asFloat / track["camera"]["height"].asFloat;  // pinch distances in square units
~noteLo = cfg["pitch"]["note_lo"].asFloat;
~noteHi = cfg["pitch"]["note_hi"].asFloat;
~pitchSpan = cfg["pitch"]["span"].collect(_.asFloat);
~snap = cfg["pitch"]["snap"].asFloat;
~glide = cfg["pitch"]["glide"].asFloat;
~voices = (cfg["voices"] ? ["0"]).collect(_.asFloat);    // semitones above the root, one oscillator each
~hysteresis = (cfg["pitch"]["hysteresis"] ? "0.25").asFloat;
// With pitch.steps, the root moves on a grid of notes: those whose pitch class above note_lo is listed.
lo = ~noteLo.asInteger;
~grid = cfg["pitch"]["steps"] !? { |steps|
	steps = steps.collect(_.asInteger);
	(lo .. ~noteHi.asInteger).select { |n| steps.includes((n - lo) % 12) }
};
~step = nil;                // current grid index
~volSpan = cfg["volume"]["span"].collect(_.asFloat);
~cutoffLo = cfg["brightness"]["cutoff_lo"].asFloat;
~cutoffHi = cfg["brightness"]["cutoff_hi"].asFloat;
~vibRate = cfg["vibrato"]["rate"].asFloat;
~vibDepth = cfg["vibrato"]["depth"].asFloat;
~delay = cfg["delay"] ? ();
~delayTime = (~delay["time"] ? "0.3").asFloat.clip(0.01, 2);
~delaySpan = (~delay["span"] ? ["0.3", "0.8"]).collect(_.asFloat);
~delayMixMax = (~delay["mix_max"] ? "0").asFloat;
~delayFbMax = (~delay["feedback_max"] ? "0").asFloat.min(0.99);
~timeout = cfg["timeout"].asFloat;
~debug = cfg["debug"] == "true";

s.options.blockSize = 64;
s.options.numInputBusChannels = 0;

s.waitForBoot {
	SynthDef(\handTheremin, { |out = 0, note = 60, amp = 0, cutoff = 3000, vib = 0, glide = 0.05,
		vibRate = 5.5, vibDepth = 0.5, delayTime = 0.3, delayMix = 0, delayFb = 0|
		var n, freq, sig, fb, echo;
		// Tracking arrives at ~30 fps; the lags turn those steps into smooth curves.
		n = Lag.kr(note, glide) + (SinOsc.kr(vibRate) * Lag.kr(vib, 0.2) * vibDepth);
		// Every voice follows the same lagged root, so a chord slides as one.
		sig = ~voices.collect { |iv|
			freq = (n + iv).midicps;
			(SinOsc.ar(freq) * 0.6) + (VarSaw.ar(freq, 0, 0.4) * 0.4)
		}.sum / ~voices.size.sqrt;
		sig = RLPF.ar(sig, Lag.kr(cutoff, 0.05).clip(80, 12000), 0.4);
		sig = sig * Lag.kr(amp, 0.03).squared * 0.5;
		// Ping-pong delay: left echoes feed the right and back. The low-pass and tanh in the loop
		// darken each repeat and keep feedback near 1 from running away.
		fb = LocalIn.ar(2) * Lag.kr(delayFb, 0.1);
		echo = DelayC.ar([sig + fb[1], fb[0]], 2, delayTime);
		echo = LPF.ar(echo, 4000).tanh;
		LocalOut.ar(echo);
		sig = (sig ! 2) + (echo * Lag.kr(delayMix, 0.1));
		sig = Limiter.ar(LeakDC.ar(sig), 0.9);
		Out.ar(out, sig);
	}).add;
	s.sync;

	~synth = Synth(\handTheremin, [\glide, ~glide, \vibRate, ~vibRate, \vibDepth, ~vibDepth,
		\delayTime, ~delayTime]);
	~hands = (Left: nil, Right: nil);   // latest 63 landmark values per hand
	~seen = (Left: 0, Right: 0);        // when each hand was last received
	~lastPost = 0;

	// point i as [x, y] in frame-height units
	~pt = { |lm, i| [lm[i * 3] * ~aspect, lm[i * 3 + 1]] };
	~dist = { |a, b| (a - b).squared.sum.sqrt };
	// thumb tip to index tip, relative to hand size (wrist to middle knuckle): ~0.1 closed, ~1 open
	~pinch = { |lm| ~dist.(~pt.(lm, 4), ~pt.(lm, 8)) / max(~dist.(~pt.(lm, 0), ~pt.(lm, 9)), 1e-3) };

	~update = {
		var now = Main.elapsedTime, args = List[], r, l;
		r = if(now - ~seen[\Right] < ~timeout) { ~hands[\Right] };
		l = if(now - ~seen[\Left] < ~timeout) { ~hands[\Left] };
		if(r.notNil) {
			var u = if(~cameraMirrored) { r[8 * 3] } { 1 - r[8 * 3] };
			var note = u.linlin(~pitchSpan[0], ~pitchSpan[1], ~noteLo, ~noteHi);
			if(~grid.isNil) { note = note.blend(note.round, ~snap) } {
				// Grid position; only move to a new step once the hand is clearly past the boundary.
				var f = u.linlin(~pitchSpan[0], ~pitchSpan[1], 0, ~grid.size - 1);
				if(~step.isNil or: { (f - ~step).abs > (0.5 + ~hysteresis) }) { ~step = f.round.asInteger };
				note = ~grid[~step];
			};
			args.addAll([\note, note, \cutoff, ~pinch.(r).linexp(0.15, 0.9, ~cutoffLo, ~cutoffHi)]);
			// right wrist height: one amount drives both the echo level and how long it rings
			{ |d| args.addAll([\delayMix, d * ~delayMixMax, \delayFb, d * ~delayFbMax]) }
				.value((1 - r[1]).linlin(~delaySpan[0], ~delaySpan[1], 0, 1));
		};
		if(l.notNil) { args.addAll([\vib, ~pinch.(l).linlin(0.15, 0.6, 1, 0)]) };
		args.addAll([\amp, if(r.notNil and: { l.notNil }) { (1 - l[1]).linlin(~volSpan[0], ~volSpan[1], 0, 1) } { 0 }]);
		~synth.set(*args);
		if(~debug and: { now - ~lastPost > 0.2 }) { ~lastPost = now; args.asArray.round(0.01).postln };
	};

	thisProcess.openUDPPort(~port);
	OSCdef(\handtrack, { |msg|
		var side = msg[2].asSymbol;
		~hands[side] = msg.copyRange(4, 66);
		~seen[side] = Main.elapsedTime;
		~update.value;
	}, '/handtrack/hand');

	// Messages stop entirely when no hands are in view (or the tracker quits): fade out then.
	~watch = Routine {
		loop {
			if(Main.elapsedTime - max(~seen[\Left], ~seen[\Right]) > ~timeout) { ~synth.set(\amp, 0) };
			0.05.wait;
		}
	}.play(SystemClock);

	~stop = { ~watch.stop; OSCdef(\handtrack).free; ~synth.free };

	"theremin ready: listening for /handtrack/hand on UDP %".format(~port).postln;
	"start the tracker with: run.py --udp 127.0.0.1:%".format(~port).postln;
};
)
