/*
 * The rig: what a PlayTrack does not and cannot contain.
 *
 * Position, orientation, the ball and the events are measured. Where a man is
 * LOOKING is not — no public release carries head or eye tracking, and none is
 * likely to. So it is modelled here, once, with limits that can be argued with,
 * rather than guessed at in five places in a renderer.
 *
 * The model is a neck on a torso on a body:
 *   torso   ±45° of twist off the hips
 *   neck    ±70° off the shoulders
 * Past their sum a man has lost sight of the ball — and we say so instead of
 * quietly pointing his face at the nearest patch of grass.
 *
 * Nothing is instant. A target is picked up REACTION seconds late and the joints
 * slew at bounded speeds, so the motion is smooth because it is physical, not
 * because it was filtered afterwards.
 */
(function (global) {
  var GRID = global.GRID || (global.GRID = {});

  var LIMITS = {
    torso: 45, neck: 70, pitch: 32,
    torsoRate: 220, neckRate: 420, pitchRate: 300, bodyRate: 300,
    reaction: 0.2, step: 1 / 60
  };
  GRID.RIG_LIMITS = LIMITS;

  var OL = { T: 1, G: 1, C: 1, OL: 1, LT: 1, LG: 1, RG: 1, RT: 1 };
  var DL = { DE: 1, DT: 1, NT: 1, DL: 1 };

  function lerp(a, b, u) { return a + (b - a) * u; }
  function lerpDeg(a, b, u) { return GRID.wrapDeg(a + GRID.shortestDeg(a, b) * u); }
  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  function toward(cur, want, maxStep) {
    var d = GRID.shortestDeg(cur, want);
    if (d > maxStep) d = maxStep; else if (d < -maxStep) d = -maxStep;
    return cur + d;
  }

  // Smooth sampling of the track between released frames. Catmull-Rom on
  // position so a 10 Hz release does not judder a helmet cam; angles the short
  // way round.
  function sampleTrack(track, t) {
    var fs = track.frames, n = fs.length;
    if (t <= fs[0].t) return snapshot(fs[0], fs[0], 0, track);
    if (t >= fs[n - 1].t) return snapshot(fs[n - 1], fs[n - 1], 0, track);
    var i = 0;
    while (i < n - 2 && fs[i + 1].t < t) i++;
    var u = (t - fs[i].t) / (fs[i + 1].t - fs[i].t);
    return snapshot(fs[i], fs[i + 1], u, track, fs[Math.max(0, i - 1)], fs[Math.min(n - 1, i + 2)]);
  }

  function snapshot(fa, fb, u, track, fprev, fnext) {
    var out = { t: lerp(fa.t, fb.t, u), players: {}, ball: null, events: fa.events || [] };
    Object.keys(track.roster).forEach(function (id) {
      var a = fa.players[id], b = fb.players[id] || a;
      if (!a) return;
      var p0 = (fprev && fprev.players[id]) || a, p3 = (fnext && fnext.players[id]) || b;
      out.players[id] = {
        x: catmull(p0.x, a.x, b.x, p3.x, u),
        y: catmull(p0.y, a.y, b.y, p3.y, u),
        s: lerp(a.s, b.s, u),
        o: lerpDeg(a.o, b.o, u),
        dir: lerpDeg(a.dir, b.dir, u),
        o_source: a.o_source
      };
    });
    if (fa.ball && fb.ball) {
      out.ball = {
        x: lerp(fa.ball.x, fb.ball.x, u),
        y: lerp(fa.ball.y, fb.ball.y, u),
        z: (fa.ball.z == null || fb.ball.z == null) ? fa.ball.z : lerp(fa.ball.z, fb.ball.z, u),
        z_source: fa.ball.z_source, carrier: fa.ball.carrier,
        from: fa.ball.from, to: fa.ball.to
      };
    } else out.ball = fa.ball;
    return out;
  }

  function catmull(p0, p1, p2, p3, u) {
    var u2 = u * u, u3 = u2 * u;
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 +
                  (-p0 + 3 * p1 - 3 * p2 + p3) * u3);
  }

  /* --------------------------------------------------------- who has it */

  // A real release gives the ball's position but never says who is holding it.
  // Nearest man inside a yard and a half, with hysteresis so it does not flicker
  // between two players in a crowd.
  function carrierChain(track) {
    var chain = [], held = null, heldFor = 0;
    track.frames.forEach(function (f) {
      var best = null, bestD = 1e9;
      if (f.ball) {
        Object.keys(f.players).forEach(function (id) {
          var p = f.players[id];
          var d = Math.pow(p.x - f.ball.x, 2) + Math.pow(p.y - f.ball.y, 2);
          if (d < bestD) { bestD = d; best = id; }
        });
      }
      var declared = f.ball && f.ball.carrier;
      var near = (bestD <= 1.5 * 1.5) ? best : null;
      var pick = declared || near;
      if (pick !== held) {
        heldFor++;
        if (declared || heldFor >= 2) { held = pick; heldFor = 0; }
      } else heldFor = 0;
      chain.push({ t: f.t, carrier: held, nearest: best, events: f.events || [] });
    });
    return chain;
  }

  // Passer and receiver of each throw, from the events plus who was holding it.
  function exchanges(track, chain) {
    var out = [];
    track.frames.forEach(function (f, i) {
      (f.events || []).forEach(function (e) {
        if (e === 'pass_forward' || e === 'lateral' || e === 'handoff') {
          out.push({ kind: e, t: f.t, from: (f.ball && f.ball.from) || chain[i].carrier, to: (f.ball && f.ball.to) || null });
        }
        if (e === 'pass_arrived' || e === 'pass_outcome_caught') {
          var last = out[out.length - 1];
          if (last && last.to == null) { last.to = chain[i].nearest; last.tArrive = f.t; }
          else if (last) last.tArrive = f.t;
        }
      });
    });
    return out;
  }

  /* --------------------------------------------------------- the model */

  GRID.Rig = function (track, opts) {
    opts = opts || {};
    var limits = Object.assign({}, LIMITS, opts.limits || {});
    var ids = Object.keys(track.roster);
    var chain = carrierChain(track);
    var swaps = exchanges(track, chain);
    var t0 = track.frames[0].t, t1 = track.frames[track.frames.length - 1].t;

    function carrierAt(t) {
      var c = chain[0];
      for (var i = 0; i < chain.length; i++) if (chain[i].t <= t) c = chain[i]; else break;
      return c.carrier;
    }
    function nextSwapFrom(id, t) {
      for (var i = 0; i < swaps.length; i++) {
        if (swaps[i].from === id && swaps[i].t >= t - 0.05 && swaps[i].t - t < 0.9) return swaps[i];
      }
      return null;
    }
    function inFlight(t) {
      for (var i = 0; i < swaps.length; i++) {
        if (swaps[i].kind === 'pass_forward' && t >= swaps[i].t && t <= (swaps[i].tArrive || swaps[i].t + 1)) return swaps[i];
      }
      return null;
    }

    // What this man would be watching. Role and phase, never "the ball" blindly:
    // a carrier looking at the ball is a man looking at his own hands.
    function target(snap, id, t) {
      var me = snap.players[id], role = (track.roster[id].role || '').toUpperCase();
      var ball = snap.ball;
      var carrier = carrierAt(t);
      var ballPt = ball ? { x: ball.x, y: ball.y, z: ball.z == null ? 1.1 : ball.z } : null;

      if (carrier === id) {
        var swap = nextSwapFrom(id, t);
        if (swap && swap.to && snap.players[swap.to]) {
          var r = snap.players[swap.to];
          return { x: r.x, y: r.y, z: 1.45, what: 'receiver' };
        }
        var rad = me.dir * Math.PI / 180;               // up the field, where he is going
        return { x: me.x + Math.cos(rad) * 8, y: me.y + Math.sin(rad) * 8, z: 1.3, what: 'upfield' };
      }

      var flight = inFlight(t);
      if (flight && flight.from === id && flight.to && snap.players[flight.to]) {
        var rc = snap.players[flight.to];
        return { x: rc.x, y: rc.y, z: 1.45, what: 'receiver' };
      }

      if ((OL[role] || DL[role]) && t > 0 && t < 1.8) {
        var foe = null, best = 1e9;
        ids.forEach(function (other) {
          if (track.roster[other].team === track.roster[id].team) return;
          var p = snap.players[other];
          if (!p) return;
          var d = Math.pow(p.x - me.x, 2) + Math.pow(p.y - me.y, 2);
          if (d < best) { best = d; foe = p; }
        });
        if (foe) return { x: foe.x, y: foe.y, z: 1.5, what: 'man across' };
      }

      return ballPt ? { x: ballPt.x, y: ballPt.y, z: ballPt.z, what: 'ball' } : null;
    }

    // Integrate the whole play once, so a scrub anywhere gives the same pose as
    // playing up to it. History-dependent smoothing cannot promise that.
    var poses = [];
    var state = {};
    ids.forEach(function (id) { state[id] = { torso: 0, neck: 0, pitch: 0, body: 0, armL: 0, armR: 0 }; });

    for (var t = t0; t <= t1 + 1e-9; t += limits.step) {
      var snap = sampleTrack(track, t);
      var delayed = sampleTrack(track, Math.max(t0, t - limits.reaction));
      var carrier = carrierAt(t);
      var out = { t: t, ball: snap.ball, carrier: carrier, players: {} };

      ids.forEach(function (id) {
        var me = snap.players[id], st = state[id];
        if (!me) return;
        var tgt = target(delayed, id, Math.max(t0, t - limits.reaction));
        var wantRel = 0, pitchWant = 0, dist = 0;
        if (tgt) {
          var dx = tgt.x - me.x, dy = tgt.y - me.y;
          dist = Math.sqrt(dx * dx + dy * dy);
          var bearing = GRID.wrapDeg(Math.atan2(dy, dx) * 180 / Math.PI);
          wantRel = GRID.shortestDeg(GRID.wrapDeg(me.o + st.body), bearing);
          // Aim at the real point: a ball two yards away and a foot above the
          // eyes is not the same angle as one thirty yards away.
          pitchWant = Math.atan2((tgt.z || 1.2) - 1.5, Math.max(0.6, dist)) * 180 / Math.PI;
        }

        var reach = limits.torso + limits.neck;
        var lost = Math.abs(wantRel) > reach;
        // Only turn a man's body when nobody measured which way it was pointing.
        var trusted = me.o_source === 'measured' || me.o_source === 'authored';
        if (lost && !trusted) {
          var over = wantRel - (wantRel > 0 ? reach : -reach);
          st.body = toward(st.body, st.body + over, limits.bodyRate * limits.step);
          wantRel = GRID.shortestDeg(GRID.wrapDeg(me.o + st.body), GRID.wrapDeg(me.o + st.body + wantRel));
          lost = false;
        }

        var wantTorso = clamp(wantRel * (limits.torso / reach), -limits.torso, limits.torso);
        var wantNeck = clamp(wantRel - wantTorso, -limits.neck, limits.neck);
        st.torso = toward(st.torso, wantTorso, limits.torsoRate * limits.step);
        st.neck = toward(st.neck, wantNeck, limits.neckRate * limits.step);
        st.pitch = toward(st.pitch, clamp(pitchWant, -limits.pitch, limits.pitch), limits.pitchRate * limits.step);

        var arms = armPose(id, t, carrier, snap);
        st.armL += (arms.L - st.armL) * 0.25;
        st.armR += (arms.R - st.armR) * 0.25;

        out.players[id] = {
          x: me.x, y: me.y, o: GRID.wrapDeg(me.o + st.body), o_source: me.o_source,
          bodyTurn: st.body, torso: st.torso, neck: st.neck, pitch: st.pitch,
          armL: st.armL, armR: st.armR,
          lost: lost, target: tgt ? tgt.what : null, targetDist: dist
        };
      });
      poses.push(out);
    }

    function armPose(id, t, carrier, snap) {
      var swap = nextSwapFrom(id, t);
      if (swap && swap.t - t < 0.42 && swap.t >= t) {
        var w = 1 - (swap.t - t) / 0.42;
        return { L: -0.5, R: -0.9 - w * 1.8 };                 // winding up
      }
      for (var i = 0; i < swaps.length; i++) {
        var s = swaps[i];
        if (s.from === id && t > s.t && t - s.t < 0.4) {
          return { L: -0.6, R: -2.7 + ((t - s.t) / 0.4) * 2.3 }; // following through
        }
        if (s.to === id && s.tArrive && s.tArrive - t < 0.55 && s.tArrive > t) {
          return { L: -1.75, R: -1.75 };                        // reaching for it
        }
      }
      if (carrier === id) return { L: -0.85, R: -1.25 };
      var role = (track.roster[id].role || '').toUpperCase();
      if ((OL[role] || DL[role]) && t > 0 && t < 1.8) return { L: -1.2, R: -1.2 };
      return { L: 0, R: 0 };
    }

    this.track = track;
    this.poses = poses;
    this.swaps = swaps;
    this.limits = limits;
    this.start = t0;
    this.end = t1;

    this.poseAt = function (t) {
      if (t <= poses[0].t) return poses[0];
      if (t >= poses[poses.length - 1].t) return poses[poses.length - 1];
      var i = Math.floor((t - t0) / limits.step);
      return poses[Math.min(poses.length - 1, Math.max(0, i))];
    };

    // What the renderer should admit to. Counts by provenance, so a play built
    // on guessed orientation cannot quietly look like a measured one.
    this.provenance = function () {
      var counts = {};
      track.frames.forEach(function (f) {
        Object.keys(f.players).forEach(function (id) {
          var s = f.players[id].o_source;
          counts[s] = (counts[s] || 0) + 1;
        });
      });
      return counts;
    };
  };
})(typeof window !== 'undefined' ? window : globalThis);
