/*
 * Turning a source of truth into the shared PlayTrack. Two of them today: the
 * hand-authored fixture, and per-frame tracking rows as the public releases
 * publish them. Anything else — the vision pipeline, a simulated alternative —
 * writes one more function here and changes nothing downstream.
 */
(function (global) {
  var GRID = global.GRID || (global.GRID = {});

  // Catmull-Rom through timed keys, [t, a, b]. Used by the fixture adapter only;
  // real tracking arrives already sampled at 10 Hz.
  function spline(keys, t) {
    var n = keys.length;
    if (t <= keys[0][0]) return { a: keys[0][1], b: keys[0][2] };
    if (t >= keys[n - 1][0]) return { a: keys[n - 1][1], b: keys[n - 1][2] };
    var i = 0;
    while (i < n - 2 && keys[i + 1][0] < t) i++;
    var p0 = keys[Math.max(0, i - 1)], p1 = keys[i], p2 = keys[i + 1], p3 = keys[Math.min(n - 1, i + 2)];
    var u = (t - p1[0]) / (p2[0] - p1[0]);
    function cr(a, b, c, d) {
      var u2 = u * u, u3 = u2 * u;
      return 0.5 * ((2 * b) + (-a + c) * u + (2 * a - 5 * b + 4 * c - d) * u2 + (-a + 3 * b - 3 * c + d) * u3);
    }
    return { a: cr(p0[1], p1[1], p2[1], p3[1]), b: cr(p0[2], p1[2], p2[2], p3[2]) };
  }

  // Angles have to be interpolated the short way round, or a man turning past
  // due north spins all the way back through his own shoulder.
  function angleAt(keys, t) {
    var n = keys.length;
    if (t <= keys[0][0]) return keys[0][1];
    if (t >= keys[n - 1][0]) return keys[n - 1][1];
    var i = 0;
    while (i < n - 1 && keys[i + 1][0] < t) i++;
    var a = keys[i], b = keys[i + 1];
    var u = (t - a[0]) / (b[0] - a[0]);
    u = u * u * (3 - 2 * u);
    return GRID.wrapDeg(a[1] + GRID.shortestDeg(a[1], b[1]) * u);
  }

  function dirFrom(prev, cur, next) {
    var a = prev || cur, b = next || cur;
    var dx = b.x - a.x, dy = b.y - a.y;
    if (dx * dx + dy * dy < 1e-6) return null;
    return GRID.wrapDeg(Math.atan2(dy, dx) * 180 / Math.PI);
  }

  /* ---------------------------------------------------------------- fixture */

  GRID.fromKeyframes = function (def) {
    var dt = 1 / GRID.RATE;
    var t0 = def.duration.start, t1 = def.duration.end;
    var roster = {}, all = def.offense.concat(def.defense);
    all.forEach(function (p) {
      roster[p.id] = {
        id: p.id, team: def.offense.indexOf(p) !== -1 ? 'offense' : 'defense',
        label: p.label || p.id, name: p.name, num: p.num, role: p.role, star: !!p.star
      };
    });

    // Positions first, so direction of travel can be differenced from them.
    var times = [], raw = {};
    for (var t = t0; t <= t1 + 1e-9; t += dt) times.push(Math.round(t * 1000) / 1000);
    all.forEach(function (p) {
      raw[p.id] = times.map(function (tt) {
        var s = spline(p.keys, tt);
        return { x: s.a, y: s.b };
      });
    });

    // Orientation derived from movement is worthless below walking pace — the
    // direction flips wildly when a man is nearly still, which is precisely when
    // a lineman is set and a receiver is waiting. Hold it, and cap how fast it
    // can swing, or every play inherits a body that spins on the spot.
    var SLOW = 0.6, DERIVED_RATE = 250;
    var lastO = {};

    var frames = times.map(function (tt, i) {
      var players = {};
      all.forEach(function (p) {
        var here = raw[p.id][i], prev = raw[p.id][i - 1], next = raw[p.id][i + 1];
        var dir = dirFrom(prev, here, next);
        var sp = 0;
        if (prev) sp = Math.sqrt(Math.pow(here.x - prev.x, 2) + Math.pow(here.y - prev.y, 2)) / dt;
        var o, src;
        if (p.face) { o = angleAt(p.face, tt); src = 'authored'; }
        else if (dir !== null && (sp > SLOW || lastO[p.id] == null)) { o = dir; src = 'from_velocity'; }
        else if (lastO[p.id] != null) { o = lastO[p.id]; src = 'from_velocity'; }
        else { o = roster[p.id].team === 'offense' ? 0 : 180; src = 'assumed'; }
        if (src !== 'authored' && lastO[p.id] != null) {
          var d = GRID.shortestDeg(lastO[p.id], o), cap = DERIVED_RATE * dt;
          if (d > cap) d = cap; else if (d < -cap) d = -cap;
          o = GRID.wrapDeg(lastO[p.id] + d);
        }
        lastO[p.id] = o;
        players[p.id] = {
          x: here.x, y: here.y, s: sp, a: 0,
          dir: dir === null ? o : dir, o: GRID.wrapDeg(o), o_source: src
        };
      });
      return { t: tt, players: players, ball: null, events: [] };
    });

    // The ball, and the events its handling implies. A real play reads these
    // from the data; the fixture has to declare them, which is the only thing
    // about it that does not generalise.
    var EV = { snap: 'ball_snap', pitch: 'lateral', pass: 'pass_forward' };
    frames.forEach(function (f) {
      var seg = def.ball[0];
      for (var i = 0; i < def.ball.length; i++) if (f.t >= def.ball[i].t0) seg = def.ball[i];
      f.ball = ballFor(seg, f, def);
      def.ball.forEach(function (s) {
        if (EV[s.type] && Math.abs(f.t - s.t0) < 1 / (2 * GRID.RATE)) f.events.push(EV[s.type]);
        if (s.type === 'pass' && Math.abs(f.t - s.t1) < 1 / (2 * GRID.RATE)) {
          f.events.push('pass_arrived');
          f.events.push('pass_outcome_caught');
        }
        if (s.type === 'handoff' && Math.abs(f.t - s.t0) < 1 / (2 * GRID.RATE)) f.events.push('handoff');
      });
    });
    var last = frames[frames.length - 1];
    if (def.meta && /touchdown/i.test(def.meta.result || '')) last.events.push('touchdown');

    function ballFor(seg, f, def) {
      function hand(id) {
        var p = f.players[id];
        if (!p) return { x: def.field.snapX, y: GRID.FIELD.midY, z: 1.0 };
        var rad = p.o * Math.PI / 180;
        // out to his right hand, a little in front of him
        return { x: p.x + Math.cos(rad) * 0.14 + Math.sin(rad) * 0.34,
                 y: p.y + Math.sin(rad) * 0.14 - Math.cos(rad) * 0.34, z: 1.02 };
      }
      if (seg.type === 'spot') return { x: seg.x, y: seg.y, z: 0.17, carrier: null };
      if (seg.type === 'carry') { var h = hand(seg.player); h.carrier = seg.player; return h; }
      var from = hand(seg.from), to = hand(seg.to);
      var u = Math.min(1, Math.max(0, (f.t - seg.t0) / (seg.t1 - seg.t0)));
      if (seg.type === 'pass') from.z += 0.5;              // released up by the ear
      return {
        x: from.x + (to.x - from.x) * u,
        y: from.y + (to.y - from.y) * u,
        // Height is not measured for a real pass either; it is synthesised from
        // the flight time and flagged as such.
        z: from.z + (to.z - from.z) * u + Math.sin(u * Math.PI) * ((seg.apex || 1.4) - 1.0),
        z_source: 'synthesised', carrier: null, from: seg.from, to: seg.to
      };
    }

    return {
      meta: Object.assign({}, def.meta, { adapter: 'keyframes' }),
      field: def.field, roster: roster, frames: frames
    };
  };

  /* --------------------------------------------------- real tracking rows */

  /*
   * Rows as the public releases ship them, one per player per frame plus one for
   * the football. `angle` converts that release's convention into ours (0 = +x,
   * increasing toward +y); it has to be calibrated against the real file the
   * first time a release is wired up, and left alone afterwards.
   */
  GRID.fromTrackingRows = function (rows, opts) {
    opts = opts || {};
    var angle = opts.angle || function (deg) { return GRID.wrapDeg(deg); };
    var flip = opts.playDirection === 'left';        // offense attacking -x
    var byFrame = {};
    rows.forEach(function (r) {
      var k = r.frameId != null ? r.frameId : r.frame;
      (byFrame[k] || (byFrame[k] = [])).push(r);
    });

    var keys = Object.keys(byFrame).map(Number).sort(function (a, b) { return a - b; });
    var snapFrame = null;
    keys.forEach(function (k) {
      byFrame[k].forEach(function (r) { if (r.event === 'ball_snap' && snapFrame === null) snapFrame = k; });
    });
    if (snapFrame === null) snapFrame = keys[0];

    var roster = {}, frames = [];
    keys.forEach(function (k) {
      var players = {}, ball = null, events = {};
      byFrame[k].forEach(function (r) {
        var isBall = (r.nflId == null || r.nflId === '' || /football/i.test(r.displayName || ''));
        var x = flip ? 120 - r.x : r.x, y = flip ? 53.3 - r.y : r.y;
        if (r.event) events[r.event] = true;
        if (isBall) { ball = { x: x, y: y, z: null, z_source: 'unmeasured', carrier: null }; return; }
        var id = String(r.nflId);
        if (!roster[id]) {
          roster[id] = { id: id, team: r.team || r.club, label: r.displayName || id,
                         name: r.displayName, num: r.jerseyNumber, role: r.position };
        }
        var o = r.o, dir = r.dir, src = 'measured';
        if (o == null || o === '') { o = dir; src = 'from_dir'; }
        if (o == null || o === '') { o = 0; src = 'assumed'; }
        var oa = angle(Number(o)), da = dir == null || dir === '' ? oa : angle(Number(dir));
        if (flip) { oa = GRID.wrapDeg(oa + 180); da = GRID.wrapDeg(da + 180); }
        players[id] = { x: x, y: y, s: Number(r.s) || 0, a: Number(r.a) || 0,
                        dir: da, o: oa, o_source: src };
      });
      frames.push({ t: (k - snapFrame) / GRID.RATE, players: players, ball: ball,
                    events: Object.keys(events) });
    });

    return {
      meta: { adapter: 'tracking', source: opts.source || 'tracking rows' },
      field: { snapX: null, goalLine: GRID.FIELD.goalLine }, roster: roster, frames: frames
    };
  };

  GRID.frameAt = function (track, t) {
    var f = track.frames[0];
    for (var i = 0; i < track.frames.length; i++) {
      if (track.frames[i].t <= t) f = track.frames[i]; else break;
    }
    return f;
  };
})(typeof window !== 'undefined' ? window : globalThis);
