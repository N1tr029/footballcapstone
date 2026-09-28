/*
 * Checks a play before anybody looks at it.
 *
 * A single hand-made play can be eyeballed. A season cannot, and the failures
 * that matter are invisible from the broadcast camera: a man whose own view
 * never contains the ball, a head that turns faster than a neck can, a camera
 * that ends up inside somebody's chest. These run headless over every play and
 * flag the ones not to publish.
 */
(function (global) {
  var GRID = global.GRID || (global.GRID = {});

  var THRESHOLDS = {
    viewHalfAngle: 45,       // degrees off the gaze axis before it is out of frame
    ballVisibleMin: 0.25,    // a carrier or receiver below this is a broken view
    lostMax: 0.6,            // fraction of the play with the ball out of reach
    occlusionRadius: 0.45,   // yards: bodies overlapping, not merely close
    occlusionMax: 0.12,      // fraction of frames with the eye inside another body
    measuredMin: 0.5         // fraction of orientation samples actually measured
  };
  GRID.QA_THRESHOLDS = THRESHOLDS;

  GRID.qa = function (track, rig, opts) {
    opts = opts || {};
    var th = Object.assign({}, THRESHOLDS, opts.thresholds || {});
    var ids = Object.keys(track.roster);
    var poses = rig.poses, step = rig.limits.step;
    var rows = [], problems = [];

    var v = GRID.validate(track);
    if (!v.ok) v.problems.forEach(function (p) { problems.push('track: ' + p); });

    // Who ever holds the ball — those are the men whose own view has to work.
    var principals = {};
    poses.forEach(function (p) { if (p.carrier) principals[p.carrier] = true; });
    rig.swaps.forEach(function (s) { if (s.to) principals[s.to] = true; });

    ids.forEach(function (id) {
      var seen = 0, counted = 0, lost = 0, occluded = 0;
      var maxNeck = 0, maxTorso = 0, maxBody = 0, prev = null;
      var sources = {};

      poses.forEach(function (pose) {
        var me = pose.players[id];
        if (!me) return;
        counted++;
        sources[me.o_source] = (sources[me.o_source] || 0) + 1;
        if (me.lost) lost++;

        if (pose.ball) {
          var gaze = me.o + me.torso + me.neck;
          var bearing = Math.atan2(pose.ball.y - me.y, pose.ball.x - me.x) * 180 / Math.PI;
          if (Math.abs(GRID.shortestDeg(GRID.wrapDeg(gaze), GRID.wrapDeg(bearing))) <= th.viewHalfAngle) seen++;
        }

        // Eye inside somebody else: the helmet cam's version of clipping.
        ids.forEach(function (other) {
          if (other === id) return;
          var op = pose.players[other];
          if (!op) return;
          var d = Math.pow(op.x - me.x, 2) + Math.pow(op.y - me.y, 2);
          if (d < th.occlusionRadius * th.occlusionRadius) occluded++;
        });

        if (prev) {
          maxNeck = Math.max(maxNeck, Math.abs(me.neck - prev.neck) / step);
          maxTorso = Math.max(maxTorso, Math.abs(me.torso - prev.torso) / step);
          maxBody = Math.max(maxBody, Math.abs(GRID.shortestDeg(prev.o, me.o)) / step);
        }
        prev = me;
      });

      var row = {
        id: id,
        label: track.roster[id].label || id,
        principal: !!principals[id],
        ballInView: counted ? seen / counted : 0,
        lost: counted ? lost / counted : 0,
        occluded: counted ? occluded / counted : 0,
        maxNeckRate: maxNeck, maxTorsoRate: maxTorso, maxBodyRate: maxBody,
        sources: sources,
        fails: [], warnings: []
      };

      if (row.principal && row.ballInView < th.ballVisibleMin) {
        row.fails.push('own view rarely contains the ball (' + Math.round(row.ballInView * 100) + '%)');
      }
      if (row.lost > th.lostMax) row.fails.push('ball out of reach for ' + Math.round(row.lost * 100) + '% of the play');
      // Linemen really do stand inside each other's space. It makes their own
      // view useless, which is worth saying — it is not a reason to bin the play.
      if (row.occluded > th.occlusionMax) {
        (row.principal ? row.fails : row.warnings)
          .push('view obstructed for ' + Math.round(row.occluded * 100) + '% of frames');
      }
      if (maxNeck > rig.limits.neckRate * 1.05) row.fails.push('neck exceeds ' + rig.limits.neckRate + ' deg/s');
      if (maxTorso > rig.limits.torsoRate * 1.05) row.fails.push('torso exceeds ' + rig.limits.torsoRate + ' deg/s');
      if (maxBody > rig.limits.bodyRate * 1.05) row.fails.push('body turns faster than ' + rig.limits.bodyRate + ' deg/s');
      rows.push(row);
    });

    var prov = rig.provenance(), total = 0, measured = 0;
    Object.keys(prov).forEach(function (k) {
      total += prov[k];
      if (k === 'measured') measured += prov[k];
    });
    var measuredFrac = total ? measured / total : 0;
    var warnings = [];
    if (measuredFrac < th.measuredMin) {
      // A reconstruction is allowed to exist. It is not allowed to pass itself
      // off as measurement, which is what the badge on the page is for.
      warnings.push('orientation is ' + Math.round(measuredFrac * 100) + '% measured — the rest is inferred, so gaze and every player view are drawings');
    }
    if (!rig.swaps.length) problems.push('no handoff, lateral or pass found — check the event column');

    var failing = rows.filter(function (r) { return r.fails.length; });
    return {
      ok: problems.length === 0 && failing.length === 0,
      warnings: warnings.concat(rows.reduce(function (acc, r) {
        return acc.concat(r.warnings.map(function (w) { return r.label + ': ' + w; }));
      }, [])),
      play: track.meta && (track.meta.name || track.meta.id),
      provenance: prov, measuredFraction: measuredFrac,
      problems: problems, players: rows, failing: failing
    };
  };

  GRID.qaText = function (report) {
    var out = [];
    out.push('play: ' + (report.play || 'unnamed'));
    out.push('orientation measured: ' + Math.round(report.measuredFraction * 100) + '%  ' +
             JSON.stringify(report.provenance));
    report.problems.forEach(function (p) { out.push('  ! ' + p); });
    (report.warnings || []).forEach(function (w) { out.push('  ~ ' + w); });
    out.push('');
    out.push(pad('player', 34) + pad('ball in view', 14) + pad('lost', 8) + pad('neck/s', 9) + 'flags');
    report.players.forEach(function (r) {
      out.push(pad((r.principal ? '* ' : '  ') + r.label, 34) +
               pad(Math.round(r.ballInView * 100) + '%', 14) +
               pad(Math.round(r.lost * 100) + '%', 8) +
               pad(Math.round(r.maxNeckRate) + '', 9) +
               (r.fails.length ? r.fails.join('; ') : ''));
    });
    out.push('');
    out.push(report.ok ? 'PASS' : 'FAIL — ' + report.failing.length + ' player views and ' +
             report.problems.length + ' play-level problems');
    return out.join('\n');
    function pad(s, n) { s = String(s); while (s.length < n) s += ' '; return s; }
  };
})(typeof window !== 'undefined' ? window : globalThis);
