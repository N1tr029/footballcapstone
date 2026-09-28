/*
 * The shared shape. Everything that draws a play draws one of these, whoever
 * produced it — the hand-authored fixture, the vision pipeline, or the play
 * model's simulated alternative.
 *
 * Field coordinates (docs/roles.md):
 *   x   0–120 yards downfield. The offense attacks +x. Goal line at x = 110.
 *   y   0–53.3 yards across, increasing toward the OFFENSE'S RIGHT.
 *   t   seconds relative to the snap.
 *
 * Angles are DEGREES, 0 = facing +x (downfield), increasing toward +y. Two of
 * them, and they are not the same thing:
 *   dir  the direction a player is MOVING
 *   o    the direction his body is POINTING
 * A receiver drifting to the corner with his head back at the ball has an `o`
 * a hundred degrees off his `dir`, and every public tracking release carries
 * both columns for exactly that reason. Guessing one from the other is what
 * made the quarterback spend the Philly Special looking at grass.
 *
 * o_source records how much to trust it:
 *   measured      it came from the data
 *   from_dir      derived from direction of travel
 *   from_velocity derived from successive positions
 *   assumed       a resting default for his side of the ball
 *   authored      drawn by hand, for fixtures — never claim this is data
 */
(function (global) {
  var GRID = global.GRID || (global.GRID = {});

  GRID.FIELD = {
    length: 120, width: 53.3, goalLine: 110, backOfEndZone: 120,
    hashes: [23.58, 29.72], midY: 53.3 / 2
  };

  GRID.O_SOURCES = ['measured', 'from_dir', 'from_velocity', 'assumed', 'authored'];

  // The events a play is cut into. Named after the tracking-data vocabulary so
  // a real play needs no translation.
  GRID.EVENTS = [
    'ball_snap', 'handoff', 'lateral', 'pass_forward', 'pass_arrived',
    'pass_outcome_caught', 'tackle', 'touchdown', 'out_of_bounds'
  ];

  GRID.RATE = 10;                                  // frames per second, as released

  function isNum(v) { return typeof v === 'number' && isFinite(v); }

  // Worth running on anything before it reaches the renderer: a play that is
  // wrong here is a play that is wrong in every view at once.
  GRID.validate = function (track) {
    var problems = [];
    if (!track || !track.frames || !track.frames.length) {
      return { ok: false, problems: ['no frames'] };
    }
    var ids = Object.keys(track.roster || {});
    if (!ids.length) problems.push('no roster');

    var prev = null, badAngles = 0, missing = 0, offField = 0;
    track.frames.forEach(function (f) {
      if (!isNum(f.t)) problems.push('frame with no time');
      if (prev !== null && f.t <= prev) problems.push('time runs backwards at t=' + f.t);
      prev = f.t;
      ids.forEach(function (id) {
        var p = f.players[id];
        if (!p) { missing++; return; }
        if (!isNum(p.x) || !isNum(p.y)) { missing++; return; }
        if (p.x < -5 || p.x > 125 || p.y < -5 || p.y > 58) offField++;
        if (!isNum(p.o) || p.o < 0 || p.o >= 360) badAngles++;
        if (GRID.O_SOURCES.indexOf(p.o_source) === -1) badAngles++;
      });
    });
    if (missing) problems.push(missing + ' player samples missing position');
    if (offField) problems.push(offField + ' samples outside the field');
    if (badAngles) problems.push(badAngles + ' samples with a bad orientation');
    if (!track.frames.some(function (f) { return f.ball; })) problems.push('no ball track');

    var events = {};
    track.frames.forEach(function (f) {
      (f.events || []).forEach(function (e) { events[e] = (events[e] || 0) + 1; });
    });
    if (!events.ball_snap) problems.push('no ball_snap event');

    return { ok: problems.length === 0, problems: problems, events: events };
  };

  // Angle helpers. Everything downstream works in radians in world space; these
  // are the only two places the conversion happens.
  GRID.degToWorldYaw = function (deg) { return -deg * Math.PI / 180; };
  GRID.wrapDeg = function (d) { d = d % 360; return d < 0 ? d + 360 : d; };
  GRID.shortestDeg = function (from, to) {
    var d = GRID.wrapDeg(to - from);
    return d > 180 ? d - 360 : d;
  };
})(typeof window !== 'undefined' ? window : globalThis);
