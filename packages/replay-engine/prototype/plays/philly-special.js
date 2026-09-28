/*
 * Philly Special — Super Bowl LII, 4th & goal from the 1, :38 left in the half.
 * Kelce snaps directly to Clement, who runs left and flips it back to Burton on
 * the reverse; Burton throws right to Foles in the corner of the end zone.
 *
 * IMPORTANT: these positions are a hand-authored approximation of the play's
 * shape — the real All-22 has never been released as tracking data. Treat it as
 * a reference animation for the replay engine, not as measured truth. Every
 * number below is in the field coordinates defined in docs/roles.md:
 *
 *   x  0–120 yards downfield. The offense attacks +x. Goal line at x = 110.
 *   y  0–53.3 yards across. y increases toward the OFFENSE'S RIGHT.
 *   t  seconds relative to the snap (t = 0). Pre-snap frames are negative.
 *
 * Positions between keyframes are interpolated by the renderer, which is what
 * makes this a keyframe form of the TrackingFrame contract rather than a
 * separate format: sample it at 10 Hz and you have TrackingFrames.
 */
(function (global) {
  const SNAP_X = 109;      // ball spotted on the 1
  const GOAL = 110;

  const play = {
    meta: {
      id: 'philly-special',
      name: 'Philly Special',
      game: 'Super Bowl LII — Philadelphia 15, New England 12',
      date: '2018-02-04',
      situation: { quarter: 2, clock: '0:38', down: 4, distance: 1, yardline: 1 },
      result: '1-yard touchdown pass, Burton to Foles. 22–12.',
      source: 'hand-authored approximation — not tracking data',
    },

    field: { snapX: SNAP_X, goalLine: GOAL, backOfEndZone: 120, width: 53.3, hashes: [23.58, 29.72] },

    // t0/t1 in seconds; the renderer attaches the ball to a carrier or arcs it.
    ball: [
      { type: 'spot', t0: -2.6, t1: 0, x: SNAP_X, y: 26.0 },
      { type: 'snap', t0: 0, t1: 0.28, from: 'KELCE', to: 'CLEMENT', apex: 0.4 },
      { type: 'carry', t0: 0.28, t1: 0.98, player: 'CLEMENT' },
      { type: 'pitch', t0: 0.98, t1: 1.22, from: 'CLEMENT', to: 'BURTON', apex: 1.1 },
      { type: 'carry', t0: 1.22, t1: 2.32, player: 'BURTON' },
      { type: 'pass', t0: 2.32, t1: 2.92, from: 'BURTON', to: 'FOLES', apex: 3.4 },
      { type: 'carry', t0: 2.92, t1: 8, player: 'FOLES' },
    ],

    offense: [
      { id: 'FOLES', num: 9, name: 'Foles', role: 'QB', star: true, keys: [
        [-2.6, 106.4, 26.6], [-1.9, 106.6, 28.4], [-1.1, 107.0, 32.0], [-0.3, 107.2, 33.9],
        [0, 107.2, 34.0], [0.7, 106.9, 35.4], [1.4, 107.1, 37.0], [2.0, 108.0, 38.6],
        [2.6, 109.4, 40.0], [2.92, 110.4, 40.6], [3.6, 112.2, 41.2], [5.0, 113.0, 41.4],
      ]},
      { id: 'CLEMENT', num: 30, name: 'Clement', role: 'RB', star: true, keys: [
        [-2.6, 104.7, 26.0], [0, 104.8, 26.0], [0.5, 105.2, 23.6], [0.98, 105.8, 21.2],
        [1.6, 106.4, 19.0], [2.6, 107.2, 17.4], [4.0, 107.6, 16.8],
      ]},
      { id: 'BURTON', num: 88, name: 'Burton', role: 'TE', star: true, keys: [
        [-2.6, 107.0, 19.2], [-0.6, 107.0, 19.2], [0, 106.9, 19.4], [0.6, 106.3, 20.4],
        [1.22, 106.0, 21.6], [1.8, 106.3, 27.0], [2.32, 106.8, 31.8], [3.0, 106.9, 33.2],
        [4.2, 107.0, 33.6],
      ]},
      { id: 'KELCE', num: 62, name: 'Kelce', role: 'C', keys: [
        [-2.6, 108.2, 26.0], [0, 108.2, 26.0], [0.6, 108.9, 25.6], [1.6, 109.4, 25.0], [3.0, 109.6, 24.6],
      ]},
      { id: 'LG', role: 'LG', keys: [[-2.6, 108.2, 23.8], [0, 108.2, 23.8], [0.8, 108.8, 23.0], [2.0, 109.2, 22.2], [3.4, 109.3, 21.8]] },
      { id: 'LT', role: 'LT', keys: [[-2.6, 108.2, 21.6], [0, 108.2, 21.6], [0.8, 108.7, 20.8], [2.0, 109.0, 20.0], [3.4, 109.1, 19.6]] },
      { id: 'RG', role: 'RG', keys: [[-2.6, 108.2, 28.2], [0, 108.2, 28.2], [0.8, 108.8, 28.8], [2.0, 109.2, 29.4], [3.4, 109.3, 29.8]] },
      { id: 'RT', role: 'RT', keys: [[-2.6, 108.2, 30.4], [0, 108.2, 30.4], [0.8, 108.8, 31.0], [2.0, 109.3, 31.8], [3.4, 109.4, 32.2]] },
      { id: 'ERTZ', role: 'TE', keys: [[-2.6, 108.2, 32.6], [0, 108.2, 32.6], [0.8, 108.9, 33.4], [1.8, 109.6, 34.6], [3.4, 109.8, 35.2]] },
      { id: 'WR-L', role: 'WR', keys: [[-2.6, 108.4, 12.0], [0, 108.4, 12.0], [1.0, 109.4, 11.4], [2.4, 110.8, 10.6], [4.0, 111.6, 10.2]] },
      { id: 'WR-R', role: 'WR', keys: [[-2.6, 108.4, 44.5], [0, 108.4, 44.5], [1.0, 109.6, 45.2], [2.2, 111.2, 46.0], [3.6, 112.4, 46.4]] },
    ],

    // The Patriots crash the inside run and chase the reverse. The right corner
    // of the end zone — where Foles is standing — is the part nobody covers.
    defense: [
      { id: 'D1', role: 'DL', keys: [[-2.6, 109.6, 24.8], [0, 109.6, 24.8], [0.8, 109.0, 23.4], [2.0, 108.4, 21.6], [3.4, 108.2, 20.4]] },
      { id: 'D2', role: 'DL', keys: [[-2.6, 109.6, 27.2], [0, 109.6, 27.2], [0.8, 109.0, 26.2], [2.0, 108.6, 24.4], [3.4, 108.4, 23.4]] },
      { id: 'D3', role: 'DL', keys: [[-2.6, 109.6, 22.4], [0, 109.6, 22.4], [0.8, 108.9, 21.0], [2.0, 108.2, 19.4], [3.4, 108.0, 18.4]] },
      { id: 'D4', role: 'DL', keys: [[-2.6, 109.6, 29.6], [0, 109.6, 29.6], [0.9, 109.1, 29.0], [2.0, 108.8, 27.4], [3.4, 108.6, 26.4]] },
      { id: 'D5', role: 'DL', keys: [[-2.6, 109.6, 31.8], [0, 109.6, 31.8], [0.9, 109.2, 31.6], [2.0, 108.9, 30.2], [3.4, 108.8, 29.2]] },
      { id: 'D6', role: 'LB', keys: [[-2.6, 111.4, 25.4], [0, 111.4, 25.4], [0.9, 110.4, 23.2], [1.9, 109.2, 20.6], [3.0, 108.6, 19.2], [4.0, 108.4, 19.0]] },
      { id: 'D7', role: 'LB', keys: [[-2.6, 111.4, 28.6], [0, 111.4, 28.6], [0.9, 110.6, 27.0], [1.9, 109.4, 24.0], [3.0, 108.6, 22.6], [4.0, 108.4, 22.4]] },
      { id: 'D8', role: 'S',  keys: [[-2.6, 112.6, 21.0], [0, 112.6, 21.0], [1.0, 111.4, 19.4], [2.2, 110.0, 18.2], [3.4, 109.2, 18.0]] },
      { id: 'D9', role: 'S',  keys: [[-2.6, 112.8, 32.2], [0, 112.8, 32.2], [1.0, 112.0, 30.6], [2.0, 110.6, 28.8], [2.8, 110.0, 30.6], [3.4, 110.2, 33.8], [4.2, 110.6, 36.4]] },
      { id: 'D10', role: 'CB', keys: [[-2.6, 112.0, 12.6], [0, 112.0, 12.6], [1.2, 112.2, 12.0], [2.6, 112.6, 11.2], [4.0, 112.8, 11.0]] },
      { id: 'D11', role: 'CB', keys: [[-2.6, 112.0, 44.8], [0, 112.0, 44.8], [1.2, 112.4, 45.4], [2.6, 113.0, 46.2], [4.0, 113.2, 46.4]] },
    ],

    // Camera keyframes. pos/look are in field coordinates; height is in yards.
    camera: [
      { t: -2.6, pos: [100.0, 18.0], h: 7.0, look: [108.6, 27.0], lh: 1.5 },
      { t: -0.2, pos: [100.8, 21.0], h: 6.2, look: [108.4, 29.2], lh: 1.5 },
      { t: 1.0,  pos: [101.0, 12.0], h: 5.4, look: [106.4, 21.0], lh: 1.3 },
      { t: 2.2,  pos: [101.2, 24.0], h: 5.2, look: [107.0, 31.0], lh: 1.4 },
      { t: 3.0,  pos: [104.0, 50.0], h: 5.2, look: [110.2, 40.4], lh: 1.5 },
      { t: 4.2,  pos: [108.0, 52.5], h: 7.2, look: [111.2, 41.0], lh: 1.4 },
      { t: 6.0,  pos: [112.5, 54.0], h: 10.5, look: [111.6, 41.0], lh: 1.2 },
    ],

    // Beats drive the caption and the strip of play phases along the bottom.
    beats: [
      { t: -2.6, phase: 'PRE',   text: 'Fourth and goal from the one. Thirty-eight seconds left in the half.' },
      { t: -1.2, phase: 'PRE',   text: 'Foles walks up the line. He is not going to take the snap.' },
      { t: 0.02, phase: 'SNAP',  text: 'Kelce snaps it straight past him to Clement.' },
      { t: 1.05, phase: 'PITCH', text: 'Clement runs left and flips it back — the reverse to Burton.' },
      { t: 2.05, phase: 'PITCH', text: 'Burton played quarterback in high school.' },
      { t: 2.40, phase: 'PASS',  text: 'Nobody in white is within ten yards of Nick Foles.' },
      { t: 2.95, phase: 'CATCH', text: 'Touchdown. The quarterback caught it.' },
      { t: 4.30, phase: 'CATCH', text: 'Philly Special. 22–12.' },
    ],

    phases: ['SNAP', 'PITCH', 'PASS', 'CATCH'],
    duration: { start: -2.6, end: 6.4 },
  };

  global.PHILLY_SPECIAL = play;
})(typeof window !== 'undefined' ? window : globalThis);
