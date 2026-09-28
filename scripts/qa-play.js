#!/usr/bin/env node
/*
 * Headless check of a play, for CI and for bulk runs over a season.
 *
 *   node scripts/qa-play.js packages/replay-engine/prototype/plays/philly-special.js
 *
 * Exits non-zero when a play should not be published, so it can gate a pipeline
 * rather than relying on somebody watching every replay.
 */
const path = require('path');
global.window = global.window || {};
const base = path.join(__dirname, '..', 'packages', 'replay-engine', 'prototype');
['contract', 'adapters', 'rig', 'qa'].forEach((m) => require(path.join(base, 'engine', m + '.js')));

const arg = process.argv[2];
if (!arg) {
  console.error('usage: node scripts/qa-play.js <play.js | tracking.json>');
  process.exit(2);
}

const G = global.window.GRID;
let track;
if (arg.endsWith('.json')) {
  const rows = require(path.resolve(arg));
  track = G.fromTrackingRows(rows.rows || rows, rows.options || {});
} else {
  require(path.resolve(arg));
  const def = global.window.PHILLY_SPECIAL || global.window.PLAY;
  if (!def) { console.error('no play definition exported on window'); process.exit(2); }
  track = G.fromKeyframes(def);
}

const rig = new G.Rig(track);
const report = G.qa(track, rig);
console.log(G.qaText(report));
process.exit(report.ok ? 0 : 1);
