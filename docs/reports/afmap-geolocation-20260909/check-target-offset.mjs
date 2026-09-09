import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import assert from 'node:assert/strict';

// This reproduces the reported coordinate separation, not surveyed accuracy.
const fixture = JSON.parse(fs.readFileSync(new URL('./evidence.json', import.meta.url)));
const args = process.argv.slice(2);
const live = args.includes('--live');
const limitArg = args.find(value => value.startsWith('--max-distance='));
const limit = limitArg ? Number(limitArg.split('=')[1]) : null;
assert(limit === null || (Number.isFinite(limit) && limit >= 0));
const get = url => {
  const response = JSON.parse(execFileSync('curl', [
    '--fail', '--silent', '--show-error', '--max-time', '20', url,
  ], { encoding: 'utf8' }));
  assert.equal(response.code, 200);
  assert.equal(response.success, true);
  return response;
};
const targetSource = fixture.sources.find(source => source.url.includes('/event/detail?'));
const mainSource = fixture.sources.find(source => source.url.includes('/detail/event/'));
let origin = fixture.sameTimeTelemetryCoordinates;
let targets = fixture.cases.find(item => item.eventId === fixture.eventId).targets;
let traceId = null;
if (live) {
  const response = get(targetSource.url);
  const main = get(mainSource.url);
  origin = JSON.parse(main.data.lngLatAlt).geometry.coordinates;
  assert.deepEqual(origin.slice(0, 2), fixture.sameTimeTelemetryCoordinates.slice(0, 2));
  targets = response.data.map(item => ({
    index: item.properties.index,
    coordinates: item.geometry.coordinates,
    imageBox: item.properties.imgxy,
  }));
  traceId = response.traceId;
}
assert(targets.length > 0);
const distance = (a, b) => {
  const r = Math.PI / 180;
  const h = Math.sin((b[1] - a[1]) * r / 2) ** 2
    + Math.cos(a[1] * r) * Math.cos(b[1] * r) * Math.sin((b[0] - a[0]) * r / 2) ** 2;
  return 2 * 6371008.8 * Math.atan2(Math.sqrt(h), Math.sqrt(1 - h));
};
const results = targets.map(item => {
  assert(item.coordinates.slice(0, 2).every(Number.isFinite));
  const box = item.imageBox;
  const validBox = [box.x, box.y, box.width, box.height, box.frame_width, box.frame_height].every(Number.isFinite)
    && box.x >= 0 && box.y >= 0 && box.width > 0 && box.height > 0
    && box.x + box.width <= box.frame_width && box.y + box.height <= box.frame_height;
  assert(validBox, `Invalid box ${item.index}`);
  return {
    index: item.index,
    coordinates: item.coordinates,
    distanceMeters: Math.round(distance(origin, item.coordinates) * 1000) / 1000,
    boxWithinDeclaredFrame: validBox,
    declaredFrame: [box.frame_width, box.frame_height],
  };
});
const exceedsLimit = limit !== null && results.some(item => item.distanceMeters > limit);
console.log(JSON.stringify({
  checkedAt: new Date().toISOString(),
  mode: live ? 'live-read-only' : 'saved-evidence',
  eventId: fixture.eventId,
  traceId,
  sameCrsAssumed: true,
  surveyedAccuracyVerified: false,
  origin,
  results,
  diagnosticLimitMeters: limit,
  verdict: limit === null ? 'MEASURED' : exceedsLimit ? 'OFFSET_REPRODUCED' : 'WITHIN_DIAGNOSTIC_LIMIT',
}, null, 2));
if (exceedsLimit) process.exitCode = 1;
