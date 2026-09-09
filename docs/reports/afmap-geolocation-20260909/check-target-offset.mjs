import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import assert from 'node:assert/strict';

// This reproduces the reported coordinate separation, not surveyed accuracy.
const fixture = JSON.parse(fs.readFileSync(new URL('./evidence.json', import.meta.url)));
const args = process.argv.slice(2);
const live = args.includes('--live');
const save = args.includes('--save');
const eventId = args.find(value => value.startsWith('--event-id='))?.slice('--event-id='.length) ?? fixture.eventId;
const selectedCase = fixture.cases.find(item => item.eventId === eventId);
assert(selectedCase, `Event ${eventId} is not in the captured evidence`);
assert(!save || live, '--save requires --live');
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
const targetUrl = new URL(targetSource.url);
targetUrl.searchParams.set('id', eventId);
const mainUrl = new URL(`/gis_server_a/portal/task/situation/detail/event/${eventId}`, mainSource.url);
let origin = selectedCase.eventCoordinates;
let targets = selectedCase.targets;
let traceId = null;
let mainTraceId = null;
let liveEventTime = null;
let targetLocation = null;
if (live) {
  const response = get(targetUrl.href);
  const main = get(mainUrl.href);
  assert.equal(main.data.id, eventId);
  assert.equal(main.data.eventTime, selectedCase.eventTime);
  origin = JSON.parse(main.data.lngLatAlt).geometry.coordinates;
  targets = response.data.map(item => ({
    index: item.properties.index,
    coordinates: item.geometry.coordinates,
    imageBox: item.properties.imgxy,
  }));
  traceId = response.traceId;
  mainTraceId = main.traceId;
  liveEventTime = main.data.eventTime;
  targetLocation = main.data.targetLocation;
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
const savedCoordinates = selectedCase.targets.map(item => [item.index, item.coordinates]).sort((a, b) => a[0] - b[0]);
const currentCoordinates = targets.map(item => [item.index, item.coordinates]).sort((a, b) => a[0] - b[0]);
const output = {
  checkedAt: new Date().toISOString(),
  mode: live ? 'live-read-only' : 'saved-evidence',
  eventId,
  eventTime: liveEventTime ?? selectedCase.eventTime,
  targetLocation,
  traceId,
  mainTraceId,
  sourceUrls: live ? {main: mainUrl.href, targets: targetUrl.href} : null,
  sameCrsAssumed: true,
  surveyedAccuracyVerified: false,
  origin,
  mainMatchesSavedCoordinates: JSON.stringify(origin) === JSON.stringify(selectedCase.eventCoordinates),
  targetsMatchSavedCoordinates: JSON.stringify(currentCoordinates) === JSON.stringify(savedCoordinates),
  results,
  diagnosticLimitMeters: limit,
  verdict: limit === null ? 'MEASURED' : exceedsLimit ? 'OFFSET_REPRODUCED' : 'WITHIN_DIAGNOSTIC_LIMIT',
};
const serialized = JSON.stringify(output, null, 2);
console.log(serialized);
if (save) fs.writeFileSync(new URL(`./live-${eventId}.json`, import.meta.url), `${serialized}\n`);
if (exceedsLimit) process.exitCode = 1;
