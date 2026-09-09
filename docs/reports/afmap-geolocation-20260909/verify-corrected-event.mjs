import assert from 'node:assert/strict';
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = new URL('./', import.meta.url);
const backup = JSON.parse(fs.readFileSync(new URL('event-correction-181008.json', root)));
const original = JSON.parse(fs.readFileSync(new URL('original-request-181008.json', root)));
const measurement = JSON.parse(execFileSync(process.execPath, [
  fileURLToPath(new URL('check-target-offset.mjs', root)), '--live',
  `--event-id=${backup.eventId}`, '--max-distance=250',
], { encoding: 'utf8' }));
const expected = JSON.parse(backup.rows[0].replacement);
assert.equal(measurement.eventId, original.eventId);
assert.equal(measurement.mainMatchesSavedCoordinates, true);
assert.equal(measurement.targetsMatchSavedCoordinates, false);
assert.equal(measurement.results.length, expected.length);
for (const point of measurement.results) {
  const saved = expected.find(feature => feature.properties.index === point.index);
  assert(saved, `Unexpected target index: ${point.index}`);
  assert.deepEqual(point.coordinates, saved.geometry.coordinates);
}
const response = JSON.parse(execFileSync('curl', [
  '--fail', '--silent', '--show-error', '--max-time', '30',
  '-H', 'Content-Type: application/json', '--data-binary', '@-',
  'http://192.168.130.238:8181/v1/geo/from-bboxes',
], { input: JSON.stringify({ ...original.request, debug: true }), encoding: 'utf8' }));
assert.equal(response.count, 2);
assert.equal(response.ok_count, 2);
const selected = response.debug_context.used_target;
assert.equal(selected.selected_pose_time_ms, 1788775808010);
assert(selected.time_validation_applied && Math.abs(selected.pose_time_delta_sec) < 0.02);
for (const point of response.results) {
  assert.equal(point.ok, true);
  const saved = expected.find(feature => feature.properties.index === point.index);
  assert(saved);
  assert.deepEqual([point.lon, point.lat], saved.geometry.coordinates.slice(0, 2));
}
const output = {
  ...measurement,
  originalRequestLiveReplay: { selected, cameraPose: response.debug_context.camera_pose },
  matchesPreparedReplacement: true,
  matchesLiveGeolocationResponse: true,
  visualAcceptanceVerified: false,
  verdict: 'PASS_STORED_COORDINATES_AND_ORIGINAL_REQUEST_REPLAY',
};
const serialized = JSON.stringify(output, null, 2) + '\n';
console.log(serialized);
if (process.argv.includes('--save')) {
  fs.writeFileSync(new URL('post-correction-181008.json', root), serialized);
}
