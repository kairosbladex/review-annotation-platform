import assert from 'node:assert/strict';
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';

const root = new URL('./', import.meta.url);
const logArg = process.argv.find(arg => arg.startsWith('--private-log='));
assert(logArg, '--private-log must name the privately retained GIS log excerpt');
const inventory = JSON.parse(fs.readFileSync(new URL('stored-geolocation-inventory.json', root)));
const original = JSON.parse(fs.readFileSync(new URL('original-request-181008.json', root)));
const backup = JSON.parse(fs.readFileSync(new URL('event-correction-181008.json', root)));
const taskId = '5c0355c1ca2e4e56a319dfc181f58fd0';
const events = inventory.allEvents.filter(event => event.taskId === taskId);
const pending = new Map();
const records = [];
const toTime = ms => new Date(ms + 8 * 3600 * 1000).toISOString().slice(0, 19).replace('T', ' ');
const get = path => {
  const result = JSON.parse(execFileSync('curl', ['--fail', '--silent', '--show-error', '--max-time', '20',
    `https://lxtest.zhuoyilingxi.com:8088/gis_server_a${path}`], { encoding: 'utf8' }));
  assert.equal(result.success, true);
  assert.equal(result.code, 200);
  return result.data;
};
for (const line of fs.readFileSync(logArg.slice('--private-log='.length), 'utf8').split('\n')) {
  if (!line.includes('ArApiUtil$HttpUtil')) continue;
  const thread = line.match(/\[(Ktd-Async-[^\]]+)\]/)?.[1].trim();
  if (!thread) continue;
  const requestPart = line.split('请求体：')[1];
  if (requestPart) {
    const request = JSON.parse(requestPart);
    if (request.log_url?.includes(taskId) && request.bboxes_xyxy) {
      pending.set(thread, { request, requestLine: Number(line.split(':')[0]) });
    }
    continue;
  }
  const responsePart = line.split('响应结果：')[1];
  if (!responsePart || !pending.has(thread)) continue;
  const response = JSON.parse(responsePart);
  if (!Array.isArray(response.results)) continue;
  const { request, requestLine } = pending.get(thread);
  pending.delete(thread);
  const matches = events.filter(event => event.eventTime === toTime(request.shot_time_ms)
    && event.measurableUniquePoints === response.results.length);
  assert.equal(matches.length, 1, 'Ambiguous event association; do not guess');
  const event = matches[0];
  const main = get(`/portal/task/situation/detail/event/${event.eventId}`);
  assert.equal(main.id, event.eventId);
  assert.equal(main.taskId, taskId);
  assert.equal(main.eventTime, event.eventTime);
  const targets = event.eventId === original.eventId ? JSON.parse(backup.rows[0].ar_geo_coordinates)
    : get(`/portal/task/situation/event/detail?id=${event.eventId}`);
  const coordinates = targets.map(item => [item.properties.index, ...item.geometry.coordinates.slice(0, 2)])
    .sort((a, b) => a[0] - b[0]);
  const originalCoordinates = response.results.filter(item => item.ok).map(item => [item.index, item.lon, item.lat])
    .sort((a, b) => a[0] - b[0]);
  assert.deepEqual(coordinates, originalCoordinates, `Stored coordinates mismatch for ${event.eventId}`);
  records.push({ eventId: event.eventId, eventTime: event.eventTime, taskId,
    mainCoordinates: JSON.parse(main.lngLatAlt).geometry.coordinates,
    sourceLog: original.sourceLog, requestLine, responseLine: Number(line.split(':')[0]),
    request, originalResults: response.results.map(({ index, ok, lon, lat, alt, anchor_uv }) => ({ index, ok, lon, lat, alt, anchor_uv })),
    storedOriginalResponseMatch: true,
  });
}
assert.equal(records.length, events.length);
assert.equal(new Set(records.map(item => item.eventId)).size, events.length);
const output = { checkedAt: new Date().toISOString(), taskId, readOnly: true, records };
fs.writeFileSync(new URL('same-task-original-requests.json', root), JSON.stringify(output, null, 2) + '\n');
console.log(JSON.stringify({ matchedEvents: records.length, originalRequestsRecovered: true, allStoredCoordinatesMatched: true }));
