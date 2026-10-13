import assert from "node:assert/strict";
import { test } from "node:test";

import {
  advanceAfterLabel,
  keepQueueSelection,
  nextPendingCandidateId,
  nextPendingSampleId,
} from "./selection.ts";

function box(id: string, label: string | null) {
  return { candidate_id: id, annotation: { label } };
}

function sample(id: string, pending: number) {
  return { sample_id: id, counts: { pending } };
}

test("next unlabeled box is the following pending candidate on the same image", () => {
  const candidates = [box("a", "true"), box("b", null), box("c", null)];
  assert.equal(nextPendingCandidateId(candidates, "a"), "b");
});

test("next unlabeled box wraps past already labeled boxes", () => {
  const candidates = [box("a", null), box("b", "false"), box("c", "true")];
  assert.equal(nextPendingCandidateId(candidates, "c"), "a");
});

test("no next box when every candidate on the image is labeled", () => {
  const candidates = [box("a", "true"), box("b", "false")];
  assert.equal(nextPendingCandidateId(candidates, "b"), null);
});

test("next pending sample skips the current image and completed images", () => {
  const queue = [sample("s1", 0), sample("s2", 2), sample("s3", 1)];
  assert.equal(nextPendingSampleId(queue, "s1"), "s2");
});

test("after labeling a box, stay on the same image and select the next unlabeled box", () => {
  const result = advanceAfterLabel({
    candidates: [box("a", "true"), box("b", null), box("c", null)],
    labeledId: "a",
    currentSampleId: "img-1",
    queueSamples: [sample("img-1", 2), sample("img-2", 4)],
  });
  assert.deepEqual(result, {
    sampleId: "img-1",
    candidateId: "b",
    kind: "next_box",
  });
});

test("after the last box on an image, move to the next sample that still has pending boxes", () => {
  const result = advanceAfterLabel({
    candidates: [box("a", "true"), box("b", "false")],
    labeledId: "b",
    currentSampleId: "img-1",
    queueSamples: [sample("img-2", 3), sample("img-3", 1)],
  });
  assert.deepEqual(result, {
    sampleId: "img-2",
    candidateId: null,
    kind: "next_sample",
  });
});

test("after the last pending image, stay put and report the queue is done", () => {
  const result = advanceAfterLabel({
    candidates: [box("a", "true")],
    labeledId: "a",
    currentSampleId: "img-1",
    queueSamples: [sample("img-1", 0)],
  });
  assert.deepEqual(result, {
    sampleId: "img-1",
    candidateId: "a",
    kind: "queue_done",
  });
});

test("queue refresh keeps the current sample even when the filter dropped it", () => {
  assert.equal(
    keepQueueSelection("img-1", [sample("img-2", 4), sample("img-3", 1)], { stealIfMissing: false }),
    "img-1",
  );
});

test("queue refresh on filter change may fall back to the first sample on the page", () => {
  assert.equal(
    keepQueueSelection("img-1", [sample("img-2", 4), sample("img-3", 1)], { stealIfMissing: true }),
    "img-2",
  );
});

test("empty queue with no current selection has nothing selected", () => {
  assert.equal(keepQueueSelection(null, [], { stealIfMissing: true }), null);
});
