# Browser behavior tests

These tests drive the real React application in headless Google Chrome with an existing Playwright installation. They replace only the HTTP API boundary with synthetic fixtures. They do not call the algorithm service or measure model accuracy.

Start the existing Vite development server on port 5178, then run:

```sh
PLAYWRIGHT_MODULE_PATH=/absolute/path/to/existing/playwright/index.mjs npm run test:browser
```

Set `FRONTEND_TEST_URL` for another server address. Google Chrome must be installed (`channel: chrome`). No additional project dependencies are required. Set `FRONTEND_SCREENSHOT_PATH` to an absolute PNG path to capture the completed retrieval view.

Coverage includes upload and single submission with an idempotency key, reference image retrieval, persisted task reopening, file/video progress, terminal polling, zero hits versus failure/cancellation, explicit cancellation, bounded disconnect recovery (including HTTP 200 snapshots with an upstream sync error), stale response rejection, uncertain creation without a repeated POST, persisted history creation/reopening, older records on page two, the disabled capability gate, nullable image metadata, partial image upload failures, and mobile layout.

## Deployed test-site acceptance

`live-workbench.mjs` uses the real browser, login, API, storage, worker and algorithm at the fixed independent test URL `http://192.168.130.238:5276`. It never intercepts requests or substitutes responses. Running without a flag only prints preparation status. `--prepare-only` generates and verifies local synthetic images and a six-second MP4, then removes those temporary fixtures without accessing the test site.

Run `--execute` only after the release has been deployed and acceptance writes are authorized:

```sh
VISION_RELEASE_DIR=/data/workspace/user/review-vision-test/releases/REPLACE_WITH_40_LOWERCASE_HEX_SHA \
VISION_EVIDENCE_DIR=/absolute/path/to/acceptance-evidence \
PLAYWRIGHT_MODULE_PATH=/absolute/path/to/existing/playwright/index.mjs \
node frontend/tests/live-workbench.mjs --execute
```

The runner validates the release directory and the four running application containers' release labels, reads the two administrator fields from the release `.private.env` through fixed SSH host `ai-lingxi`, performs a real login remotely, and retains only the real token in process/browser memory. It does not print credentials, headers, session state, raw API errors, or signed media URLs; browser tracing, inherited Playwright diagnostics and saved authentication state are not used. The report contains only hashes, created resource IDs, counts, safe status codes and browser error counters.

`VISION_PYTHON` may select an existing absolute Python interpreter with Pillow, NumPy and OpenCV. `VISION_LIVE_IMAGE` may select an existing absolute frozen image for the review and video fixture; the runner records its hash and never copies it into the repository or report. Optional `VISION_LIVE_SCREENSHOTS=1` captures synthetic-only UI views and is disabled when a real image is supplied.

The runner creates at most one review job, one retrieval task and one historical comparison. It refuses review batches containing unrelated inputs or more than three samples, waits at most 180 seconds for each task, never repeats task creation after an uncertain result, and has a 15-minute total limit. It checks all three tabs, real image upload and a pre-existing `legacy_review_agent` profile, terminal review status/counts, history creation/reopening with decoded stored images, video retrieval completion or explicit failure, retrieval reopening without another POST, and readable hit images. Model failure, missing real dependencies, browser errors and missing evidence remain failures. Zero hits are a pipeline result and do not establish recognition accuracy.
