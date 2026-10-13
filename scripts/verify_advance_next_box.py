"""Headless check: label advances to the next box on the same image, then the next pending image."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

API = "http://127.0.0.1:8190/api/v1"
UI = "http://127.0.0.1:5173"
OUT = Path("/tmp/review-advance-verify")
OUT.mkdir(parents=True, exist_ok=True)
PASSWORD = 'REDACTED_FOR_GITHUB'


def api(path: str, token: str | None = None, method: str = "GET", body: dict | None = None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"{API}{path}", data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def undo(token: str, candidate_id: str) -> None:
    sample = api(f"/samples", token)  # noqa: F841
    # fetch candidate via listing needs_review is too coarse; get by scanning labeled sample later
    pass


def main() -> None:
    token = api("/auth/login", method="POST", body={"username": "admin", "password": PASSWORD})["access_token"]
    queue = api("/samples?status_filter=needs_review&limit=20", token)["samples"]
    multi = next(item for item in queue if item["counts"]["pending"] >= 3)
    single = next(item for item in queue if item["counts"]["pending"] == 1 and item["sample_id"] != multi["sample_id"])
    multi_detail = api(f"/samples/{multi['sample_id']}", token)
    single_detail = api(f"/samples/{single['sample_id']}", token)
    labeled_ids: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(UI, wait_until="networkidle")
        page.locator('input[type="password"]').fill(PASSWORD)
        page.get_by_role("button", name="登录工作台").click()
        page.locator(".queue-item").first.wait_for(timeout=20000)
        page.screenshot(path=str(OUT / "01-logged-in.png"))

        heading = page.locator(".inspector-heading .eyebrow")
        heading.wait_for(timeout=20000)
        before = heading.inner_text()
        sample_title = page.locator(".sample-context h2").inner_text()
        assert "1 /" in before, before
        page.get_by_role("button", name="真目标").click()
        page.wait_for_function(
            """([prevHeading, prevTitle]) => {
              const heading = document.querySelector('.inspector-heading .eyebrow');
              const title = document.querySelector('.sample-context h2');
              return heading && title && heading.textContent !== prevHeading && title.textContent === prevTitle;
            }""",
            arg=[before, sample_title],
            timeout=15000,
        )
        after_first = heading.inner_text()
        after_title = page.locator(".sample-context h2").inner_text()
        notice = page.locator(".notice").inner_text()
        page.screenshot(path=str(OUT / "02-after-first-true.png"))
        print("FIRST", {"before": before, "after": after_first, "title": after_title, "notice": notice[:80]})
        assert after_title == sample_title, (sample_title, after_title)
        assert "2 /" in after_first, after_first

        page.get_by_role("button", name="真目标").click()
        page.wait_for_function(
            """(prev) => {
              const heading = document.querySelector('.inspector-heading .eyebrow');
              return heading && heading.textContent !== prev;
            }""",
            arg=after_first,
            timeout=15000,
        )
        after_second = heading.inner_text()
        page.screenshot(path=str(OUT / "03-after-second-true.png"))
        print("SECOND", {"after": after_second, "title": page.locator(".sample-context h2").inner_text()})
        assert page.locator(".sample-context h2").inner_text() == sample_title
        assert "3 /" in after_second, after_second

        # Jump-to-next-sample: click the 0/1 queue row if visible, else rely on API-selected single later.
        one_box = page.locator(".queue-item", has_text="0/1").first
        if one_box.count() > 0:
            one_box.click()
            page.wait_for_timeout(800)
            one_before_title = page.locator(".sample-context h2").inner_text()
            one_heading = heading.inner_text()
            page.get_by_role("button", name="真目标").click()
            page.wait_for_function(
                """(prevTitle) => {
                  const title = document.querySelector('.sample-context h2');
                  const notice = document.querySelector('.notice');
                  return title && (title.textContent !== prevTitle || (notice && notice.textContent.includes('下一张')));
                }""",
                arg=one_before_title,
                timeout=15000,
            )
            page.screenshot(path=str(OUT / "04-after-last-box.png"))
            print("NEXT_SAMPLE", {
                "from": one_before_title,
                "to": page.locator(".sample-context h2").inner_text(),
                "notice": page.locator(".notice").inner_text()[:120],
                "heading": heading.inner_text(),
            })
        browser.close()

    # Undo labels written on the multi-box sample so this check does not leave gold data.
    refreshed = api(f"/samples/{multi['sample_id']}", token)
    for candidate in refreshed["candidates"]:
        if candidate["annotation"]["label"] is not None:
            labeled_ids.append(candidate["candidate_id"])
            api(
                f"/candidates/{candidate['candidate_id']}/annotation",
                token,
                method="PATCH",
                body={"label": None, "note": "", "expected_revision": candidate["annotation"]["revision"]},
            )
    single_refreshed = api(f"/samples/{single['sample_id']}", token)
    for candidate in single_refreshed["candidates"]:
        if candidate["annotation"]["label"] is not None:
            labeled_ids.append(candidate["candidate_id"])
            api(
                f"/candidates/{candidate['candidate_id']}/annotation",
                token,
                method="PATCH",
                body={"label": None, "note": "", "expected_revision": candidate["annotation"]["revision"]},
            )
    print("UNDONE", labeled_ids)
    print("MULTI_AFTER", api(f"/samples/{multi['sample_id']}", token)["counts"])
    print("SINGLE_AFTER", api(f"/samples/{single['sample_id']}", token)["counts"])


if __name__ == "__main__":
    main()
