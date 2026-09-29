"""将本机下载的调查图片与页面模板组装到独立 view 目录。"""

import argparse
import shutil
from pathlib import Path


def build(snapshot_directory):
    source = Path(__file__).parent
    view = snapshot_directory / "view"
    assets = view / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    files = {
        "ar-primary.jpg": "images/0e2e65bb679c49e4ad16bcfb56199856/PRIMARYIMG-10dfa209ae474293a0bc518fd0510bc7.jpg",
        "recovered-original.jpg": "images/recovered-original-a9f4fb625d78.jpg",
        "annotated-video-event.jpg": "video-recovery/annotated.jpg",
        "recording-candidate.jpg": "video-recovery/matched-frame.jpg",
        "requested-event.jpg": "images/b3ac0a6c3cd34dc2be9821bf41614558/SHOTIMG-672f78cc09a84bd392ae3cbec2742f77.jpg",
    }
    for name, relative in files.items():
        shutil.copy2(snapshot_directory / relative, assets / name)
    shutil.copy2(source / "原图来源与取回证据.html", view / "index.html")
    shutil.copy2(source / "README.md", snapshot_directory / "原图来源与数据库映射调查报告.md")
    return view / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_directory", type=Path)
    args = parser.parse_args()
    print(build(args.snapshot_directory))
