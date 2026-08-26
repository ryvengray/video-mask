#!/usr/bin/env python3
"""Print how many task videos have each manually assigned content tag.

Run on the Controller host, for example:
  .venv/bin/python scripts/content_tag_counts.py \
    --url http://127.0.0.1:8080/api/dashboard/content-tag-statistics
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Count videos for each manual content tag")
    parser.add_argument(
        "--url", default="http://127.0.0.1:8080/api/dashboard/content-tag-statistics",
        help="Controller content-tag statistics endpoint",
    )
    parser.add_argument(
        "--status", action="append", default=[],
        help="Optional task status; repeat or comma-separate values",
    )
    parser.add_argument("--json", action="store_true", help="Print the API JSON unchanged")
    args = parser.parse_args()

    statuses = ",".join(args.status)
    url = args.url + ("&" if "?" in args.url else "?") + urllib.parse.urlencode({"status": statuses})
    with urllib.request.urlopen(url, timeout=15) as response:  # nosec B310: caller supplies Controller URL
        payload: dict[str, Any] = json.load(response)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(f"{'Content tag':<36} Videos")
    for item in payload.get("tags", []):
        print(f"{str(item['tag']):<36} {item['video_count']}")
    print(f"Total tag occurrences: {payload.get('tagged_video_occurrences', 0)}")


if __name__ == "__main__":
    main()
