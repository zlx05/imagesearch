"""Load URL entries from JSON files in the project root."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .models import URLSource


def load_urls_from_json(
    json_path: str | Path,
    max_urls: Optional[int] = None,
    exclude_sources: Optional[list[str]] = None,
) -> List[URLSource]:
    """Load URL entries from a JSON file.

    Args:
        json_path: Path to the JSON file.
        max_urls: Limit the number of URLs returned (for testing/dry runs).
        exclude_sources: Skip entries whose 'source' field matches any of
            these strings (case-insensitive, substring match).

    Returns:
        A list of URLSource objects.

    The JSON file is expected to have a "nodes" array, where each node
    has at least "id" and "url" fields.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    exclude = [s.lower() for s in (exclude_sources or [])]
    sources: List[URLSource] = []

    for node in nodes:
        if not node.get("url"):
            continue

        # Filter by excluded sources (substring match, case-insensitive)
        node_source = node.get("source", "").lower()
        if exclude and any(ex in node_source for ex in exclude):
            continue

        source = URLSource(
            id=node.get("id", ""),
            url=node.get("url", ""),
            title=node.get("title", ""),
            source=node.get("source", ""),
            engine=node.get("engine", ""),
            image_url=node.get("image_url", ""),
            possible_duplicate=node.get("possible_duplicate", False),
            reason=node.get("reason", ""),
            similarity=node.get("similarity", 0.0),
        )
        sources.append(source)

    if max_urls is not None and max_urls > 0:
        sources = sources[:max_urls]

    return sources


def discover_json_files(root_dir: str | Path) -> List[Path]:
    """Find all JSON files in the root directory (non-recursive).

    Args:
        root_dir: The root directory to search.

    Returns:
        Sorted list of Path objects for JSON files.
    """
    root = Path(root_dir)
    json_files = sorted(root.glob("*.json"))
    return json_files
