"""
Corpus Loader: Loads and indexes support documentation from local markdown files.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import yaml


def _parse_frontmatter(content: str) -> tuple[Dict, str]:
    """Parse YAML frontmatter from markdown content."""
    frontmatter = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1]) or {}
            except Exception:
                frontmatter = {}
            body = parts[2].strip()
    return frontmatter, body


def _clean_markdown(text: str) -> str:
    """Strip markdown syntax and clean text for retrieval."""
    # Remove image embeds
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Remove links but keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove headers markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _company_from_path(filepath: str, data_dir: str) -> str:
    """Infer company from file path."""
    rel = Path(filepath).relative_to(data_dir)
    parts = rel.parts
    if parts[0] == "hackerrank":
        return "HackerRank"
    elif parts[0] == "claude":
        return "Claude"
    elif parts[0] == "visa":
        return "Visa"
    return "Unknown"


def _product_area_from_path(filepath: str, data_dir: str) -> str:
    """Derive product_area from directory path and breadcrumbs."""
    rel = Path(filepath).relative_to(data_dir)
    parts = rel.parts
    # e.g. hackerrank/screen/... -> screen
    #      claude/claude/conversation-management/... -> conversation_management
    #      visa/support/consumer/... -> consumer_support
    if len(parts) > 1:
        area_parts = list(parts[1:-1])  # dirs between company and filename
        # Normalize
        area = "_".join(p.replace("-", "_") for p in area_parts) if area_parts else parts[0]
        return area[:80]  # cap length
    return parts[0]


def load_corpus(data_dir: str) -> List[Dict]:
    """
    Load all markdown files from data_dir.
    Returns list of dicts with keys: id, title, content, raw_content, company, product_area, filepath.
    """
    data_path = Path(data_dir)
    docs = []
    for md_file in sorted(data_path.rglob("*.md")):
        try:
            raw = md_file.read_text(encoding="utf-8", errors="replace")
            frontmatter, body = _parse_frontmatter(raw)
            title = frontmatter.get("title", "") or md_file.stem
            breadcrumbs = frontmatter.get("breadcrumbs", [])
            company = _company_from_path(str(md_file), data_dir)
            product_area = _product_area_from_path(str(md_file), data_dir)

            # Use breadcrumbs to refine product_area
            if breadcrumbs and len(breadcrumbs) >= 2:
                product_area = "_".join(str(b).lower().replace(" ", "_").replace("-", "_")
                                        for b in breadcrumbs[1:])

            content = _clean_markdown(body)
            full_text = f"{title}\n{content}"

            docs.append({
                "id": str(md_file.relative_to(data_dir)),
                "title": title,
                "content": content,
                "full_text": full_text,
                "company": company,
                "product_area": product_area,
                "breadcrumbs": breadcrumbs,
                "filepath": str(md_file),
            })
        except Exception as e:
            print(f"Warning: could not load {md_file}: {e}")
    return docs
