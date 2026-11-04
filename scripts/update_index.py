"""Validate review front matter and replace only the README archive block."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import html
from pathlib import Path
import re
import sys
from urllib.parse import quote, urlsplit, urlunsplit

try:
    import yaml
except ImportError:
    raise SystemExit("Missing PyYAML. Run: python -m pip install -r requirements.txt")

START = "<!-- PAPER_ARCHIVE:START -->"
END = "<!-- PAPER_ARCHIVE:END -->"
DEFAULT_ROOT = Path(__file__).resolve().parents[1]


class IndexError(ValueError):
    """An archive input is missing or invalid."""


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently overwriting metadata."""


def unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise IndexError("front matter keys must be strings")
        if key in result:
            raise IndexError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping
)


@dataclass(frozen=True)
class Review:
    path: Path
    title: str
    written: date
    venue: str
    distinction: str
    categories: tuple[str, ...]
    paper: str
    status: str


def scalar(data: dict, key: str, *, optional: bool = False) -> str:
    value = data.get(key, "" if optional else None)
    if not isinstance(value, str) or (not optional and not value.strip()):
        raise IndexError(f"{key}: expected {'a string' if optional else 'a non-empty string'}")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise IndexError(f"{key}: must be a single line without control characters")
    return value.strip()


def http_url(value: str, key: str) -> str:
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and not parsed.username
            and not parsed.password
            and not any(char.isspace() for char in value)
        )
        _ = parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise IndexError(f"{key}: expected an absolute http(s) URL without credentials or whitespace")
    return value


def read_review(path: Path) -> Review:
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise IndexError("missing YAML front matter opening ---")
    try:
        closing = lines.index("---", 1)
    except ValueError:
        raise IndexError("missing YAML front matter closing ---") from None
    data = yaml.load("\n".join(lines[1:closing]), Loader=UniqueKeyLoader)
    if not isinstance(data, dict):
        raise IndexError("front matter must be a YAML mapping")
    if not "\n".join(lines[closing + 1:]).strip():
        raise IndexError("review body is empty")

    title = scalar(data, "title")
    venue = scalar(data, "venue")
    distinction = scalar(data, "distinction", optional=True)
    status = scalar(data, "status")
    if status not in {"draft", "published"}:
        raise IndexError("status: expected draft or published")
    raw_date = data.get("date")
    if type(raw_date) is date:
        written = raw_date
    elif isinstance(raw_date, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_date):
        try:
            written = date.fromisoformat(raw_date)
        except ValueError:
            raise IndexError("date: invalid calendar date") from None
    else:
        raise IndexError("date: expected YYYY-MM-DD (the review date)")
    categories = data.get("categories")
    if not isinstance(categories, list) or not categories:
        raise IndexError("categories: expected a non-empty YAML list")
    clean_categories = tuple(scalar({"category": item}, "category") for item in categories)
    if len({item.casefold() for item in clean_categories}) != len(clean_categories):
        raise IndexError("categories: duplicate tags")
    paper = http_url(scalar(data, "paper"), "paper")
    code = scalar(data, "code", optional=True)
    if code:
        http_url(code, "code")
    return Review(path, title, written, venue, distinction, clean_categories, paper, status)


def paper_identity(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    if host in {"arxiv.org", "www.arxiv.org"}:
        path = re.sub(r"^/pdf/", "/abs/", path)
        path = re.sub(r"\.pdf$", "", path)
        path = re.sub(r"v\d+$", "", path)
    return urlunsplit((parsed.scheme.lower(), host, path, parsed.query, ""))


def collect_reviews(root: Path) -> list[Review]:
    directory = root / "reviews"
    if not directory.is_dir():
        raise IndexError("reviews/ directory is missing")
    reviews = []
    errors = []
    seen_urls: dict[str, Path] = {}
    seen_titles: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".md":
            continue
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                raise IndexError("review must be a regular file inside reviews/")
            review = read_review(path)
            identity = paper_identity(review.paper)
            title_key = " ".join(review.title.casefold().split())
            existing = seen_urls.get(identity) or seen_titles.get(title_key)
            if existing is not None:
                raise IndexError(f"duplicate paper; already in {existing.relative_to(root).as_posix()}")
            seen_urls[identity] = path
            seen_titles[title_key] = path
            reviews.append(review)
        except (IndexError, OSError, UnicodeError, yaml.YAMLError) as exc:
            errors.append(f"{path.relative_to(root).as_posix()}: {exc}")
    if errors:
        raise IndexError("\n".join(errors))
    return sorted(
        (review for review in reviews if review.status == "published"),
        key=lambda review: (-review.written.toordinal(), review.title.casefold(), review.path.as_posix()),
    )


def cell(value: str) -> str:
    # Escape raw HTML and Markdown metacharacters inside table/link text.
    value = html.escape(value, quote=False)
    for char in "\\`*_[]|":
        value = value.replace(char, "\\" + char)
    return value


def destination(value: str) -> str:
    return quote(value, safe="/:?=&%+#@~!$;',-")


def render_archive(reviews: list[Review], root: Path) -> str:
    lines = [f"Total papers: {len(reviews)}", "", "| Date | Paper | Venue | Category | Review |", "|---|---|---|---|---|"]
    for review in reviews:
        venue = review.venue + (f" · {review.distinction}" if review.distinction else "")
        path = destination(review.path.relative_to(root).as_posix())
        lines.append(
            f"| {review.written.isoformat()} | [{cell(review.title)}]({destination(review.paper)}) "
            f"| {cell(venue)} | {', '.join(cell(tag) for tag in review.categories)} | [Review]({path}) |"
        )
    return "\n".join(lines)


def update(root: Path, *, check: bool = False) -> bool:
    reviews = collect_reviews(root)
    readme = root / "README.md"
    original = readme.read_bytes()
    text = original.decode("utf-8")
    if text.count(START) != 1 or text.count(END) != 1 or text.index(START) > text.index(END):
        raise IndexError("README.md must contain exactly one archive START and END marker")
    before, rest = text.split(START)
    old_archive, after = rest.split(END)
    # Markers must stand alone; never modify a malformed boundary.
    if before.split("\n")[-1].strip() or after.split("\n")[0].strip():
        raise IndexError("README archive markers must each occupy their own line")
    newline = "\r\n" if "\r\n" in old_archive else "\n"
    archive = render_archive(reviews, root).replace("\n", newline)
    updated = (before + START + newline + archive + newline + END + after).encode("utf-8")
    changed = updated != original
    if changed and not check:
        readme.write_bytes(updated)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="repository directory")
    parser.add_argument("--check", action="store_true", help="validate without writing; exit 1 if stale")
    args = parser.parse_args(argv)
    try:
        changed = update(args.root.resolve(), check=args.check)
    except (IndexError, OSError, UnicodeError) as exc:
        print(f"Index error: {exc}", file=sys.stderr)
        return 2
    if args.check and changed:
        print("README archive is out of date. Run: python scripts/update_index.py", file=sys.stderr)
        return 1
    print("README archive updated." if changed else "README archive is up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
