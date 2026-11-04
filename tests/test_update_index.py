"""Exercise archive publishing, preservation, validation, and CLI behavior."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import date
import io
from pathlib import Path
import tempfile
import unittest

import yaml

from scripts import update_index as index


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "reviews").mkdir()
        self.readme = self.root / "README.md"
        self.before = "# Keep this title\n\n한국어 설명\n\n"
        self.after = "\n\n[Custom link](keep.md)\n"
        self.readme.write_bytes(
            (self.before + index.START + "\nstale\n" + index.END + self.after).encode("utf-8")
        )

    def review(self, name="memory/paper.md", **changes):
        metadata = {
            "title": "A memory paper", "date": "2026-11-04", "venue": "ICLR 2026",
            "distinction": "Oral", "categories": ["Agent Memory", "Personalization"],
            "paper": "https://arxiv.org/abs/2601.12345", "code": "", "status": "published",
        }
        metadata.update(changes)
        path = self.root / "reviews" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("---\n" + yaml.safe_dump(metadata, allow_unicode=True) + "---\n\n# Explanation\n", encoding="utf-8")
        return path

    def output(self):
        return self.readme.read_text(encoding="utf-8")

    def test_published_table_with_source_review_and_multiple_tags(self):
        self.review()
        self.assertTrue(index.update(self.root))
        text = self.output()
        self.assertIn("Total papers: 1", text)
        self.assertIn("[A memory paper](https://arxiv.org/abs/2601.12345)", text)
        self.assertIn("ICLR 2026 · Oral", text)
        self.assertIn("Agent Memory, Personalization", text)
        self.assertIn("[Review](reviews/memory/paper.md)", text)

    def test_draft_excluded_and_empty_archive_valid(self):
        self.review(status="draft")
        index.update(self.root)
        self.assertIn("Total papers: 0", self.output())
        self.assertNotIn("A memory paper", self.output())

    def test_sort_by_review_date_then_title_not_venue_year(self):
        self.review("a.md", title="Old review", date="2026-01-01", venue="ICML 2026")
        self.review("b.md", title="Zebra", paper="https://example.com/b", venue="ICCV 2023")
        self.review("c.md", title="Alpha", paper="https://example.com/c")
        index.update(self.root)
        text = self.output()
        self.assertLess(text.index("[Alpha]"), text.index("[Zebra]"))
        self.assertLess(text.index("[Zebra]"), text.index("[Old review]"))

    def test_preserves_surrounding_bytes_and_is_idempotent(self):
        self.review()
        index.update(self.root)
        first = self.readme.read_bytes()
        self.assertTrue(first.startswith(self.before.encode("utf-8")))
        self.assertTrue(first.endswith(self.after.encode("utf-8")))
        self.assertFalse(index.update(self.root))
        self.assertEqual(first, self.readme.read_bytes())

    def test_crlf_and_bom_preserved(self):
        original = b"\xef\xbb\xbf" + self.readme.read_bytes().replace(b"\n", b"\r\n")
        self.readme.write_bytes(original)
        index.update(self.root)
        result = self.readme.read_bytes()
        self.assertTrue(result.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\n", result.replace(b"\r\n", b""))
        self.assertFalse(index.update(self.root))

    def test_check_is_read_only_and_reports_stale(self):
        original = self.readme.read_bytes()
        self.assertTrue(index.update(self.root, check=True))
        self.assertEqual(original, self.readme.read_bytes())
        index.update(self.root)
        self.assertFalse(index.update(self.root, check=True))

    def test_invalid_metadata_never_changes_readme_even_for_drafts(self):
        for changes in [
            {"title": ""}, {"title": 10}, {"title": "two\nlines"},
            {"date": "2026-02-30"}, {"date": "2026-1-1"}, {"date": 20261104},
            {"status": "ready"}, {"status": True}, {"venue": ""},
            {"distinction": None}, {"categories": "Memory"}, {"categories": []},
            {"categories": [1]}, {"categories": ["Memory", "memory"]},
            {"paper": "javascript:alert(1)"}, {"paper": "https:///missing-host"},
            {"paper": "https://user:pass@example.com/paper"},
            {"paper": "https://example.com:invalid/paper"},
            {"code": "not-a-url"}, {"code": None},
        ]:
            with self.subTest(changes=changes):
                self.review(status="draft", **{key: value for key, value in changes.items() if key != "status"})
                if "status" in changes:
                    self.review(**changes)
                original = self.readme.read_bytes()
                with self.assertRaises(index.IndexError):
                    index.update(self.root)
                self.assertEqual(original, self.readme.read_bytes())

    def test_missing_required_fields(self):
        path = self.review()
        text = path.read_text(encoding="utf-8")
        metadata = yaml.safe_load(text.split("---")[1])
        for key in ("title", "date", "venue", "categories", "paper", "status"):
            with self.subTest(key=key):
                reduced = {name: value for name, value in metadata.items() if name != key}
                path.write_text("---\n" + yaml.safe_dump(reduced) + "---\n# Body\n", encoding="utf-8")
                with self.assertRaises(index.IndexError):
                    index.update(self.root)

    def test_optional_fields_and_yaml_date(self):
        path = self.review(date=date(2026, 11, 4))
        text = path.read_text(encoding="utf-8").replace("code: ''\n", "").replace("distinction: Oral\n", "")
        path.write_text(text, encoding="utf-8")
        review = index.read_review(path)
        self.assertEqual(date(2026, 11, 4), review.written)
        self.assertEqual("", review.distinction)

    def test_broken_yaml_and_empty_body(self):
        path = self.review()
        examples = [
            "# No metadata", "---\ntitle: unclosed", "---\n[]\n---\n# body",
            "---\ntitle: [\n---\n# body", "---\n- title\n---\n# body",
            "---\ntitle: a\ntitle: b\n---\n# body",
            "---\n1: bad key\n---\n# body", "---\n\n---\n# body",
            "---\nstatus: draft\n---\n\n",
            "---\na: !!python/object/apply:os.system [echo unsafe]\n---\n# body",
        ]
        for text in examples:
            with self.subTest(text=text):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises((index.IndexError, yaml.YAMLError)):
                    index.update(self.root)

    def test_duplicate_paper_and_alternative_arxiv_version(self):
        self.review()
        self.review("agents/copy.md", title="Same paper, another name", paper="https://arxiv.org/pdf/2601.12345v2.pdf")
        with self.assertRaisesRegex(index.IndexError, "duplicate paper"):
            index.update(self.root)

    def test_duplicate_title_across_different_urls_and_statuses(self):
        self.review()
        self.review("agents/copy.md", title="A MEMORY paper", paper="https://example.com/copy", status="draft")
        with self.assertRaisesRegex(index.IndexError, "duplicate paper"):
            index.update(self.root)

    def test_reports_all_bad_files_together(self):
        self.review("a.md", title="")
        self.review("b.md", categories=[])
        with self.assertRaises(index.IndexError) as error:
            index.update(self.root)
        self.assertIn("reviews/a.md", str(error.exception))
        self.assertIn("reviews/b.md", str(error.exception))

    def test_removing_or_unpublishing_review_removes_row(self):
        path = self.review()
        index.update(self.root)
        self.review(status="draft")
        index.update(self.root)
        self.assertIn("Total papers: 0", self.output())
        self.review()
        index.update(self.root)
        path.unlink()
        index.update(self.root)
        self.assertIn("Total papers: 0", self.output())

    def test_markdown_text_and_path_escaping(self):
        self.review("memory/한 글 (draft).md", title="A [mask] | <tag> *paper*", paper="https://example.com/paper_(1)")
        index.update(self.root)
        text = self.output()
        self.assertIn(r"A \[mask\] \| &lt;tag&gt; \*paper\*", text)
        self.assertIn("paper_%281%29", text)
        self.assertIn("%ED%95%9C%20%EA%B8%80%20%28draft%29.md", text)

    def test_missing_duplicate_reversed_or_inline_markers(self):
        for text in ["no markers", index.START, index.END + index.START,
                     index.START + index.START + index.END,
                     "prefix " + index.START + "\n" + index.END,
                     index.START + "\n" + index.END + "suffix"]:
            with self.subTest(text=text):
                self.readme.write_text(text, encoding="utf-8")
                with self.assertRaises(index.IndexError):
                    index.update(self.root)
                self.assertEqual(text, self.output())

    def test_missing_readme_or_reviews(self):
        self.readme.unlink()
        with self.assertRaises(FileNotFoundError):
            index.update(self.root)
        (self.root / "reviews").rmdir()
        with self.assertRaisesRegex(index.IndexError, "reviews/"):
            index.update(self.root)

    def test_bom_review_and_non_markdown_files(self):
        path = self.review("memory/paper.MD")
        path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
        (path.parent / ".gitkeep").touch()
        (path.parent / "notes.txt").write_text("not front matter", encoding="utf-8")
        index.update(self.root)
        self.assertIn("Total papers: 1", self.output())

    def test_cli_exit_codes(self):
        args = ["--root", str(self.root)]
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertEqual(1, index.main(args + ["--check"]))
            self.assertEqual(0, index.main(args))
            self.assertEqual(0, index.main(args + ["--check"]))
            self.review(status="wrong")
            self.assertEqual(2, index.main(args))


if __name__ == "__main__":
    unittest.main()
