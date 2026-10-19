
"""第71课：来源冲突、原文比对、PDF与OCR候选测试。"""

import tempfile
import unittest

from pathlib import Path

from app.knowledge.source_provenance import (
    SourceCatalog,
    verify_excerpt,
    suspected_ocr,
)


class ProvenanceTests(unittest.TestCase):

    def setUp(self):

        self.tmp = (
            tempfile.TemporaryDirectory()
        )

        self.root = (
            Path(self.tmp.name)
            / "books"
        )

        (self.root / "one").mkdir(
            parents=True
        )

        (self.root / "two").mkdir()

        self.quote = (
            "用户自发探索机器学习原理，"
            "而不是被动完成任务。"
        ) * 4

        (
            self.root / "one" / "兴趣.txt"
        ).write_text(
            self.quote,
            encoding="utf-8",
        )

        (
            self.root / "two" / "兴趣.txt"
        ).write_text(
            "不同教材" * 20,
            encoding="utf-8",
        )

        (
            self.root / "image.png"
        ).write_bytes(
            b"not used"
        )

        (
            self.root / "missing_source.txt"
        ).write_text(
            "另一本教材" * 8,
            encoding="utf-8",
        )

        self.catalog = SourceCatalog(
            self.root
        )

    def tearDown(self):

        self.tmp.cleanup()

    # ========================================
    # 1. 同名文件必须判定歧义
    # ========================================

    def test_duplicate_names_do_not_silently_map(
        self,
    ):

        status, path = (
            self.catalog.resolve(
                "兴趣.txt"
            )
        )

        self.assertEqual(
            status,
            "ambiguous_source",
        )

        self.assertIsNone(path)

        status, path = (
            self.catalog.resolve(
                "one/兴趣.txt"
            )
        )

        self.assertEqual(
            status,
            "resolved",
        )

        self.assertEqual(
            self.catalog.display(path),
            "one/兴趣.txt",
        )

    # ========================================
    # 2. 精确原文匹配
    # ========================================

    def test_match_full_original_excerpt(
        self,
    ):

        result = verify_excerpt(
            self.catalog,
            "one/兴趣.txt",
            self.quote,
        )

        self.assertEqual(
            result.status,
            "matched_original_text",
        )

        self.assertEqual(
            result.resolved_path,
            "one/兴趣.txt",
        )

    # ========================================
    # 3. 找不到文本不能当作已核验
    # ========================================

    def test_missing_quote_not_claimed_verified(
        self,
    ):

        result = verify_excerpt(
            self.catalog,
            "one/兴趣.txt",
            "完全不存在的实验结论" * 4,
        )

        self.assertEqual(
            result.status,
            "text_not_located",
        )

    # ========================================
    # 4. 图片需要OCR，当前不自动执行
    # ========================================

    def test_image_needs_ocr_but_no_ocr_runs(
        self,
    ):

        result = verify_excerpt(
            self.catalog,
            "image.png",
            self.quote,
        )

        self.assertEqual(
            result.status,
            "needs_ocr",
        )

        candidates = suspected_ocr(
            self.catalog,
            [self.root / "image.png"],
        )

        self.assertEqual(
            candidates[0]["reason"],
            "image_requires_ocr",
        )

    # ========================================
    # 5. 非法来源和过短片段
    # ========================================

    def test_bad_source_and_short_excerpt(
        self,
    ):

        self.assertEqual(
            verify_excerpt(
                self.catalog,
                "wrong.pdf",
                self.quote,
            ).status,
            "source_not_found",
        )

        self.assertEqual(
            verify_excerpt(
                self.catalog,
                "one/兴趣.txt",
                "短文本",
            ).status,
            "text_too_short",
        )

    # ========================================
    # 6. DOCX原文匹配
    # ========================================

    def test_docx_text_if_available(self):

        try:
            from docx import Document

        except ImportError:
            self.skipTest(
                "python-docx未安装"
            )

        path = (
            self.root / "word.docx"
        )

        doc = Document()

        doc.add_paragraph(
            self.quote
        )

        doc.save(path)

        catalog = SourceCatalog(
            self.root
        )

        self.assertEqual(
            verify_excerpt(
                catalog,
                "word.docx",
                self.quote,
            ).status,
            "matched_original_text",
        )

    # ========================================
    # 7. PDF正文与真实页码
    # ========================================

    def test_pdf_text_if_available(self):

        try:
            import fitz

        except ImportError:
            self.skipTest(
                "PyMuPDF未安装"
            )

        quote = (
            "A verified source paragraph "
            "describing active study and learning."
        )

        path = (
            self.root / "paper.pdf"
        )

        pdf = fitz.open()

        page = pdf.new_page()

        page.insert_text(
            (50, 70),
            quote,
        )

        pdf.save(
            str(path)
        )

        pdf.close()

        catalog = SourceCatalog(
            self.root
        )

        result = verify_excerpt(
            catalog,
            "paper.pdf",
            quote,
        )

        self.assertEqual(
            result.status,
            "matched_original_text",
        )

        self.assertEqual(
            result.located_at,
            "page=1",
        )

    # ========================================
    # 8. 单页扫描PDF不会索引越界
    # ========================================

    def test_blank_pdf_is_only_ocr_candidate(
        self,
    ):

        try:
            import fitz

        except ImportError:
            self.skipTest(
                "PyMuPDF未安装"
            )

        path = (
            self.root / "scan.pdf"
        )

        pdf = fitz.open()

        pdf.new_page()

        pdf.save(
            str(path)
        )

        pdf.close()

        catalog = SourceCatalog(
            self.root
        )

        self.assertEqual(
            verify_excerpt(
                catalog,
                "scan.pdf",
                self.quote,
            ).status,
            "needs_ocr",
        )

        items = suspected_ocr(
            catalog,
            [path],
        )

        self.assertEqual(
            items[0]["reason"],
            "sampled_pages_no_extractable_text",
        )

    # ========================================
    # 9. 缺少教材目录时报告错误
    # ========================================

    def test_missing_original_folder(self):

        with self.assertRaises(
            FileNotFoundError
        ):

            SourceCatalog(
                self.root / "notthere"
            )


if __name__ == "__main__":
    unittest.main()
