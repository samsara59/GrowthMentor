
from dataclasses import dataclass, asdict
from typing import Literal


@dataclass(frozen=True)
class DocumentRecord:
    """统一的教材文本记录。"""

    source: str
    location: str
    text: str
    method: Literal["pdf_text", "docx_text", "ocr"]

    def __post_init__(self):
        if not self.text.strip():
            raise ValueError("教材文本不能为空")

    def to_dict(self):
        return asdict(self)
