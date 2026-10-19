
from pathlib import Path

from app.ingestion.pdf_reader import read_pdf


# 定位项目根目录
ROOT = Path(__file__).resolve().parents[1]

# 查找 books 中的 PDF
pdf_files = sorted((ROOT / "books").glob("*.pdf"))

if not pdf_files:
    print("没有找到 PDF 教材")

else:
    # 先测试第一份 PDF
    pdf_path = ROOT / "books" / "【29行动力解析课】-音频.pdf"

    print("正在读取：", pdf_path.name)

    records = read_pdf(pdf_path)

    print("成功提取的页面数量：", len(records))

    # 查看前两个有文字的页面
    for record in records[:2]:
        print("\n------------------------")
        print("来源：", record.source)
        print("位置：", record.location)
        print("方式：", record.method)
        print("内容：")
        print(record.text[:300])
