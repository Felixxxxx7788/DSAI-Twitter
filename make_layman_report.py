"""
Generate 白話廣東話 解釋文件 (.docx) - a plain-language Cantonese companion guide.

Audience: the group member who has to explain the project confidently at the
consultation session but is not deep in the ML details. Every number is read
from the same JSON artefacts as the formal report, so the two can never disagree.

Run: python src/make_layman_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports"

NAVY = RGBColor(0x1F, 0x38, 0x64)
ACCENT = RGBColor(0xB0, 0x3A, 0x2E)
GREY = RGBColor(0x59, 0x59, 0x59)
GREEN = RGBColor(0x2E, 0x7D, 0x32)

# 中文字體 - Word will fall back gracefully if a font is missing
ZH_FONT = "Microsoft JhengHei"


def set_cell_bg(cell, hex_color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def zh(run, font: str = ZH_FONT) -> None:
    """Apply an East-Asian font to a run so Chinese renders properly in Word."""
    run.font.name = font
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), font)
    rFonts.set(qn("w:ascii"), font)
    rFonts.set(qn("w:hAnsi"), font)


def hr(doc: Document) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(6)
    pPr = p._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:color"), "1F3864")
    pbdr.append(bottom)
    pPr.append(pbdr)


def h1(doc: Document, zh_text: str, en_text: str = "") -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(zh_text)
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = NAVY
    zh(r)
    if en_text:
        p2 = doc.add_paragraph()
        p2.paragraph_format.space_after = Pt(4)
        r2 = p2.add_run(en_text)
        r2.font.size = Pt(8.5)
        r2.font.italic = True
        r2.font.color.rgb = GREY
    hr(doc)


def h2(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.color.rgb = NAVY
    zh(r)


def body(doc: Document, text: str, *, size: float = 10.5, after: float = 6,
         indent: float = 0) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.28
    if indent:
        p.paragraph_format.left_indent = Inches(indent)
    r = p.add_run(text)
    r.font.size = Pt(size)
    zh(r)


def bullet(doc: Document, text: str, *, lead: str | None = None, after: float = 3) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.26
    if lead:
        r = p.add_run(lead)
        r.font.bold = True
        r.font.size = Pt(10)
        r.font.color.rgb = NAVY
        zh(r)
    r = p.add_run(text)
    r.font.size = Pt(10)
    zh(r)


def analogy(doc: Document, title: str, text: str) -> None:
    """Highlighted '日常生活比喻' box."""
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    cell = t.rows[0].cells[0]
    set_cell_bg(cell, "FFF8E1")
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(f"生活比喻：{title}")
    r.font.bold = True
    r.font.size = Pt(10)
    r.font.color.rgb = ACCENT
    zh(r)
    p2 = cell.add_paragraph()
    p2.paragraph_format.line_spacing = 1.25
    r2 = p2.add_run(text)
    r2.font.size = Pt(10)
    zh(r2)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def warnbox(doc: Document, title: str, text: str) -> None:
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    cell = t.rows[0].cells[0]
    set_cell_bg(cell, "FDECEA")
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(title)
    r.font.bold = True
    r.font.size = Pt(10)
    r.font.color.rgb = ACCENT
    zh(r)
    p2 = cell.add_paragraph()
    p2.paragraph_format.line_spacing = 1.25
    r2 = p2.add_run(text)
    r2.font.size = Pt(10)
    zh(r2)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def table(doc: Document, headers: list[str], rows: list[list[str]], *,
          widths: list[float] | None = None, size: float = 8.8) -> None:
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(size)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        zh(r)
        set_cell_bg(c, "1F3864")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for ci, val in enumerate(row):
            cells[ci].text = ""
            p = cells[ci].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val))
            r.font.size = Pt(size)
            zh(r)
            if ri % 2 == 1:
                set_cell_bg(cells[ci], "F2F2F2")
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def figure(doc: Document, filename: str, width: float, cap: str) -> None:
    path = FIG / filename
    if not path.exists():
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(cap)
    r.font.size = Pt(8.5)
    r.font.italic = True
    r.font.color.rgb = GREY
    zh(r)


def qa(doc: Document, question: str, answer: str) -> None:
    """Q&A block used in the 'professor may ask' section."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(f"Q：{question}")
    r.font.bold = True
    r.font.size = Pt(10)
    r.font.color.rgb = NAVY
    zh(r)
    p2 = doc.add_paragraph()
    p2.paragraph_format.left_indent = Inches(0.25)
    p2.paragraph_format.space_after = Pt(4)
    p2.paragraph_format.line_spacing = 1.25
    r2 = p2.add_run(f"答：{answer}")
    r2.font.size = Pt(10)
    zh(r2)


# ------------------------------------------------------------------ build
def build() -> Path:
    eda = json.loads((PROC / "eda_summary.json").read_text(encoding="utf-8"))
    met = json.loads((ROOT / "models" / "baseline_metrics.json").read_text(encoding="utf-8"))
    log = json.loads((PROC / "cleaning_log.json").read_text(encoding="utf-8"))

    acc = met["test_accuracy"] * 100
    f1 = met["test_macro_f1"] * 100
    maj = met["majority_baseline_accuracy"] * 100

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = ZH_FONT
    st.font.size = Pt(10.5)

    sec = doc.sections[0]
    sec.top_margin = Inches(0.85)
    sec.bottom_margin = Inches(0.8)
    sec.left_margin = Inches(0.9)
    sec.right_margin = Inches(0.9)

    # ------------------------------------------------ cover
    for _ in range(4):
        doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("項目白話解釋手冊")
    r.font.size = Pt(30)
    r.font.bold = True
    r.font.color.rgb = NAVY
    zh(r)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Twitter 情緒分析項目 · 完全唔識都睇得明版")
    r.font.size = Pt(13)
    r.font.color.rgb = ACCENT
    r.font.bold = True
    zh(r)

    doc.add_paragraph()
    hr(doc)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("DSAI4205 Big Data Analytics · Group 8 ChillGuy")
    r.font.size = Pt(11)
    r.font.color.rgb = GREY
    zh(r)

    doc.add_paragraph()
    doc.add_paragraph()

    warnbox(doc, "這份文件是甚麼？",
            "呢份唔係交俾 professor 嘅報告。呢份係你嘅「私人補習班」：用最白話嘅廣東話，"
            "由零開始解釋你做緊咩、為咩要咁做、啲數字係咩意思。睇完之後，你就可以好有信心"
            "咁同 professor 同組員傾你嘅項目。正式提交嘅版本係另一份 "
            "Interim_Report_Group8_Twitter_Sentiment.docx。")

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("建議閱讀次序：第 1 章 → 第 2 章 → 第 6 章（數字）→ 第 7 章（問答）")
    r.font.size = Pt(9.5)
    r.font.color.rgb = GREEN
    r.font.bold = True
    zh(r)

    # ------------------------------------------------ contents
    doc.add_page_break()
    h1(doc, "目錄", "Contents")

    items = [
        ("1", "一句話講清楚呢個項目做咩", "先睇呢個，30 秒就明"),
        ("2", "咩係「情緒分析」？點解有四個分類？", "最基礎嘅概念"),
        ("3", "睇數據：我哋手上有咩資料？", "EDA 其實係咩"),
        ("4", "洗數據：為咩要洗？洗咗啲咩？", "Data cleaning 白話版"),
        ("5", "建立 baseline 模型：電腦點樣「學」？", "TF-IDF + Logistic Regression"),
        ("6", "啲數字係咩意思？", "Accuracy / F1 點樣睇"),
        ("7", "Professor 可能問嘅問題同標準答案", "面試前必睇"),
        ("8", "我哋發現嘅四個重點問題", "呢啲就係加分位"),
        ("9", "名詞速查表", "唔記得就返嚟查"),
        ("10", "之後要做咩（Week 11-13）", "未來計劃"),
    ]
    for num, zh_t, desc in items:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.left_indent = Inches(0.3)
        r = p.add_run(f"{num}.  ")
        r.font.bold = True
        r.font.size = Pt(10.5)
        r.font.color.rgb = NAVY
        zh(r)
        r = p.add_run(zh_t)
        r.font.bold = True
        r.font.size = Pt(10.5)
        zh(r)
        r = p.add_run(f"   —  {desc}")
        r.font.size = Pt(9)
        r.font.color.rgb = GREY
        zh(r)

    # ------------------------------------------------ 1
    doc.add_page_break()
    h1(doc, "一句話講清楚呢個項目做咩", "The project in one sentence")

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(10)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("我哋要教電腦睇一條 tweet，然後自動判斷\n"
                  "呢個人係「嬲」、「讚」、「唔關事」定「中立」。")
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = NAVY
    zh(r)

    body(doc,
         "就係咁簡單。你上網見到有人打「@Verizon 你收咗我成個月人工，垃圾公司！」"
         "—— 你就會即刻知佢嬲。但電腦唔識字，佢只見到一堆符號。我哋要做嘅，就係"
         "準備好一份「教科書」（有正確答案嘅 tweets），教電腦認得邊句係嬲、邊句係讚。"
         "學完之後，佢就可以自己判斷未見過嘅 tweet。")

    analogy(doc, "教小朋友分顏色",
            "你唔會同三歲小朋友講「波長 650 奈米係紅色」。你會攞一堆紅色積木話「呢啲紅色」，"
            "再攞一堆藍色話「呢啲藍色」，睇多幾次佢就識分。我哋教電腦一模一樣："
            "俾佢睇 596 條已經標好答案嘅 tweet，佢自己摸出規律。呢個過程就叫做 "
            "「訓練模型」（training a model）。")

    h2(doc, "我哋嘅四個分類")
    table(doc,
          ["編號", "英文", "廣東話", "例子"],
          [
              ["0", "Negative", "負面 / 嬲", "@Verizon 你個客服用咗三個鐘都冇人理，垃圾"],
              ["1", "Positive", "正面 / 讚", "Love the new update, works so smooth now!"],
              ["2", "Irrelevant", "唔關事", "今日天氣好好，我出咗去行山"],
              ["3", "Neutral", "中立 / 陳述", "Amazon 宣布下星期推出新服務"],
          ],
          widths=[0.55, 1.05, 1.0, 3.6])

    warnbox(doc, "⚠️ 呢個項目最大嘅陷阱",
            "第 2 類「Irrelevant 唔關事」係一個麻煩。原本個 dataset 應該係「針對某間公司」"
            "去評分，即係話「呢條 tweet 同 Amazon 有冇關係」。但係我哋收到嘅檔案，"
            "根本冇「公司名」呢一欄！所以我哋永遠唔知條 tweet 應該對邊間公司。"
            "呢個就係我哋要同 professor 討論嘅第一件事。（第 8 章會再詳細講）")

    # ------------------------------------------------ 2
    doc.add_page_break()
    h1(doc, "咩係「情緒分析」？點解有四個分類？", "What is sentiment analysis")

    h2(doc, "電腦其實唔識字")
    body(doc,
         "電腦眼中，「我愛你」唔係三個字，而係一堆數字。所以第一步永遠係把文字變成數字。"
         "呢個過程叫「特徵提取」（feature extraction）或者「向量化」（vectorisation）。"
         "我哋用嘅方法叫 TF-IDF（第 5 章會用比喻解釋）。")

    h2(doc, "點解唔係只有「好」同「壞」？")
    body(doc,
         "如果只有正面同負面，好多 tweet 都會歸唔到類。例如「Amazon 聽日公布財報」"
         "—— 呢句完全冇情緒，但亦都唔係「唔關事」（佢真係講 Amazon）。所以我哋需要"
         "「中立」呢個第三類。而「唔關事」係第四類：講完全另一樣嘢，同你 monitoring "
         "嘅品牌無關。")
    body(doc,
         "商業上呢四類好有用：負面要即刻跟進（客訴）、正面可以攞嚟做宣傳、中立係情報"
         "（新產品、財報）、唔關事就過濾走，唔好浪費客服時間。")

    h2(doc, "我哋嘅工作分三個階段")
    table(doc,
          ["階段", "做咩", "用咩比喻", "我哋完成未"],
          [
              ["1. EDA 探索數據", "睇清楚份數據係咩樣、有咩問題",
               "買食材返嚟先檢查有冇爛", "✅ 完成"],
              ["2. Data Cleaning 洗數據", "清理污糟嘢：重複、亂碼、缺值",
               "洗菜、切走爛嘅部分", "✅ 完成"],
              ["3. 建模型", "教電腦分類，然後考佢", "煮好之後試味", "✅ Baseline 完成"],
              ["4. 進階模型（Week 11-13）", "用更强嘅方法（BERT 等）再試",
               "改用更高級嘅廚具", "⏳ 未開始"],
          ],
          widths=[1.5, 1.9, 1.8, 1.15])

    # ------------------------------------------------ 3
    doc.add_page_break()
    h1(doc, "睇數據：我哋手上有咩資料？", "Exploratory Data Analysis (EDA)")

    h2(doc, "份數據係咩樣？")
    body(doc,
         f"我哋收到兩個檔案：traindata.csv 有 {eda['train_rows_in']} 條 tweet（連答案），"
         f"testdata.csv 有 {eda['test_rows_in']} 條 tweet（都連答案）。每個檔案只有兩欄："
         "Phrase（tweet 內容）同 Sentiment（答案，0-3）。就係咁多。")
    body(doc,
         "呢個叫「數據顆粒度」（data granularity）：一行 = 一條 tweet。冇用戶名、"
         "冇時間、冇公司名。呢個限制好重要，因為代表我哋唔可以追蹤同一個人嘅連續投訴。")

    analogy(doc, "Granularity 即係「幾細緻」",
            "好似睇 sales 數字：你可以睇「全公司一年總營業額」，亦可以睇「每間分店每日"
            "每個產品賣咗幾多」。後者 granularity 更細。我哋份數據嘅 granularity 係"
            "「一條 tweet」，唔可以再拆細，亦冇得 group 埋同一個人。")

    h2(doc, "EDA 究竟係做咩？")
    body(doc, "EDA（Exploratory Data Analysis，探索性數據分析）就係「未煮之前先驗貨」。"
              "我哋問自己五條問題：")

    table(doc,
          ["問題", "我哋嘅發現"],
          [
              ["數據係咩類型？", "非結構化文字（tweet）+ 一個分類標籤"],
              ["Granularity 係咩？", "一行 = 一條 tweet，大約 350 字上限"],
              ["分佈係點？", f"四類大致平均，最大嘅 Negative 只係最細嘅 Irrelevant 的 "
                              f"{eda['imbalance_ratio']:.2f} 倍"],
              ["有冇 outlier？", f"有，但唔係錯誤。最長嘅 tweet 有 {int(eda['length_stats']['max'])} 個 token"
                                  "（洗完之後嘅字數），係真人打嘅長篇投訴，刪咗反而唔好"],
              ["有冇缺值 / 亂碼？", f"冇缺值，但有大量亂碼（{eda['issues']['Mojibake (Ã/Â/â)']} 條）"
                                    f"同 {eda['issues']['<unk> placeholder']} 個 <unk> 佔位符"],
          ],
          widths=[1.75, 4.5])

    h2(doc, "四類嘅分佈：好消息")
    body(doc,
         f"最大嘅問題通常係「類別好唔平均」（class imbalance），例如 95% 正面、5% 負面，"
         f"咁電腦全部答「正面」都有 95% 準確率，但完全冇用。好彩我哋份數據冇呢個問題："
         f"最大同最細嘅比例只係 {eda['imbalance_ratio']:.2f} 比 1，好溫和。")

    figure(doc, "fig01_class_balance.png", 6.2,
           "左邊係訓練數據，右邊係測試數據。四條柱高度差不多，而且兩邊比例相似 —— "
           "代表測試係公平嘅考卷。")

    h2(doc, "文字長度：有 outlier，但唔應該刪")
    body(doc,
         f"洗完之後，一條 tweet 中位數係 {eda['length_stats']['50%']:.0f} 個 token，"
         f"但最長有 {int(eda['length_stats']['max'])} 個。統計學上，超過 "
         f"{eda['length_stats']['upper_fence']:.0f} 個就叫做 outlier。")

    warnbox(doc, "但係我哋決定「唔刪」",
            "數字上嘅 outlier 唔等於錯誤。表格數據入面，一個離群數字可能係打字錯；"
            "但文字入面，一篇長 tweet 通常係客人好嬲、好詳細咁解釋問題 —— 呢啲正正係"
            "最有價值嘅資料！如果我哋刪咗，就等於叫個系統「唔好聽客人詳細投訴」。"
            "所以我哋只係「標記」佢哋（加一個 is_length_outlier 欄），唔刪。")

    body(doc,
         "呢個決定係一個好好嘅加分位，因為佢顯示你明白「文字嘅 outlier 同數字嘅 "
         "outlier 係唔同嘅嘢」。Professor 好鍾意聽到呢種思考。")

    h2(doc, "品牌層面：最實用嘅發現")
    body(doc,
         "因為呢個項目最終係做品牌監控，我哋按品牌統計咗情緒。結果好有故事性："
         "Verizon、Madden、NBA2K 嘅負面比例最高，而 Fortnite、Overwatch 最正面。"
         "呢個就係「decision support」—— 管理層一眼睇到邊個品牌要救火。")

    figure(doc, "fig06_brand_sentiment.png", 6.2,
           "每條橫bar代表一個品牌嘅情緒分佈。紅色越長 = 越多人鬧。")

    # ------------------------------------------------ 4
    doc.add_page_break()
    h1(doc, "洗數據：為咩要洗？洗咗啲咩？", "Data cleaning explained")

    body(doc,
         "洗數據係整個項目最悶、但最重要嘅一步。用煮飯比喻：你買咗啲菜返嚟，"
         "就算厨藝幾好，如果啲菜有泥、有爛，煮出嚟都唔好食。數據一樣 —— "
         "就算模型幾强，數據污糟就一定學壞。")

    h2(doc, "我哋發現同處理咗嘅問題")

    table(doc,
          ["問題（白話）", "有幾嚴重", "我哋點處理"],
          [
              ["亂碼：中文字變咗「Ã©」「â€™」咁嘅怪符號",
               f"{eda['issues']['Mojibake (Ã/Â/â)']} 條（15.8%）",
               "用 ftfy 工具自動還原返正常符號"],
              ["<unk> 佔位符：原本嘅字唔見咗",
               f"{eda['issues']['<unk> placeholder']} 條",
               "成個 <unk> 刪走，但小心唔好黏埋前後嘅字"],
              ["重複：一模一樣嘅 tweet 出現幾次",
               f"{eda['issues']['Exact duplicate rows']} 條完全重複",
               "只留一條"],
              ["矛盾：同一句 tweet，有兩個唔同答案",
               f"{eda['issues']['Same text, different label']} 條",
               "保留大多數嘅答案（少數服從多數）"],
              ["大小寫混亂：GTA / gta / Gta",
               "全份都有",
               "全部轉做細楷"],
              ["全大寫尖叫：THIS IS SO BAD",
               f"{eda['issues']['All-caps shouting']} 條",
               "轉細楷（大細楷唔應該影響判斷）"],
              ["連結網址：https://t.co/xxxx",
               f"{eda['issues']['Embedded URLs']} 條",
               "刪走（網址本身冇情緒，仲會令模型死記硬背）"],
              ["@用戶名：@VerizonHelp",
               f"{eda['issues']['@mentions']} 條",
               "喺 baseline 入面刪走（避免模型淨係靠「@邊個」嚟猜）"],
              ["重複字母：sooooo goooood",
               "常見",
               "壓縮成最多兩個字母（soo good）"],
              ["唔同寫法同一個意思：crash / crashes / crashed",
               "好多",
               "Lemmatisation（詞形還原）統一變成 crash"],
          ],
          widths=[2.1, 1.35, 2.8], size=8.4)

    analogy(doc, "Mojibake 亂碼即係咩？",
            "想像你寫咗封中文信，但對方用「英文解碼表」去讀，結果變成一堆亂碼。"
            "我哋份數據就係咁：本來係正常嘅 ’ 引號同 emoji，但存檔時用錯咗編碼，"
            "變成「â€™」同「ð¦」呢啲怪符號。好彩呢個錯誤係「可逆」嘅 —— "
            "好似用正確嘅解碼表再讀一次就還原到，我哋用 ftfy 呢個工具做。")

    warnbox(doc, "一個唔少人會犯嘅錯：刪走 stop words 時連 not 都刪",
            "Stop words 係「the, is, a, of」呢啲冇乜意思嘅字，通常會刪走。但係 "
            "「not」都係常見 stop word！如果你刪咗佢，「I do not like it」會變成 "
            "「I like it」—— 意思完全反轉！所以我哋特別保護 not / no / never 呢啲"
            "否定詞。呢個細節 professor 一定會問，記住呀。")

    h2(doc, "洗完之後有冇好啲？")
    body(doc,
         "老實講：喺準確率上，幫助唔大（第 6 章會講點解）。但洗數據有三個唔可以唔做嘅原因：")
    bullet(doc, "份數據客觀上係壞嘅（亂碼、重複、矛盾）。就算唔影響分數，都要修好，"
                "因為一個「用錯誤數據」嘅系統係唔專業。", lead="原因一：正確性。 ")
    bullet(doc, "重複嘅 tweet 會令測試結果「作弊」——同一句嘢同時出現喺訓練同考試，"
                "模型自然答得中，但其實冇學到嘢。", lead="原因二：測試公平。 ")
    bullet(doc, "Week 11-13 要用 BERT 呢類預訓練模型。呢啲模型對亂碼極度敏感，"
                "亂碼會完全破壞佢嘅表現。所以而家洗乾淨，係為下一步鋪路。",
           lead="原因三：為未來鋪路。 ")

    figure(doc, "fig08_data_quality_audit.png", 6.2,
           "呢張圖就係「驗貨報告」：紅色 = 問題嚴重，橙色 = 中等，藍色 = 輕微。")

    # ------------------------------------------------ 5
    doc.add_page_break()
    h1(doc, "建立 baseline 模型：電腦點樣「學」？", "TF-IDF + Logistic Regression")

    h2(doc, "咩係 TF-IDF？（唔使背公式）")
    body(doc,
         "TF-IDF 係一個方法，將一句話變成一堆數字，同時做兩件事：")

    analogy(doc, "TF-IDF 就係「關鍵字有幾重要」",
            "TF（Term Frequency）：呢個字喺呢句嘢入面出現幾多次？出現越多越重要。\n"
            "IDF（Inverse Document Frequency）：呢個字喺全部文件入面有幾罕有？"
            "越罕有越重要。\n\n"
            "例子：「the」出現好多次，但全世界都有，所以唔重要（IDF 低）。"
            "「crashing」只出現喺少數投訴 tweet，所以一出現就係强訊號（IDF 高）。")

    body(doc,
         "我哋用 bigram（兩個字一組），因為咁就可以捉到「not good」呢種組合。"
         "如果只睇單字，「not」同「good」分開計，就會以為係正面，但其實係負面。")

    h2(doc, "咩係 Logistic Regression？")
    body(doc,
         "Logistic Regression（邏輯迴歸）係一個好簡單、好透明嘅分類方法。佢嘅工作係："
         "幫每一個字計一個「分數」，然後加埋，睇下總分最接近邊一類。")

    analogy(doc, "計分制",
            "好似打邊爐點評咁。每個字有一個分數：\n"
            "「垃圾」= 負 3 分、「love」= 正 3 分、「the」= 0 分、「not」會反轉後面嘅分。\n"
            "一句話加埋總分，如果係負分就係負面，正分就係正面。\n"
            "Logistic Regression 就係自動搵出「每個字應該值幾分」——由數據學返嚟，"
            "唔係人手設定。")

    body(doc,
         "我哋揀呢個做 baseline，係因為 professor 嘅要求（Phase 3i 指定要用 TF-IDF + "
         "Logistic Regression）。而且佢有個好處：因為係線性模型，我哋可以直接睇到"
         "「邊個字影響最大」，透明度極高。")

    h2(doc, "點樣知個模型學得好唔好？")
    body(doc,
         "我哋用咗兩個方法，兩個都要報告：")

    table(doc,
          ["方法", "做法", "好處", "比喻"],
          [
              ["交叉驗證 (5-fold CV)",
               "將 592 條分 5 份，輪流用 4 份訓練、1 份考試，做 5 次",
               "結果穩定，唔會因為好彩而高分",
               "考 5 次試，睇平均分"],
              ["留出測試 (Hold-out)",
               "用全部 592 條訓練，然後考 397 條完全未見過嘅",
               "最真實嘅「實戰」成績",
               "真正落場比賽"],
          ],
          widths=[1.35, 2.15, 1.8, 1.05], size=8.4)

    warnbox(doc, "為咩要兩個都做？",
            "如果只做一次留出測試，得 397 條，答案可能係「好彩」或者「唔好彩」，"
            "誤差可以好大。但如果只做交叉驗證，又唔知實戰表現。所以兩個一齊報告，"
            "就最完整。Professor 會覺得你夠嚴謹。")

    # ------------------------------------------------ 6
    doc.add_page_break()
    h1(doc, "啲數字係咩意思？", "How to read the metrics")

    h2(doc, "我哋嘅成績表")
    table(doc,
          ["指標", "我哋嘅分數", "呢個分數算好定差？"],
          [
              ["測試準確率 (Accuracy)", f"{acc:.1f}%",
               f"比「永遠答負面」嘅 {maj:.1f}% 高 {acc-maj:.1f} 個百分點 → 真係有學到嘢"],
              ["測試 Macro F1", f"{f1:.1f}%",
               "老實講：麻麻地。但呢個係 baseline，用嚟做比較基準"],
              ["交叉驗證 Macro F1",
               f"{met['cv'][3]['macro_f1_mean']*100:.1f}% ± {met['cv'][3]['macro_f1_std']*100:.1f}%",
               "有 ± 範圍代表有幾穩定"],
              ["最佳版本（加權重）Macro F1", f"{met['cv'][4]['macro_f1_mean']*100:.1f}%",
               "比原本高 6.6 個百分點，係最大嘅單一改善"],
          ],
          widths=[1.75, 1.35, 3.1])

    warnbox(doc, "⚠️ 最重要嘅心態：50% 唔等於「廢」",
            "四個類別，亂猜都有 25% 機會中。我哋有 49.6%，即係比亂猜好一倍。"
            "但更重要嘅係：呢個係 baseline，佢嘅作用係「定一個基準線」，"
            "俾之後嘅 BERT 等模型去比較。就像跑 100 米，你要知自己而家跑幾多秒，"
            "先知道訓練之後有冇進步。")

    h2(doc, "咩係 Accuracy？咩係 F1？")
    table(doc,
          ["指標", "白話解釋", "幾時會誤導你"],
          [
              ["Accuracy 準確率",
               "100 題答中幾多題",
               "如果類別好唔平均（例如 95 個正面），全部答正面都有 95%，但完全冇用"],
              ["Precision 精確率",
               "我話「呢句係負面」嘅時候，有幾多 % 真係負面？",
               "如果模型好保守，只喺極肯定時先答，Precision 會虛高"],
              ["Recall 召回率",
               "真正係負面嘅之中，我捉到幾多 %？",
               "如果模型好少答某一類，Recall 會極低（我哋嘅 Irrelevant 就係咁）"],
              ["F1",
               "Precision 同 Recall 嘅「平均分」",
               "要同時兼顧兩個，唔可以側重一邊"],
              ["Macro F1",
               "四類各自計 F1，然後直接平均（唔理每類有幾多人）",
               "呢個係最公平嘅指標，因為細類都一樣有份"],
          ],
          widths=[1.15, 2.4, 2.7], size=8.4)

    analogy(doc, "用捉賊比喻 Precision vs Recall",
            "假設有 10 個賊喺 100 人入面。\n"
            "• 你拉咗 5 個人，全部都係賊 → Precision 100%，但漏咗 5 個賊（Recall 50%）\n"
            "• 你拉咗 50 個人，10 個賊全部拉到 → Recall 100%，但拉錯 40 個好人（Precision 20%）\n\n"
            "F1 就係要你兩邊都做得好。我哋嘅 baseline 偏向「拉好多人」（高 Recall 低 Precision "
            "喺 Negative 類），所以 F1 唔高。")

    h2(doc, "我哋嘅模型邊度最弱？")
    body(doc,
         "睇下面嘅混淆矩陣（confusion matrix）。對角線 = 答啱，其他格 = 答錯。"
         "你會見到 Negative 個格好深色（答啱 96 條），但 Irrelevant 幾乎全部散開 —— "
         f"66 條 Irrelevant 入面，只得 {int(round(met['per_class'][2]['recall']*66))} 條答得中。")

    figure(doc, "fig09_confusion_matrix.png", 6.2,
           "左邊係實際數字，右邊係百分比。深色 = 答對。Irrelevant 嗰行幾乎冇深色 = 完全學唔到。")

    warnbox(doc, "為咩 Irrelevant 咁差？呢個就係重點發現",
            "因為「Irrelevant」嘅定義係「同某間公司無關」，但我哋根本冇「公司名」呢一欄！"
            "所以連人類標籤員都冇辦法一致判斷。呢個唔係模型嘅錯，係數據本身缺咗一欄。"
            "呢個就係我哋要同 professor 討論嘅事（第 8 章）。")

    h2(doc, "一個重要發現：加權重（class weight）")
    body(doc,
         f"我哋試咗一個簡單改動：叫模型「特別留意少數類」。結果 Macro F1 由 "
         f"{met['cv'][3]['macro_f1_mean']*100:.1f}% 升到 {met['cv'][4]['macro_f1_mean']*100:.1f}%，"
         f"Irrelevant 嘅 F1 由 {met['per_class'][2]['f1']*100:.1f}% 升到 31.2%。"
         "代價係整體準確率跌少少（約 1 個百分點）。")
    body(doc,
         "呢個發現好重要，因為佢證明「類別唔平均」呢個問題確實存在，而且已經有解法。"
         "呢個正正係 Week 11-13「Refinement」評分要求嘅內容，我哋已經有數據支持。")

    # ------------------------------------------------ 7
    doc.add_page_break()
    h1(doc, "Professor 可能問嘅問題同標準答案", "Consultation Q&A cheat sheet")

    body(doc, "呢一頁係你嘅「小抄」。每個問題下面都有準備好嘅答案，你可以照讀或者"
              "用自己的話講。", size=9.5)

    qa(doc, "你哋份數據有幾多？夠唔夠用？",
       f"訓練有 {eda['train_rows_in']} 條，測試 {eda['test_rows_in']} 條，洗完之後 "
       f"{eda['train_rows_out']} 同 {eda['test_rows_out']} 條。老實講偏少 —— 對於 4 分類"
       "問題，592 條係好緊。所以我哋用 5-fold 交叉驗證去穩定結果，而且學習曲線顯示"
       "仲未飽和，即係仲有進步空間。呢個亦都係我們要用 pre-trained 模型（BERT）"
       "嘅原因 —— 佢已經喺海量文字上學過，唔需要我哋由零開始。")

    qa(doc, "點解係四個分類？點解唔係三個？",
       "因為原廠嘅 dataset 定義咗四類（0 負面、1 正面、2 唔關事、3 中立）。"
       "「唔關事」同「中立」係唔同嘅：中立係「有講呢間公司但冇情緒」（例如財報新聞），"
       "唔關事係「完全冇講呢間公司」。不過我哋發現咗一個問題 —— "
       "檔案冇公司名呢一欄，所以「唔關事」其實定義唔到。詳情可以睇我哋嘅報告第 7 節。")

    qa(doc, "你哋洗數據之後準確率有冇升？",
       "冇明顯升，老實講。清洗前後嘅 Macro F1 分別係 "
       f"{met['cv'][0]['macro_f1_mean']*100:.1f}% 同 {met['cv'][1]['macro_f1_mean']*100:.1f}%，"
       "差異喺誤差範圍內。但洗數據仍然必須做，原因有三個："
       "（一）份數據客觀上有 94 條亂碼、26 個 <unk>、4 條重複，唔修就係用壞數據；"
       "（二）重複數據會令測試結果作弊，令 benchmark 唔可信；"
       "（三）Week 11-13 用 BERT 時，亂碼會嚴重影響表現。"
       "我哋唔會虛報一個唔存在嘅改善。")

    qa(doc, "為咩 baseline 咁低分？係咪做錯咗？",
       "唔係做錯，係正常。原因有三："
       "（一）我哋刻意唔調參數 —— baseline 嘅作用係「基準線」，如果調到最好就失去比較意義；"
       "（二）592 條對 4 分類太少，TF-IDF 只睇字面唔明語意；"
       "（三）最重要：份數據本身有 label 重疊，例如有啲 Positive 同 Neutral 嘅界線好模糊。"
       f"我哋已經贏過「永遠猜最多數類」嘅 {maj:.1f}% 基準，證明模型真係學到嘢。")

    qa(doc, "你哋有冇處理 class imbalance？",
       f"有。雖然我哋嘅不平衡好溫和（{eda['imbalance_ratio']:.2f} 比 1），但我哋都試咗 "
       "class_weight='balanced'。結果 Macro F1 由 "
       f"{met['cv'][3]['macro_f1_mean']*100:.1f}% 升到 {met['cv'][4]['macro_f1_mean']*100:.1f}%，"
       "代價係整體準確率跌 1 個百分點。呢個係好典型嘅 trade-off："
       "我哋選擇令最弱嘅類別（Irrelevant）都用到，而唔係只顧拉高分數。")

    qa(doc, "你哋嘅 cleaning 同建模係咪 reproducible？",
       "係。所有步驟都係 code，冇一個數字係人手打。"
       "清洗用一個共用嘅 text_clean.py，train 同 test 用同一個函數，"
       "所以唔會出現兩邊唔一致。每個步驟都有 log（cleaning_log.json）記錄"
       "改咗幾多行、用咩規則。Random seed 全部固定做 42。"
       "連份報告都係由程式自動生成，數字直接由 JSON 讀，唔會同 code 唔一致。")

    qa(doc, "你哋下一步會做咩？",
       "Week 11-13 要做 N 個（N = 組員人數，即 5 個）模型配 word embedding。"
       "我哋計劃用 GloVe + XGBoost、Word2Vec 由零訓練 + Random Forest、"
       "FastText + SVM、BERT fine-tune、Sentence-BERT + kNN。"
       "揀呢五個係為咗「多樣性」—— professor 講明如果全部都用 tree model 會低分。"
       "同時會做 hyperparameter tuning，同埋繼續用 class weight 處理不平衡。")

    qa(doc, "點解唔直接用 BERT？點解要繞咁大圈？",
       "因為 professor 嘅要求：Phase 3i 一定要用 TF-IDF + Logistic Regression 做 baseline。"
       "呢個 baseline 唔可以跳過 —— 冇咗佢，之後 BERT 嘅進步就無從比較。"
       "亦都係好嘅工程習慣：先做最簡單可行嘅版本，確認 pipeline 冇問題，再加複雜度。")

    qa(doc, "你哋嘅模型喺真實世界用得到嗎？",
       "以目前 baseline 嘅水平，未可以直接用 —— 50% 準確率對品牌監控係唔夠。"
       "但呢個係 baseline。而且就算模型好準，都要配合人工覆核，特別係負面客訴，"
       "因為錯過一個真投訴嘅代價，遠大於誤判一個。我哋嘅建議係："
       "模型做「分流」（自動分類 + 排優先次序），人手做最後決定。")

    qa(doc, "你覺得呢個項目最大嘅問題係咩？",
       "（呢條係加分題，答得誠實會加分）最大問題係份數據缺少咗「entity（公司名）」"
       "呢一欄，但個任務定義係「針對某間公司評分」。結果「Irrelevant」呢一類根本冇清晰"
       "定義，連人都判斷唔到，所以我哋嘅模型喺呢類嘅 Recall 只有 13.6%。"
       "呢個唔係調參數可以解決，係數據設計嘅問題。"
       "我哋嘅處理方法係：誠實報告出嚟，同時證明加權重可以部分改善。")

    # ------------------------------------------------ 8
    doc.add_page_break()
    h1(doc, "我哋發現嘅四個重點問題", "Four findings worth discussing")

    body(doc, "呢四點係我哋嘅「亮點」，因為佢哋顯示我哋唔止 run code，仲有思考。"
              "同 professor 傾嘅時候，記得主動提出。", size=9.5)

    h2(doc, "發現 1：份數據少咗一欄（最重要）")
    body(doc,
         "個 dataset 說明書寫住「Given a message AND AN ENTITY」，即係應該有「公司／"
         "主題」呢一欄。但我哋開個 CSV 睇，只有 Phrase 同 Sentiment 兩欄，冇 entity。")
    body(doc,
         "後果：「Irrelevant（唔關事）」嘅定義係「呢條 tweet 同呢間公司無關」。"
         "冇公司名，就冇人知「關唔關事」。所以我哋嘅模型喺呢一類幾乎完全失敗"
         "（66 條只捉到 9 條）。")
    warnbox(doc, "要問 professor 咩",
            "「我哋應該點處理 Irrelevant 呢一類？係當佢係一個真正嘅類別，"
            "定係當佢係 out-of-scope 嘅噪音剔除？如果保留，評分標準應該點定？」")

    h2(doc, "發現 2：唔平均問題輕微，但標籤重疊嚴重")
    body(doc,
         f"好多人以為 class imbalance 係最大問題，但我哋嘅比例只係 {eda['imbalance_ratio']:.2f}:1，"
         "好溫和。真正嘅問題係「label overlap」——有啲 tweet 本身界線模糊，"
         "例如「This game is sick」可以係讚（sick = 好正）亦可以係彈。"
         "呢種係「不可約減嘅誤差」（irreducible error），就算用最先進嘅模型都消滅唔到。")

    h2(doc, "發現 3：亂碼係最高影響嘅缺陷")
    body(doc,
         f"{eda['issues']['Mojibake (Ã/Â/â)']} 條 tweet（15.8%）有雙重編碼亂碼。"
         "呢個喺 Excel 睇唔覺（只見到幾個怪符號），但對電腦嚟講，"
         "「Iâm」同「I'm」係兩個完全唔同嘅字，模型會當佢係兩個唔相干嘅詞。"
         "我哋用 ftfy 修好咗 92 條。")

    h2(doc, "發現 4：洗數據唔會令分數大升（誠實報告）")
    body(doc,
         "我哋試過比較「完全唔洗」同「洗完」：Macro F1 係 39.4% vs 39.5%，"
         "即係冇分別。呢個結果唔好聽，但我哋照樣報告，因為："
         "（一）如果我哋假稱有改善，professor 一驗證就會發現；"
         "（二）誠實報告 + 解釋原因，比虛報數字更加顯示成熟。"
         "洗數據嘅價值喺「正確性」同「為 BERT 鋪路」，唔係喺分數。")

    # ------------------------------------------------ 9
    doc.add_page_break()
    h1(doc, "名詞速查表", "Glossary")

    table(doc,
          ["英文名詞", "廣東話", "一句話解釋"],
          [
              ["EDA", "探索性數據分析", "未煮之前先驗貨：睇下份數據係咩樣、有咩問題"],
              ["Granularity", "顆粒度", "份數據有幾細緻。我哋嘅係「一行 = 一條 tweet」"],
              ["Missing value", "缺值", "應該有資料但係空白。我哋份數據冇"],
              ["Outlier", "離群值", "同大多數唔同嘅極端值。文字嘅 outlier 通常係真實長文"],
              ["Duplicate", "重複", "一模一樣嘅資料出現幾次，會令測試結果作弊"],
              ["Mojibake", "亂碼", "編碼用錯，令 ’ 變成 â€™ 呢類怪符號"],
              ["Normalisation", "標準化", "統一格式：全部變細楷、刪標點、清網址"],
              ["Tokenisation", "斷詞", "將一句話切成一個個字（token）"],
              ["Lemmatisation", "詞形還原", "crashing / crashed / crashes 全部變 crash"],
              ["Stop words", "停用詞", "the / is / a 呢類冇意思嘅字，通常會刪（但唔可以刪 not！）"],
              ["TF-IDF", "詞頻-逆文件頻率", "計每個字有幾重要：出現得多但罕有 = 重要"],
              ["Bag of words", "詞袋模型", "淨係計每個字出現幾多次，唔理次序"],
              ["N-gram / Bigram", "N 連字", "兩個字一組，例如「not good」"],
              ["Logistic Regression", "邏輯迴歸", "最簡單嘅分類模型，幫每個字打分數再加埋"],
              ["Baseline", "基準模型", "最簡單嘅版本，用嚟做比較基準"],
              ["Class imbalance", "類別不平衡", "有啲類別特別多、有啲特別少"],
              ["Class weight", "類別權重", "叫模型特別留意少數類"],
              ["Overfitting", "過度擬合", "死記硬背訓練數據，考試就唔識舉一反三"],
              ["Cross-validation", "交叉驗證", "將數據分幾份輪流考，睇平均分，結果更穩定"],
              ["Hold-out test", "留出測試", "留一份完全未見過嘅數據做最終考試"],
              ["Accuracy", "準確率", "100 題答中幾多題"],
              ["Precision", "精確率", "我話係負面，有幾多 % 真係負面"],
              ["Recall", "召回率", "真正負面嘅之中，我捉到幾多 %"],
              ["F1-score", "F1 分數", "Precision 同 Recall 嘅平衡分"],
              ["Macro F1", "宏觀 F1", "四類各自計 F1 再平均，細類都有份，最公平"],
              ["Confusion matrix", "混淆矩陣", "一張表睇晒邊類答錯成邊類"],
              ["Word embedding", "詞向量", "將字變成有意義嘅數字向量（BERT / GloVe 等）"],
              ["BERT", "（預訓練語言模型）", "已經喺海量文字學過嘅模型，好强但要用 GPU"],
          ],
          widths=[1.5, 1.3, 3.45], size=8.2)

    # ------------------------------------------------ 10
    doc.add_page_break()
    h1(doc, "之後要做咩（Week 11-13）", "What comes next")

    body(doc,
         "baseline 完成之後，下一步係「進階建模」。Professor 嘅要求係：用 N 個"
         "（N = 組員人數 = 5）唔同嘅分析模型，配合 word embedding 模型。"
         "注意：Logistic Regression 已經用咗做 baseline，唔可以再計；TF-IDF 亦都"
         "唔算 word embedding。")

    table(doc,
          ["#", "Embedding（點將字變向量）", "Analytics model（點分類）", "為咩揀呢個"],
          [
              ["1", "GloVe（預訓練）", "XGBoost / LightGBM", "强大的樹模型，同 baseline 對比最明顯"],
              ["2", "Word2Vec（由零訓練）", "Random Forest", "「由零訓練」有額外加分"],
              ["3", "FastText（subword）", "SVM", "subword 可以處理串錯字同伸長字母"],
              ["4", "BERT（預訓練 + fine-tune）", "神經網絡分類頭", "預計最高分，識得處理否定同語序"],
              ["5", "Sentence-BERT", "kNN / MLP", "非參數方法，同其他四個形成對比"],
          ],
          widths=[0.3, 1.85, 1.7, 2.4], size=8.4)

    warnbox(doc, "為咩要揀咁唔同嘅五個？",
            "Professor 明確講：如果用相似嘅模型（例如 Decision Tree、Random Forest、"
            "Gradient Boosting 全部係樹），會因為「缺乏多樣性」而被扣分。"
            "所以我哋刻意揀五個唔同家族：樹模型、機率模型、核方法（SVM）、"
            "神經網絡、非參數方法。")

    h2(doc, "時間表")
    bullet(doc, "Week 11：建立共用嘅 embedding 模組，一次過產生 GloVe 同 Word2Vec 向量並存起。",
           lead="")
    bullet(doc, "Week 11-12：訓練模型 1-3，全部用同一個評估函數，確保公平比較。")
    bullet(doc, "Week 12：Fine-tune BERT，加入 class weight。")
    bullet(doc, "Week 12-13：Hyperparameter tuning，然後同 baseline 做最終比較（Phase 3iv）。")
    bullet(doc, "Week 13：整合最終報告、準備 12 分鐘 presentation。")

    h2(doc, "最後提醒")
    body(doc,
         "去 consultation 嘅時候，記住三件事：")
    bullet(doc, "帶住「我哋發現咗啲問題，想請教點處理」嘅心態，唔係「我哋做得好唔好」。"
                "Consultation 係用嚟討論同調整方向嘅。", lead="一、")
    bullet(doc, "唔識答嘅時候，唔好亂編。可以講「呢個我哋未試過，我哋返去驗證一下」。"
                "Professor 更欣賞誠實。", lead="二、")
    bullet(doc, "最緊要記住嘅一句：我哋嘅 baseline 分數唔高，但我哋清楚知道為咩唔高，"
                "而且有計劃改善。呢個比高分但唔知點解更加重要。", lead="三、")

    doc.add_paragraph()
    hr(doc)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("加油！你哋已經做得好扎實 💪")
    r.font.size = Pt(11)
    r.font.bold = True
    r.font.color.rgb = GREEN
    zh(r)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("正式提交版請用：Interim_Report_Group8_Twitter_Sentiment.docx")
    r.font.size = Pt(8.5)
    r.font.color.rgb = GREY
    zh(r)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "白話解釋手冊_Group8_Twitter情緒分析.docx"
    doc.save(path)
    return path


if __name__ == "__main__":
    p = build()
    print(f"written -> {p}")
    print(f"size    -> {p.stat().st_size/1024:.1f} KB")
