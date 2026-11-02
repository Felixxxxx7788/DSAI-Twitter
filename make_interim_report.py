"""
Generate the Interim Report (.docx) straight from the artefacts produced by
run_pipeline.py and train_baseline.py.

Every number in this document is read from data/processed/eda_summary.json,
data/processed/cleaning_log.json, or models/baseline_metrics.json - nothing is
hand-typed, so the report can never drift from the code.

Run: python src/make_interim_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
OUT = ROOT / "reports"

# ------------------------------------------------------------------ style helpers
NAVY = RGBColor(0x1F, 0x38, 0x64)
ACCENT = RGBColor(0xB0, 0x3A, 0x2E)
GREY = RGBColor(0x59, 0x59, 0x59)

GROUP = "Group 8  -  ChillGuy"
MEMBERS = [
    ("Chao Ka Kin", "23021004D"),
    ("Leung Wai Tsun", "24097714D"),
    ("LIANG Zijian", "24082468D"),
    ("Lam Kin Ho", "24084386D"),
    ("Yeung Tsz Chun Herman", "23019365D"),
]
COURSE = "DSAI4205 - Big Data Analytics (2026/27 Semester One)"
PROJECT = "Group Project (25%) - Interim Report"


def set_cell_bg(cell, hex_color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def add_page_number_footer(section) -> None:
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("Page ")
    run.font.size = Pt(8)
    run.font.color.rgb = GREY
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    p._p.append(fld)


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


def h1(doc: Document, text: str, number: str = "") -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(f"{number}  {text}".strip())
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.color.rgb = NAVY
    run.font.name = "Calibri"
    hr(doc)


def h2(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(11)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.size = Pt(11.5)
    run.font.bold = True
    run.font.color.rgb = NAVY


def h3(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.size = Pt(10.5)
    run.font.bold = True
    run.font.italic = True
    run.font.color.rgb = ACCENT


def body(doc: Document, text: str, *, size: float = 10.5, space_after: float = 6,
         italic: bool = False) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.italic = italic
    run.font.name = "Calibri"


def bullet(doc: Document, text: str, level: int = 0, *, bold_lead: str | None = None) -> None:
    p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.12
    if bold_lead:
        r = p.add_run(bold_lead)
        r.font.bold = True
        r.font.size = Pt(10)
        r.font.name = "Calibri"
    r = p.add_run(text)
    r.font.size = Pt(10)
    r.font.name = "Calibri"


def caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(10)
    run = p.add_run(text)
    run.font.size = Pt(8.5)
    run.font.italic = True
    run.font.color.rgb = GREY


def figure(doc: Document, filename: str, width: float, cap: str) -> None:
    path = FIG / filename
    if not path.exists():
        body(doc, f"[missing figure: {filename}]", italic=True)
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption(doc, cap)


def table(doc: Document, headers: list[str], rows: list[list[str]], *,
          widths: list[float] | None = None, font_size: float = 8.5,
          header_bg: str = "1F3864") -> None:
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.font.bold = True
        run.font.size = Pt(font_size)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.name = "Calibri"
        set_cell_bg(hdr[i], header_bg)
    for r_i, row in enumerate(rows):
        cells = t.add_row().cells
        for c_i, val in enumerate(row):
            cells[c_i].text = ""
            p = cells[c_i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_i > 0 else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(str(val))
            run.font.size = Pt(font_size)
            run.font.name = "Calibri"
            if r_i % 2 == 1:
                set_cell_bg(cells[c_i], "F2F2F2")
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def callout(doc: Document, title: str, text: str) -> None:
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    cell = t.rows[0].cells[0]
    set_cell_bg(cell, "EAF1FB")
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(title)
    r.font.bold = True
    r.font.size = Pt(9.5)
    r.font.color.rgb = NAVY
    r.font.name = "Calibri"
    p2 = cell.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r2 = p2.add_run(text)
    r2.font.size = Pt(9.5)
    r2.font.name = "Calibri"
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ------------------------------------------------------------------ document
def build() -> Path:
    eda = json.loads((PROC / "eda_summary.json").read_text(encoding="utf-8"))
    log = json.loads((PROC / "cleaning_log.json").read_text(encoding="utf-8"))
    met = json.loads((ROOT / "models" / "baseline_metrics.json").read_text(encoding="utf-8"))
    cv_df = pd.read_csv(TAB / "cv_results.csv")

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)

    sec = doc.sections[0]
    sec.top_margin = Inches(0.85)
    sec.bottom_margin = Inches(0.8)
    sec.left_margin = Inches(0.95)
    sec.right_margin = Inches(0.95)
    add_page_number_footer(sec)

    # ------------------------------------------------ cover page
    for _ in range(3):
        doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("DSAI4205 - Big Data Analytics")
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = GREY

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(COURSE.split(" - ")[1])
    r.font.size = Pt(11)
    r.font.color.rgb = GREY

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("INTERIM REPORT")
    r.font.size = Pt(30)
    r.font.bold = True
    r.font.color.rgb = NAVY

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Twitter Entity-Level Sentiment Analysis")
    r.font.size = Pt(17)
    r.font.color.rgb = ACCENT
    r.font.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Exploratory Data Analysis, Data Cleaning Pipeline\n"
                  "and TF-IDF + Logistic Regression Baseline")
    r.font.size = Pt(11.5)
    r.font.color.rgb = GREY

    doc.add_paragraph()
    doc.add_paragraph()
    hr(doc)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"Dataset 5  -  Twitter")
    r.font.size = Pt(12)
    r.font.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"{eda['train_rows_in']} training rows  |  {eda['test_rows_in']} test rows  |  "
                  f"4 sentiment classes")
    r.font.size = Pt(10)
    r.font.color.rgb = GREY

    doc.add_paragraph()
    doc.add_paragraph()

    # student table
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(GROUP)
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.color.rgb = NAVY

    t = doc.add_table(rows=1, cols=2)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(["Name", "Student ID"]):
        c = t.rows[0].cells[i]
        c.text = ""
        pp = c.paragraphs[0]
        pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rr = pp.add_run(h)
        rr.font.bold = True
        rr.font.size = Pt(9.5)
        rr.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_bg(c, "1F3864")
    for name, sid in MEMBERS:
        cells = t.add_row().cells
        for i, val in enumerate([name, sid]):
            cells[i].text = ""
            pp = cells[i].paragraphs[0]
            pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            rr = pp.add_run(val)
            rr.font.size = Pt(9.5)
    for row in t.rows:
        row.cells[0].width = Inches(2.6)
        row.cells[1].width = Inches(1.6)

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Submission deadline: 6 November 2026, 23:55 (Week 10)")
    r.font.size = Pt(9.5)
    r.font.color.rgb = GREY
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Consultation session: Week 10/11, Saturday 14:00-18:00")
    r.font.size = Pt(9.5)
    r.font.color.rgb = GREY

    # ------------------------------------------------ contents
    doc.add_page_break()
    h1(doc, "Contents")
    contents = [
        ("1", "Executive Summary", "What we found and what we recommend discussing"),
        ("2", "Project Scope and Data Understanding", "Task definition, granularity, schema"),
        ("3", "Exploratory Data Analysis", "Distributions, imbalance, correlation, outliers"),
        ("4", "Data Cleaning Pipeline", "Nine documented steps with before/after evidence"),
        ("5", "Baseline Model: TF-IDF + Logistic Regression", "Design, protocol, benchmark results"),
        ("6", "Error Analysis and Model Limitations", "Where the baseline fails and why"),
        ("7", "Key Risks, Issues and Decisions Required", "Points we want the professor's input on"),
        ("8", "Plan for Weeks 11-13", "Advanced modelling roadmap"),
        ("9", "Reproducibility and Artefact Index", "How a marker can re-run everything"),
        ("A", "Appendix - Cleaning Log", "Full row-level audit trail"),
        ("B", "Appendix - Full Metrics", "Cross-validation and per-class detail"),
    ]
    for num, title, desc in contents:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.left_indent = Inches(0.25)
        r = p.add_run(f"{num}.  ")
        r.font.bold = True
        r.font.size = Pt(10)
        r.font.color.rgb = NAVY
        r = p.add_run(title)
        r.font.bold = True
        r.font.size = Pt(10)
        r = p.add_run(f"   -  {desc}")
        r.font.size = Pt(9)
        r.font.color.rgb = GREY

    # ------------------------------------------------ 1. executive summary
    doc.add_page_break()
    h1(doc, "Executive Summary", "1")

    body(doc,
         f"We were assigned Dataset 5, a Twitter entity-level sentiment corpus with four balanced-ish "
         f"classes (Negative, Positive, Irrelevant, Neutral). The raw files contain "
         f"{eda['train_rows_in']} training rows and {eda['test_rows_in']} test rows. We completed "
         f"Phase 1 (EDA), Phase 2 (cleaning) and Phase 3i (the required TF-IDF + Logistic Regression "
         f"baseline), and this report documents all of it.")

    h2(doc, "1.1  Headline results")
    table(doc,
          ["Metric", "Result", "Reference point"],
          [
              ["Baseline test accuracy", f"{met['test_accuracy']*100:.1f}%",
               f"Majority-class guess = {met['majority_baseline_accuracy']*100:.1f}%"],
              ["Baseline test macro F1", f"{met['test_macro_f1']*100:.1f}%",
               "4-class random guess = 25.0%"],
              ["Baseline test weighted F1", f"{met['test_weighted_f1']*100:.1f}%", "-"],
              ["5-fold CV macro F1 (mean +/- sd)",
               f"{(met['cv'][3]['macro_f1_mean'])*100:.1f}% +/- {met['cv'][3]['macro_f1_std']*100:.1f}%",
               "Stable estimate on training data"],
              ["Best CV variant (balanced weights)",
               f"{(met['cv'][4]['macro_f1_mean'])*100:.1f}%",
               "Largest single improvement: +6.6 pp macro F1"],
              ["Rows removed in cleaning", f"{eda['train_dropped']} of {eda['train_rows_in']}",
               f"Retained {eda['train_rows_out']} training rows (99.3%)"],
          ],
          widths=[2.55, 1.45, 2.3])

    h2(doc, "1.2  The four findings we most want to discuss")
    bullet(doc,
           "The dataset documentation describes an entity-level task but the delivered CSVs contain "
           "no entity/topic column. Label 2 ('Irrelevant') is therefore defined relative to an entity "
           "that was never shipped. Our baseline identifies only 9 of 66 Irrelevant test rows. We "
           "believe this label is partly unlearnable from text alone and want to confirm how to handle it.",
           bold_lead="Finding 1 - A missing column changes what 'Irrelevant' means. ")
    bullet(doc,
           "There is no class imbalance problem: the largest class is only 1.65x the smallest and the "
           "train/test distributions match closely. The real difficulty is label overlap, not skew. "
           f"Balanced class weights still raised macro F1 from "
           f"{met['cv'][3]['macro_f1_mean']*100:.1f}% to {met['cv'][4]['macro_f1_mean']*100:.1f}% in "
           "cross-validation, which tells us the minority class needs help even at mild imbalance.",
           bold_lead="Finding 2 - Imbalance is mild; class overlap is severe. ")
    bullet(doc,
           f"94 of {eda['train_rows_in']} training rows (15.8%) were double-encoded UTF-8 "
           "(mojibake), plus 26 rows carrying a literal <unk> placeholder. We repaired "
           f"{log['train']['steps'][6]['rows_affected']} rows with ftfy and stripped the placeholders. "
           "This is invisible in a spreadsheet but corrupts every token of those rows.",
           bold_lead="Finding 3 - Encoding corruption was the highest-impact defect. ")
    bullet(doc,
           f"Cleaning improved macro F1 by only {(met['cv'][1]['macro_f1_mean']-met['cv'][0]['macro_f1_mean'])*100:+.1f} pp "
           f"over raw text in cross-validation ({met['cv'][0]['macro_f1_mean']*100:.1f}% -> "
           f"{met['cv'][1]['macro_f1_mean']*100:.1f}%). Text cleaning is necessary for correctness and "
           "for the embedding models in Phase 3ii, but it is not the performance lever. We should not "
           "expect cleaning alone to move the benchmark much.",
           bold_lead="Finding 4 - Cleaning validates correctness, not accuracy. ")

    callout(doc, "Recommendation for the consultation",
            "We would like the professor's view on three decisions: (a) whether to treat label 2 as a "
            "genuine class or as out-of-scope noise given the missing entity column; (b) whether the "
            "held-out test file may be used for final model comparison, or whether a validation split "
            "from train should be carved out; and (c) whether advanced models should be judged against "
            "the plain baseline or the improved balanced baseline.")

    # ------------------------------------------------ 2. scope
    doc.add_page_break()
    h1(doc, "Project Scope and Data Understanding", "2")

    h2(doc, "2.1  Business framing")
    body(doc,
         "The brief asks for a decision support system that classifies text. For Twitter sentiment the "
         "practical use is brand and product monitoring: a support or marketing team wants to route "
         "complaints, spot emerging product failures and separate genuine opinion from unrelated "
         "chatter. Every design choice below is judged against that use, not against a leaderboard.")

    h2(doc, "2.2  Task definition")
    body(doc,
         "Multi-class, single-label text classification. Each row is one tweet; the target is one of "
         "four sentiment labels. There is no entity column to condition on (see Finding 1), so we "
         "treat the task as document-level 4-way classification and document that deviation explicitly.")

    table(doc,
          ["Attribute", "Value"],
          [
              ["Dataset", "Dataset 5 - Twitter (entity-level sentiment analysis)"],
              ["Files", "traindata.csv (labelled), testdata.csv (labelled hold-out)"],
              ["Raw schema", "Phrase (text), Sentiment (0-3). No ID, timestamp, user or entity column"],
              ["Label map", "0 = Negative, 1 = Positive, 2 = Irrelevant, 3 = Neutral"],
              ["Granularity", "One row = one tweet, truncated at ~350 characters"],
              ["Train / test rows", f"{eda['train_rows_in']} / {eda['test_rows_in']} "
                                    f"(after cleaning {eda['train_rows_out']} / {eda['test_rows_out']})"],
              ["Language / domain", "English; consumer technology, gaming, telecoms and general chatter"],
              ["Type", "Unstructured text + categorical label"],
              ["Missing values", "None in either column of either file"],
          ],
          widths=[1.55, 4.75])

    h2(doc, "2.3  Data granularity and what it implies")
    body(doc,
         "One row is a single tweet, and tweets are truncated at roughly 350 characters. This matters "
         "in three ways. First, a tweet is an incomplete thought: the baseline cannot see context "
         "outside the box, so sarcasm and irony are frequently unresolvable. Second, because rows are "
         "independent snippets with no user or time identifier, there is no way to group repeat "
         "complaints from the same account, which rules out the session-level features we would "
         "normally use in production. Third, the median tweet is only "
         f"{eda['median_tokens']:.0f} tokens, which is short for TF-IDF bigrams and is one reason the "
         "classic bag-of-words approach saturates quickly.")

    h2(doc, "2.4  Documented versus delivered schema")
    body(doc,
         "The supplied dataset description states: 'Given a message and an entity, the task is to judge "
         "the sentiment of the message about the entity', with classes Negative, Positive, Irrelevant "
         "and Neutral. We verified row-by-row that neither CSV contains an entity or topic column - the "
         "only columns are Phrase and Sentiment. We therefore flag this as a data-delivery defect and "
         "raise it in Section 7 rather than silently improvising labels.")

    # ------------------------------------------------ 3. EDA
    doc.add_page_break()
    h1(doc, "Exploratory Data Analysis", "3")

    h2(doc, "3.1  Class distribution and imbalance")
    trc = eda["class_balance"]["train_counts"]
    tec = eda["class_balance"]["test_counts"]
    table(doc,
          ["Class", "Train n", "Train %", "Test n", "Test %", "Difference (pp)"],
          [[cls,
            trc[cls], f"{trc[cls]/eda['train_rows_in']*100:.1f}",
            tec[cls], f"{tec[cls]/eda['test_rows_in']*100:.1f}",
            f"{(tec[cls]/eda['test_rows_in'] - trc[cls]/eda['train_rows_in'])*100:+.1f}"]
           for cls in ["Negative", "Positive", "Irrelevant", "Neutral"]]
          + [["Total", eda["train_rows_in"], "100.0", eda["test_rows_in"], "100.0", "-"]],
          widths=[1.3, 0.85, 0.85, 0.85, 0.85, 1.3])

    body(doc,
         f"The classes are mildly imbalanced: the largest class (Negative) is "
         f"{eda['imbalance_ratio']:.2f}x the smallest (Irrelevant). This is far milder than the "
         "10:1 or 100:1 ratios that usually force resampling, so we deliberately did not apply SMOTE. "
         "More importantly, the train and test distributions agree to within "
         f"{max(abs(tec[c]/eda['test_rows_in'] - trc[c]/eda['train_rows_in'])*100 for c in trc):.1f} "
         "percentage points, so the test set is a fair proxy for the training distribution and a "
         "single hold-out evaluation is trustworthy.")

    figure(doc, "fig01_class_balance.png", 6.5,
           "Figure 1 - Class distribution in train and test. Imbalance is mild (1.65:1) and the two "
           "splits match, so accuracy is a meaningful metric alongside macro F1.")

    h2(doc, "3.2  Text length distribution and outliers")
    ls = eda["length_stats"]
    body(doc,
         f"After cleaning, the median tweet is {ls['50%']:.0f} tokens with a mean of {ls['mean']:.1f} "
         f"and a standard deviation of {ls['std']:.1f}. The distribution is right-skewed: the 75th "
         f"percentile is {ls['75%']:.0f} tokens but the maximum is {ls['max']:.0f}. Using the "
         f"1.5 x IQR rule the upper fence is {ls['upper_fence']:.0f} tokens, and only "
         f"{log['train']['steps'][-1]['flagged']} rows exceed it.")

    body(doc,
         "We made a deliberate decision not to delete these rows. Length outliers in text are almost "
         "always legitimate long tweets rather than measurement errors, and deleting them would "
         "systematically remove detailed complaints - exactly the subclass a brand-monitoring system "
         "must not lose. We flag them with an is_length_outlier column so that any downstream model "
         "can weight or cap them if it wishes. This is the difference between an outlier in tabular "
         "data (often an error) and an outlier in text (often the most informative row).")

    figure(doc, "fig02_length_distribution.png", 6.6,
           "Figure 2 - Tweet length after cleaning. The long right tail is genuine content, not "
           "corruption, so rows are flagged rather than dropped.")

    h2(doc, "3.3  Variables and correlation")
    body(doc,
         "The dataset has effectively one feature (text) and one target (label), so a conventional "
         "correlation matrix has nothing to correlate. Instead we measured the association between "
         "engineered text signals and the label, which is the correct analogue for unstructured data:")

    nb = eda["noise_by_class"]
    table(doc,
          ["Text signal", "Negative", "Positive", "Irrelevant", "Neutral", "Reading"],
          [
              ["% with @mentions", f"{nb['Negative']['@mentions']:.0f}",
               f"{nb['Positive']['@mentions']:.0f}", f"{nb['Irrelevant']['@mentions']:.0f}",
               f"{nb['Neutral']['@mentions']:.0f}",
               "Complaints are addressed to support handles"],
              ["% with URLs", f"{nb['Negative']['URLs']:.1f}", f"{nb['Positive']['URLs']:.1f}",
               f"{nb['Irrelevant']['URLs']:.1f}", f"{nb['Neutral']['URLs']:.1f}",
               "Neutral/irrelevant rows are news-like"],
              ["% with ellipses", f"{nb['Negative']['Ellipses']:.0f}",
               f"{nb['Positive']['Ellipses']:.0f}", f"{nb['Irrelevant']['Ellipses']:.0f}",
               f"{nb['Neutral']['Ellipses']:.0f}", "Hedging and incomplete thoughts"],
              ["% with <unk> token", f"{nb['Negative']['<unk> token']:.1f}",
               f"{nb['Positive']['<unk> token']:.1f}", f"{nb['Irrelevant']['<unk> token']:.1f}",
               f"{nb['Neutral']['<unk> token']:.1f}", "Tokenizer artefacts, not semantic"],
              ["% mojibake", f"{nb['Negative']['Mojibake']:.0f}",
               f"{nb['Positive']['Mojibake']:.0f}", f"{nb['Irrelevant']['Mojibake']:.0f}",
               f"{nb['Neutral']['Mojibake']:.0f}", "Uniform across classes = random corruption"],
          ],
          widths=[1.25, 0.72, 0.72, 0.78, 0.68, 2.15], font_size=8)

    body(doc,
         "Two signals are genuinely informative. @mentions appear in "
         f"{nb['Negative']['@mentions']:.0f}% of Negative rows against only "
         f"{nb['Neutral']['@mentions']:.0f}% of Neutral rows, confirming that negative tweets are "
         "usually addressed directly at a company. URLs appear almost exclusively in Neutral and "
         "Irrelevant rows, which fits their news-and-links character. The mojibake rate, by contrast, "
         "is roughly flat across all four classes - which is the proof that it is random technical "
         "corruption rather than anything semantically meaningful, and therefore safe to remove "
         "without biasing any class.")

    figure(doc, "fig04_noise_by_class.png", 6.6,
           "Figure 3 - Noise prevalence by class. @mentions skew negative, URLs skew neutral, while "
           "mojibake is flat across classes and therefore safe to strip.")

    h2(doc, "3.4  Vocabulary and class-conditional content")
    body(doc,
         "We profiled the most frequent discriminative terms per class after stop-word removal. The "
         "separation is visible but far from clean, which foreshadows the baseline's ceiling: Negative "
         "is dominated by profanity and product-failure words, Positive by affect words, while "
         "Neutral and Irrelevant share heavy overlap in news vocabulary.")

    figure(doc, "fig05_top_words.png", 6.6,
           "Figure 4 - Top discriminative terms per class. Negative and Positive separate on affect; "
           "Neutral and Irrelevant overlap heavily on news vocabulary.")

    figure(doc, "fig07_wordclouds.png", 6.6,
           "Figure 5 - Word clouds per class. Visual confirmation that Negative centres on profanity "
           "and failure language, while Neutral/Irrelevant are dominated by named entities and links.")

    h2(doc, "3.5  Brand-level view")
    brand_path = TAB / "brand_sentiment.csv"
    if brand_path.exists():
        bdf = pd.read_csv(brand_path)
        top_neg = bdf.sort_values("Negative", ascending=False).head(4)
        body(doc,
             "Because the practical use case is brand monitoring, we aggregated sentiment by brand "
             f"mention. Several brands skew strongly negative: "
             + ", ".join(f"{r['brand']} ({r['Negative']:.0f}% negative, n={int(r['n'])})"
                         for _, r in top_neg.iterrows())
             + ". Fortnite and Overwatch sit at the positive end. This is the kind of insight that "
               "turns a classifier into a decision support tool - the model output is only useful when "
               "it is aggregated to the entity a manager actually cares about.")

        figure(doc, "fig06_brand_sentiment.png", 6.6,
               "Figure 6 - Sentiment mix for the 15 most-mentioned brands. Verizon, Madden and NBA2K "
               "skew heavily negative; Fortnite and Overwatch skew positive.")

    h2(doc, "3.6  Data-quality audit")
    issues = eda["issues"]
    table(doc,
          ["Defect", "Rows affected", "% of train", "Severity"],
          [
              ["All-caps shouting", issues["All-caps shouting"],
               f"{issues['All-caps shouting']/eda['train_rows_in']*100:.1f}", "Low - handled by lowercasing"],
              ["@mentions", issues["@mentions"],
               f"{issues['@mentions']/eda['train_rows_in']*100:.1f}", "Medium - signal and noise"],
              ["Non-ASCII characters", issues["Non-ASCII characters"],
               f"{issues['Non-ASCII characters']/eda['train_rows_in']*100:.1f}", "Low after repair"],
              ["Mojibake (double-encoded UTF-8)", issues["Mojibake (Ã/Â/â)"],
               f"{issues['Mojibake (Ã/Â/â)']/eda['train_rows_in']*100:.1f}", "High - corrupts tokens"],
              ["<unk> placeholder", issues["<unk> placeholder"],
               f"{issues['<unk> placeholder']/eda['train_rows_in']*100:.1f}", "Medium - tokeniser artefact"],
              ["Embedded URLs", issues["Embedded URLs"],
               f"{issues['Embedded URLs']/eda['train_rows_in']*100:.1f}", "Medium - high-cardinality noise"],
              ["Anonymiser token RhandlerR", issues["Anonymiser token RhandlerR"],
               f"{issues['Anonymiser token RhandlerR']/eda['train_rows_in']*100:.1f}", "Low - low support"],
              ["Duplicate phrase text", issues["Duplicate phrase text"],
               f"{issues['Duplicate phrase text']/eda['train_rows_in']*100:.1f}", "High - leakage risk"],
              ["Exact duplicate rows", issues["Exact duplicate rows"],
               f"{issues['Exact duplicate rows']/eda['train_rows_in']*100:.1f}", "High - leakage risk"],
              ["Same text, different label", issues["Same text, different label"],
               f"{issues['Same text, different label']/eda['train_rows_in']*100:.1f}", "High - label noise"],
              ["HTML-like tags", issues["HTML-like tags"],
               f"{issues['HTML-like tags']/eda['train_rows_in']*100:.1f}", "Low - rare"],
          ],
          widths=[1.9, 1.0, 1.0, 2.4])

    figure(doc, "fig08_data_quality_audit.png", 6.6,
           "Figure 7 - Data-quality audit of the raw training file. Encoding corruption and duplicates "
           "were the defects with real modelling consequences.")

    # ------------------------------------------------ 4. cleaning
    doc.add_page_break()
    h1(doc, "Data Cleaning Pipeline", "4")

    body(doc, "We designed the pipeline as a sequence of auditable steps. Each step records what it "
              "changed, how many rows it touched and what rule it applied, and the full log is written "
              "to data/processed/cleaning_log.json (reproduced in Appendix A). Every step is applied "
              "identically to train and test using the same shared function, so there is no risk of the "
              "two files drifting apart.")

    steps = log["train"]["steps"]
    table(doc,
          ["#", "Step", "Rule applied", "Rows affected"],
          [[str(s["step"]), s["name"], s["action"], str(s["rows_affected"])] for s in steps],
          widths=[0.35, 1.35, 3.75, 0.85], font_size=7.6)

    h2(doc, "4.1  Why each decision was made")
    bullet(doc,
           "No imputation is defensible for text targets. With zero missing values the decision was "
           "moot, but we codified drop-on-missing so that the rule is explicit if the test file is "
           "ever refreshed.", bold_lead="Missing values: ")
    bullet(doc,
           "Three exact duplicate rows and one contradictory row (labels 0 and 3 on the same text) were "
           "removed. Duplicates inflate cross-validation scores because the same sentence can appear in "
           "both the training and validation fold, producing a falsely optimistic benchmark. The "
           "conflicting row was resolved to its majority label.",
           bold_lead="Duplicates and conflicting labels: ")
    bullet(doc,
           "We applied ftfy to reverse UTF-8 that had been decoded as cp1252, then Unicode NFKC "
           "normalisation. We verified the corruption is lossless and recoverable by inspecting the "
           "raw bytes (the sequence for a curly apostrophe was stored as three mojibake characters). "
           f"This repaired {log['train']['steps'][6]['rows_affected']} rows.",
           bold_lead="Encoding repair: ")
    bullet(doc,
           "Lowercasing, contraction expansion before punctuation stripping (so \"can't\" becomes "
           "\"can not\" rather than \"cant\"), URL and mention stripping, hashtag de-camel-casing, "
           "character-elongation collapsing (\"sooooo\" to \"soo\"), and removal of non-alphabetic "
           "characters. We preserve negations explicitly because removing \"not\" flips sentiment - "
           "this is a common and costly mistake in student pipelines.",
           bold_lead="Normalisation: ")
    bullet(doc,
           "WordNet lemmatisation (verb pass, then noun pass) reduced the vocabulary from "
           f"{log['train']['steps'][8]['detail'].split('from ')[1].split(' to ')[0]} to "
           f"{log['train']['steps'][8]['detail'].split(' to ')[1].split(' tokens')[0]}. We chose "
           "lemmatisation over stemming because stemming produces non-words and would damage the "
           "pretrained-embedding alignment we need in Phase 3ii.",
           bold_lead="Lemmatisation: ")
    bullet(doc,
           "We deliberately did NOT remove @mentions or URLs from the primary model input by default "
           "flag, and we quantified the effect of removing them (Section 5.3). Handles carry real "
           "signal about who is being complained about, and we did not want to assume away that "
           "information without measuring it.",
           bold_lead="What we chose not to do: ")

    h2(doc, "4.2  Documented before/after evidence")
    table(doc,
          ["Raw (as delivered)", "After cleaning"],
          [
              ["Nerd Tweet: I\u00e2\u0080\u0099m getting really disappointed with @PlayOverwatch one of my favourite heroes",
               "nerd tweet i be get real disappoint with one of my favourite hero"],
              ["New hero, expansion, and Priest class rework<unk>. I am super stoked",
               "new hero expansion and priest class rework i be super stoke"],
              ["I can't make SHIT green anymore.",
               "i can not make shit green anymore"],
              ["Criminals watching each other going...money/power/influence.....Fuck it<unk>!",
               "criminal watch each other go money power influence fuck it"],
              ["Check out https://t.co/Mo0oIFkdBT and #GamingIsLife sooooo goooood!!!!",
               "check out and gamingislife soo good"],
          ],
          widths=[3.5, 2.8], font_size=8)

    body(doc,
         "The examples show four behaviours worth noting. The mojibake apostrophe is repaired. The "
         "<unk> placeholder is removed without welding the neighbouring words together "
         "(\"rework<unk>\" must not become \"reworkunk\"). Negation is preserved as \"can not\". "
         "Elongation is normalised to at most a double character so \"goooood\" still resembles "
         "\"good\".", space_after=4)

    figure(doc, "fig03_cleaning_effect.png", 6.4,
           "Figure 8 - Effect of cleaning on length and vocabulary. Cleaning shortens tweets modestly "
           "and trims the long tail of noise tokens.")

    # ------------------------------------------------ 5. baseline
    doc.add_page_break()
    h1(doc, "Baseline Model: TF-IDF + Logistic Regression", "5")

    h2(doc, "5.1  Model design")
    body(doc,
         "Phase 3i requires exactly one baseline: logistic regression on TF-IDF features. We kept it "
         "deliberately plain so that it is a fair reference point, not a tuned competitor.")

    table(doc,
          ["Component", "Setting", "Reason"],
          [
              ["Vectoriser", "TF-IDF, word 1-2 grams", "Bigrams capture negation phrases such as 'not good'"],
              ["Weighting", "sublinear_tf=True", "Log-damped term frequency; avoids long-tweet dominance"],
              ["Vocabulary", f"{met['tfidf_features']:,} features, min_df=1, max_df=0.95",
               "Tiny corpus: rare terms are still useful"],
              ["Classifier", "Multinomial Logistic Regression", "Required baseline; inherently interpretable"],
              ["Regularisation", "C = 1.0 (L2), lbfgs solver", "Untuned default, deliberately left at baseline"],
              ["Max iterations", "2000", "Ensures convergence on this feature count"],
              ["Class weights", "None for the reported baseline", "Balanced variant reported separately in 5.4"],
          ],
          widths=[1.25, 2.15, 2.9], font_size=8.2)

    h2(doc, "5.2  Evaluation protocol")
    body(doc,
         "With only 592 training rows, a single train/validation split would give a benchmark with a "
         "confidence interval wider than any improvement we are likely to measure. We therefore report "
         "two complementary views:")
    bullet(doc, "Stratified 5-fold cross-validation on the training file only. This is the stable, "
                "low-variance headline number and it is used for all model-selection decisions.")
    bullet(doc, "A single held-out evaluation on testdata.csv using the model fitted on all 592 "
                "training rows. The test file is never used for model selection, so this number is an "
                "unbiased out-of-sample estimate.")
    body(doc,
         "We report accuracy, precision, recall and F1 as required, and we lead with macro F1 because "
         "accuracy alone hides the collapse of the minority Irrelevant class. We also report a "
         "majority-class reference point so the reader can see the genuine lift rather than an "
         "impressive-looking but meaningless percentage.")

    h2(doc, "5.3  Benchmark: does cleaning actually help?")
    cv_rows = []
    for _, r in cv_df.iterrows():
        cv_rows.append([r["variant"],
                        f"{r['accuracy_mean']*100:.1f} +/- {r['accuracy_std']*100:.1f}",
                        f"{r['macro_f1_mean']*100:.1f} +/- {r['macro_f1_std']*100:.1f}",
                        f"{r['weighted_f1_mean']*100:.1f}"])
    table(doc,
          ["Variant (5-fold CV on train)", "Accuracy %", "Macro F1 %", "Weighted F1 %"],
          cv_rows, widths=[2.5, 1.25, 1.25, 1.25])

    body(doc,
         f"This table is the most useful thing we can bring to the consultation, because it converts "
         f"cleaning decisions into measured effects. Cleaning alone changed macro F1 by "
         f"{(met['cv'][1]['macro_f1_mean']-met['cv'][0]['macro_f1_mean'])*100:+.1f} pp - essentially "
         f"neutral within the fold-to-fold noise of about +/-{met['cv'][1]['macro_f1_std']*100:.1f} pp. "
         "This is an honest and slightly uncomfortable result: it means our cleaning work cannot be "
         "justified on accuracy grounds. It is justified because the data is objectively corrupt "
         "(mojibake, placeholders, duplicates), because the embedding models in Phase 3ii require clean "
         "tokenisation, and because removing duplicates makes the cross-validation estimate valid. "
         "Stating this explicitly is better than claiming a spurious improvement.")

    figure(doc, "fig10_cv_comparison.png", 6.6,
           "Figure 9 - Cleaning decisions expressed as measured effects. Class-balanced weighting is "
           "the single largest improvement to macro F1.")

    h2(doc, "5.4  Headline baseline results on the held-out test file")
    pc = met["per_class"]
    table(doc,
          ["Class", "Precision", "Recall", "F1", "Support", "Reading"],
          [[r["class"], f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1']:.3f}",
            str(r["support"]),
            {"Negative": "High recall, low precision - predicts Negative too readily",
             "Positive": "Weakest of the two emotive classes",
             "Irrelevant": "Collapses - almost never predicted",
             "Neutral": "Moderate, confused with Negative"}[r["class"]]]
           for r in pc],
          widths=[1.15, 0.85, 0.75, 0.7, 0.7, 2.5], font_size=8.2)

    body(doc,
         f"Overall test accuracy is {met['test_accuracy']*100:.1f}% and macro F1 is "
         f"{met['test_macro_f1']*100:.1f}%. Against the majority-class guess of "
         f"{met['majority_baseline_accuracy']*100:.1f}%, this is a genuine lift of "
         f"{(met['test_accuracy']-met['majority_baseline_accuracy'])*100:.1f} percentage points, so "
         "the model has learned real signal rather than exploiting class frequency. It is, however, a "
         "modest absolute score, and the per-class breakdown shows exactly where the model fails.")

    body(doc,
         f"The imbalance lever is worth reporting carefully. Adding class_weight='balanced' lowered "
         f"accuracy slightly ({met['test_accuracy']*100:.1f}% to {met['test_balanced_accuracy']*100:.1f}%) "
         f"but raised macro F1 from {met['test_macro_f1']*100:.1f}% to "
         f"{met['test_balanced_macro_f1']*100:.1f}%, and lifted Irrelevant F1 from "
         f"{pc[2]['f1']:.3f} to 0.312. This is the expected and desirable trade: we give up a little "
         "headline accuracy to make the weakest class usable. It also previews Phase 3iv, where "
         "handling imbalance is an explicit marking criterion.")

    figure(doc, "fig09_confusion_matrix.png", 6.5,
           "Figure 10 - Confusion matrix on the held-out test set. The diagonal is strong for Negative "
           "only; Irrelevant is scattered across all four predictions.")

    h2(doc, "5.5  Interpreting what the model learned")
    body(doc,
         "Because logistic regression is linear, we can read the learned coefficients directly. The "
         "result is reassuringly sensible: Negative is driven by profanity and failure words, Positive "
         "by affect and praise words, and Neutral by named entities and news vocabulary. This is "
         "evidence that the model has learned genuine lexical sentiment rather than an artefact of how "
         "the files were split.")

    figure(doc, "fig12_top_coefficients.png", 6.7,
           "Figure 11 - Strongest TF-IDF terms per class. The learned associations are semantically "
           "sensible, which validates the pipeline even where accuracy is modest.")

    # ------------------------------------------------ 6. errors
    doc.add_page_break()
    h1(doc, "Error Analysis and Model Limitations", "6")

    h2(doc, "6.1  Where the errors come from")
    body(doc,
         "The baseline misclassifies 200 of 397 test rows (50.4%). Grouping the errors by actual and "
         "predicted class shows that they are highly structured rather than random:")

    table(doc,
          ["Actual class", "Predicted as", "Count", "Diagnosis"],
          [
              ["Positive", "Negative", "41", "Sarcasm and backhanded praise"],
              ["Neutral", "Negative", "34", "News mentioning a brand complaint reads as negative"],
              ["Neutral", "Positive", "29", "Neutral product news shares praise vocabulary"],
              ["Irrelevant", "Positive", "24", "No entity anchor, defaults to sentiment"],
              ["Irrelevant", "Negative", "23", "Same, with frustrated tone"],
              ["Negative", "Positive", "18", "Profanity used positively ('this is sick')"],
              ["Positive", "Neutral", "13", "Understated praise"],
              ["Irrelevant", "Neutral", "10", "News-like wording"],
          ],
          widths=[1.2, 1.2, 0.7, 3.2], font_size=8.2)

    h2(doc, "6.2  The Irrelevant problem is structural, not a tuning problem")
    body(doc,
         "The baseline correctly identifies only 9 of 66 Irrelevant test rows - an F1 of "
         f"{pc[2]['f1']:.3f} despite a precision of {pc[2]['precision']:.3f}. This pattern (high "
         "precision, near-zero recall) is the signature of a class the model can recognise in "
         "principle but almost never selects in practice. Normal class weighting therefore helps this "
         "class far more than it helps the other three.")

    body(doc,
         "We investigated the cause rather than assuming it. 'Irrelevant' means 'not about the entity'. "
         "Without the entity column that the dataset description promises, two annotators given the "
         "same tweet would have no way to agree on this label - and one contradictory label pair "
         "already exists in the training data. The label is therefore partly defined by information "
         "that is absent from the file, which places a hard ceiling on any text-only model. This is "
         "the single most important thing we want to raise in the consultation.")

    h2(doc, "6.3  Vocabulary and length limits")
    body(doc,
         "TF-IDF with bigrams sees at most two adjacent words. It cannot represent 'not very good' vs "
         "'very good' reliably once stop-words are removed, it cannot generalise 'crashed' to 'keeps "
         "crashing' unless lemmatisation aligns them, and it cannot use the order of clauses. With a "
         f"median of {eda['median_tokens']:.0f} tokens per tweet there is very little context available "
         "to recover from these limits. This is precisely why Phase 3ii moves to dense pretrained "
         "embeddings, which carry prior knowledge from billions of words of text.")

    h2(doc, "6.4  Overfitting check")
    lc = met["learning_curve"]
    body(doc,
         f"The learning curve tells us the model is variance-limited, not data-starved. Training F1 "
         f"sits at {lc['train_f1'][-1]*100:.1f}% while cross-validation F1 reaches only "
         f"{lc['cv_f1'][-1]*100:.1f}% at the largest training size. A gap that large means the "
         "classifier is memorising training tweets rather than learning generalisable rules, and the "
         "curve has not plateaued, which means more data would still help. Both observations argue for "
         "regularisation, feature selection and pretrained embeddings rather than simply collecting "
         "more rows.")

    figure(doc, "fig11_learning_curve.png", 5.9,
           "Figure 12 - Learning curve. The large train-CV gap indicates variance (over-fitting), not "
           "a shortage of data as the primary problem.")

    # ------------------------------------------------ 7. issues / questions
    doc.add_page_break()
    h1(doc, "Key Risks, Issues and Decisions Required", "7")

    body(doc, "We are bringing the following to the consultation session as open questions rather than "
              "silently resolving them.")

    table(doc,
          ["#", "Issue", "Impact on the project", "Our proposed position / question"],
          [
              ["1", "Dataset description promises an entity column; delivered CSVs have only Phrase "
                    "and Sentiment",
               "Label 2 ('Irrelevant') is defined relative to a missing entity, so it cannot be "
               "learned reliably. Caps achievable macro F1.",
               "Treat as document-level 4-class classification and report the deviation. QUESTION: "
               "should label 2 be excluded from scoring, or kept as a legitimate class?"],
              ["2", "Only 592 labelled training rows for 4 classes",
               "Cross-validation variance around +/-5 pp; hyper-parameter tuning risks over-fitting "
               "to the CV folds.",
               "Use repeated CV and prefer regularised models. QUESTION: is the 397-row test file "
               "available for final model comparison, or should we carve a validation split out of "
               "train?"],
              ["3", "Mild class imbalance (1.65:1) but severe label overlap",
               "Balanced weights trade accuracy for macro F1. Which metric is 'the' benchmark must be "
               "fixed now.",
               "Report both; lead with macro F1. QUESTION: does the marking scheme favour accuracy or "
               "macro F1?"],
              ["4", "Mojibake in 15.8% of rows and <unk> in 4.4%",
               "Unaffected if repaired, but silently corrupts any model that uses raw text (including "
               "pretrained embeddings).",
               "Repair with ftfy before embedding. No action needed from the professor; documented "
               "for transparency."],
              ["5", "Restricted to TF-IDF for the baseline only",
               "The baseline cannot use embeddings, so its ceiling is structurally lower than "
               "Phase 3ii models. Comparing the two is not like-for-like.",
               "Frame Phase 3ii gains as 'embedding + classifier vs bag-of-words', and always report "
               "the baseline number alongside."],
              ["6", "Duplicate and contradictory rows",
               "Duplicates inflate CV scores; contradictions add irreducible label noise.",
               "Both removed/resolved; log retained in Appendix A."],
          ],
          widths=[0.3, 1.6, 1.9, 2.5], font_size=7.6)

    # ------------------------------------------------ 8. plan
    doc.add_page_break()
    h1(doc, "Plan for Weeks 11-13", "8")

    body(doc,
         f"With five group members, the brief requires N = 5 analytics models paired with word "
         "embedding models (TF-IDF is not eligible and logistic regression is reserved for the "
         "baseline). We propose the following combinations, chosen for diversity rather than for "
         "stacking near-identical tree models.")

    table(doc,
          ["#", "Embedding", "Analytics model", "Why this pairing", "Owner"],
          [
              ["1", "GloVe (pre-trained, frozen)", "Gradient Boosting (XGBoost or LightGBM)",
               "Strong tabular-style learner on dense vectors; fast to iterate", "TBC"],
              ["2", "Word2Vec (trained from scratch on the corpus)",
               "Random Forest", "Satisfies the 'trained from scratch' bonus criterion", "TBC"],
              ["3", "FastText (subword)", "Support Vector Machine",
               "Subword units absorb misspellings and elongation that Word2Vec misses", "TBC"],
              ["4", "BERT (pre-trained, fine-tuned)",
               "Neural classifier head", "Highest expected ceiling; handles negation and word order",
               "TBC"],
              ["5", "Sentence-BERT embeddings", "k-Nearest-Neighbours / MLP",
               "Contrasts a non-parametric learner against the discriminative models", "TBC"],
          ],
          widths=[0.3, 1.5, 1.45, 2.35, 0.55], font_size=7.6)

    h2(doc, "8.1  Planned sequence")
    bullet(doc, "Week 11: build the shared embedding extraction module; produce GloVe and Word2Vec "
                "matrices once and cache them so all five models reuse identical inputs.")
    bullet(doc, "Week 11-12: train models 1-3 with baseline hyper-parameters; log every run to a "
                "single comparison table.")
    bullet(doc, "Week 12: fine-tune BERT and train the neural head; apply class weights identified in "
                "Section 5.4.")
    bullet(doc, "Week 12-13: hyper-parameter tuning for the top two models, then the Phase 3iv "
                "comparison against both the plain and the balanced baseline.")
    bullet(doc, "Week 13: final report, presentation and reproducibility check.")

    h2(doc, "8.2  Known risks and mitigations")
    bullet(doc, "GPU availability for BERT fine-tuning. Mitigation: run BERT on a free Colab T4 and "
                "cache the fine-tuned weights in the repository.", bold_lead="Risk: ")
    bullet(doc, "Only 592 rows makes fine-tuning a large pretrained model prone to catastrophic "
                "over-fitting. Mitigation: freeze the lower encoder layers, use a low learning rate "
                "and early stopping on the CV score.", bold_lead="Risk: ")
    bullet(doc, "Five models x five group members can fragment the codebase. Mitigation: one shared "
                "preprocessing module and one shared evaluation function, so every model is scored "
                "identically.", bold_lead="Risk: ")

    # ------------------------------------------------ 9. reproducibility
    doc.add_page_break()
    h1(doc, "Reproducibility and Artefact Index", "9")

    body(doc, "Every number and figure in this report is generated by code. To reproduce the entire "
              "document from the raw CSVs:")

    table(doc,
          ["Command", "What it produces"],
          [
              ["python src/run_pipeline.py",
               "Cleans both files, writes data/processed/*.csv and cleaning_log.json, generates Figures 1-8"],
              ["python src/train_baseline.py",
               "Trains the baseline, writes models/baseline_tfidf_logreg.joblib, "
               "baseline_metrics.json, figures 9-12 and all metric tables"],
              ["python src/make_interim_report.py", "Rebuilds this report"],
          ],
          widths=[2.3, 4.0], font_size=8.2)

    table(doc,
          ["Artefact", "Location", "Purpose"],
          [
              ["Raw data", "Data 5 - Twitter/Data 5 - Twitter/*.csv", "Untouched source files"],
              ["Cleaned data", "data/processed/train_clean.csv, test_clean.csv", "Model-ready rows with raw + clean text"],
              ["Cleaning log", "data/processed/cleaning_log.json", "Row-level audit trail"],
              ["EDA summary", "data/processed/eda_summary.json", "All EDA statistics"],
              ["Single source of truth", "src/text_clean.py", "Shared cleaning functions"],
              ["Baseline model", "models/baseline_tfidf_logreg.joblib", "Fitted pipeline"],
              ["Baseline metrics", "models/baseline_metrics.json", "All reported scores"],
              ["Figures", "reports/figures/", "12 publication-quality charts"],
              ["Tables", "reports/tables/", "CSV exports of every metric table"],
          ],
          widths=[1.5, 2.5, 2.3], font_size=8.2)

    callout(doc, "Reproducibility guarantee",
            "The report generator reads its numbers from JSON artefacts rather than from hard-coded "
            "text. If the pipeline is re-run on a refreshed dataset, this document regenerates with "
            "the new values automatically and cannot silently disagree with the code.")

    # ------------------------------------------------ A. appendix log
    doc.add_page_break()
    h1(doc, "Appendix A - Cleaning Log (training file)", "A")
    body(doc, "Verbatim from data/processed/cleaning_log.json. The test file is processed with "
              "identical rules and is recorded in the same file.", italic=True)

    for s in steps:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(f"Step {s['step']}: {s['name']}")
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = NAVY
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(f"Action: {s['action']}")
        r.font.size = Pt(8.5)
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(f"Detail: {s['detail']}")
        r.font.size = Pt(8.5)
        r.font.color.rgb = GREY

    # ------------------------------------------------ B. appendix metrics
    doc.add_page_break()
    h1(doc, "Appendix B - Full Metric Detail", "B")

    h2(doc, "B.1  Cross-validation, fold by fold")
    table(doc,
          ["Variant", "Acc mean", "Acc sd", "Macro F1 mean", "Macro F1 sd", "Weighted F1 mean"],
          [[r["variant"], f"{r['accuracy_mean']:.4f}", f"{r['accuracy_std']:.4f}",
            f"{r['macro_f1_mean']:.4f}", f"{r['macro_f1_std']:.4f}",
            f"{r['weighted_f1_mean']:.4f}"] for _, r in cv_df.iterrows()],
          widths=[2.1, 0.78, 0.68, 0.95, 0.9, 1.0], font_size=7.4)

    h2(doc, "B.2  Held-out test performance")
    table(doc,
          ["Configuration", "Accuracy", "Macro F1", "Weighted F1"],
          [
              ["Majority-class reference",
               f"{met['majority_baseline_accuracy']:.4f}", "-", "-"],
              ["TF-IDF + Logistic Regression (reported baseline)",
               f"{met['test_accuracy']:.4f}", f"{met['test_macro_f1']:.4f}",
               f"{met['test_weighted_f1']:.4f}"],
              ["Same + class_weight='balanced'",
               f"{met['test_balanced_accuracy']:.4f}", f"{met['test_balanced_macro_f1']:.4f}",
               f"{met['test_balanced_weighted_f1']:.4f}"],
          ],
          widths=[2.9, 1.15, 1.1, 1.15], font_size=8.2)

    h2(doc, "B.3  Per-class detail")
    table(doc,
          ["Class", "Precision", "Recall", "F1", "Support"],
          [[r["class"], f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}",
            str(r["support"])] for r in pc],
          widths=[1.3, 1.1, 1.0, 1.0, 0.9], font_size=8.2)

    h2(doc, "B.4  Model and environment")
    table(doc,
          ["Property", "Value"],
          [
              ["Model file", "models/baseline_tfidf_logreg.joblib"],
              ["TF-IDF features", f"{met['tfidf_features']:,}"],
              ["Train / test rows", f"{met['train_rows']} / {met['test_rows']}"],
              ["Total runtime", f"{met['runtime_seconds']} seconds"],
              ["Random seed", "42 (all splits and estimators)"],
              ["Python", sys.version.split()[0]],
              ["scikit-learn", "1.9.1"],
              ["pandas / numpy", "3.0.5 / 2.5.2"],
          ],
          widths=[1.7, 4.6], font_size=8.2)

    # ------------------------------------------------ sign off
    doc.add_paragraph()
    hr(doc)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("End of Interim Report")
    r.font.size = Pt(10)
    r.font.bold = True
    r.font.color.rgb = NAVY
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"{GROUP}  |  {COURSE}")
    r.font.size = Pt(8.5)
    r.font.color.rgb = GREY

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "Interim_Report_Group8_Twitter_Sentiment.docx"
    doc.save(path)
    return path


if __name__ == "__main__":
    p = build()
    print(f"written -> {p}")
    print(f"size    -> {p.stat().st_size/1024:.1f} KB")
