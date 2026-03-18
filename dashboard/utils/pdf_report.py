from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import io

# ── Color palette ─────────────────────────────────────────────────────────────
DARK_BG    = colors.HexColor("#0f172a")
DARK_CARD  = colors.HexColor("#1e293b")
ACCENT     = colors.HexColor("#2563a8")
ACCENT2    = colors.HexColor("#3b82f6")
TEXT_LIGHT = colors.HexColor("#f1f5f9")
TEXT_MUTED = colors.HexColor("#94a3b8")
RED        = colors.HexColor("#e74c3c")
GREEN      = colors.HexColor("#2ecc71")
ORANGE     = colors.HexColor("#f39c12")
WHITE      = colors.white

def _styles():
    s = getSampleStyleSheet()
    base = dict(fontName="Helvetica", textColor=TEXT_LIGHT)
    return {
        "title":    ParagraphStyle("title",    fontSize=22, fontName="Helvetica-Bold", textColor=WHITE,      spaceAfter=2),
        "subtitle": ParagraphStyle("subtitle", fontSize=10, fontName="Helvetica",      textColor=TEXT_MUTED, spaceAfter=12),
        "section":  ParagraphStyle("section",  fontSize=11, fontName="Helvetica-Bold", textColor=ACCENT2,    spaceBefore=14, spaceAfter=6),
        "body":     ParagraphStyle("body",     fontSize=9,  fontName="Helvetica",      textColor=TEXT_LIGHT, spaceAfter=4),
        "small":    ParagraphStyle("small",    fontSize=8,  fontName="Helvetica",      textColor=TEXT_MUTED),
    }

def _kpi_table(kpis):
    """kpis: list of (label, value) tuples, max 5"""
    cell_w = 45 * mm
    data = [[Paragraph(f'<font size="8" color="#94a3b8">{l}</font>', ParagraphStyle("x", alignment=TA_CENTER)),
             ] for l, v in kpis]
    val_row = [Paragraph(f'<font size="14"><b><font color="#ffffff">{v}</font></b></font>',
                          ParagraphStyle("x", alignment=TA_CENTER)) for l, v in kpis]
    lbl_row = [Paragraph(f'<font size="8" color="#94a3b8">{l}</font>',
                          ParagraphStyle("x", alignment=TA_CENTER)) for l, v in kpis]
    tbl = Table([lbl_row, val_row], colWidths=[cell_w] * len(kpis))
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), DARK_CARD),
        ("ROUNDEDCORNERS", [6]),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[DARK_CARD]),
        ("TOPPADDING",   (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0), (-1,-1), 8),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("LINEAFTER",    (0,0), (-2,-1), 0.5, DARK_BG),
    ]))
    return tbl

def _data_table(df, max_rows=15):
    """Render DataFrame as styled table"""
    cols = [c for c in ["timestamp","source","gender","tenure","contract",
                         "internet_service","monthly_charges","churn_probability",
                         "churn_flag","risk_level"] if c in df.columns]
    df_show = df[cols].head(max_rows).copy()
    if "timestamp" in df_show.columns:
        df_show["timestamp"] = df_show["timestamp"].dt.strftime("%m-%d %H:%M")
    if "churn_probability" in df_show.columns:
        df_show["churn_probability"] = df_show["churn_probability"].map("{:.4f}".format)

    header = [Paragraph(f'<font size="6.5" color="#93c5fd"><b>{c.replace("_"," ").upper()}</b></font>',
                         ParagraphStyle("h", alignment=TA_CENTER)) for c in cols]
    rows = [header]
    for _, row in df_show.iterrows():
        r = []
        for c in cols:
            val = str(row[c]) if row[c] is not None else ""
            color = "#ffffff"
            if c == "risk_level":
                color = "#e74c3c" if val=="HIGH" else "#f39c12" if val=="MEDIUM" else "#2ecc71"
            r.append(Paragraph(f'<font size="6.5" color="{color}">{val}</font>',
                                ParagraphStyle("d", alignment=TA_CENTER)))
        rows.append(r)

    # Proporsional column widths
    col_widths_map = {
        "timestamp":         28*mm,
        "source":            18*mm,
        "gender":            18*mm,
        "tenure":            15*mm,
        "contract":          30*mm,
        "internet_service":  28*mm,
        "monthly_charges":   22*mm,
        "churn_probability": 24*mm,
        "churn_flag":        16*mm,
        "risk_level":        18*mm,
    }
    col_widths = [col_widths_map.get(c, 18*mm) for c in cols]
    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND",    (0,0), (-1,0),  ACCENT),
        ("BACKGROUND",    (0,1), (-1,-1), DARK_CARD),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [DARK_CARD, colors.HexColor("#263548")]),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("LINEBELOW",     (0,0), (-1,0),  0.5, ACCENT2),
        ("LINEBELOW",     (0,1), (-1,-2), 0.3, DARK_BG),
    ]
    # highlight risk_level column if present
    if "risk_level" in cols:
        ri = cols.index("risk_level")
        for i, (_, row) in enumerate(df_show.iterrows(), start=1):
            bg = RED if row.get("risk_level")=="HIGH" else ORANGE if row.get("risk_level")=="MEDIUM" else GREEN
            style.append(("BACKGROUND", (ri,i), (ri,i), bg))
            style.append(("TEXTCOLOR",  (ri,i), (ri,i), WHITE))
    tbl.setStyle(TableStyle(style))
    return tbl

def generate_pdf(page_title, kpis, df, extra_stats=None):
    """
    Generate PDF report bytes.
    page_title : str
    kpis       : list of (label, value) — max 5
    df         : DataFrame with prediction data
    extra_stats: list of (label, value) for secondary stats row
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=landscape(A4),
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=15*mm, bottomMargin=15*mm,
    )
    ST = _styles()
    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("AcmeTel Churn Prediction", ST["title"]))
    story.append(Paragraph(f"{page_title}  ·  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", ST["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=10))

    # ── KPI row ───────────────────────────────────────────────────────────────
    story.append(Paragraph("KEY METRICS", ST["section"]))
    story.append(_kpi_table(kpis))

    if extra_stats:
        story.append(Spacer(1, 6))
        story.append(_kpi_table(extra_stats))

    # ── Recent predictions table ───────────────────────────────────────────────
    story.append(Spacer(1, 10))
    story.append(Paragraph("RECENT PREDICTIONS", ST["section"]))
    story.append(Paragraph(f"Showing latest {min(15, len(df))} of {len(df):,} records", ST["small"]))
    story.append(Spacer(1, 4))
    story.append(_data_table(df))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=DARK_CARD))
    story.append(Paragraph("AcmeTel Churn Prediction API  ·  Confidential", ST["small"]))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
