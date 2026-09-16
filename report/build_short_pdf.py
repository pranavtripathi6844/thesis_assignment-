#!/usr/bin/env python3
"""Build a 2–3 page PDF report from committed results."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "report" / "Water_Scene_Classification_Report.pdf"

NAVY = colors.HexColor("#1F3A5F")
TEAL = colors.HexColor("#2C5F6E")
RULE = colors.HexColor("#C5CDD6")
HEADER_BG = colors.HexColor("#E8EEF4")
ROW_ALT = colors.HexColor("#F7F9FB")
MUTED = colors.HexColor("#4A5560")


def styles():
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle(
            "T",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=14.5,
            leading=18,
            textColor=NAVY,
            spaceAfter=2 * mm,
            alignment=TA_CENTER,
        ),
        "sub": ParagraphStyle(
            "S",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=9,
            leading=12,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=6 * mm,
        ),
        "h": ParagraphStyle(
            "H",
            parent=base["Heading2"],
            fontName="Times-Bold",
            fontSize=11,
            leading=14,
            textColor=NAVY,
            spaceBefore=3.5 * mm,
            spaceAfter=1.8 * mm,
        ),
        "body": ParagraphStyle(
            "B",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=9.2,
            leading=12.2,
            alignment=TA_JUSTIFY,
            spaceAfter=2.2 * mm,
        ),
        "cap": ParagraphStyle(
            "C",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=8,
            leading=10.4,
            textColor=MUTED,
            alignment=TA_LEFT,
            spaceBefore=1 * mm,
            spaceAfter=3 * mm,
        ),
        "cell": ParagraphStyle(
            "Cell",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=7.6,
            leading=9.6,
            alignment=TA_CENTER,
        ),
        "cellL": ParagraphStyle(
            "CellL",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=7.6,
            leading=9.6,
            alignment=TA_LEFT,
        ),
        "th": ParagraphStyle(
            "TH",
            parent=base["Normal"],
            fontName="Times-Bold",
            fontSize=7.5,
            leading=9.4,
            alignment=TA_CENTER,
            textColor=NAVY,
        ),
        "thL": ParagraphStyle(
            "THL",
            parent=base["Normal"],
            fontName="Times-Bold",
            fontSize=7.5,
            leading=9.4,
            alignment=TA_LEFT,
            textColor=NAVY,
        ),
        "foot": ParagraphStyle(
            "F",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=8,
            leading=10,
            textColor=MUTED,
        ),
        "li": ParagraphStyle(
            "LI",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=9.2,
            leading=12.0,
            leftIndent=0,
        ),
    }
    return s


def P(text, st):
    return Paragraph(text, st)


def make_table(rows, col_widths):
    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))
    tbl.setStyle(TableStyle(style))
    return tbl


def header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 8 * mm, w, 8 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Times-Roman", 8)
    canvas.drawString(16 * mm, h - 5.4 * mm, "EuroSAT100 water-scene classification")
    canvas.drawRightString(w - 16 * mm, h - 5.4 * mm, "Short experimental report")
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, w, 8 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Times-Roman", 8)
    canvas.drawString(16 * mm, 3.2 * mm, "Pranav Tripathi  ·  thesis assignment")
    canvas.drawRightString(w - 16 * mm, 3.2 * mm, f"Page {doc.page}")
    canvas.restoreState()


def fitted_image(path: Path, max_w, max_h):
    img = Image(str(path))
    iw, ih = img.imageWidth, img.imageHeight
    scale = min(max_w / iw, max_h / ih)
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale
    img.hAlign = "CENTER"
    return img


def build():
    st = styles()
    page_w, _ = A4
    usable = page_w - 32 * mm

    def th(text, left=False):
        return P(text, st["thL"] if left else st["th"])

    def td(text, left=False):
        return P(text, st["cellL"] if left else st["cell"])

    story = []
    story.append(P("Water-scene classification under a 100-label budget", st["title"]))
    story.append(
        P(
            "EuroSAT100 (Sentinel-2, 13 bands, 64×64)  ·  binary River ∪ SeaLake vs other classes  ·  "
            "physics, tiny CNN, frozen ResNet-18",
            st["sub"],
        )
    )

    story.append(P("1. Problem", st["h"]))
    story.append(
        P(
            "We ask a small-sample remote-sensing question: given one Sentinel-2 chip, is the scene a "
            "<b>water scene</b> or not? EuroSAT100 provides exactly 100 labelled 64×64 tiles with 13 "
            "spectral bands. We map the original ten land-cover classes to a binary label: "
            "<b>positive</b> if the official class is River or SeaLake (20 chips), <b>negative</b> otherwise "
            "(80 chips). Table&nbsp;1 records the official 60/20/20 split. Only four water chips fall in "
            "validation and four in test, so a single accuracy number is not a sufficient summary.",
            st["body"],
        )
    )
    story.append(
        P(
            "This is <b>scene classification, not water segmentation</b>. Each tile has one label — the "
            "dominant land-cover class — not a pixel mask. A Highway chip that contains a river is still "
            "a <b>true negative</b> under these labels: the scene is a road scene, even if water is visible. "
            "A River chip still includes banks and vegetation. At 10&nbsp;m GSD the tile covers 640&nbsp;m "
            "× 640&nbsp;m; mixed pixels are expected. We do not estimate flood extent or shoreline.",
            st["body"],
        )
    )

    t1 = make_table(
        [
            [th("Split", True), th("Tiles"), th("Water"), th("Non-water"), th("Positive rate")],
            [td("Train", True), td("60"), td("12"), td("48"), td("0.20")],
            [td("Validation", True), td("20"), td("4"), td("16"), td("0.20")],
            [td("Test", True), td("20"), td("4"), td("16"), td("0.20")],
            [td("Total", True), td("100"), td("20"), td("80"), td("0.20")],
        ],
        [usable * 0.28, usable * 0.18, usable * 0.18, usable * 0.18, usable * 0.18],
    )
    story.append(
        KeepTogether(
            [
                t1,
                P(
                    "<b>Table 1.</b> Official EuroSAT100 split used throughout. Source: "
                    "<font face='Courier'>results/split_summary.json</font> in the repository.",
                    st["cap"],
                ),
            ]
        )
    )
    story.append(
        P(
            "Two facts drive the design. Water reflects in green and absorbs in the near-infrared and "
            "short-wave infrared, so spectral indices are a natural first model. And 80% of chips are "
            "non-water, so a classifier that always answers “no water” already reaches <b>80% accuracy</b>. "
            "Accuracy alone cannot be the headline metric.",
            st["body"],
        )
    )

    story.append(P("2. Three-stage methodology", st["h"]))
    story.append(
        P(
            "We compare three <b>separate</b> models on the same 100 chips. They are not fused, stacked, "
            "or voted at inference. Neural nets have to beat physics; they are not assumed to win. "
            "Cuts and hyperparameters are chosen on validation only. The test set is touched once. "
            "Thresholds come from unique validation scores (higher cut on ties). NDWI/MNDWI are used as "
            "<b>raw</b> scores, with no test-set scaling.",
            st["body"],
        )
    )
    story.append(
        ListFlowable(
            [
                ListItem(
                    P(
                        "<b>Stage 1 — Physics (reference).</b> "
                        "NDWI = (B<sub>03</sub>−B<sub>08</sub>)/(B<sub>03</sub>+B<sub>08</sub>) and "
                        "MNDWI = (B<sub>03</sub>−B<sub>11</sub>)/(B<sub>03</sub>+B<sub>11</sub>). "
                        "Each tile is summarised by mean, max, and wet-pixel fractions (eight features), "
                        "then a regularised logistic regression (about ten coefficients) is fit. "
                        "A one-parameter index threshold is also reported. "
                        "<i>Motivation:</i> with 60 training images the safest model is few parameters "
                        "plus a real spectral prior.",
                        st["li"],
                    ),
                    leftIndent=8,
                    bulletColor=NAVY,
                ),
                ListItem(
                    P(
                        "<b>Stage 2 — Tiny CNN from scratch.</b> "
                        "Three conv–BN–ReLU blocks, global average pooling, dropout, ~24–25k weights, "
                        "class-weighted loss, early stopping on validation loss, no augmentation. "
                        "<i>Motivation:</i> mean NDWI ignores shape (narrow river vs lake). This tests "
                        "spatial learning under a tight budget. We call it capacity-reduced, not proven "
                        "capacity-matched.",
                        st["li"],
                    ),
                    leftIndent=8,
                    bulletColor=NAVY,
                ),
                ListItem(
                    P(
                        "<b>Stage 3 — Frozen pretrained ResNet-18.</b> "
                        "Sentinel-2 MoCo (SSL4EO-S12) backbone is frozen; a logistic head is fit on "
                        "512-D embeddings (~513 trainable parameters). Unused bands are zero-masked "
                        "so every band subset uses the same stem. "
                        "<i>Motivation:</i> fine-tuning eleven million weights on 60 labels is not a "
                        "reasonable estimator. Unlabelled pretraining is allowed; extra labelled "
                        "EuroSAT images are not.",
                        st["li"],
                    ),
                    leftIndent=8,
                    bulletColor=NAVY,
                ),
            ],
            bulletType="1",
            start="1",
            leftIndent=12,
            bulletFontName="Times-Bold",
            bulletFontSize=9,
        )
    )
    story.append(Spacer(1, 1.5 * mm))

    fig1 = fitted_image(RESULTS / "fig_examples.png", usable, 54 * mm)
    story.append(
        KeepTogether(
            [
                fig1,
                P(
                    "<b>Figure 1.</b> Train examples (2–98% display stretch). Top: true colour "
                    "(B<sub>04</sub>–B<sub>03</sub>–B<sub>02</sub>). Bottom: water composite "
                    "(SWIR1–NIR–green). SeaLake is dark in the composite because water absorbs NIR/SWIR; "
                    "the river stays dark while vegetation is bright. File: "
                    "<font face='Courier'>results/fig_examples.png</font>.",
                    st["cap"],
                ),
            ]
        )
    )

    story.append(
        KeepTogether(
            [
                P("3. Evaluation metrics", st["h"]),
                P(
                    "The class prior is 0.80 non-water. We keep accuracy as a sanity check against that floor "
                    "and put the weight on metrics that treat water as the positive class.",
                    st["body"],
                ),
            ]
        )
    )
    story.append(
        P(
            "<b>Accuracy</b> — <i>pro:</i> easy to explain; anything below 0.80 is worse than always "
            "predicting non-water. <i>con:</i> dominated by the negative class. "
            "<b>Precision</b> — <i>pro:</i> penalises false water calls. <i>con:</i> can look high if "
            "the model is timid. "
            "<b>Recall</b> — <i>pro:</i> how many water scenes we catch. <i>con:</i> one miss among four "
            "positives moves recall by 0.25; a Wilson 95% interval on 3/4 is about 0.30–0.95 "
            "(Table&nbsp;2). "
            "<b>F1</b> — <i>pro:</i> balances precision and recall at one operating point. <i>con:</i> "
            "depends on the validation cut, which is noisy with four val positives. "
            "<b>ROC-AUC</b> — <i>pro:</i> threshold-free ranking. <i>con:</i> optimistic with 80% "
            "negatives. "
            "<b>PR-AUC (AP)</b> — <i>pro:</i> ranking for the positive class; chance equals 0.20. "
            "<i>con:</i> still has no operating point. Several models in Table&nbsp;2 have AP = 1.0 "
            "and F1 = 0.86: they <b>order</b> the twenty chips correctly but the <b>val-chosen cut</b> "
            "misses one water scene.",
            st["body"],
        )
    )

    t2 = make_table(
        [
            [
                th("Model", True),
                th("Acc."),
                th("Acc. 95% CI"),
                th("F1"),
                th("AP"),
                th("Recall 95% CI"),
            ],
            [
                td("Majority (always non-water)", True),
                td("0.80"),
                td("0.58–0.92"),
                td("0.00"),
                td("0.20"),
                td("0.00–0.49"),
            ],
            [
                td("NDWI threshold", True),
                td("0.95"),
                td("0.76–0.99"),
                td("0.86"),
                td("0.89"),
                td("0.30–0.95"),
            ],
            [
                td("Logistic, NDWI/MNDWI stats", True),
                td("0.95"),
                td("0.76–0.99"),
                td("0.86"),
                td("1.00"),
                td("0.30–0.95"),
            ],
            [
                td("Tiny CNN, water bands", True),
                td("1.00"),
                td("0.84–1.00"),
                td("1.00"),
                td("1.00"),
                td("0.51–1.00"),
            ],
            [
                td("Frozen probe, RGB only", True),
                td("0.80"),
                td("0.58–0.92"),
                td("0.33"),
                td("0.65"),
                td("0.05–0.70"),
            ],
            [
                td("Frozen probe, water bands", True),
                td("0.95"),
                td("0.76–0.99"),
                td("0.86"),
                td("1.00"),
                td("0.30–0.95"),
            ],
            [
                td("Frozen probe, all 13 bands", True),
                td("1.00"),
                td("0.84–1.00"),
                td("1.00"),
                td("1.00"),
                td("0.51–1.00"),
            ],
        ],
        [usable * 0.32, usable * 0.10, usable * 0.16, usable * 0.10, usable * 0.10, usable * 0.22],
    )
    story.append(
        KeepTogether(
            [
                t2,
                P(
                    "<b>Table 2.</b> Official 20-image test set (4 water). AP is average precision "
                    "(PR-AUC). Intervals are Wilson 95% CIs on accuracy and recall. Source: "
                    "<font face='Courier'>results/official_metrics.md</font>. Full band-ablation and "
                    "random-4 arms are not repeated here; they are in the same file on GitHub.",
                    st["cap"],
                ),
            ]
        )
    )

    story.append(P("4. Results", st["h"]))
    story.append(
        P(
            "Table&nbsp;2 is the official split. Majority accuracy is 0.80 with F1 = 0, as expected. "
            "A raw NDWI threshold already reaches F1 = 0.86. Logistic regression on the eight index "
            "statistics reaches AP = 1.0 and F1 = 0.86: perfect ranking, one threshold error. The tiny "
            "CNN on water bands and the all-band frozen probe can look perfect on <i>this</i> 20-chip "
            "split (F1 = 1.0). RGB-only embeddings do not (F1 = 0.33). Those perfect scores sit inside "
            "wide intervals (recall 0.51–1.00 for 4/4). Pairwise McNemar tests on the same twenty chips "
            "are not significant (all p ≥ 0.125); with four positives they cannot establish a winner. "
            "The McNemar table is omitted here and is stored as "
            "<font face='Courier'>results/mcnemar.json</font>.",
            st["body"],
        )
    )
    story.append(
        P(
            "Figure&nbsp;2 and Table&nbsp;3 report stratified five-fold cross-validation on the "
            "<b>same</b> 100 chips — a stability check, not extra labelled data. This is the more "
            "honest summary. Logistic index features give F1 0.80 ± 0.17. The frozen probe is in the "
            "same band (water bands 0.83 ± 0.13; all bands 0.77 ± 0.21). The tiny CNN is worse and "
            "much noisier (0.63 ± 0.34). Neural complexity did not buy stability.",
            st["body"],
        )
    )

    story.append(PageBreak())
    fig2 = fitted_image(RESULTS / "fig_cv_stability.png", usable, 62 * mm)
    story.append(
        KeepTogether(
            [
                fig2,
                P(
                    "<b>Figure 2.</b> Stratified 5-fold F1 and PR-AUC (mean ± std) on the same 100 chips. "
                    "The tiny CNN has the largest error bar. Physics logistic and the frozen probe sit "
                    "together near F1 ≈ 0.8. File: "
                    "<font face='Courier'>results/fig_cv_stability.png</font>.",
                    st["cap"],
                ),
            ]
        )
    )

    t3 = make_table(
        [
            [th("Model", True), th("CV F1 (mean ± sd)"), th("CV AP (mean ± sd)")],
            [td("Majority", True), td("0.00 ± 0.00"), td("0.20 ± 0.00")],
            [td("Logistic, index statistics", True), td("0.80 ± 0.17"), td("0.96 ± 0.08")],
            [td("Tiny CNN, water bands", True), td("0.63 ± 0.34"), td("0.82 ± 0.25")],
            [td("Frozen probe, water bands", True), td("0.83 ± 0.13"), td("0.98 ± 0.04")],
            [td("Frozen probe, all 13 bands", True), td("0.77 ± 0.21"), td("0.96 ± 0.09")],
        ],
        [usable * 0.40, usable * 0.30, usable * 0.30],
    )
    story.append(
        KeepTogether(
            [
                t3,
                P(
                    "<b>Table 3.</b> Supplementary 5-fold CV on all 100 chips (inner validation for "
                    "thresholds). Source: <font face='Courier'>results/cv_f1.json</font>.",
                    st["cap"],
                ),
            ]
        )
    )

    story.append(P("5. Reading", st["h"]))
    story.append(
        P(
            "We treated this as a small-sample remote-sensing study, not an accuracy contest. Physics "
            "first, because the spectrum of water is known; a small CNN next, to test spatial learning; "
            "a frozen Sentinel-2 ResNet last, to use pretraining without fitting millions of weights. "
            "Accuracy is checked against the 0.80 floor in Table&nbsp;2. F1 and AP are the headlines "
            "and are kept separate because one is a cut and the other is a ranking. The evidence in "
            "Tables&nbsp;2–3 and Figure&nbsp;2 is that <b>spectral features are a competitive reference</b>, "
            "that a tiny CNN is not shown to be reliable at this n, and that four test positives cannot "
            "support a superiority claim. If more labels arrive, the same ladder still applies: keep "
            "the index model, keep the frozen probe, and only then unfreeze the backbone.",
            st["body"],
        )
    )
    story.append(
        P(
            "Remaining artefacts (full official-split dump, band-ablation bars, PR/ROC curves, "
            "confusion matrices, training curves, per-band reflectance table, error galleries) are "
            "not reproduced here. They are on GitHub: "
            "https://github.com/pranavtripathi6844/thesis_assignment- in the results/ folder.",
            st["foot"],
        )
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="Water-scene classification under a 100-label budget",
        author="Pranav Tripathi",
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
