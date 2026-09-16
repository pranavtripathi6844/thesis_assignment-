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
    return {
        "title": ParagraphStyle(
            "T",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=16,
            leading=20,
            textColor=NAVY,
            spaceAfter=2 * mm,
            alignment=TA_CENTER,
        ),
        "sub": ParagraphStyle(
            "S",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=10,
            leading=13,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=6 * mm,
        ),
        "h": ParagraphStyle(
            "H",
            parent=base["Heading2"],
            fontName="Times-Bold",
            fontSize=12,
            leading=15,
            textColor=NAVY,
            spaceBefore=3.8 * mm,
            spaceAfter=2.0 * mm,
        ),
        "body": ParagraphStyle(
            "B",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=10,
            leading=13.4,
            alignment=TA_JUSTIFY,
            spaceAfter=2.6 * mm,
        ),
        "cap": ParagraphStyle(
            "C",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=8.2,
            leading=10.8,
            textColor=MUTED,
            alignment=TA_LEFT,
            spaceBefore=1.2 * mm,
            spaceAfter=3.2 * mm,
        ),
        "cell": ParagraphStyle(
            "Cell",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=8,
            leading=10.2,
            alignment=TA_CENTER,
        ),
        "cellL": ParagraphStyle(
            "CellL",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=8,
            leading=10.2,
            alignment=TA_LEFT,
        ),
        "th": ParagraphStyle(
            "TH",
            parent=base["Normal"],
            fontName="Times-Bold",
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=NAVY,
        ),
        "thL": ParagraphStyle(
            "THL",
            parent=base["Normal"],
            fontName="Times-Bold",
            fontSize=8,
            leading=10,
            alignment=TA_LEFT,
            textColor=NAVY,
        ),
        "foot": ParagraphStyle(
            "F",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=8.5,
            leading=11.2,
            textColor=MUTED,
        ),
    }


def P(text, st):
    return Paragraph(text, st)


def make_table(rows, col_widths):
    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.2),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
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
    canvas.drawString(16 * mm, h - 5.4 * mm, "Water scene classification")
    canvas.drawRightString(w - 16 * mm, h - 5.4 * mm, "Short report")
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, w, 8 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Times-Roman", 8)
    canvas.drawString(16 * mm, 3.2 * mm, "Pranav Tripathi")
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
    story.append(P("Classifying water scenes in satellite images", st["title"]))
    story.append(P("Pranav Tripathi  ·  thesis assignment  ·  EuroSAT dataset", st["sub"]))

    story.append(P("Introduction", st["h"]))
    story.append(
        P(
            "I started from a straightforward question: if I am given a small satellite photograph, "
            "can I tell whether the scene is mainly water or not? I used a public set of Sentinel-2 "
            "tiles. Each tile is 64 by 64 pixels and has thirteen spectral bands. There are only one "
            "hundred labelled tiles in total. Twenty of them are water scenes (rivers or lakes). "
            "The rest are other land-cover types such as crops, forest, roads or buildings.",
            st["body"],
        )
    )
    story.append(
        P(
            "It is important to be clear about the label. I am doing scene classification, not "
            "segmentation. Each tile gets one answer: water scene or not. That answer follows the "
            "original land-cover class of the whole tile. So a highway photograph that happens to "
            "show a river is still a non-water example for me. The scene is a road. If my model "
            "calls it water because it saw the river, that is a mistake under these labels, even "
            "though a mapping product might want the opposite. I am not drawing a shoreline or "
            "measuring how much of the tile is wet.",
            st["body"],
        )
    )
    story.append(
        P(
            "Table 1 is the split I used. Sixty tiles for training, twenty to make decisions, twenty "
            "held out for a final look. That leaves only four water tiles in the test set. From the "
            "beginning I treated that as a warning: one lucky score on twenty images would not mean "
            "the problem was solved.",
            st["body"],
        )
    )

    t1 = make_table(
        [
            [th("Split", True), th("Tiles"), th("Water"), th("Not water")],
            [td("Training", True), td("60"), td("12"), td("48")],
            [td("Validation", True), td("20"), td("4"), td("16")],
            [td("Test", True), td("20"), td("4"), td("16")],
            [td("All", True), td("100"), td("20"), td("80")],
        ],
        [usable * 0.34, usable * 0.22, usable * 0.22, usable * 0.22],
    )
    story.append(
        KeepTogether(
            [
                t1,
                P(
                    "Table 1. How the hundred tiles are divided. The full counts are in "
                    "results/split_summary.json on GitHub.",
                    st["cap"],
                ),
            ]
        )
    )

    story.append(P("How I approached it", st["h"]))
    story.append(
        P(
            "I did not begin with a large neural network. Eighty of the hundred tiles are not water, "
            "so a method that always says “not water” is already right most of the time. I wanted "
            "something that actually finds water, and I wanted each step to be justified before I "
            "added complexity. I therefore built three models, one after the other, on the same "
            "hundred tiles. They are compared, not combined. Nothing is stacked or voted at the end.",
            st["body"],
        )
    )
    story.append(
        P(
            "The first idea came from the physics of the sensor. Water looks bright-ish in green "
            "light and very dark in the near-infrared and short-wave infrared. Remote sensing has "
            "used that fact for a long time, through indices such as NDWI, which compares the green "
            "band to the near-infrared band, and MNDWI, which compares green to short-wave infrared. "
            "For each tile I summarised those index maps with a few numbers: the mean, the maximum, "
            "and how much of the tile looks wet. A small logistic regression then learns a line in "
            "that space. I also tried a single cut on the mean index, chosen only on the validation "
            "tiles. I treated this spectral model as the reference. If a neural network could not "
            "beat it, there was no reason to prefer the network.",
            st["body"],
        )
    )
    story.append(
        P(
            "The second step was about shape. A thin river and a lake can have a similar average "
            "index but look different on the ground. So I trained a small convolutional network from "
            "scratch: a few layers, dropout, and early stopping on the validation loss. I kept it "
            "small on purpose. I was not trying to prove that the size is perfect for sixty examples, "
            "only that a network with far fewer weights than a ResNet can be tested fairly here. I "
            "did not add extra images or flips. The constraint was to stay with the original hundred "
            "tiles.",
            st["body"],
        )
    )
    story.append(
        P(
            "The third step used a ResNet that had already been trained on unlabelled Sentinel-2 "
            "imagery. I froze that backbone and trained only a linear head on top of the embeddings. "
            "The large network is then a feature extractor, not something I am fitting on sixty "
            "labels. I showed it different band sets — visible colour, the water-related bands, or "
            "all thirteen — by hiding unused channels rather than rebuilding the network each time. "
            "The question here is whether a pretrained satellite representation adds anything once "
            "the spectral model is already on the table.",
            st["body"],
        )
    )
    fig1 = fitted_image(RESULTS / "fig_examples.png", usable, 50 * mm)
    story.append(
        KeepTogether(
            [
                P(
                    "Figure 1 is why the first step felt natural. In true colour a lake can look like a dark "
                    "grey surface. In the infrared composite the same water goes almost black, while plants "
                    "stay bright. That is the signal the spectral model is built on.",
                    st["body"],
                ),
                fig1,
                P(
                    "Figure 1. Four training tiles. Top row is true colour. Bottom row is a water "
                    "composite using short-wave infrared, near-infrared and green. The lake and the "
                    "river go dark in the lower row; vegetation does not. Image file: "
                    "results/fig_examples.png.",
                    st["cap"],
                ),
            ]
        )
    )

    story.append(
        KeepTogether(
            [
                P("How I decided to score the models", st["h"]),
                P(
                    "Because most tiles are not water, I refused to rank methods by accuracy alone. "
                    "Accuracy is easy to explain, and it is useful as a check: anything below the "
                    "“always not water” answer is worse than doing nothing. But it can look strong "
                    "while the model never finds a river. Precision tells me whether a water call is "
                    "trustworthy, yet a timid model can look precise by almost never calling water. "
                    "Recall tells me how many real water scenes I catch, but with only four test "
                    "positives a single miss moves that number a lot. F1 sits between precision and "
                    "recall and is the thresholded number I quote most often. I also kept a ranking "
                    "score that does not depend on the cut, because choosing the cut on four "
                    "validation water tiles is itself unstable. The ranking score can be perfect "
                    "while the cut still misses one scene. I see that as a result, not as a nuisance.",
                    st["body"],
                ),
            ]
        )
    )
    story.append(
        P(
            "Cuts and other choices were made on the validation tiles only. The test tiles were "
            "used once. I also repeated the training in five folds on the same hundred tiles, not "
            "to pretend I had more data, but to see whether a good-looking test score survived a "
            "different split. Pairwise tests between models on twenty images are in the repository; "
            "they do not pick a winner, which is what I expected with four positives.",
            st["body"],
        )
    )

    t2 = make_table(
        [
            [th("Model", True), th("Accuracy"), th("F1"), th("Ranking score")],
            [td("Always predict not water", True), td("0.80"), td("0.00"), td("0.20")],
            [td("Simple water-index cut", True), td("0.95"), td("0.86"), td("0.89")],
            [td("Logistic on water-index features", True), td("0.95"), td("0.86"), td("1.00")],
            [td("Small CNN", True), td("1.00"), td("1.00"), td("1.00")],
            [td("Frozen ResNet, colour only", True), td("0.80"), td("0.33"), td("0.65")],
            [td("Frozen ResNet, all bands", True), td("1.00"), td("1.00"), td("1.00")],
        ],
        [usable * 0.46, usable * 0.18, usable * 0.18, usable * 0.18],
    )
    story.append(
        KeepTogether(
            [
                t2,
                P(
                    "Table 2. Official twenty-tile test set. The ranking score is average precision. "
                    "The full table with intervals and extra band sets is "
                    "results/official_metrics.md on GitHub.",
                    st["cap"],
                ),
            ]
        )
    )

    story.append(P("What I found", st["h"]))
    story.append(
        P(
            "I read the results as a chain, not as a leaderboard. The first number that matters is "
            "<b>0.80</b>. That is the accuracy of always saying “not water”. Table 2 starts there so "
            "that every later score has to mean more than “most tiles are land”.",
            st["body"],
        )
    )
    story.append(
        P(
            "The second number is <b>0.86</b>. That is the F1 of the spectral logistic model, and "
            "also of a plain cut on the water index. So before any neural network, the physics I "
            "started with already finds water without drowning in false alarms. The ranking score "
            "for that logistic model is perfect on these twenty tiles: it puts the four water scenes "
            "at the top of the list. The F1 is not perfect because the cut chosen on validation "
            "still misses one of them. That is how I learned not to treat a single threshold as "
            "the whole story.",
            st["body"],
        )
    )
    story.append(
        P(
            "The third number looks better at first: the small CNN can reach <b>1.00</b> on this "
            "particular test split. If I had stopped there I would have said the network won. I "
            "did not stop there. When I reused the same hundred tiles in five folds, that CNN "
            "dropped to about <b>0.63</b> on average and jumped around from fold to fold. The "
            "perfect test score was a small split, not a stable method. Figure 2 is that check. "
            "The spectral model and the frozen ResNet stay near each other. The small CNN is the "
            "one with the long error bar.",
            st["body"],
        )
    )
    fig2 = fitted_image(RESULTS / "fig_cv_stability.png", usable, 55 * mm)
    story.append(
        KeepTogether(
            [
                P(
                    "The fourth number is the one I trust for the pretrained model: around <b>0.8</b> in "
                    "cross-validation, in the same range as the spectral logistic model. Colour-only "
                    "features are clearly weaker in Table 2, which matches Figure 1: visible light is not "
                    "where water stands out. Adding the infrared bands brings the frozen network back in "
                    "line with physics. It does not clearly overtake it.",
                    st["body"],
                ),
                fig2,
                P(
                    "Figure 2. The same hundred tiles, split five ways. The small CNN moves around "
                    "much more than the spectral model or the frozen ResNet. Image file: "
                    "results/fig_cv_stability.png.",
                    st["cap"],
                ),
            ]
        )
    )

    t3 = make_table(
        [
            [th("Model", True), th("Average F1 across folds")],
            [td("Always predict not water", True), td("0.00")],
            [td("Logistic on water-index features", True), td("0.80, with a spread of 0.17")],
            [td("Small CNN", True), td("0.63, with a spread of 0.34")],
            [td("Frozen ResNet, water-related bands", True), td("0.83, with a spread of 0.13")],
            [td("Frozen ResNet, all bands", True), td("0.77, with a spread of 0.21")],
        ],
        [usable * 0.58, usable * 0.42],
    )
    story.append(
        KeepTogether(
            [
                t3,
                P(
                    "Table 3. Cross-validation on the original hundred tiles. Full fold values are in "
                    "results/cv_f1.json on GitHub.",
                    st["cap"],
                ),
            ]
        )
    )

    story.append(P("Closing", st["h"]))
    story.append(
        P(
            "I approached the problem as a sequence of decisions. First use what is already known "
            "about water in Sentinel-2. Then ask whether a small network can add shape. Then ask "
            "whether a frozen satellite ResNet adds a representation I could not learn from sixty "
            "labels. I judged them with F1 and a ranking score because accuracy is too easy when "
            "eighty tiles are land. The chain of results is: the trivial answer is already 0.80 "
            "accurate; physics already reaches an F1 of 0.86; a 1.00 on twenty tiles did not survive "
            "resampling; a frozen ResNet lands in the same place as physics. With this many labels I "
            "would keep the spectral model as the reference. If more labelled tiles arrived later, "
            "I would keep the same order and only then unfreeze the large network.",
            st["body"],
        )
    )
    story.append(
        P(
            "The other plots and tables from the experiments — band comparisons, precision–recall "
            "curves, confusion matrices, training curves, and the tiles each model got wrong — are "
            "not repeated here. They are in the results folder of the GitHub repository "
            "https://github.com/pranavtripathi6844/thesis_assignment-.",
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
        title="Classifying water scenes in satellite images",
        author="Pranav Tripathi",
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
