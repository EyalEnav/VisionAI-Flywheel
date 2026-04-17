#!/usr/bin/env python3
"""
VisionAI-Flywheel - VLM Dataset PDF Report Generator

Generates a per-clip PDF report with video frame thumbnails and VSS accuracy scores.

Usage:
    python3 scripts/generate_dataset_report.py \\
        --records data/annotations.json \\
        --output  vlm_dataset_report.pdf \\
        --clip-dir /path/to/vst/clip_storage

Requirements:
    pip install fpdf2 pillow
    sudo apt install ffmpeg

Record JSON schema:
    [{"idx": 1, "name": "batch_01_falling", "tag": "falling",
      "annotation": "ground truth description...",
      "vss": "vss model response..."}]
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
from collections import Counter

try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
except ImportError:
    print("ERROR: fpdf2 not installed. Run: pip install fpdf2", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CLIP_DIR = (
    "/home/ubuntu/video-search-and-summarization/"
    "deployments/data-dir/data_log/vst/clip_storage"
)
FRAMES_DIR = "/tmp/pdf_frames"

TAG_COLORS = {
    "falling":    (220, 90,  60),
    "limp":       (200, 140, 40),
    "violence":   (180, 50,  50),
    "drunk":      (100, 80,  180),
    "mixed":      (80,  140, 180),
    "crowd_push": (200, 60,  120),
    "walking":    (60,  160, 80),
}
SCORE_COLORS = {
    1: (200, 50,  50),
    2: (220, 120, 40),
    3: (220, 200, 40),
    4: (100, 180, 50),
    5: (40,  160, 40),
}
BAD_PHRASES = [
    "does not exist", "unavailable", "cannot proceed", "not found",
    "no video", "invalid", "digital", "simulation", "floating",
    "no movement", "stationary", "no incidents",
]

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_vss(vss: str, annotation: str, tag: str) -> int:
    """Heuristic VSS accuracy score 1-5 vs ground-truth annotation."""
    vss_l = vss.lower()
    ann_l = annotation.lower()
    bads = sum(1 for p in BAD_PHRASES if p in vss_l)
    action_words = [w for w in ann_l.split() if len(w) > 4][:8]
    hits = sum(1 for w in action_words if w in vss_l)
    tag_hit = 1 if tag.lower() in vss_l else 0
    score = 3 + hits * 0.4 + tag_hit * 0.5 - bads * 0.7
    return min(5, max(1, round(score)))


def star(s: int) -> str:
    return "*" * s + "o" * (5 - s)


def safe(t: str) -> str:
    return t.encode("latin-1", errors="replace").decode("latin-1")


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frame(name: str, out_path: str, clip_dir: str) -> bool:
    vid = os.path.join(clip_dir, f"{name}.mp4")
    if not os.path.exists(vid):
        return False
    subprocess.run(
        ["ffmpeg", "-y", "-ss", "2", "-i", vid,
         "-frames:v", "1", "-vf", "scale=300:-1", out_path],
        capture_output=True,
    )
    return os.path.exists(out_path) and os.path.getsize(out_path) > 100


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

class DatasetPDF(FPDF):
    def header(self):
        self.set_fill_color(15, 20, 35)
        self.rect(0, 0, 210, 20, "F")
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(255, 255, 255)
        self.set_y(5)
        self.cell(0, 10, "VLM Fine-Tuning Dataset - VSS Evaluation Report", align="C")
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"VisionAI-Flywheel | Page {self.page_no()}", align="C")


def build_pdf(records: list, output: str, clip_dir: str):
    os.makedirs(FRAMES_DIR, exist_ok=True)
    scores = [score_vss(r["vss"], r["annotation"], r["tag"]) for r in records]
    tag_counts = Counter(r["tag"] for r in records)
    avg_score = sum(scores) / len(scores) if scores else 0

    pdf = DatasetPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---- Cover ----
    pdf.set_y(28)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(30, 40, 60)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 9, f"  Dataset Overview - {len(records)} Annotated Clips",
             fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)
    for tag, cnt in sorted(tag_counts.items()):
        c = TAG_COLORS.get(tag, (100, 100, 100))
        pdf.set_fill_color(*c)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(28, 6, f"  {tag}", fill=True)
        pdf.set_fill_color(240, 240, 240)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(10, 6, f" {cnt}", fill=True)
        pdf.cell(5, 6, "")
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, f"  Avg VSS Score: {avg_score:.1f}/5.0   |   Total pairs: {len(records)}   |   Status: reviewed",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    # ---- Per-clip pages ----
    for i, rec in enumerate(records):
        pdf.add_page()
        tag = rec["tag"]
        score = scores[i]
        color = TAG_COLORS.get(tag, (100, 100, 100))

        pdf.set_fill_color(*color)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_y(22)
        pdf.cell(0, 9,
                 safe(f"  #{rec['idx']:02d}  {rec['name']}   [{tag.upper()}]   VSS Score: {star(score)} ({score}/5)"),
                 fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

        frame_path = os.path.join(FRAMES_DIR, f"{rec['name']}.jpg")
        has_frame = extract_frame(rec["name"], frame_path, clip_dir)
        IMG_W, IMG_H = 80, 55
        TEXT_X = 15 + IMG_W + 6
        TEXT_W = 210 - TEXT_X - 10
        y_start = pdf.get_y()

        if has_frame:
            try:
                pdf.image(frame_path, x=12, y=y_start, w=IMG_W, h=IMG_H)
            except Exception as e:
                print(f"  [WARN] {rec['name']}: {e}", file=sys.stderr)

        pdf.set_xy(TEXT_X, y_start)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(30, 100, 30)
        pdf.cell(TEXT_W, 5, "Ground Truth Annotation:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(0, 0, 0)
        for line in textwrap.wrap(safe(rec["annotation"]), 52):
            pdf.set_x(TEXT_X)
            pdf.cell(TEXT_W, 5, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.ln(2)
        pdf.set_x(TEXT_X)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(150, 60, 20)
        pdf.cell(TEXT_W, 5, "VSS Response (truncated):", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(60, 60, 60)
        vss_short = safe(rec["vss"][:320] + ("..." if len(rec["vss"]) > 320 else ""))
        for line in textwrap.wrap(vss_short, 60)[:6]:
            pdf.set_x(TEXT_X)
            pdf.cell(TEXT_W, 4.5, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_y(max(pdf.get_y(), y_start + IMG_H) + 5)
        pdf.set_x(12)
        sc = SCORE_COLORS.get(score, (100, 100, 100))
        pdf.set_fill_color(*sc)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 9)
        bar_w = score * 30
        pdf.cell(bar_w, 7, f"  {star(score)}  {score}/5", fill=True)
        pdf.set_fill_color(220, 220, 220)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(150 - bar_w, 7, "", fill=True)
        pdf.ln(3)

        pdf.set_x(12)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 5, "Full VSS Response:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(50, 50, 50)
        for line in textwrap.wrap(safe(rec["vss"]), 100)[:12]:
            pdf.set_x(12)
            pdf.cell(0, 4, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)

    # ---- Summary table ----
    pdf.add_page()
    pdf.set_y(22)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(30, 40, 60)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 9, "  VSS Score Summary Table", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(60, 60, 80)
    pdf.set_text_color(255, 255, 255)
    for col, w in [("#", 10), ("Name", 55), ("Tag", 22), ("Score", 40), ("Annotation", 60)]:
        pdf.cell(w, 6, col, fill=True, border=1)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    for i, rec in enumerate(records):
        score = scores[i]
        bg = (245, 245, 245) if i % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*bg)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.cell(10,  5.5, str(rec["idx"]),         fill=True, border=1)
        pdf.cell(55,  5.5, rec["name"][:32],         fill=True, border=1)
        c = TAG_COLORS.get(rec["tag"], (100, 100, 100))
        pdf.set_fill_color(*c)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(22,  5.5, rec["tag"],               fill=True, border=1)
        sc = SCORE_COLORS.get(score, (180, 180, 180))
        pdf.set_fill_color(*sc)
        pdf.cell(40,  5.5, f"{star(score)} {score}/5", fill=True, border=1)
        pdf.set_fill_color(*bg)
        pdf.set_text_color(0, 0, 0)
        ann_s = safe(rec["annotation"][:45] + ("..." if len(rec["annotation"]) > 45 else ""))
        pdf.cell(60,  5.5, ann_s,                    fill=True, border=1)
        pdf.ln()

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 9)
    high = sum(1 for s in scores if s >= 4)
    med  = sum(1 for s in scores if s == 3)
    low  = sum(1 for s in scores if s <= 2)
    pdf.cell(0, 6,
             f"Average VSS Score: {avg_score:.2f}/5.0  |  "
             f"High (4-5): {high}  |  Medium (3): {med}  |  Low (1-2): {low}")

    pdf.output(output)
    size_kb = os.path.getsize(output) // 1024
    print(f"[OK] PDF saved: {output}  ({size_kb} KB, {len(records)} clips)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output",   default="vlm_dataset_report.pdf",
                        help="Output PDF path (default: vlm_dataset_report.pdf)")
    parser.add_argument("--records",  default=None,
                        help="JSON annotation records file (or pipe via stdin)")
    parser.add_argument("--clip-dir", default=DEFAULT_CLIP_DIR,
                        help="Directory containing .mp4 clip files")
    args = parser.parse_args()

    if args.records:
        with open(args.records) as f:
            records = json.load(f)
    elif not sys.stdin.isatty():
        records = json.load(sys.stdin)
    else:
        parser.error("Provide --records or pipe JSON to stdin.")

    build_pdf(records, args.output, args.clip_dir)


if __name__ == "__main__":
    main()
