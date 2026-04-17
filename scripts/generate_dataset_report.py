#!/usr/bin/env python3
"""
VisionAI-Flywheel - VLM Dataset PDF Report Generator

Generates a per-clip PDF report with 3 video frame thumbnails (start/mid/end)
and VSS accuracy scores.

Usage:
    python3 scripts/generate_dataset_report.py \
        --records data/annotations.json \
        --output  vlm_dataset_report.pdf \
        --clip-dir /path/to/vst/clip_storage

Requirements:
    pip install fpdf2 pillow
    sudo apt install ffmpeg
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

DEFAULT_CLIP_DIR = (
    "/home/ubuntu/video-search-and-summarization/"
    "deployments/data-dir/data_log/vst/clip_storage"
)
FRAMES_DIR = "/tmp/pdf_frames3"

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


def score_vss(vss: str, annotation: str, tag: str) -> int:
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


def get_video_duration(vid_path: str) -> float:
    """Return video duration in seconds using ffprobe."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", vid_path],
        capture_output=True, text=True
    )
    try:
        return float(r.stdout.strip())
    except Exception:
        return 4.0  # fallback


def extract_frames(name: str, clip_dir: str) -> list:
    """
    Extract 3 frames: start (10%), mid (50%), end (90%) of video duration.
    Returns list of existing file paths (may be shorter if extraction fails).
    """
    vid = os.path.join(clip_dir, f"{name}.mp4")
    if not os.path.exists(vid):
        return []

    duration = get_video_duration(vid)
    timestamps = {
        "start": max(0.1, duration * 0.10),
        "mid":   duration * 0.50,
        "end":   max(0.1, duration * 0.90),
    }

    paths = []
    for label, t in timestamps.items():
        out = os.path.join(FRAMES_DIR, f"{name}_{label}.jpg")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(t), "-i", vid,
             "-frames:v", "1", "-vf", "scale=240:-1", out],
            capture_output=True,
        )
        if os.path.exists(out) and os.path.getsize(out) > 100:
            paths.append((label, out))
    return paths  # list of (label, path)


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
    pdf.cell(0, 7,
             f"  Avg VSS Score: {avg_score:.1f}/5.0   |   Total pairs: {len(records)}   |   Status: reviewed",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    # ---- Per-clip pages ----
    for i, rec in enumerate(records):
        pdf.add_page()
        tag = rec["tag"]
        score = scores[i]
        color = TAG_COLORS.get(tag, (100, 100, 100))

        # Title bar
        pdf.set_fill_color(*color)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_y(22)
        pdf.cell(0, 9,
                 safe(f"  #{rec['idx']:02d}  {rec['name']}   [{tag.upper()}]   VSS Score: {star(score)} ({score}/5)"),
                 fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

        # Extract 3 frames
        frames = extract_frames(rec["name"], clip_dir)

        # Place frames side by side: start | mid | end
        IMG_W = 58
        IMG_H = 40
        GAP = 4
        FRAME_AREA_H = IMG_H + 10  # image + label
        y_frames = pdf.get_y()

        if frames:
            for fi, (label, fpath) in enumerate(frames):
                x = 12 + fi * (IMG_W + GAP)
                try:
                    pdf.image(fpath, x=x, y=y_frames, w=IMG_W, h=IMG_H)
                except Exception as e:
                    print(f"  [WARN] {rec['name']} {label}: {e}", file=sys.stderr)
                # Label below image
                pdf.set_xy(x, y_frames + IMG_H + 1)
                pdf.set_font("Helvetica", "B", 7)
                pdf.set_text_color(80, 80, 80)
                pdf.cell(IMG_W, 4, label.upper(), align="C")

        pdf.set_y(y_frames + FRAME_AREA_H + 4)
        pdf.set_text_color(0, 0, 0)

        # Ground truth annotation
        pdf.set_x(12)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(30, 100, 30)
        pdf.cell(0, 5, "Ground Truth Annotation:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(12)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(0, 0, 0)
        for line in textwrap.wrap(safe(rec["annotation"]), 95):
            pdf.set_x(12)
            pdf.cell(0, 5, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.ln(2)
        pdf.set_x(12)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(150, 60, 20)
        pdf.cell(0, 5, "VSS Response:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(60, 60, 60)
        vss_text = safe(rec["vss"])
        for line in textwrap.wrap(vss_text, 100)[:10]:
            pdf.set_x(12)
            pdf.cell(0, 4.5, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Score bar
        pdf.ln(3)
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
        pdf.set_text_color(0, 0, 0)

    # ---- Summary table ----
    pdf.add_page()
    pdf.set_y(22)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(30, 40, 60)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 9, "  VSS Score Summary Table", fill=True,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
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
        pdf.cell(10, 5.5, str(rec["idx"]),   fill=True, border=1)
        pdf.cell(55, 5.5, rec["name"][:32],  fill=True, border=1)
        c = TAG_COLORS.get(rec["tag"], (100, 100, 100))
        pdf.set_fill_color(*c)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(22, 5.5, rec["tag"],        fill=True, border=1)
        sc = SCORE_COLORS.get(score, (180, 180, 180))
        pdf.set_fill_color(*sc)
        pdf.cell(40, 5.5, f"{star(score)} {score}/5", fill=True, border=1)
        pdf.set_fill_color(*bg)
        pdf.set_text_color(0, 0, 0)
        ann_s = safe(rec["annotation"][:45] + ("..." if len(rec["annotation"]) > 45 else ""))
        pdf.cell(60, 5.5, ann_s,             fill=True, border=1)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output",   default="vlm_dataset_report.pdf")
    parser.add_argument("--records",  default=None)
    parser.add_argument("--clip-dir", default=DEFAULT_CLIP_DIR)
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
