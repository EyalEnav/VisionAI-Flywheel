#!/usr/bin/env python3
"""
VisionAI-Flywheel - VLM Dataset PDF Report Generator v3
- 3 frames per clip: start / mid / end
- Clickable video link per clip (VST stream URL)
"""

import argparse, json, os, subprocess, sys, textwrap
from collections import Counter

try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
except ImportError:
    print("ERROR: pip install fpdf2", file=sys.stderr); sys.exit(1)

DEFAULT_CLIP_DIR = (
    "/home/ubuntu/video-search-and-summarization/"
    "deployments/data-dir/data_log/vst/clip_storage"
)
VST_BASE = "http://3.140.195.112:30888"
FRAMES_DIR = "/tmp/pdf_frames3"

TAG_COLORS = {
    "falling":     (220, 90,  60),
    "limp":        (200, 140, 40),
    "violence":    (180, 50,  50),
    "drunk":       (100, 80,  180),
    "mixed":       (80,  140, 180),
    "crowd_push":  (200, 60,  120),
    "walking":     (60,  160, 80),
    "panic":       (220, 60,  200),
    "suspicious":  (140, 100, 20),
    "trespassing": (30,  120, 180),
}
SCORE_COLORS = {
    1: (200, 50, 50), 2: (220, 120, 40),
    3: (220, 200, 40), 4: (100, 180, 50), 5: (40, 160, 40),
}
BAD_PHRASES = [
    "does not exist", "unavailable", "cannot proceed", "not found",
    "no video", "invalid", "digital", "simulation", "floating",
    "no movement", "stationary", "no incidents", "not available",
    "is not available", "does not contain",
]

def score_vss(vss, annotation, tag):
    vss_l, ann_l = vss.lower(), annotation.lower()
    bads = sum(1 for p in BAD_PHRASES if p in vss_l)
    words = [w for w in ann_l.split() if len(w) > 4][:8]
    hits = sum(1 for w in words if w in vss_l)
    tag_hit = 1 if tag.lower() in vss_l else 0
    return min(5, max(1, round(3 + hits * 0.4 + tag_hit * 0.5 - bads * 0.7)))

def star(s):
    return "*" * s + "o" * (5 - s)

def safe(t):
    return t.encode("latin-1", errors="replace").decode("latin-1")

def get_duration(vid):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", vid],
        capture_output=True, text=True)
    try: return float(r.stdout.strip())
    except: return 4.0

def extract_frames(name, clip_dir):
    vid = os.path.join(clip_dir, f"{name}.mp4")
    if not os.path.exists(vid): return []
    dur = get_duration(vid)
    frames = []
    for label, pct in [("START", 0.10), ("MID", 0.50), ("END", 0.90)]:
        t = max(0.1, dur * pct)
        out = os.path.join(FRAMES_DIR, f"{name}_{label}.jpg")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(t), "-i", vid,
             "-frames:v", "1", "-vf", "scale=240:-1", out],
            capture_output=True)
        if os.path.exists(out) and os.path.getsize(out) > 100:
            frames.append((label, out))
    return frames

def video_url(rec, vst_base):
    """Build direct VST stream URL if video_id available, else fallback."""
    vid_id = rec.get("video_id", "")
    sensor_id = rec.get("sensor_id", "")
    if vid_id and sensor_id:
        return f"{vst_base}/vst/api/v1/sensor/{sensor_id}/video/{vid_id}/stream"
    elif vid_id:
        return f"{vst_base}/vst/api/v1/video/{vid_id}/stream"
    return ""

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


def build_pdf(records, output, clip_dir, vst_base):
    os.makedirs(FRAMES_DIR, exist_ok=True)
    scores = [score_vss(r["vss"], r["annotation"], r["tag"]) for r in records]
    tag_counts = Counter(r["tag"] for r in records)
    avg_score = sum(scores) / len(scores) if scores else 0

    pdf = DatasetPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Cover page ────────────────────────────────────────────────────────────
    pdf.set_y(28)
    pdf.set_font("Helvetica", "B", 12)
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
        pdf.cell(30, 6, f"  {tag}", fill=True)
        pdf.set_fill_color(240, 240, 240)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(10, 6, f" {cnt}", fill=True)
        pdf.cell(6, 6, "")
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7,
             f"  Avg VSS Score: {avg_score:.1f}/5.0  |  "
             f"Total pairs: {len(records)}  |  Status: reviewed",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    # ── Per-clip pages ────────────────────────────────────────────────────────
    for i, rec in enumerate(records):
        pdf.add_page()
        tag = rec["tag"]
        score = scores[i]
        color = TAG_COLORS.get(tag, (100, 100, 100))
        url = video_url(rec, vst_base)

        # Title bar
        pdf.set_fill_color(*color)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_y(22)
        pdf.cell(0, 9,
                 safe(f"  #{rec['idx']:02d}  {rec['name']}   [{tag.upper()}]   {star(score)} ({score}/5)"),
                 fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

        # ── 3 frames ──────────────────────────────────────────────────────────
        frames = extract_frames(rec["name"], clip_dir)
        IMG_W, IMG_H, GAP = 58, 40, 4
        y_frames = pdf.get_y()

        for fi, (label, fpath) in enumerate(frames):
            x = 12 + fi * (IMG_W + GAP)
            try:
                pdf.image(fpath, x=x, y=y_frames, w=IMG_W, h=IMG_H)
            except Exception as e:
                print(f"  [WARN] {rec['name']} {label}: {e}", file=sys.stderr)
            pdf.set_xy(x, y_frames + IMG_H + 1)
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(IMG_W, 4, label, align="C")

        # ── Video link button ─────────────────────────────────────────────────
        if url:
            btn_x = 12 + 3 * (IMG_W + GAP) + 4   # right of the 3 images
            btn_y = y_frames + 8
            btn_w = 210 - btn_x - 12
            pdf.set_xy(btn_x, btn_y)
            pdf.set_fill_color(30, 80, 180)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(btn_w, 10, "  >> Watch Video", fill=True, border=0,
                     align="C", link=url)
            pdf.set_xy(btn_x, btn_y + 13)
            pdf.set_font("Helvetica", "", 6)
            pdf.set_text_color(80, 80, 200)
            # Show truncated URL under button
            short_url = url[:55] + "..." if len(url) > 55 else url
            pdf.cell(btn_w, 4, safe(short_url), align="C", link=url)
        else:
            # No URL - show clip name
            btn_x = 12 + 3 * (IMG_W + GAP) + 4
            btn_y = y_frames + 8
            btn_w = 210 - btn_x - 12
            pdf.set_xy(btn_x, btn_y)
            pdf.set_font("Helvetica", "I", 7)
            pdf.set_text_color(140, 140, 140)
            pdf.cell(btn_w, 6, safe(rec['name']), align="C")

        pdf.set_y(y_frames + IMG_H + 8)
        pdf.set_text_color(0, 0, 0)

        # ── Ground truth ──────────────────────────────────────────────────────
        pdf.set_x(12)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(30, 100, 30)
        pdf.cell(0, 5, "Ground Truth Annotation:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
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
        for line in textwrap.wrap(safe(rec["vss"]), 100)[:10]:
            pdf.set_x(12)
            pdf.cell(0, 4.5, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # ── Score bar ─────────────────────────────────────────────────────────
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

    # ── Summary table ─────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_y(22)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(30, 40, 60)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 9, "  VSS Score Summary Table",
             fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(60, 60, 80)
    pdf.set_text_color(255, 255, 255)
    for col, w in [("#", 8), ("Name", 50), ("Tag", 22), ("Score", 35), ("Annotation", 55), ("Video", 22)]:
        pdf.cell(w, 6, col, fill=True, border=1)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    for i, rec in enumerate(records):
        score = scores[i]
        url = video_url(rec, vst_base)
        bg = (245, 245, 245) if i % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*bg)
        pdf.set_font("Helvetica", "", 7)
        pdf.cell(8,  5.5, str(rec["idx"]),      fill=True, border=1)
        pdf.cell(50, 5.5, rec["name"][:28],     fill=True, border=1)
        c = TAG_COLORS.get(rec["tag"], (100, 100, 100))
        pdf.set_fill_color(*c)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(22, 5.5, rec["tag"],           fill=True, border=1)
        sc = SCORE_COLORS.get(score, (180, 180, 180))
        pdf.set_fill_color(*sc)
        pdf.cell(35, 5.5, f"{star(score)} {score}/5", fill=True, border=1)
        pdf.set_fill_color(*bg)
        pdf.set_text_color(0, 0, 0)
        ann_s = safe(rec["annotation"][:40] + ("..." if len(rec["annotation"]) > 40 else ""))
        pdf.cell(55, 5.5, ann_s,                fill=True, border=1)
        # Video link cell
        if url:
            pdf.set_fill_color(30, 80, 180)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(22, 5.5, "  PLAY", fill=True, border=1, link=url)
        else:
            pdf.set_fill_color(200, 200, 200)
            pdf.set_text_color(120, 120, 120)
            pdf.cell(22, 5.5, "  -", fill=True, border=1)
        pdf.set_text_color(0, 0, 0)
        pdf.ln()

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 9)
    high = sum(1 for s in scores if s >= 4)
    med  = sum(1 for s in scores if s == 3)
    low  = sum(1 for s in scores if s <= 2)
    pdf.cell(0, 6,
             f"Avg: {avg_score:.2f}/5.0  |  High (4-5): {high}  |  "
             f"Medium (3): {med}  |  Low (1-2): {low}")

    pdf.output(output)
    kb = os.path.getsize(output) // 1024
    print(f"[OK] {output}  ({kb} KB, {len(records)} clips)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output",   default="vlm_dataset_report_v3.pdf")
    p.add_argument("--records",  default=None)
    p.add_argument("--clip-dir", default=DEFAULT_CLIP_DIR)
    p.add_argument("--vst-base", default=VST_BASE)
    args = p.parse_args()
    if args.records:
        records = json.load(open(args.records))
    elif not sys.stdin.isatty():
        records = json.load(sys.stdin)
    else:
        p.error("Provide --records or pipe JSON")
    build_pdf(records, args.output, args.clip_dir, args.vst_base)

if __name__ == "__main__":
    main()
