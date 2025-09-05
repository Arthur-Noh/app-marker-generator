#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Marker PNG bulk generator with final output scaled to 1.25x 'original' size.

- Input layouts are @3x images under assets/raws/layouts:
  {type}_{selected}.png  e.g. detail_short_selected.png

- Final output size is computed from the provided original (1x) geometry
  multiplied by 1.25x, regardless of the input layout pixel size (which is 3x).

- Operator / other icons are fitted into a SQUARE box (transparent padding),
  preserving their aspect ratio, then centered on the balloon.

- Font: assets/font/NotoSansKR-Regular.otf
- Ranges: 0~10  (FF/SS = "00".."10")

Usage:
  python src/generate.py
  python src/generate.py --type detail_short
  python src/generate.py --operator ss --selected selected --lp lp
  python src/generate.py --dry-run
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

# ====== Paths ======
BASE_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = BASE_DIR / "assets"
RAW_DIR = ASSETS_DIR / "raws"
LAYOUT_DIR = RAW_DIR / "layouts"
OP_ICON_DIR = RAW_DIR / "operator_icons"
OTHER_ICON_DIR = RAW_DIR / "other_icons"
BADGE_DIR = RAW_DIR / "badges"
FONT_PATH = ASSETS_DIR / "font" / "NotoSansKR-Regular.otf"
RESULTS_DIR = ASSETS_DIR / "results"
META_DIR = RESULTS_DIR / "_meta"

# ====== Fixed dimensions (original 1x) & scale policy ======
# Provided original sizes (width, height, innerHeight). Final output = 1.25x these originals.
GEOM_1X = {
    "simple": {
        "unselected": {"width": 28.0, "height": 32.25, "innerHeight": 28.0},
        "selected":   {"width": 34.0, "height": 38.63, "innerHeight": 34.0},
    },
    "detail_short": {
        "unselected": {"width": 50.0, "height": 35.0,  "innerHeight": 29.68},
        "selected":   {"width": 56.0, "height": 40.0,  "innerHeight": 33.92},
    },
    "detail_long": {
        "unselected": {"width": 70.0, "height": 35.0,  "innerHeight": 29.68},
        "selected":   {"width": 80.0, "height": 40.0,  "innerHeight": 33.92},
    },
}
FINAL_SCALE = 1.25  # 최종 산출물은 1.25배(원본 1x 대비)

# ====== Fixed enums & ranges ======
TYPES = ["simple", "detail_short", "detail_long"]
SELECTED_STATES = ["selected", "unselected"]
LP_STATES = ["lp", "no-lp"]
FAST_VALUES = [f"{i:02d}" for i in range(0, 11)]  # "00".."10"
SLOW_VALUES = FAST_VALUES.copy()
FALLBACK_OPERATOR = "unlinked"

# ====== Layout anchors (ratios) ======
# - All positions are ratios on the @3x canvas (0..1)
# - icon/ badge sizes are derived from innerHeight (on @3x canvas) via *_on_inner ratios
ANCHORS = {
    "simple": {
        "operator_icon_center": (0.5, 0.45),
        "icon_side_on_inner": 0.95,   # icon square side ~= 0.95 * innerHeight(@3x)
        "badge_left_top": (0.10, 0.10),
        "badge_width_on_height": 0.28,  # badge width = 0.28 * H(@3x)
    },
    "detail_short": {
        "operator_icon_center": (0.16, 0.42),
        "icon_side_on_inner": 0.95,
        "fast_text_center": (0.54, 0.40),
        "slow_text_center": (0.80, 0.40),
        "font_on_inner": 0.78,  # font size ~= 0.78 * innerHeight(@3x)
        "badge_left_top": (0.12, 0.15),
        "badge_width_on_height": 0.22,
    },
    "detail_long": {
        "operator_icon_center": (0.14, 0.44),
        "icon_side_on_inner": 0.95,
        "fast_text_center": (0.55, 0.45),
        "slow_text_center": (0.82, 0.45),
        "font_on_inner": 0.72,
        "badge_left_top": (0.12, 0.15),
        "badge_width_on_height": 0.20,
    }
}

# ====== Text style ======
TEXT_FILL = (255, 255, 255, 255)
TEXT_STROKE = (0, 0, 0, 255)
TEXT_STROKE_WIDTH = 3

def fast_label(ff: str) -> str:
    return str(int(ff))

def slow_label(ss: str) -> str:
    return str(int(ss))

# ====== Utils ======
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_rgba(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    return im

def scale_to_width(im: Image.Image, target_w: int) -> Image.Image:
    w, h = im.size
    if w == target_w:
        return im
    ratio = target_w / float(w)
    return im.resize((target_w, max(1, int(h * ratio))), Image.LANCZOS)

def scale_to_fit(im: Image.Image, target_w: int, target_h: int) -> Image.Image:
    w, h = im.size
    if w <= 0 or h <= 0:
        return im
    ratio = min(target_w / float(w), target_h / float(h))
    new_w = max(1, int(w * ratio))
    new_h = max(1, int(h * ratio))
    return im.resize((new_w, new_h), Image.LANCZOS)

def make_square_fit(im: Image.Image, side: int) -> Image.Image:
    """Preserve aspect; fit entire image into a square 'side'x'side' with transparent padding."""
    fitted = scale_to_fit(im, side, side)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    x = (side - fitted.size[0]) // 2
    y = (side - fitted.size[1]) // 2
    canvas.alpha_composite(fitted, (x, y))
    return canvas

def paste_center(base: Image.Image, overlay: Image.Image, center_xy: Tuple[float, float]) -> None:
    W, H = base.size
    x = int(W * center_xy[0]) - overlay.size[0] // 2
    y = int(H * center_xy[1]) - overlay.size[1] // 2
    base.alpha_composite(overlay, (x, y))

def paste_topleft_with_width_ratio(base: Image.Image, overlay: Image.Image, left_top_ratio: Tuple[float, float], width_ratio_on_height: float) -> None:
    """Badge: width = width_ratio_on_height * H(@3x). Keeps aspect ratio."""
    W, H = base.size
    target_w = int(H * width_ratio_on_height)
    ov = scale_to_width(overlay, max(1, target_w))
    x = int(W * left_top_ratio[0])
    y = int(H * left_top_ratio[1])
    base.alpha_composite(ov, (x, y))

def draw_centered_text(base: Image.Image, text: str, center_xy: Tuple[float, float], font: ImageFont.FreeTypeFont) -> None:
    W, H = base.size
    x = int(W * center_xy[0])
    y = int(H * center_xy[1])
    draw = ImageDraw.Draw(base)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=TEXT_STROKE_WIDTH)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pos = (x - tw // 2, y - th // 2)
    draw.text(pos, text, font=font, fill=TEXT_FILL, stroke_fill=TEXT_STROKE, stroke_width=TEXT_STROKE_WIDTH)

# ====== Geometry helpers ======
def geom_for(marker_type: str, selected: str) -> dict:
    return GEOM_1X[marker_type][selected]

def final_output_size(marker_type: str, selected: str) -> Tuple[int, int]:
    g = geom_for(marker_type, selected)
    w = int(round(g["width"] * FINAL_SCALE))
    h = int(round(g["height"] * FINAL_SCALE))
    return max(1, w), max(1, h)

def inner_height_at_3x(marker_type: str, selected: str) -> int:
    """innerHeight (1x) -> multiply by 3 for the layout canvas scale."""
    g = geom_for(marker_type, selected)
    return int(round(g["innerHeight"] * 3.0))

# ====== Path helpers ======
def layout_path(marker_type: str, selected_state: str) -> Path:
    # e.g. detail_short_selected.png
    return LAYOUT_DIR / f"{marker_type}_{selected_state}.png"

def output_path_simple(operator: str, selected_state: str, lp_state: str) -> Path:
    return RESULTS_DIR / "simple" / operator / selected_state / lp_state / "base.png"

def output_path_detail(marker_type: str, operator: str, selected_state: str, lp_state: str, ff: str, ss: str) -> Path:
    return RESULTS_DIR / marker_type / operator / selected_state / lp_state / f"f{ff}_s{ss}.png"

def read_operator_codes() -> List[str]:
    if not OP_ICON_DIR.exists():
        return []
    return sorted([p.stem for p in OP_ICON_DIR.glob("*.png")])

def pick_operator_icon(operator: str) -> Path:
    cand = OP_ICON_DIR / f"{operator}.png"
    if cand.exists():
        return cand
    fb = OTHER_ICON_DIR / f"{FALLBACK_OPERATOR}.png"
    return fb if fb.exists() else cand

# ====== Composition ======
def compose_simple(operator: str, selected: str, lp: str, dry: bool=False) -> Optional[Path]:
    mtype = "simple"
    base_img_path = layout_path(mtype, selected)
    out_fp = output_path_simple(operator, selected, lp)

    if dry:
        print("[DRY] SIMPLE", operator, selected, lp, "->", out_fp)
        return out_fp

    ensure_dir(out_fp.parent)
    base = load_rgba(base_img_path)  # @3x canvas

    # operator icon into square box (based on innerHeight@3x)
    inner3 = inner_height_at_3x(mtype, selected)
    icon_side = int(round(inner3 * ANCHORS[mtype]["icon_side_on_inner"]))
    icon = load_rgba(pick_operator_icon(operator))
    icon_sq = make_square_fit(icon, max(1, icon_side))
    paste_center(base, icon_sq, ANCHORS[mtype]["operator_icon_center"])

    # lp badge
    if lp == "lp":
        badge_fp = BADGE_DIR / "luckypass.png"
        if badge_fp.exists():
            badge = load_rgba(badge_fp)
            paste_topleft_with_width_ratio(base, badge, ANCHORS[mtype]["badge_left_top"], ANCHORS[mtype]["badge_width_on_height"])

    # final resize to 1.25x of 1x original
    final_w, final_h = final_output_size(mtype, selected)
    base = base.resize((final_w, final_h), Image.LANCZOS)
    base.save(out_fp, format="PNG", optimize=True)
    return out_fp

def compose_detail(marker_type: str, operator: str, selected: str, lp: str, ff: str, ss: str, dry: bool=False) -> Optional[Path]:
    base_img_path = layout_path(marker_type, selected)
    out_fp = output_path_detail(marker_type, operator, selected, lp, ff, ss)

    if dry:
        print("[DRY]", marker_type, operator, selected, lp, ff, ss, "->", out_fp)
        return out_fp

    ensure_dir(out_fp.parent)
    base = load_rgba(base_img_path)  # @3x canvas

    # operator icon as square (innerHeight@3x)
    inner3 = inner_height_at_3x(marker_type, selected)
    icon_side = int(round(inner3 * ANCHORS[marker_type]["icon_side_on_inner"]))
    icon = load_rgba(pick_operator_icon(operator))
    icon_sq = make_square_fit(icon, max(1, icon_side))
    paste_center(base, icon_sq, ANCHORS[marker_type]["operator_icon_center"])

    # badge
    if lp == "lp":
        badge_fp = BADGE_DIR / "luckypass.png"
        if badge_fp.exists():
            badge = load_rgba(badge_fp)
            paste_topleft_with_width_ratio(base, badge, ANCHORS[marker_type]["badge_left_top"], ANCHORS[marker_type]["badge_width_on_height"])

    # text (fast/slow) with font size based on innerHeight@3x
    font_px = max(8, int(round(inner3 * ANCHORS[marker_type]["font_on_inner"])))
    font = ImageFont.truetype(str(FONT_PATH), font_px)
    draw_centered_text(base, fast_label(ff), ANCHORS[marker_type]["fast_text_center"], font)
    draw_centered_text(base, slow_label(ss), ANCHORS[marker_type]["slow_text_center"], font)

    # final resize to 1.25x of 1x original
    final_w, final_h = final_output_size(marker_type, selected)
    base = base.resize((final_w, final_h), Image.LANCZOS)
    base.save(out_fp, format="PNG", optimize=True)
    return out_fp

# ====== Generation ======
def generate_all(
    marker_types: List[str],
    operators: List[str],
    selected_states: List[str],
    lp_states: List[str],
    fast_values: List[str],
    slow_values: List[str],
    dry_run: bool=False
) -> int:
    count = 0
    for t in marker_types:
        for op in operators:
            for sel in selected_states:
                for lp in lp_states:
                    if t == "simple":
                        compose_simple(op, sel, lp, dry=dry_run); count += 1
                    else:
                        for ff in fast_values:
                            for ss in slow_values:
                                compose_detail(t, op, sel, lp, ff, ss, dry=dry_run); count += 1
    return count

def write_manifest(
    marker_types: List[str],
    operators: List[str],
    selected_states: List[str],
    lp_states: List[str],
    fast_values: List[str],
    slow_values: List[str],
):
    ensure_dir(META_DIR)
    manifest = {
        "types": marker_types,
        "states": selected_states,
        "lp": lp_states,
        "fastValues": fast_values,
        "slowValues": slow_values,
        "fallbackOperator": FALLBACK_OPERATOR,
        "template": {
            "simple": "assets/results/simple/{operator}/{state}/{lp}/base.png",
            "detail": "assets/results/{type}/{operator}/{state}/{lp}/f{FF}_s{SS}.png"
        },
        "finalScale": FINAL_SCALE,
        "geomOriginal1x": GEOM_1X
    }
    with open(META_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

# ====== CLI ======
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Generate marker PNGs into assets/results/ with 1.25x final size.")
    p.add_argument("--type", choices=TYPES + ["all"], default="all")
    p.add_argument("--operator", action="append", help="Generate only for specific operator code(s).")
    p.add_argument("--selected", choices=["selected", "unselected", "both"], default="both")
    p.add_argument("--lp", choices=["lp", "no-lp", "both"], default="both")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-manifest", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    marker_types = TYPES if args.type == "all" else [args.type]
    operators = args.operator if args.operator else (read_operator_codes() or [FALLBACK_OPERATOR])
    selected_states = SELECTED_STATES if args.selected == "both" else [args.selected]
    lp_states = LP_STATES if args.lp == "both" else [args.lp]

    ensure_dir(RESULTS_DIR)
    total = generate_all(marker_types, operators, selected_states, lp_states, FAST_VALUES, SLOW_VALUES, dry_run=args.dry_run)

    if not args.no_manifest and not args.dry_run:
        write_manifest(marker_types, operators, selected_states, lp_states, FAST_VALUES, SLOW_VALUES)

    print(f"Done. {'(dry-run) ' if args.dry_run else ''}Generated plans/files: {total}")

if __name__ == "__main__":
    main()
