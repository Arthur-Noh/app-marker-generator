#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate EXACTLY ONE marker PNG (debug)

- detail_short filename: fast -> f{FF}.png, slow -> s{SS}.png
- LP badge: canvas grows when lp==lp → +4px (1x) to left/top (+12px @3x)
- Operator icon/text: type-specific anchors + pixel shifts (independent)
- Text: no stroke, black; unselected=10px Regular, selected=12px Medium(500 if available)
- Labels: '급{n}', '완{n}'
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import argparse
import sys

# ===== Paths =====
BASE_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = BASE_DIR / "assets"
RAW_DIR = ASSETS_DIR / "raws"
LAYOUT_DIR = RAW_DIR / "layouts"
OP_ICON_DIR = RAW_DIR / "operator_icons"
OTHER_ICON_DIR = RAW_DIR / "other_icons"
BADGE_DIR = RAW_DIR / "badges"
FONT_REGULAR = ASSETS_DIR / "font" / "NotoSansKR-Regular.otf"
FONT_MEDIUM  = ASSETS_DIR / "font" / "NotoSansKR-Medium.otf"  # 선택 상태에서 있으면 사용
RESULTS_DIR = ASSETS_DIR / "results"

# ===== Geometry (1x) & final scale =====
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
FINAL_SCALE = 1.25  # final vs 1x

# ===== LP badge absolute size (1x) & offsets =====
BADGE_SIZE_1X         = (30, 15)   # 1x
CANVAS_PAD_1X_WHEN_LP = (4, 4)     # lp일 때 캔버스 (좌,상) 확장(1x)

# 타입별 배지 오프셋(1x) — simple만 더 왼쪽
BADGE_OFFSET_1X_BY_TYPE = {
    "simple":       (-4, -4),
    "detail_short": ( 0, -4),
    "detail_long":  ( 0, -4),
}

def to_canvas_px_3x(px_1x: int) -> int:
    return int(round(px_1x * 3))

def badge_size_canvas() -> Tuple[int,int]:
    w1x, h1x = BADGE_SIZE_1X
    return to_canvas_px_3x(w1x), to_canvas_px_3x(h1x)

def badge_offset_canvas_for(marker_type: str) -> Tuple[int,int]:
    ox, oy = BADGE_OFFSET_1X_BY_TYPE.get(marker_type, (0, -4))
    return to_canvas_px_3x(ox), to_canvas_px_3x(oy)

def canvas_pad_when_lp() -> Tuple[int,int]:
    px, py = CANVAS_PAD_1X_WHEN_LP
    return to_canvas_px_3x(px), to_canvas_px_3x(py)

# ===== Enums & defaults =====
TYPES = ["simple", "detail_short", "detail_long"]
SELECTED_STATES = ["selected", "unselected"]
LP_STATES = ["lp", "no-lp"]
FALLBACK_OPERATOR = "unlinked"

# ===== Anchors (ratios on base @3x) =====
# 아이콘/텍스트를 각각 독립 좌표(x,y)로 배치
ANCHORS = {
    "simple": {
        "operator_icon_center": (0.50, 0.50),
        "icon_side_on_inner": 0.75,   # 작게 (기존 0.95 -> 0.85)
    },
    "detail_short": {
        "operator_icon_center": (0.16, 0.42),
        "text_single_center":   (0.67, 0.40),
        "icon_side_on_inner":   0.75,
    },
    "detail_long": {
        "operator_icon_center": (0.14, 0.44),  # 비율은 유지, 대신 픽셀 시프트로 오른쪽 이동
        "fast_text_center":     (0.55, 0.45),
        "slow_text_center":     (0.82, 0.45),
        "icon_side_on_inner":   0.75,
    }
}

# 아이콘 X 시프트(오른쪽 +, 1x px)
ICON_RIGHT_SHIFT_1X = {
    "simple": 0,
    "detail_short": 10,
    "detail_long": 10,   # → detail_long 아이콘을 더 오른쪽으로
}

# 아이콘 Y 시프트(위로 -, 1x px)
ICON_Y_SHIFT_1X = {
    "simple": -2,
    "detail_short": -2,
    "detail_long":  -2,
}

# 텍스트 시프트(좌/상, 1x px) — detail_long 텍스트만 왼쪽/위쪽으로 더 이동
TEXT_SHIFT_X_1X = {
    "simple": 0,
    "detail_short": 0,
    "detail_long": -4,  # ← 왼쪽으로
}
TEXT_SHIFT_Y_1X = {
    "simple": -2,
    "detail_short": -3,
    "detail_long": -4,  # ↑ 위로
}

# ===== Text style =====
TEXT_FILL = (17, 17, 17, 255)  # 검정(#111)

def canvas_font_px_from_final(final_px: int) -> int:
    return max(8, int(round(final_px * 3.0 / FINAL_SCALE)))

def pick_font_path(selected: str) -> Path:
    if selected == "selected" and FONT_MEDIUM.exists():
        return FONT_MEDIUM
    return FONT_REGULAR

def fast_label(n: int) -> str:
    return f"급{int(n)}"

def slow_label(n: int) -> str:
    return f"완{int(n)}"

# ===== Utils =====
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_rgba(path: Path) -> Image.Image:
    if not path.exists():
        sys.exit(f"[ERR] Not found: {path}")
    return Image.open(path).convert("RGBA")

def scale_to_fit(im: Image.Image, target_w: int, target_h: int) -> Image.Image:
    w, h = im.size
    ratio = min(target_w / float(w), target_h / float(h))
    return im.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.LANCZOS)

def make_square_fit(im: Image.Image, side: int) -> Image.Image:
    fitted = scale_to_fit(im, side, side)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    x = (side - fitted.size[0]) // 2
    y = (side - fitted.size[1]) // 2
    canvas.paste(fitted, (x, y), fitted)
    return canvas

def center_from_anchor(base_origin: Tuple[int,int], base_size: Tuple[int,int],
                       anchor_x: float, anchor_y: float,
                       extra_dx_px: int=0, extra_dy_px: int=0) -> Tuple[int,int]:
    ox, oy = base_origin; bw, bh = base_size
    cx = ox + int(bw * anchor_x) + extra_dx_px
    cy = oy + int(bh * anchor_y) + extra_dy_px
    return cx, cy

def paste_center_at(canvas: Image.Image, overlay: Image.Image, center_xy: Tuple[int,int]) -> None:
    x = center_xy[0] - overlay.size[0] // 2
    y = center_xy[1] - overlay.size[1] // 2
    canvas.paste(overlay, (x, y), overlay)

def paste_topleft_abs(canvas: Image.Image, overlay: Image.Image, topleft_xy: Tuple[int,int]) -> None:
    canvas.paste(overlay, topleft_xy, overlay)

def draw_centered_text_at(canvas: Image.Image, text: str, center_xy: Tuple[int,int], font: ImageFont.FreeTypeFont) -> None:
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
    pos = (center_xy[0] - tw // 2, center_xy[1] - th // 2)
    draw.text(pos, text, font=font, fill=TEXT_FILL)

# ===== Geometry helpers =====
def final_output_size(marker_type: str, selected: str, lp: str) -> Tuple[int, int]:
    g = GEOM_1X[marker_type][selected]
    add_w_1x = CANVAS_PAD_1X_WHEN_LP[0] if lp == "lp" else 0
    add_h_1x = CANVAS_PAD_1X_WHEN_LP[1] if lp == "lp" else 0
    w = int(round((g["width"]  + add_w_1x) * FINAL_SCALE))
    h = int(round((g["height"] + add_h_1x) * FINAL_SCALE))
    return max(1, w), max(1, h)

def inner_height_at_3x(marker_type: str, selected: str) -> int:
    g = GEOM_1X[marker_type][selected]
    return int(round(g["innerHeight"] * 3.0))

# ===== Paths =====
def layout_path(marker_type: str, selected_state: str) -> Path:
    return LAYOUT_DIR / f"{marker_type}_{selected_state}.png"

def output_path_simple(operator: str, selected_state: str, lp_state: str) -> Path:
    return RESULTS_DIR / "simple" / operator / selected_state / lp_state / "base.png"

def output_filename_detail(marker_type: str, ff: int, ss: int, mode_short: str) -> str:
    if marker_type == "detail_short":
        return f"f{ff:02d}.png" if mode_short == "fast" else f"s{ss:02d}.png"
    return f"f{ff:02d}_s{ss:02d}.png"

def output_path_detail(marker_type: str, operator: str, selected_state: str, lp_state: str, ff: int, ss: int, mode_short: str) -> Path:
    return RESULTS_DIR / marker_type / operator / selected_state / lp_state / output_filename_detail(marker_type, ff, ss, mode_short)

def pick_operator_icon(operator: str) -> Path:
    cand = OP_ICON_DIR / f"{operator}.png"
    if cand.exists(): return cand
    fb = OTHER_ICON_DIR / f"{FALLBACK_OPERATOR}.png"
    return fb if fb.exists() else cand

# ===== Compose (one image) =====
def compose_simple(operator: str, selected: str, lp: str) -> Optional[Path]:
    mtype = "simple"
    layout = load_rgba(layout_path(mtype, selected))  # base @3x
    bw, bh = layout.size

    pad_left, pad_top = canvas_pad_when_lp() if lp == "lp" else (0, 0)
    canvas = Image.new("RGBA", (bw + pad_left, bh + pad_top), (0, 0, 0, 0))
    base_origin = (pad_left, pad_top)
    canvas.paste(layout, base_origin, layout)

    # icon
    inner3 = inner_height_at_3x(mtype, selected)
    icon_side = int(round(inner3 * ANCHORS[mtype]["icon_side_on_inner"]))
    icon_sq = make_square_fit(load_rgba(pick_operator_icon(operator)), max(1, icon_side))
    shift_x = to_canvas_px_3x(ICON_RIGHT_SHIFT_1X[mtype])
    shift_y = to_canvas_px_3x(ICON_Y_SHIFT_1X[mtype])
    ax, ay = ANCHORS[mtype]["operator_icon_center"]
    center = center_from_anchor(base_origin, (bw, bh), ax, ay, extra_dx_px=shift_x, extra_dy_px=shift_y)
    paste_center_at(canvas, icon_sq, center)

    # LP badge
    if lp == "lp":
        badge_w, badge_h = badge_size_canvas()
        badge = load_rgba(BADGE_DIR / "luckypass.png").resize((badge_w, badge_h), Image.LANCZOS)
        bx_off, by_off = badge_offset_canvas_for(mtype)
        bx = base_origin[0] + bx_off
        by = base_origin[1] + by_off
        paste_topleft_abs(canvas, badge, (bx, by))

    # final resize
    out_fp = output_path_simple(operator, selected, lp)
    ensure_dir(out_fp.parent)
    final_w, final_h = final_output_size(mtype, selected, lp)
    canvas.resize((final_w, final_h), Image.LANCZOS).save(out_fp, format="PNG", optimize=True)
    print("[OK]", out_fp)
    return out_fp

def compose_detail(marker_type: str, operator: str, selected: str, lp: str, ff: int, ss: int, mode_short: str="fast") -> Optional[Path]:
    layout = load_rgba(layout_path(marker_type, selected))  # base @3x
    bw, bh = layout.size

    pad_left, pad_top = canvas_pad_when_lp() if lp == "lp" else (0, 0)
    canvas = Image.new("RGBA", (bw + pad_left, bh + pad_top), (0, 0, 0, 0))
    base_origin = (pad_left, pad_top)
    canvas.paste(layout, base_origin, layout)

    # icon
    inner3 = inner_height_at_3x(marker_type, selected)
    icon_side = int(round(inner3 * ANCHORS[marker_type]["icon_side_on_inner"]))
    icon_sq = make_square_fit(load_rgba(pick_operator_icon(operator)), max(1, icon_side))
    shift_x = to_canvas_px_3x(ICON_RIGHT_SHIFT_1X[marker_type])
    shift_y = to_canvas_px_3x(ICON_Y_SHIFT_1X[marker_type])
    ax, ay = ANCHORS[marker_type]["operator_icon_center"]
    center = center_from_anchor(base_origin, (bw, bh), ax, ay, extra_dx_px=shift_x, extra_dy_px=shift_y)
    paste_center_at(canvas, icon_sq, center)

    # LP badge
    if lp == "lp":
        badge_w, badge_h = badge_size_canvas()
        badge = load_rgba(BADGE_DIR / "luckypass.png").resize((badge_w, badge_h), Image.LANCZOS)
        bx_off, by_off = badge_offset_canvas_for(marker_type)
        bx = base_origin[0] + bx_off
        by = base_origin[1] + by_off
        paste_topleft_abs(canvas, badge, (bx, by))

    # text
    final_px  = 12 if selected == "selected" else 10
    font_px   = canvas_font_px_from_final(final_px)
    font_path = pick_font_path(selected)
    font      = ImageFont.truetype(str(font_path), font_px)
    tdx = to_canvas_px_3x(TEXT_SHIFT_X_1X[marker_type])
    tdy = to_canvas_px_3x(TEXT_SHIFT_Y_1X[marker_type])

    if marker_type == "detail_short":
        text = fast_label(ff) if mode_short == "fast" else slow_label(ss)
        tx, ty = ANCHORS[marker_type]["text_single_center"]
        cxy = center_from_anchor(base_origin, (bw, bh), tx, ty, extra_dx_px=tdx, extra_dy_px=tdy)
        draw_centered_text_at(canvas, text, cxy, font)
    else:
        fx, fy = ANCHORS[marker_type]["fast_text_center"]
        sx, sy = ANCHORS[marker_type]["slow_text_center"]
        c1 = center_from_anchor(base_origin, (bw, bh), fx, fy, extra_dx_px=tdx, extra_dy_px=tdy)
        c2 = center_from_anchor(base_origin, (bw, bh), sx, sy, extra_dx_px=tdx, extra_dy_px=tdy)
        draw_centered_text_at(canvas, fast_label(ff), c1, font)
        draw_centered_text_at(canvas, slow_label(ss), c2, font)

    # final resize
    out_fp = output_path_detail(marker_type, operator, selected, lp, ff, ss, mode_short)
    ensure_dir(out_fp.parent)
    final_w, final_h = final_output_size(marker_type, selected, lp)
    canvas.resize((final_w, final_h), Image.LANCZOS).save(out_fp, format="PNG", optimize=True)
    print("[OK]", out_fp)
    return out_fp

# ===== CLI (single only) =====
def parse_args():
    p = argparse.ArgumentParser(description="Generate exactly ONE marker image (type-specific positioning).")
    p.add_argument("--type", choices=TYPES, default="detail_short")
    p.add_argument("--operator", required=False, help="operator code (ex: ss, ke). If omitted, tries first icon or 'unlinked'.")
    p.add_argument("--selected", choices=SELECTED_STATES, default="selected")
    p.add_argument("--lp", choices=LP_STATES, default="lp")
    p.add_argument("--fast", type=int, default=3, help="0~99 (detail_* only)")
    p.add_argument("--slow", type=int, default=7, help="0~99 (detail_* only)")
    p.add_argument("--mode", choices=["fast","slow"], default="fast", help="detail_short에서 어떤 값만 표기할지")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()

def autodetect_operator() -> str:
    pngs = sorted(OP_ICON_DIR.glob("*.png"))
    if pngs:
        return pngs[0].stem
    return FALLBACK_OPERATOR

def main():
    args = parse_args()
    operator = args.operator or autodetect_operator()
    ff = max(0, min(99, args.fast))
    ss = max(0, min(99, args.slow))

    if args.dry_run:
        print("[DRY] one:", dict(type=args.type, operator=operator, selected=args.selected, lp=args.lp, fast=ff, slow=ss, mode=args.mode))
        print("=> filename:", output_filename_detail(args.type, ff, ss, args.mode) if args.type!='simple' else "base.png")
        return

    if args.type == "simple":
        compose_simple(operator, args.selected, args.lp)
    else:
        compose_detail(args.type, operator, args.selected, args.lp, ff, ss, mode_short=args.mode)

if __name__ == "__main__":
    main()
