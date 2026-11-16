import streamlit as st
import streamlit.components.v1 as components
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import tempfile
import os
import io
import zipfile
import urllib.request
import datetime
import subprocess
import imageio_ffmpeg
import base64
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="ゴルフスイングAI自動分析ツール", layout="wide")

MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"

def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("初回起動のためAI骨格解析モデルをダウンロードしています（約30MB・初回のみ）..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

P_DEFINITIONS = [
    ("P1", "アドレス", True, True),
    ("P2", "テイクバック", False, False),
    ("P3", "ハーフウェイバック", False, False),
    ("P4", "トップ", True, True),
    ("P5", "切り返し", False, False),
    ("P6", "デリバリー", False, False),
    ("P7", "インパクト", True, True),
    ("P8", "フォロー", False, False),
    ("P9", "ハイフォロー", False, False),
    ("P10", "フィニッシュ", False, False),
]

P_NAMES_LIST = [p[0] for p in P_DEFINITIONS]
P_LABEL_MAP = {p[0]: f"{p[0]} {p[1]}" for p in P_DEFINITIONS}

DEFAULT_STYLES = {
    "head": {"color": "#00FFFF", "width": 2, "style": "実線"},
    "spine": {"color": "#FFFF00", "width": 4, "style": "実線"},
    "spine_tilt": {"color": "#FFA500", "width": 4, "style": "実線"},
    "shoulders": {"color": "#00FF00", "width": 3, "style": "実線"},
    "shaft": {"color": "#FFFFFF", "width": 3, "style": "実線"},
    "hip": {"color": "#FFD700", "width": 3, "style": "点線"},
    "ankles": {"color": "#FFFF00", "width": 3, "style": "実線"},
    "vertical": {"color": "#00DCFF", "width": 2, "style": "点線"},
}

def is_frame_in_selected_positions(frame_idx, selected_p_codes, current_p_frames):
    if not selected_p_codes:
        return True
    target_indices = {current_p_frames.get(p_code) for p_code in selected_p_codes}
    return frame_idx in target_indices

def filter_custom_shapes_for_frame(frame_idx, custom_shapes, global_selected_p_codes, current_p_frames):
    if not custom_shapes:
        return []
    visible_shapes = []
    global_visible = is_frame_in_selected_positions(frame_idx, global_selected_p_codes, current_p_frames)
    for sh in custom_shapes:
        target_p = sh.get("target_p")
        if target_p:
            if frame_idx == current_p_frames.get(target_p):
                visible_shapes.append(sh)
        else:
            if global_visible:
                visible_shapes.append(sh)
    return visible_shapes

def get_gothic_font(size=21):
    font_candidates = [
        "C:/Windows/Fonts/YuGothB.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in font_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()

def calculate_spine_angle_horizontal(shoulder_mid, hip_mid):
    dx = abs(shoulder_mid[0] - hip_mid[0])
    dy = abs(hip_mid[1] - shoulder_mid[1])
    angle_rad = math.atan2(dy, dx)
    return int(round(math.degrees(angle_rad)))

def calculate_spine_tilt_frontal(shoulder_mid, hip_mid, target_dir_x):
    dx = shoulder_mid[0] - hip_mid[0]
    dy = abs(hip_mid[1] - shoulder_mid[1])
    if dy == 0:
        return 0
    angle_deg = math.degrees(math.atan2(abs(dx), dy))
    signed_angle = int(round(angle_deg))
    if signed_angle == 0:
        return 0
    if (dx * target_dir_x) > 0:
        return signed_angle
    else:
        return -signed_angle

def calculate_line_angle_horizontal(p1, p2):
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    if dx == 0 and dy == 0:
        return 0
    return int(round(math.degrees(math.atan2(dy, dx))))

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return (0, 0, 255)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return (255, 255, 255)
    return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))

def auto_detect_p1_to_p10(frame_data_list, fps):
    num_frames = len(frame_data_list)
    if num_frames < 10:
        step = max(1, num_frames // 10)
        return {f"P{i}": min((i - 1) * step, max(0, num_frames - 1)) for i in range(1, 11)}

    valid_frames = [(i, d) for i, d in enumerate(frame_data_list) if d is not None]
    if len(valid_frames) < 10:
        return {f"P{i}": int((num_frames - 1) * (i - 1) / 9) for i in range(1, 11)}

    search_limit = max(10, int(num_frames * 0.75))
    candidates_for_top = [(i, d) for i, d in valid_frames if i < search_limit]
    if not candidates_for_top:
        candidates_for_top = valid_frames
    p4_idx = min(candidates_for_top, key=lambda x: x[1]["wrist_mid"][1])[0]

    p1_idx = 0
    min_speed = 99999.0
    start_addr = max(1, int(fps * 0.05))
    end_addr = max(start_addr + 1, p4_idx - max(3, int(fps * 0.15)))
    for i in range(start_addr, end_addr):
        curr = frame_data_list[i]
        prev = frame_data_list[i - 1]
        if curr is not None and prev is not None:
            if curr["wrist_mid"][1] > curr["hip_mid"][1] * 0.85:
                speed = math.dist(prev["wrist_mid"], curr["wrist_mid"])
                if speed < min_speed:
                    min_speed = speed
                    p1_idx = i

    impact_start = min(p4_idx + max(2, int(fps * 0.08)), num_frames - 1)
    impact_end = min(p4_idx + int(fps * 1.2), num_frames)
    impact_candidates = [
        (i, frame_data_list[i])
        for i in range(impact_start, impact_end)
        if frame_data_list[i] is not None
    ]
    if impact_candidates:
        p7_idx = max(impact_candidates, key=lambda x: x[1]["wrist_mid"][1])[0]
    else:
        p7_idx = min(p4_idx + max(3, int(fps * 0.25)), num_frames - 1)

    finish_start = min(p7_idx + max(2, int(fps * 0.1)), num_frames - 1)
    finish_candidates = [
        (i, frame_data_list[i])
        for i in range(finish_start, num_frames)
        if frame_data_list[i] is not None
    ]
    if finish_candidates:
        p10_idx = min(finish_candidates, key=lambda x: x[1]["wrist_mid"][1])[0]
    else:
        p10_idx = num_frames - 1

    p2_idx = int(round(p1_idx + (p4_idx - p1_idx) * 0.45))
    p3_idx = int(round(p1_idx + (p4_idx - p1_idx) * 0.75))
    p5_idx = int(round(p4_idx + (p7_idx - p4_idx) * 0.35))
    p6_idx = int(round(p4_idx + (p7_idx - p4_idx) * 0.70))
    p8_idx = int(round(p7_idx + (p10_idx - p7_idx) * 0.30))
    p9_idx = int(round(p7_idx + (p10_idx - p7_idx) * 0.65))

    return {
        "P1": p1_idx,
        "P2": p2_idx,
        "P3": p3_idx,
        "P4": p4_idx,
        "P5": p5_idx,
        "P6": p6_idx,
        "P7": p7_idx,
        "P8": p8_idx,
        "P9": p9_idx,
        "P10": p10_idx,
    }

def draw_styled_line(img, p1, p2, color_bgr, thickness=3, style="実線", dash_len=14):
    if style != "点線":
        cv2.line(img, p1, p2, color_bgr, thickness, cv2.LINE_AA)
        return
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if dist < 1:
        return
    step = dash_len * 2
    num_steps = int(dist // step) + 1
    for i in range(num_steps):
        t_start = (i * step) / dist
        t_end = min(1.0, (i * step + dash_len) / dist)
        if t_start >= 1.0:
            break
        sx = int(round(p1[0] + (p2[0] - p1[0]) * t_start))
        sy = int(round(p1[1] + (p2[1] - p1[1]) * t_start))
        ex = int(round(p1[0] + (p2[0] - p1[0]) * t_end))
        ey = int(round(p1[1] + (p2[1] - p1[1]) * t_end))
        cv2.line(img, (sx, sy), (ex, ey), color_bgr, thickness, cv2.LINE_AA)

def draw_styled_circle(img, center, radius, color_bgr, thickness=2, style="実線"):
    if style != "点線" or radius <= 0:
        cv2.circle(img, center, radius, color_bgr, thickness, cv2.LINE_AA)
        return
    circumference = 2 * math.pi * radius
    dash_px = 12
    num_dashes = max(8, int(circumference // (dash_px * 2)))
    angle_step = 360.0 / num_dashes
    for i in range(num_dashes):
        start_ang = i * angle_step
        end_ang = start_ang + angle_step * 0.55
        cv2.ellipse(img, center, (radius, radius), 0, start_ang, end_ang, color_bgr, thickness, cv2.LINE_AA)

def draw_outlined_text(draw, pos, text_str, font, text_color=(255, 255, 255), stroke_width=2):
    draw.text(
        pos,
        text_str,
        font=font,
        fill=text_color,
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0)
    )

def draw_custom_shapes(img, custom_shapes):
    if not custom_shapes:
        return
    for sh in custom_shapes:
        stype = sh.get("type")
        bgr = hex_to_bgr(sh.get("color", "#ff0000"))
        thick = max(1, int(sh.get("width", 3)))
        style = sh.get("style", "実線")
        if stype == "line":
            p1 = (int(sh["x1"]), int(sh["y1"]))
            p2 = (int(sh["x2"]), int(sh["y2"]))
            draw_styled_line(img, p1, p2, bgr, thick, style)
        elif stype == "circle":
            cx, cy = int(sh["cx"]), int(sh["cy"])
            r = max(1, int(sh["r"]))
            draw_styled_circle(img, (cx, cy), r, bgr, thick, style)
        elif stype == "free":
            pts = sh.get("pts", [])
            if len(pts) >= 2:
                if style == "点線":
                    for i in range(0, len(pts) - 1, 2):
                        p1 = (int(pts[i][0]), int(pts[i][1]))
                        p2 = (int(pts[i+1][0]), int(pts[i+1][1]))
                        cv2.line(img, p1, p2, bgr, thick, cv2.LINE_AA)
                else:
                    arr = np.array([[int(p[0]), int(p[1])] for p in pts], dtype=np.int32)
                    cv2.polylines(img, [arr], isClosed=False, color=bgr, thickness=thick, lineType=cv2.LINE_AA)

def draw_all_annotations(
    frame, data, ref_data, phase_label=None,
    show_head=True, show_spine=True, show_spine_tilt=False, show_shoulders=True,
    show_shaft=True, show_hip_wall=True, show_ankles=False, show_vertical_90=False,
    shaft_p1=(0, 0), shaft_p2=(100, 100),
    hip_x=100, ankle_l_x=100, ankle_r_x=200,
    styles=None, custom_shapes=None, draw_border=True
):
    annotated = frame.copy()
    h, w, _ = annotated.shape
    st_cfg = styles or DEFAULT_STYLES

    if data is not None and ref_data is not None:
        init_head = ref_data["head_center"]
        head_radius = ref_data["head_radius"]
        init_shoulder = ref_data["shoulder_mid"]
        init_hip = ref_data["hip_mid"]
        init_l_ankle = ref_data["l_ankle"]
        init_r_ankle = ref_data["r_ankle"]

        head_top_y = max(10, init_head[1] - int(head_radius * 1.3))
        ground_y = min(h - 10, max(init_l_ankle[1], init_r_ankle[1]) + 10)

        if show_vertical_90:
            s = st_cfg["vertical"]
            draw_styled_line(annotated, (init_hip[0], head_top_y), (init_hip[0], ground_y), hex_to_bgr(s["color"]), s["width"], s["style"])

        if show_hip_wall:
            s = st_cfg["hip"]
            draw_styled_line(annotated, (hip_x, head_top_y), (hip_x, ground_y), hex_to_bgr(s["color"]), s["width"], s["style"], dash_len=12)

        if show_shaft:
            s = st_cfg["shaft"]
            vx = shaft_p2[0] - shaft_p1[0]
            vy = shaft_p2[1] - shaft_p1[1]
            ext_p2 = (int(shaft_p1[0] + vx * 2.2), int(shaft_p1[1] + vy * 2.2))
            draw_styled_line(annotated, shaft_p1, ext_p2, hex_to_bgr(s["color"]), s["width"], s["style"])

        if show_ankles:
            s = st_cfg["ankles"]
            bgr = hex_to_bgr(s["color"])
            draw_styled_line(annotated, (ankle_l_x, head_top_y), (ankle_l_x, ground_y), bgr, s["width"], s["style"])
            draw_styled_line(annotated, (ankle_r_x, head_top_y), (ankle_r_x, ground_y), bgr, s["width"], s["style"])

        if show_head:
            s = st_cfg["head"]
            draw_styled_circle(annotated, init_head, head_radius, hex_to_bgr(s["color"]), s["width"], s["style"])
            cv2.circle(annotated, data["head_center"], 7, (0, 0, 255), -1, cv2.LINE_AA)

        if show_shoulders:
            s = st_cfg["shoulders"]
            bgr = hex_to_bgr(s["color"])
            draw_styled_line(annotated, data["l_sh"], data["r_sh"], bgr, s["width"], s["style"])
            cv2.circle(annotated, data["l_sh"], max(4, s["width"] + 2), bgr, -1, cv2.LINE_AA)
            cv2.circle(annotated, data["r_sh"], max(4, s["width"] + 2), bgr, -1, cv2.LINE_AA)

        if show_spine:
            s = st_cfg["spine"]
            bgr = hex_to_bgr(s["color"])
            if phase_label and phase_label != "P1":
                cv2.line(annotated, init_hip, init_shoulder, (160, 160, 160), 2, cv2.LINE_AA)
            draw_styled_line(annotated, data["hip_mid"], data["shoulder_mid"], bgr, s["width"], s["style"])
            cv2.circle(annotated, data["shoulder_mid"], max(5, s["width"] + 2), bgr, -1, cv2.LINE_AA)
            cv2.circle(annotated, data["hip_mid"], max(5, s["width"] + 2), bgr, -1, cv2.LINE_AA)

        if show_spine_tilt:
            s = st_cfg["spine_tilt"]
            bgr = hex_to_bgr(s["color"])
            draw_styled_line(annotated, (data["hip_mid"][0], max(10, data["shoulder_mid"][1] - 35)), data["hip_mid"], (190, 190, 190), 2, "点線", dash_len=8)
            draw_styled_line(annotated, data["hip_mid"], data["shoulder_mid"], bgr, s["width"], s["style"])
            cv2.circle(annotated, data["shoulder_mid"], max(5, s["width"] + 2), bgr, -1, cv2.LINE_AA)
            cv2.circle(annotated, data["hip_mid"], max(5, s["width"] + 2), bgr, -1, cv2.LINE_AA)

    draw_custom_shapes(annotated, custom_shapes)

    has_custom_line_angle = any(
        sh.get("type") == "line" and sh.get("show_angle", True)
        for sh in (custom_shapes or [])
    )
    need_pil_text = (data is not None and ref_data is not None) or has_custom_line_angle

    if need_pil_text:
        if data is not None and ref_data is not None and phase_label and draw_border:
            cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (0, 165, 255), 10)

        font_scale = max(0.65, h / 1080.0)
        font_size = int(21 * font_scale)
        stroke_w = max(2, int(round(2.5 * font_scale)))
        font = get_gothic_font(font_size)

        pil_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        for sh in (custom_shapes or []):
            if sh.get("type") == "line" and sh.get("show_angle", True):
                p1 = (int(sh["x1"]), int(sh["y1"]))
                p2 = (int(sh["x2"]), int(sh["y2"]))
                if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) >= 8:
                    line_deg = calculate_line_angle_horizontal(p1, p2)
                    mid_lx = int((p1[0] + p2[0]) / 2)
                    mid_ly = int((p1[1] + p2[1]) / 2)
                    offset_x = int(12 * font_scale)
                    draw_outlined_text(
                        draw,
                        (mid_lx + offset_x, mid_ly - int(font_size * 0.5)),
                        f"{line_deg}°",
                        font,
                        text_color=hex_to_rgb(sh.get("color", "#ff3b30")),
                        stroke_width=stroke_w
                    )

        if data is not None and ref_data is not None:
            if show_spine:
                curr_spine = data["spine_angle"]
                mid_sx = int((data["hip_mid"][0] + data["shoulder_mid"][0]) / 2)
                mid_sy = int((data["hip_mid"][1] + data["shoulder_mid"][1]) / 2)
                offset_x = int(12 * font_scale)
                draw_outlined_text(
                    draw,
                    (mid_sx + offset_x, mid_sy - int(font_size * 0.5)),
                    f"{curr_spine}°",
                    font,
                    text_color=hex_to_rgb(st_cfg["spine"]["color"]),
                    stroke_width=stroke_w
                )

            if show_spine_tilt:
                target_dir_x = -1 if ref_data["l_sh"][0] < ref_data["r_sh"][0] else 1
                tilt_deg = calculate_spine_tilt_frontal(data["shoulder_mid"], data["hip_mid"], target_dir_x)
                mid_sx = int((data["hip_mid"][0] + data["shoulder_mid"][0]) / 2)
                mid_sy = int((data["hip_mid"][1] + data["shoulder_mid"][1]) / 2)
                offset_x = int(14 * font_scale)
                draw_outlined_text(
                    draw,
                    (mid_sx + offset_x, mid_sy - int(font_size * 0.5)),
                    f"{tilt_deg}°",
                    font,
                    text_color=hex_to_rgb(st_cfg["spine_tilt"]["color"]),
                    stroke_width=stroke_w
                )

            if show_shoulders:
                shoulder_angle_deg = calculate_line_angle_horizontal(data["l_sh"], data["r_sh"])
                right_end_sh = data["l_sh"] if data["l_sh"][0] >= data["r_sh"][0] else data["r_sh"]
                offset_x = int(12 * font_scale)
                pos_x = min(w - int(font_size * 2.4), right_end_sh[0] + offset_x)
                pos_y = right_end_sh[1] - int(font_size * 0.5)
                draw_outlined_text(
                    draw,
                    (pos_x, pos_y),
                    f"{shoulder_angle_deg}°",
                    font,
                    text_color=hex_to_rgb(st_cfg["shoulders"]["color"]),
                    stroke_width=stroke_w
                )

            if show_shaft:
                shaft_angle_deg = calculate_line_angle_horizontal(shaft_p1, shaft_p2)
                mid_px = int((shaft_p1[0] + shaft_p2[0]) / 2)
                mid_py = int((shaft_p1[1] + shaft_p2[1]) / 2)
                offset_x = int(14 * font_scale)
                draw_outlined_text(
                    draw,
                    (mid_px + offset_x, mid_py - int(font_size * 0.5)),
                    f"{shaft_angle_deg}°",
                    font,
                    text_color=hex_to_rgb(st_cfg["shaft"]["color"]),
                    stroke_width=stroke_w
                )

            if phase_label:
                p_font_size = int(32 * font_scale)
                p_stroke_w = max(2, int(round(3.0 * font_scale)))
                p_font = get_gothic_font(p_font_size)
                draw_outlined_text(
                    draw,
                    (26, 22),
                    phase_label,
                    p_font,
                    text_color=(255, 200, 60),
                    stroke_width=p_stroke_w
                )

        annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return annotated

def render_interactive_drag_editor(
    frame_bgr, frame_data, ref_data, orig_w, orig_h, active_p_code,
    show_head, show_spine, show_spine_tilt, show_shoulders, show_shaft, show_hip_wall, show_ankles,
    shaft_p1, shaft_p2, hip_x, ankle_l_x, ankle_r_x,
    styles, custom_shapes
):
    bg = frame_bgr.copy()
    head_top_y = 40
    ground_y = orig_h - 30
    spine_deg = 60
    st_cfg = styles or DEFAULT_STYLES
    cur_data = frame_data or ref_data

    if ref_data is not None:
        head_top_y = max(10, ref_data["head_center"][1] - int(ref_data["head_radius"] * 1.3))
        ground_y = min(orig_h - 10, max(ref_data["l_ankle"][1], ref_data["r_ankle"][1]) + 10)
        if show_head:
            s = st_cfg["head"]
            draw_styled_circle(bg, ref_data["head_center"], ref_data["head_radius"], hex_to_bgr(s["color"]), s["width"], s["style"])
            if cur_data is not None:
                cv2.circle(bg, cur_data["head_center"], 7, (0, 0, 255), -1, cv2.LINE_AA)

    if cur_data is not None:
        spine_deg = cur_data["spine_angle"]
        if show_shoulders:
            s = st_cfg["shoulders"]
            draw_styled_line(bg, cur_data["l_sh"], cur_data["r_sh"], hex_to_bgr(s["color"]), s["width"], s["style"])
        if show_spine:
            s = st_cfg["spine"]
            draw_styled_line(bg, cur_data["hip_mid"], cur_data["shoulder_mid"], hex_to_bgr(s["color"]), s["width"], s["style"])
        if show_spine_tilt:
            s = st_cfg["spine_tilt"]
            draw_styled_line(bg, cur_data["hip_mid"], cur_data["shoulder_mid"], hex_to_bgr(s["color"]), s["width"], s["style"])

    disp_w = 540
    scale = disp_w / orig_w
    disp_h = int(orig_h * scale)
    bg_small = cv2.resize(bg, (disp_w, disp_h))
    _, buffer = cv2.imencode('.jpg', bg_small, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    disp_shapes = []
    for sh in (custom_shapes or []):
        t = sh.get("type")
        if t == "line":
            disp_shapes.append({
                "type": "line",
                "x1": sh["x1"] * scale, "y1": sh["y1"] * scale,
                "x2": sh["x2"] * scale, "y2": sh["y2"] * scale,
                "color": sh.get("color", "#ff0000"),
                "width": sh.get("width", 3),
                "style": sh.get("style", "実線"),
                "showAngle": sh.get("show_angle", True),
                "targetP": sh.get("target_p", None)
            })
        elif t == "circle":
            disp_shapes.append({
                "type": "circle",
                "cx": sh["cx"] * scale, "cy": sh["cy"] * scale,
                "r": sh["r"] * scale,
                "color": sh.get("color", "#ff0000"),
                "width": sh.get("width", 3),
                "style": sh.get("style", "実線"),
                "targetP": sh.get("target_p", None)
            })
        elif t == "free":
            disp_shapes.append({
                "type": "free",
                "pts": [[p[0] * scale, p[1] * scale] for p in sh.get("pts", [])],
                "color": sh.get("color", "#ff0000"),
                "width": sh.get("width", 3),
                "style": sh.get("style", "実線"),
                "targetP": sh.get("target_p", None)
            })

    init_state = {
        "activeP": active_p_code,
        "sp1": [round(shaft_p1[0] * scale, 1), round(shaft_p1[1] * scale, 1)],
        "sp2": [round(shaft_p2[0] * scale, 1), round(shaft_p2[1] * scale, 1)],
        "hipX": round(hip_x * scale, 1),
        "ankLX": round(ankle_l_x * scale, 1),
        "ankRX": round(ankle_r_x * scale, 1),
        "topY": round(head_top_y * scale, 1),
        "botY": round(ground_y * scale, 1),
        "midY": round(((head_top_y + ground_y) / 2) * scale, 1),
        "showShaft": show_shaft,
        "showHip": show_hip_wall,
        "showAnkles": show_ankles,
        "shaftStyle": st_cfg["shaft"],
        "hipStyle": st_cfg["hip"],
        "anklesStyle": st_cfg["ankles"],
        "spineDeg": spine_deg,
        "scale": scale,
        "shapes": disp_shapes
    }

    html_code = f"""
    <div style="font-family: sans-serif; user-select: none;">
      <div style="margin-bottom: 8px; padding: 8px 12px; background: #1e293b; color: #f8fafc; border-radius: 6px; font-size: 13px;">
        👆 <b>背景: <span style="color:#fbbf24;">{active_p_code}</span> ｜ 丸ツマミ移動＆自由描画（自動保存されます）</b><br>
        <span style="color:#4ade80;">● 緑丸</span>：ボール位置 
        <span style="color:#38bdf8;">● 青丸</span>：グリップ側 
        <span style="color:#fbbf24;">● 黄丸</span>：お尻／両足首（左右）
      </div>

      <div style="margin-bottom: 8px; display: flex; flex-wrap: wrap; align-items: center; gap: 6px; background: #f1f5f9; padding: 6px 8px; border-radius: 6px; border: 1px solid #cbd5e1;">
        <button type="button" class="tool-btn active" data-tool="move" style="padding: 5px 10px; border-radius: 4px; border: 1px solid #2563eb; background: #2563eb; color: #fff; font-weight: bold; cursor: pointer; font-size: 12px;">✋ ツマミ移動</button>
        <button type="button" class="tool-btn" data-tool="line" style="padding: 5px 10px; border-radius: 4px; border: 1px solid #94a3b8; background: #fff; color: #1e293b; font-weight: bold; cursor: pointer; font-size: 12px;">📏 直線</button>
        <button type="button" class="tool-btn" data-tool="circle" style="padding: 5px 10px; border-radius: 4px; border: 1px solid #94a3b8; background: #fff; color: #1e293b; font-weight: bold; cursor: pointer; font-size: 12px;">⭕ 円</button>
        <button type="button" class="tool-btn" data-tool="free" style="padding: 5px 10px; border-radius: 4px; border: 1px solid #94a3b8; background: #fff; color: #1e293b; font-weight: bold; cursor: pointer; font-size: 12px;">✏️ フリーハンド</button>
        <input type="color" id="drawColor" value="#ff3b30" title="線の色" style="width: 28px; height: 26px; border: none; cursor: pointer; background: none;">
        <select id="drawWidth" title="線の太さ" style="padding: 4px; border-radius: 4px; border: 1px solid #94a3b8; font-size: 12px;">
          <option value="2">細 (2px)</option>
          <option value="3" selected>標準 (3px)</option>
          <option value="5">太 (5px)</option>
          <option value="7">極太 (7px)</option>
        </select>
        <select id="drawStyle" title="線の種類" style="padding: 4px; border-radius: 4px; border: 1px solid #94a3b8; font-size: 12px;">
          <option value="実線" selected>実線</option>
          <option value="点線">点線</option>
        </select>
        <label style="display: inline-flex; align-items: center; gap: 3px; font-size: 12px; font-weight: bold; color: #1e293b; cursor: pointer; background: #fff; padding: 4px 7px; border-radius: 4px; border: 1px solid #94a3b8;">
          <input type="checkbox" id="drawLineAngle" checked> 角度表示
        </label>
        <label title="チェックを入れると、今描いた図形を「{active_p_code}」のコマ専用として記録します" style="display: inline-flex; align-items: center; gap: 3px; font-size: 12px; font-weight: bold; color: #0f172a; cursor: pointer; background: #fef3c7; padding: 4px 7px; border-radius: 4px; border: 1px solid #f59e0b;">
          <input type="checkbox" id="bindCurrentPCheck" checked> {active_p_code}のみ表示
        </label>
        <button type="button" id="undoBtn" style="padding: 5px 8px; border-radius: 4px; border: 1px solid #94a3b8; background: #fff; color: #1e293b; cursor: pointer; font-size: 12px;">↩ 戻す</button>
        <button type="button" id="clearBtn" style="padding: 5px 8px; border-radius: 4px; border: 1px solid #ef4444; background: #fee2e2; color: #b91c1c; cursor: pointer; font-size: 12px;">🗑 消去</button>
      </div>

      <div style="position: relative; width: {disp_w}px; height: {disp_h}px; border: 2px solid #475569; border-radius: 6px; overflow: hidden;">
        <canvas id="golfCanvas" width="{disp_w}" height="{disp_h}" style="display: block; cursor: grab; touch-action: none;"></canvas>
      </div>
      <div style="margin-top: 8px; display: flex; align-items: center; gap: 14px;">
        <div id="angleReadout" style="font-weight: bold; font-size: 15px; color: #16a34a;">--°（垂直: 90°）</div>
        <div id="syncStatus" style="font-size: 13px; color: #2563eb; font-weight: bold;">✅ 座標同期済み</div>
      </div>
    </div>

    <script>
      const canvas = document.getElementById('golfCanvas');
      const ctx = canvas.getContext('2d');
      const angleReadout = document.getElementById('angleReadout');
      const syncStatus = document.getElementById('syncStatus');
      const drawColorInput = document.getElementById('drawColor');
      const drawWidthSelect = document.getElementById('drawWidth');
      const drawStyleSelect = document.getElementById('drawStyle');
      const drawLineAngleCheck = document.getElementById('drawLineAngle');
      const bindCurrentPCheck = document.getElementById('bindCurrentPCheck');
      const toolBtns = document.querySelectorAll('.tool-btn');
      const undoBtn = document.getElementById('undoBtn');
      const clearBtn = document.getElementById('clearBtn');

      const state = {json.dumps(init_state)};
      const img = new Image();
      img.src = "data:image/jpeg;base64,{img_b64}";

      let currentTool = 'move';
      let dragging = null;
      let activeShape = null;
      const HANDLE_R = 12;

      toolBtns.forEach(btn => {{
        btn.addEventListener('click', () => {{
          currentTool = btn.dataset.tool;
          toolBtns.forEach(b => {{
            b.style.background = '#fff';
            b.style.color = '#1e293b';
            b.style.borderColor = '#94a3b8';
          }});
          btn.style.background = '#2563eb';
          btn.style.color = '#fff';
          btn.style.borderColor = '#2563eb';
          canvas.style.cursor = (currentTool === 'move') ? 'grab' : 'crosshair';
        }});
      }});

      undoBtn.addEventListener('click', () => {{
        if (state.shapes.length > 0) {{
          state.shapes.pop();
          render();
          syncToParentUrl();
        }}
      }});

      clearBtn.addEventListener('click', () => {{
        state.shapes = [];
        render();
        syncToParentUrl();
      }});

      function syncToParentUrl() {{
        const s = state.scale;
        const params = new URLSearchParams(window.parent.location.search);
        params.set('sp1_x', Math.round(state.sp1[0] / s));
        params.set('sp1_y', Math.round(state.sp1[1] / s));
        params.set('sp2_x', Math.round(state.sp2[0] / s));
        params.set('sp2_y', Math.round(state.sp2[1] / s));
        params.set('hip_x', Math.round(state.hipX / s));
        params.set('ank_l_x', Math.round(state.ankLX / s));
        params.set('ank_r_x', Math.round(state.ankRX / s));

        const origShapes = state.shapes.map(sh => {{
          if (sh.type === 'line') {{
            return {{
              type: 'line',
              x1: Math.round(sh.x1 / s), y1: Math.round(sh.y1 / s),
              x2: Math.round(sh.x2 / s), y2: Math.round(sh.y2 / s),
              color: sh.color, width: sh.width, style: sh.style || '実線',
              show_angle: sh.showAngle !== false,
              target_p: sh.targetP || null
            }};
          }} else if (sh.type === 'circle') {{
            return {{
              type: 'circle',
              cx: Math.round(sh.cx / s), cy: Math.round(sh.cy / s),
              r: Math.round(sh.r / s),
              color: sh.color, width: sh.width, style: sh.style || '実線',
              target_p: sh.targetP || null
            }};
          }} else if (sh.type === 'free') {{
            const simp = [];
            for (let i = 0; i < sh.pts.length; i++) {{
              const p = sh.pts[i];
              if (simp.length === 0 || Math.hypot(p[0] - simp[simp.length-1][0], p[1] - simp[simp.length-1][1]) >= 2.5) {{
                simp.push([Math.round(p[0] / s), Math.round(p[1] / s)]);
              }}
            }}
            return {{
              type: 'free',
              pts: simp,
              color: sh.color,
              width: sh.width,
              style: sh.style || '実線',
              target_p: sh.targetP || null
            }};
          }}
        }});
        params.set('shapes', JSON.stringify(origShapes));

        const newUrl = window.parent.location.pathname + '?' + params.toString();
        window.parent.history.replaceState(null, '', newUrl);
        syncStatus.textContent = "✅ 位置・描画を記録しました（そのまま動画・静止画を作成できます）";
      }}

      function updateReadout() {{
        const dx = Math.abs(state.sp2[0] - state.sp1[0]);
        const dy = Math.abs(state.sp2[1] - state.sp1[1]);
        const deg = (dx === 0 && dy === 0) ? 0 : Math.round(Math.atan2(dy, dx) * 180 / Math.PI);
        angleReadout.textContent = deg + "°（垂直: 90°）";
        return deg;
      }}

      function drawHandle(x, y, fill) {{
        ctx.beginPath();
        ctx.arc(x, y, HANDLE_R, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = "#ffffff";
        ctx.stroke();
      }}

      function drawShapeItem(sh) {{
        const isCurrentP = !sh.targetP || sh.targetP === state.activeP;
        ctx.save();
        ctx.globalAlpha = isCurrentP ? 1.0 : 0.28;
        if (sh.style === '点線') {{
          ctx.setLineDash([8, 8]);
        }} else {{
          ctx.setLineDash([]);
        }}
        ctx.strokeStyle = sh.color;
        ctx.lineWidth = sh.width;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        if (sh.type === 'line') {{
          ctx.beginPath();
          ctx.moveTo(sh.x1, sh.y1);
          ctx.lineTo(sh.x2, sh.y2);
          ctx.stroke();

          if (sh.showAngle !== false && Math.hypot(sh.x2 - sh.x1, sh.y2 - sh.y1) >= 6) {{
            const dx = Math.abs(sh.x2 - sh.x1);
            const dy = Math.abs(sh.y2 - sh.y1);
            const deg = Math.round(Math.atan2(dy, dx) * 180 / Math.PI);
            const midX = (sh.x1 + sh.x2) / 2 + 10;
            const midY = (sh.y1 + sh.y2) / 2;
            const labelTxt = sh.targetP ? (deg + "° [" + sh.targetP + "]") : (deg + "°");
            ctx.setLineDash([]);
            ctx.font = "bold 12px sans-serif";
            ctx.lineWidth = 3;
            ctx.strokeStyle = "#000000";
            ctx.strokeText(labelTxt, midX, midY + 1);
            ctx.fillStyle = sh.color;
            ctx.fillText(labelTxt, midX, midY + 1);
          }}
        }} else if (sh.type === 'circle') {{
          ctx.beginPath();
          ctx.arc(sh.cx, sh.cy, sh.r, 0, Math.PI * 2);
          ctx.stroke();
        }} else if (sh.type === 'free' && sh.pts.length >= 2) {{
          ctx.beginPath();
          ctx.moveTo(sh.pts[0][0], sh.pts[0][1]);
          for (let i = 1; i < sh.pts.length; i++) {{
            ctx.lineTo(sh.pts[i][0], sh.pts[i][1]);
          }}
          ctx.stroke();
        }}
        ctx.restore();
      }}

      function render() {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);

        if (state.showHip) {{
          ctx.save();
          ctx.setLineDash(state.hipStyle.style === '点線' ? [8, 8] : []);
          ctx.beginPath();
          ctx.moveTo(state.hipX, state.topY);
          ctx.lineTo(state.hipX, state.botY);
          ctx.strokeStyle = state.hipStyle.color;
          ctx.lineWidth = state.hipStyle.width;
          ctx.stroke();
          ctx.restore();
          drawHandle(state.hipX, state.midY, state.hipStyle.color);
        }}

        if (state.showAnkles) {{
          ctx.save();
          ctx.setLineDash(state.anklesStyle.style === '点線' ? [8, 8] : []);
          ctx.beginPath();
          ctx.moveTo(state.ankLX, state.topY);
          ctx.lineTo(state.ankLX, state.botY);
          ctx.moveTo(state.ankRX, state.topY);
          ctx.lineTo(state.ankRX, state.botY);
          ctx.strokeStyle = state.anklesStyle.color;
          ctx.lineWidth = state.anklesStyle.width;
          ctx.stroke();
          ctx.restore();
          drawHandle(state.ankLX, state.midY, state.anklesStyle.color);
          drawHandle(state.ankRX, state.midY, state.anklesStyle.color);
        }}

        if (state.showShaft) {{
          const vx = state.sp2[0] - state.sp1[0];
          const vy = state.sp2[1] - state.sp1[1];
          const extX = state.sp1[0] + vx * 2.2;
          const extY = state.sp1[1] + vy * 2.2;
          ctx.save();
          ctx.setLineDash(state.shaftStyle.style === '点線' ? [8, 8] : []);
          ctx.beginPath();
          ctx.moveTo(state.sp1[0], state.sp1[1]);
          ctx.lineTo(extX, extY);
          ctx.strokeStyle = state.shaftStyle.color;
          ctx.lineWidth = state.shaftStyle.width;
          ctx.stroke();
          ctx.restore();

          const deg = updateReadout();
          const midX = (state.sp1[0] + state.sp2[0]) / 2 + 10;
          const midY = (state.sp1[1] + state.sp2[1]) / 2;
          ctx.font = "bold 12px sans-serif";
          ctx.lineWidth = 3;
          ctx.strokeStyle = "#000000";
          ctx.strokeText(deg + "°", midX, midY + 1);
          ctx.fillStyle = state.shaftStyle.color;
          ctx.fillText(deg + "°", midX, midY + 1);

          drawHandle(state.sp1[0], state.sp1[1], "#00ff64");
          drawHandle(state.sp2[0], state.sp2[1], "#00bfff");
        }} else {{
          updateReadout();
        }}

        state.shapes.forEach(drawShapeItem);
        if (activeShape) {{
          drawShapeItem(activeShape);
        }}
      }}

      function getPos(e) {{
        const rect = canvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return [
          Math.max(0, Math.min(canvas.width, clientX - rect.left)),
          Math.max(0, Math.min(canvas.height, clientY - rect.top))
        ];
      }}

      function dist(x1, y1, x2, y2) {{
        return Math.hypot(x2 - x1, y2 - y1);
      }}

      function onDown(e) {{
        const [mx, my] = getPos(e);
        if (currentTool === 'move') {{
          const hitR = HANDLE_R + 8;
          if (state.showShaft && dist(mx, my, state.sp1[0], state.sp1[1]) <= hitR) {{
            dragging = 'sp1';
          }} else if (state.showShaft && dist(mx, my, state.sp2[0], state.sp2[1]) <= hitR) {{
            dragging = 'sp2';
          }} else if (state.showHip && dist(mx, my, state.hipX, state.midY) <= hitR) {{
            dragging = 'hip';
          }} else if (state.showAnkles && dist(mx, my, state.ankLX, state.midY) <= hitR) {{
            dragging = 'ankL';
          }} else if (state.showAnkles && dist(mx, my, state.ankRX, state.midY) <= hitR) {{
            dragging = 'ankR';
          }}
          if (dragging) {{
            canvas.style.cursor = 'grabbing';
            syncStatus.textContent = "調整中...";
            e.preventDefault();
          }}
        }} else {{
          const col = drawColorInput.value;
          const wid = parseInt(drawWidthSelect.value, 10);
          const stl = drawStyleSelect.value;
          const showAng = drawLineAngleCheck.checked;
          const targetP = bindCurrentPCheck.checked ? state.activeP : null;
          if (currentTool === 'line') {{
            activeShape = {{ type: 'line', x1: mx, y1: my, x2: mx, y2: my, color: col, width: wid, style: stl, showAngle: showAng, targetP: targetP }};
          }} else if (currentTool === 'circle') {{
            activeShape = {{ type: 'circle', cx: mx, cy: my, r: 1, color: col, width: wid, style: stl, targetP: targetP }};
          }} else if (currentTool === 'free') {{
            activeShape = {{ type: 'free', pts: [[mx, my]], color: col, width: wid, style: stl, targetP: targetP }};
          }}
          syncStatus.textContent = "描画中...";
          e.preventDefault();
        }}
      }}

      function onMove(e) {{
        if (!dragging && !activeShape) return;
        const [mx, my] = getPos(e);
        if (dragging) {{
          if (dragging === 'sp1') {{
            state.sp1 = [mx, my];
          }} else if (dragging === 'sp2') {{
            state.sp2 = [mx, my];
          }} else if (dragging === 'hip') {{
            state.hipX = mx;
          }} else if (dragging === 'ankL') {{
            state.ankLX = mx;
          }} else if (dragging === 'ankR') {{
            state.ankRX = mx;
          }}
        }} else if (activeShape) {{
          if (activeShape.type === 'line') {{
            activeShape.x2 = mx;
            activeShape.y2 = my;
            const dx = Math.abs(activeShape.x2 - activeShape.x1);
            const dy = Math.abs(activeShape.y2 - activeShape.y1);
            const deg = (dx === 0 && dy === 0) ? 0 : Math.round(Math.atan2(dy, dx) * 180 / Math.PI);
            angleReadout.textContent = "描画直線: " + deg + "°（垂直: 90°）";
          }} else if (activeShape.type === 'circle') {{
            activeShape.r = Math.hypot(mx - activeShape.cx, my - activeShape.cy);
          }} else if (activeShape.type === 'free') {{
            activeShape.pts.push([mx, my]);
          }}
        }}
        render();
        e.preventDefault();
      }}

      function onUp() {{
        if (dragging) {{
          dragging = null;
          canvas.style.cursor = 'grab';
          syncToParentUrl();
        }}
        if (activeShape) {{
          state.shapes.push(activeShape);
          activeShape = null;
          render();
          syncToParentUrl();
        }}
      }}

      canvas.addEventListener('mousedown', onDown);
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onUp);

      canvas.addEventListener('touchstart', onDown, {{ passive: false }});
      window.addEventListener('touchmove', onMove, {{ passive: false }});
      window.addEventListener('touchend', onUp);

      img.onload = () => {{
        render();
        syncToParentUrl();
      }};
    </script>
    """
    components.html(html_code, height=disp_h + 150)

def analyze_skeleton(input_path):
    ensure_model_exists()
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frames = []
    frame_data_list = []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / fps) * 1000)
            frame_idx += 1
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                lm = results.pose_landmarks[0]
                def get_px(idx):
                    return (int(lm[idx].x * width), int(lm[idx].y * height))
                nose, l_ear, r_ear = get_px(0), get_px(7), get_px(8)
                l_sh, r_sh = get_px(11), get_px(12)
                l_wr, r_wr = get_px(15), get_px(16)
                l_hip, r_hip = get_px(23), get_px(24)
                l_ankle, r_ankle = get_px(27), get_px(28)

                head_center = (int((nose[0] + l_ear[0] + r_ear[0]) / 3), int((nose[1] + l_ear[1] + r_ear[1]) / 3))
                shoulder_mid = (int((l_sh[0] + r_sh[0]) / 2), int((l_sh[1] + r_sh[1]) / 2))
                hip_mid = (int((l_hip[0] + r_hip[0]) / 2), int((l_hip[1] + r_hip[1]) / 2))
                wrist_mid = (int((l_wr[0] + r_wr[0]) / 2), int((l_wr[1] + r_wr[1]) / 2))

                shoulder_width = math.dist(l_sh, r_sh)
                head_radius = max(int(shoulder_width * 0.45), 35)
                spine_angle = calculate_spine_angle_horizontal(shoulder_mid, hip_mid)

                frame_data_list.append({
                    "head_center": head_center,
                    "head_radius": head_radius,
                    "l_sh": l_sh,
                    "r_sh": r_sh,
                    "shoulder_mid": shoulder_mid,
                    "hip_mid": hip_mid,
                    "wrist_mid": wrist_mid,
                    "l_ankle": l_ankle,
                    "r_ankle": r_ankle,
                    "spine_angle": spine_angle
                })
            else:
                frame_data_list.append(None)
    cap.release()
    return frames, frame_data_list, fps, width, height

def build_sequence_sheet(bgr_images, max_cols=5, gap=10):
    if not bgr_images:
        return None
    n = len(bgr_images)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    h, w, _ = bgr_images[0].shape

    sheet_w = cols * w + (cols + 1) * gap
    sheet_h = rows * h + (rows + 1) * gap
    sheet = np.full((sheet_h, sheet_w, 3), 24, dtype=np.uint8)

    for idx, img in enumerate(bgr_images):
        r = idx // cols
        c = idx % cols
        y0 = gap + r * (h + gap)
        x0 = gap + c * (w + gap)
        resized = img if (img.shape[0] == h and img.shape[1] == w) else cv2.resize(img, (w, h))
        sheet[y0:y0 + h, x0:x0 + w] = resized
    return sheet

def export_selected_snapshots(
    frames, frame_data_list, selected_list, p1_idx,
    show_head, show_spine, show_spine_tilt, show_shoulders, show_shaft, show_hip_wall, show_ankles, show_vertical_90,
    shaft_p1, shaft_p2, hip_x, ankle_l_x, ankle_r_x,
    styles, custom_shapes, orig_name,
    line_positions=None, current_p_frames=None
):
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = orig_name.rsplit('.', 1)[0]
    ref_data = frame_data_list[p1_idx] or next((d for d in frame_data_list if d is not None), None)
    pos_map = line_positions or {}
    p_map = current_p_frames or {}

    saved_items = []
    bgr_list = []
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for p_code, p_name, f_idx in selected_list:
            f_head = show_head and is_frame_in_selected_positions(f_idx, pos_map.get("head"), p_map)
            f_spine = show_spine and is_frame_in_selected_positions(f_idx, pos_map.get("spine"), p_map)
            f_spine_tilt = show_spine_tilt and is_frame_in_selected_positions(f_idx, pos_map.get("spine_tilt"), p_map)
            f_shoulders = show_shoulders and is_frame_in_selected_positions(f_idx, pos_map.get("shoulders"), p_map)
            f_shaft = show_shaft and is_frame_in_selected_positions(f_idx, pos_map.get("shaft"), p_map)
            f_hip = show_hip_wall and is_frame_in_selected_positions(f_idx, pos_map.get("hip"), p_map)
            f_ankles = show_ankles and is_frame_in_selected_positions(f_idx, pos_map.get("ankles"), p_map)
            f_vert = show_vertical_90 and is_frame_in_selected_positions(f_idx, pos_map.get("vertical"), p_map)
            f_shapes = filter_custom_shapes_for_frame(f_idx, custom_shapes, pos_map.get("custom_shapes"), p_map)

            img_bgr = draw_all_annotations(
                frames[f_idx], frame_data_list[f_idx], ref_data, p_code,
                show_head=f_head, show_spine=f_spine, show_spine_tilt=f_spine_tilt,
                show_shoulders=f_shoulders, show_shaft=f_shaft, show_hip_wall=f_hip,
                show_ankles=f_ankles, show_vertical_90=f_vert,
                shaft_p1=shaft_p1, shaft_p2=shaft_p2,
                hip_x=hip_x, ankle_l_x=ankle_l_x, ankle_r_x=ankle_r_x,
                styles=styles, custom_shapes=f_shapes,
                draw_border=False
            )
            bgr_list.append(img_bgr)
            file_name = f"{base_name}_{p_code}_{timestamp}.png"
            save_path = os.path.join(output_dir, file_name)
            is_ok, png_buf = cv2.imencode(".png", img_bgr)
            if is_ok:
                png_bytes = png_buf.tobytes()
                with open(save_path, "wb") as f:
                    f.write(png_bytes)
                zf.writestr(file_name, png_bytes)
                saved_items.append({
                    "p_code": p_code,
                    "p_name": p_name,
                    "file_name": file_name,
                    "save_path": save_path,
                    "png_bytes": png_bytes
                })

        sheet_bytes = None
        sheet_name = None
        if len(bgr_list) >= 2:
            sheet_bgr = build_sequence_sheet(bgr_list, max_cols=5, gap=10)
            if sheet_bgr is not None:
                sheet_name = f"{base_name}_sequence_{timestamp}.png"
                sheet_path = os.path.join(output_dir, sheet_name)
                ok_s, s_buf = cv2.imencode(".png", sheet_bgr)
                if ok_s:
                    sheet_bytes = s_buf.tobytes()
                    with open(sheet_path, "wb") as f:
                        f.write(sheet_bytes)
                    zf.writestr(sheet_name, sheet_bytes)

    return saved_items, sheet_bytes, sheet_name, zip_buffer.getvalue(), f"{base_name}_snapshots_{timestamp}.zip"

def render_and_save_video(
    frames, frame_data_list, fps, width, height,
    selected_phases_dict, p1_idx, pause_seconds,
    show_head, show_spine, show_spine_tilt, show_shoulders, show_shaft, show_hip_wall, show_ankles, show_vertical_90,
    shaft_p1, shaft_p2, hip_x, ankle_l_x, ankle_r_x,
    styles, custom_shapes, orig_name,
    line_positions=None, current_p_frames=None
):
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = orig_name.rsplit('.', 1)[0]
    raw_path = os.path.join(output_dir, f"temp_{base_name}_{timestamp}.mp4")
    save_path = os.path.join(output_dir, f"analyzed_{base_name}_{timestamp}.mp4")

    ref_data = frame_data_list[p1_idx] or next((d for d in frame_data_list if d is not None), None)
    pos_map = line_positions or {}
    p_map = current_p_frames or {}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))

    pause_frame_count = max(1, int(round(fps * pause_seconds)))
    for i, frame in enumerate(frames):
        phase_label = selected_phases_dict.get(i, None)

        f_head = show_head and is_frame_in_selected_positions(i, pos_map.get("head"), p_map)
        f_spine = show_spine and is_frame_in_selected_positions(i, pos_map.get("spine"), p_map)
        f_spine_tilt = show_spine_tilt and is_frame_in_selected_positions(i, pos_map.get("spine_tilt"), p_map)
        f_shoulders = show_shoulders and is_frame_in_selected_positions(i, pos_map.get("shoulders"), p_map)
        f_shaft = show_shaft and is_frame_in_selected_positions(i, pos_map.get("shaft"), p_map)
        f_hip = show_hip_wall and is_frame_in_selected_positions(i, pos_map.get("hip"), p_map)
        f_ankles = show_ankles and is_frame_in_selected_positions(i, pos_map.get("ankles"), p_map)
        f_vert = show_vertical_90 and is_frame_in_selected_positions(i, pos_map.get("vertical"), p_map)
        f_shapes = filter_custom_shapes_for_frame(i, custom_shapes, pos_map.get("custom_shapes"), p_map)

        annotated_frame = draw_all_annotations(
            frame, frame_data_list[i], ref_data, phase_label,
            show_head=f_head, show_spine=f_spine, show_spine_tilt=f_spine_tilt,
            show_shoulders=f_shoulders, show_shaft=f_shaft, show_hip_wall=f_hip,
            show_ankles=f_ankles, show_vertical_90=f_vert,
            shaft_p1=shaft_p1, shaft_p2=shaft_p2,
            hip_x=hip_x, ankle_l_x=ankle_l_x, ankle_r_x=ankle_r_x,
            styles=styles, custom_shapes=f_shapes
        )
        if phase_label is not None and pause_seconds > 0:
            for _ in range(pause_frame_count):
                out.write(annotated_frame)
        else:
            out.write(annotated_frame)
    out.release()

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y", "-i", raw_path,
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        save_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        os.remove(raw_path)
    else:
        os.replace(raw_path, save_path)

    with open(save_path, "rb") as f:
        video_bytes = f.read()
    return video_bytes, save_path

if "chk_head" not in st.session_state:
    st.session_state["chk_head"] = True
if "chk_spine" not in st.session_state:
    st.session_state["chk_spine"] = True
if "chk_spine_tilt" not in st.session_state:
    st.session_state["chk_spine_tilt"] = False
if "chk_shoulders" not in st.session_state:
    st.session_state["chk_shoulders"] = True
if "chk_shaft" not in st.session_state:
    st.session_state["chk_shaft"] = True
if "chk_hip_wall" not in st.session_state:
    st.session_state["chk_hip_wall"] = True
if "chk_ankles" not in st.session_state:
    st.session_state["chk_ankles"] = False
if "chk_vertical_90" not in st.session_state:
    st.session_state["chk_vertical_90"] = False

st.title("🏌️‍♂️ ゴルフスイングAI自動分析ツール")

col_side, col_main = st.columns([1.05, 2.35])

with col_side:
    st.subheader("1. 撮影モード一括切り替え")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        if st.button("📷 後方撮影モード", use_container_width=True):
            st.session_state["chk_head"] = True
            st.session_state["chk_spine"] = True
            st.session_state["chk_spine_tilt"] = False
            st.session_state["chk_shoulders"] = True
            st.session_state["chk_shaft"] = True
            st.session_state["chk_hip_wall"] = True
            st.session_state["chk_ankles"] = False
            st.rerun()
    with p_col2:
        if st.button("🧍 正面撮影モード", use_container_width=True):
            st.session_state["chk_head"] = True
            st.session_state["chk_spine"] = False
            st.session_state["chk_spine_tilt"] = True
            st.session_state["chk_shoulders"] = True
            st.session_state["chk_shaft"] = False
            st.session_state["chk_hip_wall"] = False
            st.session_state["chk_ankles"] = True
            st.rerun()

    st.subheader("2. 基準ライン表示・スタイル・ポジション指定")
    st.caption("ポジションを複数選択すると、そのコマだけに線を表示します（未選択＝全フレーム表示）。")

    LINE_ITEMS = [
        ("head", "頭の固定枠", "chk_head"),
        ("spine", "背骨の前傾角度ライン（後方用）", "chk_spine"),
        ("spine_tilt", "背骨の傾きライン（正面用・90°=0°）", "chk_spine_tilt"),
        ("shoulders", "両肩を結ぶライン", "chk_shoulders"),
        ("shaft", "シャフトプレーン", "chk_shaft"),
        ("hip", "ヒップライン（後方用）", "chk_hip_wall"),
        ("ankles", "両足首の外側垂線2本（正面用）", "chk_ankles"),
        ("vertical", "垂直基準線（90°）", "chk_vertical_90"),
    ]

    line_enabled = {}
    line_positions = {}

    for key_name, label, chk_key in LINE_ITEMS:
        is_on = st.checkbox(label, key=chk_key)
        line_enabled[key_name] = is_on
        if is_on:
            selected_pos = st.multiselect(
                f"{label}の表示ポジション",
                options=P_NAMES_LIST,
                default=[],
                placeholder="全フレーム表示（クリックでP1〜P10を選択）",
                key=f"pos_sel_{key_name}",
                label_visibility="collapsed"
            )
            line_positions[key_name] = selected_pos
        else:
            line_positions[key_name] = []

    st.markdown("**共通手描き図形の表示ポジション**")
    line_positions["custom_shapes"] = st.multiselect(
        "共通手描き図形の表示ポジション",
        options=P_NAMES_LIST,
        default=[],
        placeholder="全フレーム表示（P専用チェック外し時）",
        key="pos_sel_custom_shapes",
        label_visibility="collapsed"
    )

    show_head = line_enabled["head"]
    show_spine = line_enabled["spine"]
    show_spine_tilt = line_enabled["spine_tilt"]
    show_shoulders = line_enabled["shoulders"]
    show_shaft = line_enabled["shaft"]
    show_hip_wall = line_enabled["hip"]
    show_ankles = line_enabled["ankles"]
    show_vertical_90 = line_enabled["vertical"]

    def render_line_style_editor(label, key_name):
        d = DEFAULT_STYLES[key_name]
        st.markdown(f"**{label}**")
        c_col, w_col, s_col = st.columns([1, 1.3, 1.4])
        with c_col:
            col_val = st.color_picker("色", value=d["color"], key=f"st_col_{key_name}", label_visibility="collapsed")
        with w_col:
            wid_val = st.selectbox("太さ", [1, 2, 3, 4, 5, 6, 8], index=[1, 2, 3, 4, 5, 6, 8].index(d["width"]), format_func=lambda x: f"{x}px", key=f"st_wid_{key_name}", label_visibility="collapsed")
        with s_col:
            stl_val = st.selectbox("種類", ["実線", "点線"], index=["実線", "点線"].index(d["style"]), key=f"st_stl_{key_name}", label_visibility="collapsed")
        return {"color": col_val, "width": wid_val, "style": stl_val}

    with st.expander("🎨 線の色・太さ・種類（実線／点線）を変更", expanded=False):
        current_styles = {
            "head": render_line_style_editor("頭の固定枠", "head"),
            "spine": render_line_style_editor("背骨の前傾角度（後方）", "spine"),
            "spine_tilt": render_line_style_editor("背骨の傾き（正面）", "spine_tilt"),
            "shoulders": render_line_style_editor("両肩を結ぶライン", "shoulders"),
            "shaft": render_line_style_editor("シャフトプレーン", "shaft"),
            "hip": render_line_style_editor("ヒップライン", "hip"),
            "ankles": render_line_style_editor("両足首の外側垂線2本", "ankles"),
            "vertical": render_line_style_editor("垂直基準線（90°）", "vertical"),
        }

    st.markdown("---")
    pause_seconds = st.slider("各フェーズの一時停止時間（秒）", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

with col_main:
    st.subheader("3. 動画のアップロード")
    uploaded_file = st.file_uploader("スイング動画を選択またはドラッグ＆ドロップ（MP4 / MOV）")

    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("current_file_id") != file_id:
            with st.spinner("AIが動画全体の骨格を解析しています..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                    tmp_in.write(uploaded_file.read())
                    input_path = tmp_in.name

                frames, frame_data_list, fps, width, height = analyze_skeleton(input_path)
                os.remove(input_path)

                p_indices = auto_detect_p1_to_p10(frame_data_list, fps)
                st.session_state["current_file_id"] = file_id
                st.session_state["frames"] = frames
                st.session_state["frame_data_list"] = frame_data_list
                st.session_state["fps"] = fps
                st.session_state["width"] = width
                st.session_state["height"] = height
                st.session_state["p_indices"] = p_indices
                st.session_state["last_video_bytes"] = None
                st.session_state["last_save_path"] = None
                st.session_state["last_snapshots"] = None
                st.session_state["custom_shapes"] = []

                st.query_params.clear()
                p1_init = p_indices["P1"]
                ref0 = frame_data_list[p1_init] or next((d for d in frame_data_list if d is not None), None)
                if ref0 is not None:
                    front_dir = 1 if ref0["wrist_mid"][0] >= ref0["hip_mid"][0] else -1
                    back_dir = -front_dir
                    ankle_mid_x = (ref0["l_ankle"][0] + ref0["r_ankle"][0]) // 2
                    ground_y = min(height - 10, max(ref0["l_ankle"][1], ref0["r_ankle"][1]) + 10)
                    st.session_state["shaft_p1"] = (max(10, min(width - 10, ankle_mid_x + front_dir * int(width * 0.16))), ground_y)
                    st.session_state["shaft_p2"] = ref0["wrist_mid"]
                    st.session_state["hip_x"] = max(10, min(width - 10, ref0["hip_mid"][0] + back_dir * int(width * 0.06)))
                    if ref0["l_ankle"][0] < ref0["r_ankle"][0]:
                        st.session_state["ankle_l_x"] = max(10, ref0["l_ankle"][0] - 20)
                        st.session_state["ankle_r_x"] = min(width - 10, ref0["r_ankle"][0] + 20)
                    else:
                        st.session_state["ankle_l_x"] = max(10, ref0["r_ankle"][0] - 20)
                        st.session_state["ankle_r_x"] = min(width - 10, ref0["l_ankle"][0] + 20)

        frames = st.session_state["frames"]
        frame_data_list = st.session_state["frame_data_list"]
        fps = st.session_state["fps"]
        width = st.session_state["width"]
        height = st.session_state["height"]
        total_frames = len(frames)
        p_indices = st.session_state["p_indices"]

        if total_frames > 0:
            def get_latest_state():
                qp = st.query_params
                sp1_default = st.session_state.get("shaft_p1", (width // 2, int(height * 0.85)))
                sp2_default = st.session_state.get("shaft_p2", (width // 2, int(height * 0.55)))
                hip_default = st.session_state.get("hip_x", width // 3)
                ank_l_default = st.session_state.get("ankle_l_x", width // 3)
                ank_r_default = st.session_state.get("ankle_r_x", (width * 2) // 3)
                shapes_default = st.session_state.get("custom_shapes", [])
                try:
                    sp1 = (int(qp.get("sp1_x", sp1_default[0])), int(qp.get("sp1_y", sp1_default[1])))
                    sp2 = (int(qp.get("sp2_x", sp2_default[0])), int(qp.get("sp2_y", sp2_default[1])))
                    hip_v = int(qp.get("hip_x", hip_default))
                    ank_l = int(qp.get("ank_l_x", ank_l_default))
                    ank_r = int(qp.get("ank_r_x", ank_r_default))
                    shapes_raw = qp.get("shapes", None)
                    shapes_list = json.loads(shapes_raw) if shapes_raw else shapes_default
                    return sp1, sp2, hip_v, ank_l, ank_r, shapes_list
                except Exception:
                    return sp1_default, sp2_default, hip_default, ank_l_default, ank_r_default, shapes_default

            shaft_p1, shaft_p2, hip_x_val, ankle_l_val, ankle_r_val, custom_shapes_val = get_latest_state()

            for p_code, _, default_pause, default_snap in P_DEFINITIONS:
                s_key = f"slider_{file_id}_{p_code}"
                if s_key not in st.session_state:
                    st.session_state[s_key] = p_indices[p_code]
                chk_p_key = f"chk_pause_{file_id}_{p_code}"
                if chk_p_key not in st.session_state:
                    st.session_state[chk_p_key] = default_pause
                chk_s_key = f"chk_snap_{file_id}_{p_code}"
                if chk_s_key not in st.session_state:
                    st.session_state[chk_s_key] = default_snap

            def step_frame(s_key, delta, max_f):
                st.session_state[s_key] = max(0, min(max_f, st.session_state[s_key] + delta))

            def set_all_snapshots(val):
                for p_code, _, _, _ in P_DEFINITIONS:
                    st.session_state[f"chk_snap_{file_id}_{p_code}"] = val

            current_p_frames = {
                p_code: st.session_state[f"slider_{file_id}_{p_code}"]
                for p_code, _, _, _ in P_DEFINITIONS
            }

            def get_flags_for_frame(target_f_idx):
                flags = {
                    k: line_enabled[k] and is_frame_in_selected_positions(target_f_idx, line_positions[k], current_p_frames)
                    for k in line_enabled
                }
                flags["custom_shapes"] = filter_custom_shapes_for_frame(
                    target_f_idx, custom_shapes_val, line_positions["custom_shapes"], current_p_frames
                )
                return flags

            p1_current = st.session_state[f"slider_{file_id}_P1"]
            ref_preview = frame_data_list[p1_current] or next((d for d in frame_data_list if d is not None), None)

            st.subheader("4. 各ポジションのコマ微調整と出力選択（P1〜P10）")
            st.caption("「⏸ ポーズ」で動画一時停止、「📸 静止画」で切り出し対象ポジションを複数選択できます。")

            sel_bar1, sel_bar2, _ = st.columns([1.3, 1.3, 2.4])
            with sel_bar1:
                st.button("📸 静止画：すべて選択", on_click=set_all_snapshots, args=(True,), use_container_width=True)
            with sel_bar2:
                st.button("📸 静止画：すべて解除", on_click=set_all_snapshots, args=(False,), use_container_width=True)

            checked_phases = {}
            selected_snapshots_list = []
            for row_start in range(0, len(P_DEFINITIONS), 3):
                chunk = P_DEFINITIONS[row_start:row_start + 3]
                cols = st.columns(3)
                for col_i, (p_code, p_name, _, _) in enumerate(chunk):
                    s_key = f"slider_{file_id}_{p_code}"
                    chk_p_key = f"chk_pause_{file_id}_{p_code}"
                    chk_s_key = f"chk_snap_{file_id}_{p_code}"
                    with cols[col_i]:
                        with st.container(border=True):
                            st.markdown(f"**{p_code} {p_name}**")
                            c_chk1, c_chk2 = st.columns(2)
                            with c_chk1:
                                is_paused = st.checkbox("⏸ ポーズ", key=chk_p_key)
                            with c_chk2:
                                is_snap = st.checkbox("📸 静止画", key=chk_s_key)

                            f_idx = st.slider(
                                f"{p_code} コマ番号",
                                0,
                                total_frames - 1,
                                key=s_key,
                                label_visibility="collapsed"
                            )
                            btn_prev, btn_next = st.columns(2)
                            with btn_prev:
                                st.button(
                                    "◀",
                                    key=f"prev_{file_id}_{p_code}",
                                    on_click=step_frame,
                                    args=(s_key, -1, total_frames - 1),
                                    use_container_width=True
                                )
                            with btn_next:
                                st.button(
                                    "▶",
                                    key=f"next_{file_id}_{p_code}",
                                    on_click=step_frame,
                                    args=(s_key, 1, total_frames - 1),
                                    use_container_width=True
                                )

                            f_flags = get_flags_for_frame(f_idx)
                            prev_img = draw_all_annotations(
                                frames[f_idx], frame_data_list[f_idx], ref_preview, None,
                                f_flags["head"], f_flags["spine"], f_flags["spine_tilt"],
                                f_flags["shoulders"], f_flags["shaft"], f_flags["hip"],
                                f_flags["ankles"], f_flags["vertical"],
                                shaft_p1, shaft_p2, hip_x_val, ankle_l_val, ankle_r_val,
                                styles=current_styles, custom_shapes=f_flags["custom_shapes"]
                            )
                            st.image(
                                cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB),
                                caption=f"{p_code} ({f_idx}コマ / {round(f_idx/fps, 2)}秒)",
                                use_container_width=True
                            )

                            if is_paused:
                                checked_phases[f_idx] = p_code
                            if is_snap:
                                selected_snapshots_list.append((p_code, p_name, f_idx))

            p1_idx = st.session_state[f"slider_{file_id}_P1"]

            st.subheader("5. 丸ツマミ移動＆自由描画（直線・円・フリーハンド）")
            editor_p_code = st.selectbox(
                "背景として表示するポジションを選択（P1〜P10）",
                options=P_NAMES_LIST,
                format_func=lambda x: P_LABEL_MAP[x],
                key=f"editor_bg_p_{file_id}"
            )
            editor_f_idx = st.session_state[f"slider_{file_id}_{editor_p_code}"]

            render_interactive_drag_editor(
                frames[editor_f_idx], frame_data_list[editor_f_idx], ref_preview, width, height, editor_p_code,
                show_head, show_spine, show_spine_tilt, show_shoulders, show_shaft, show_hip_wall, show_ankles,
                shaft_p1, shaft_p2, hip_x_val, ankle_l_val, ankle_r_val,
                current_styles, custom_shapes_val
            )

            snap_count = len(selected_snapshots_list)
            act_col1, act_col2 = st.columns(2)
            with act_col1:
                create_video_clicked = st.button("🎬 選択ポーズで分析動画を作成", type="primary", use_container_width=True)
            with act_col2:
                export_img_clicked = st.button(f"📸 選択した静止画像を生成（{snap_count}枚）", use_container_width=True)

            if create_video_clicked:
                shaft_p1, shaft_p2, hip_x_val, ankle_l_val, ankle_r_val, custom_shapes_val = get_latest_state()
                st.session_state["shaft_p1"] = shaft_p1
                st.session_state["shaft_p2"] = shaft_p2
                st.session_state["hip_x"] = hip_x_val
                st.session_state["ankle_l_x"] = ankle_l_val
                st.session_state["ankle_r_x"] = ankle_r_val
                st.session_state["custom_shapes"] = custom_shapes_val

                latest_p_frames = {
                    p_code: st.session_state[f"slider_{file_id}_{p_code}"]
                    for p_code, _, _, _ in P_DEFINITIONS
                }

                with st.spinner("分析動画を書き出してブラウザ再生用に変換しています..."):
                    video_bytes, save_path = render_and_save_video(
                        frames, frame_data_list, fps, width, height,
                        checked_phases, p1_idx, pause_seconds,
                        show_head, show_spine, show_spine_tilt, show_shoulders, show_shaft, show_hip_wall, show_ankles, show_vertical_90,
                        shaft_p1, shaft_p2, hip_x_val, ankle_l_val, ankle_r_val,
                        current_styles, custom_shapes_val, uploaded_file.name,
                        line_positions=line_positions, current_p_frames=latest_p_frames
                    )
                    st.session_state["last_video_bytes"] = video_bytes
                    st.session_state["last_save_path"] = save_path
                    st.rerun()

            if export_img_clicked:
                if not selected_snapshots_list:
                    st.warning("切り出すポジションの「📸 静止画」チェックボックスに少なくとも1つチェックを入れてください。")
                else:
                    shaft_p1, shaft_p2, hip_x_val, ankle_l_val, ankle_r_val, custom_shapes_val = get_latest_state()
                    st.session_state["shaft_p1"] = shaft_p1
                    st.session_state["shaft_p2"] = shaft_p2
                    st.session_state["hip_x"] = hip_x_val
                    st.session_state["ankle_l_x"] = ankle_l_val
                    st.session_state["ankle_r_x"] = ankle_r_val
                    st.session_state["custom_shapes"] = custom_shapes_val

                    latest_p_frames = {
                        p_code: st.session_state[f"slider_{file_id}_{p_code}"]
                        for p_code, _, _, _ in P_DEFINITIONS
                    }

                    saved_items, sheet_bytes, sheet_name, zip_bytes, zip_name = export_selected_snapshots(
                        frames, frame_data_list, selected_snapshots_list, p1_idx,
                        show_head, show_spine, show_spine_tilt, show_shoulders, show_shaft, show_hip_wall, show_ankles, show_vertical_90,
                        shaft_p1, shaft_p2, hip_x_val, ankle_l_val, ankle_r_val,
                        current_styles, custom_shapes_val, uploaded_file.name,
                        line_positions=line_positions, current_p_frames=latest_p_frames
                    )
                    st.session_state["last_snapshots"] = {
                        "items": saved_items,
                        "sheet_bytes": sheet_bytes,
                        "sheet_name": sheet_name,
                        "zip_bytes": zip_bytes,
                        "zip_name": zip_name
                    }
                    st.rerun()

            if st.session_state.get("last_snapshots") is not None:
                snap_data = st.session_state["last_snapshots"]
                items = snap_data["items"]
                st.success(f"選択した {len(items)} 枚の静止画像を生成し、`output` フォルダへ保存しました！")

                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        label="📥 個別画像＋連続写真シートを一括ダウンロード（ZIP）",
                        data=snap_data["zip_bytes"],
                        file_name=snap_data["zip_name"],
                        mime="application/zip",
                        use_container_width=True
                    )
                if snap_data.get("sheet_bytes") is not None:
                    with dl_col2:
                        st.download_button(
                            label="🖼️ 1枚にまとめた連続写真シートをダウンロード（PNG）",
                            data=snap_data["sheet_bytes"],
                            file_name=snap_data["sheet_name"],
                            mime="image/png",
                            use_container_width=True
                        )
                    st.image(snap_data["sheet_bytes"], caption="連続写真シート（選択ポジション一覧）", use_container_width=True)

                snap_cols = st.columns(min(4, len(items)))
                for idx_s, item in enumerate(items):
                    with snap_cols[idx_s % len(snap_cols)]:
                        st.image(item["png_bytes"], caption=f"{item['p_code']} ({item['p_name']})", use_container_width=True)
                        st.download_button(
                            label=f"⬇ {item['p_code']}.png",
                            data=item["png_bytes"],
                            file_name=item["file_name"],
                            mime="image/png",
                            key=f"dl_snap_{item['file_name']}"
                        )

            if st.session_state.get("last_video_bytes") is not None:
                st.success(f"分析動画が完成し、自動保存されました！\n保存先: `{st.session_state['last_save_path']}`")
                st.video(st.session_state["last_video_bytes"])
                st.download_button(
                    label="📥 分析済み動画をダウンロード（MP4）",
                    data=st.session_state["last_video_bytes"],
                    file_name=os.path.basename(st.session_state["last_save_path"]),
                    mime="video/mp4"
                )
