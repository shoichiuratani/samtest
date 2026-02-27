"""
SAM2 POPラベル トラッキングアプリ v2.0
Streamlit + SAM2 を使用した高精度動画オブジェクトトラッキング
マウスドローによるROI設定 + SAM2.1 Tiny モデル
"""

import streamlit as st
import numpy as np
import cv2
import os
import json
import tempfile
import shutil
import time
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch

# ページ設定
st.set_page_config(
    page_title="SAM2 POPラベル トラッキング",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .step-card {
        background: #f8f9fa;
        border-left: 4px solid #2d6a9f;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .step-badge {
        background: #2d6a9f;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 10px;
        font-size: 14px;
    }
    .roi-info-card {
        background: #e8f4fd;
        border: 1px solid #90caf9;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .success-banner {
        background: linear-gradient(135deg, #1b5e20 0%, #388e3c 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .warning-banner {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 12px;
        border-radius: 6px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: bold !important;
        color: #2d6a9f !important;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .canvas-hint {
        background: #e3f2fd;
        border: 1px solid #90caf9;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 14px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 定数設定
# ============================================================
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "sam2.1_hiera_tiny.pt")
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
MAX_ROIS = 8
COLORS = [
    (255, 80,  80),    # 赤
    (80,  220, 80),    # 緑
    (80,  120, 255),   # 青
    (255, 210, 0),     # 黄
    (255, 130, 0),     # オレンジ
    (190, 60,  255),   # 紫
    (0,   220, 220),   # シアン
    (255, 80,  200),   # ピンク
]
COLOR_NAMES = ["赤", "緑", "青", "黄", "オレンジ", "紫", "シアン", "ピンク"]

# 日本語フォントパス
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# デモ動画パス
DEMO_VIDEO_PATH = "/home/user/uploaded_files/メディア3.mp4"
DEMO_RESULT_VIDEO = os.path.join(os.path.dirname(__file__), "pop_tracking_h264.mp4")

# ============================================================
# セッション状態の初期化
# ============================================================
def init_session():
    defaults = {
        "video_path": None,
        "video_frames": None,
        "video_fps": 30.0,
        "video_w": 0,
        "video_h": 0,
        "total_frames": 0,
        "rois": [],
        "preview_frame_idx": 0,
        "tracking_done": False,
        "tracking_results": None,
        "output_video_path": None,
        "temp_dir": None,
        "frames_dir": None,
        "canvas_display_w": 700,
        "canvas_display_h": 400,
        "scale_x": 1.0,
        "scale_y": 1.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ============================================================
# ユーティリティ関数
# ============================================================

@st.cache_resource
def load_sam2_predictor():
    """SAM2ビデオプレディクターをロード（キャッシュ）"""
    from sam2.build_sam import build_sam2_video_predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(
        MODEL_CONFIG,
        CHECKPOINT_PATH,
        device=device
    )
    return predictor


@st.cache_resource
def load_sam2_image_predictor():
    """SAM2 Image Predictorをロード（キャッシュ）"""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_model = build_sam2(MODEL_CONFIG, CHECKPOINT_PATH, device=device)
    predictor = SAM2ImagePredictor(sam_model)
    return predictor


def get_japanese_font(size: int = 20):
    """日本語フォントを取得"""
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def put_text_ja(img_bgr: np.ndarray, text: str, pos: tuple, color: tuple,
                font_size: int = 20, bg: bool = True) -> np.ndarray:
    """PILを使って日本語テキストをOpenCV画像に描画"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = get_japanese_font(font_size)
    x, y = pos

    if bg:
        bbox = draw.textbbox((x, y), text, font=font)
        pad = 3
        draw.rectangle(
            [bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad],
            fill=color
        )
        # テキストは白
        draw.text((x, y), text, font=font, fill=(255, 255, 255))
    else:
        draw.text((x, y), text, font=font, fill=color)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def extract_frames(video_path: str, temp_dir: str,
                   scale: float = 0.5) -> tuple:
    """動画からフレームを抽出 (スケールダウン可能)"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # SAM2処理用のスケール
    proc_w = int(orig_w * scale)
    proc_h = int(orig_h * scale)

    frames_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    frames_rgb = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 処理用リサイズ
        if scale != 1.0:
            frame = cv2.resize(frame, (proc_w, proc_h))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_rgb.append(frame_rgb)
        cv2.imwrite(os.path.join(frames_dir, f"{idx:05d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        idx += 1
    cap.release()
    return frames_rgb, fps, proc_w, proc_h, total, frames_dir, orig_w, orig_h


def draw_rois_on_frame(frame: np.ndarray, rois: list, alpha: float = 0.25) -> np.ndarray:
    """フレームにROIを描画（日本語ラベル対応）"""
    result = frame.copy()
    overlay = frame.copy()
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    for roi in rois:
        x1, y1, x2, y2 = roi["box"]
        color_rgb = COLORS[roi["id"] % len(COLORS)]
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), color_bgr, -1)
        cv2.rectangle(result_bgr, (x1, y1), (x2, y2), color_bgr, 3)

    cv2.addWeighted(overlay_bgr, alpha, result_bgr, 1 - alpha, 0, result_bgr)

    # テキストラベルをPILで描画
    for roi in rois:
        x1, y1, x2, y2 = roi["box"]
        color_rgb = COLORS[roi["id"] % len(COLORS)]
        label = roi.get("label", f"ROI {roi['id']+1}")
        result_bgr = put_text_ja(result_bgr, label, (x1, max(0, y1 - 26)),
                                  color_rgb, font_size=18)

    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def apply_masks_to_frame(frame: np.ndarray, masks_dict: dict, rois: list,
                          mask_opacity: float = 0.40) -> np.ndarray:
    """SAM2マスクをフレームに適用して可視化"""
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    overlay = frame_bgr.copy()
    result = frame_bgr.copy()

    for obj_id, mask in masks_dict.items():
        if mask is None or mask.sum() == 0:
            continue
        color_rgb = COLORS[obj_id % len(COLORS)]
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        roi = next((r for r in rois if r["id"] == obj_id), None)
        label = roi.get("label", f"ROI {obj_id+1}") if roi else f"Obj {obj_id}"

        mask_bool = mask.astype(bool)

        # 半透明カラーマスク
        overlay[mask_bool] = color_bgr

        # 輪郭線（太め）
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color_bgr, 2)

        # ラベルをマスク重心に表示
        if mask_bool.sum() > 0:
            ys, xs = np.where(mask_bool)
            cx, cy = int(xs.mean()), int(ys.mean())
            result = put_text_ja(result, label, (max(0, cx - 30), max(0, cy - 12)),
                                  color_rgb, font_size=16)

    cv2.addWeighted(overlay, mask_opacity, result, 1 - mask_opacity, 0, result)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def run_sam2_tracking(frames_dir: str, rois: list,
                      progress_callback=None) -> dict:
    """SAM2でトラッキング実行（プログレスコールバック付き）"""
    predictor = load_sam2_predictor()

    with torch.inference_mode():
        inference_state = predictor.init_state(video_path=frames_dir)
        predictor.reset_state(inference_state)

        # ROIをSAM2に登録（フレーム0）
        for roi in rois:
            x1, y1, x2, y2 = roi["box"]
            box = np.array([x1, y1, x2, y2], dtype=np.float32)
            obj_id = roi["id"]
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                box=box,
            )

        # 全フレームで伝播
        results = {}
        frame_list = list(predictor.propagate_in_video(inference_state))
        n_total = len(frame_list)

        for i, (frame_idx, obj_ids, mask_logits) in enumerate(frame_list):
            results[frame_idx] = {}
            for j, obj_id in enumerate(obj_ids):
                mask = (mask_logits[j] > 0.0).squeeze().cpu().numpy()
                results[frame_idx][int(obj_id)] = mask.astype(bool)
            if progress_callback and n_total > 0:
                progress_callback(int((i + 1) / n_total * 100))

    return results


def create_output_video(frames: list, tracking_results: dict, rois: list,
                        fps: float, output_path: str,
                        mask_opacity: float = 0.40) -> None:
    """トラッキング結果を動画に書き出し (H.264 via ffmpeg)"""
    h, w = frames[0].shape[:2]
    # まず mp4v で書き出し
    tmp_path = output_path + "_tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

    for i, frame in enumerate(frames):
        vis_frame = frame.copy()
        if i in tracking_results:
            vis_frame = apply_masks_to_frame(
                vis_frame, tracking_results[i], rois, mask_opacity
            )
        # フレーム番号
        frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame_bgr, f"Frame: {i+1}/{len(frames)}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        out.write(frame_bgr)
    out.release()

    # ffmpeg で H.264 に変換
    try:
        cmd = [
            "ffmpeg", "-y", "-i", tmp_path,
            "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
            "-pix_fmt", "yuv420p", output_path
        ]
        subprocess.run(cmd, capture_output=True, timeout=120)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            os.remove(tmp_path)
        else:
            # ffmpeg失敗時はmp4vのまま
            shutil.move(tmp_path, output_path)
    except Exception:
        if os.path.exists(tmp_path):
            shutil.move(tmp_path, output_path)


# ============================================================
# UI: ヘッダー
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🎯 SAM2 POPラベル トラッキングシステム</h1>
    <p style="margin:0; opacity:0.85; font-size:16px;">
        Segment Anything Model 2.1 による高精度マルチオブジェクトトラッキング
        &nbsp;|&nbsp; マウスドローでROI設定
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# サイドバー
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ 設定")

    st.markdown("### 🤖 モデル情報")
    model_loaded = os.path.exists(CHECKPOINT_PATH)
    if model_loaded:
        st.success("✅ SAM2.1 Tiny モデル Ready")
        device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        st.info(f"🖥️ デバイス: {device}")
    else:
        st.error("❌ モデルが見つかりません")

    st.divider()

    st.markdown("### 📋 操作手順")
    st.markdown("""
    <div class="step-card">
        <span class="step-badge">1</span><b>動画アップロード</b><br>
        <small>MP4/AVI/MOV 形式</small>
    </div>
    <div class="step-card">
        <span class="step-badge">2</span><b>ROI設定</b><br>
        <small>🖱️ キャンバス上でドラッグ描画<br>または座標手動入力</small>
    </div>
    <div class="step-card">
        <span class="step-badge">3</span><b>トラッキング実行</b><br>
        <small>SAM2が全フレームを自動解析</small>
    </div>
    <div class="step-card">
        <span class="step-badge">4</span><b>結果確認・DL</b><br>
        <small>フレーム閲覧・動画ダウンロード</small>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🔧 処理設定")
    scale_factor = st.select_slider(
        "処理解像度スケール",
        options=[0.25, 0.33, 0.5, 0.67, 1.0],
        value=0.5,
        help="小さくすると高速・精度低下、大きくすると高精度・低速"
    )
    mask_opacity = st.slider("マスク透明度", 0.1, 0.7, 0.40, 0.05)
    default_label_prefix = st.text_input("ラベルプレフィックス", value="POPラベル")

    if st.session_state.rois:
        st.divider()
        st.markdown("### 📌 登録済みROI一覧")
        for roi in st.session_state.rois:
            color = COLORS[roi["id"] % len(COLORS)]
            hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            x1, y1, x2, y2 = roi["box"]
            st.markdown(f"""
            <div class="roi-info-card">
                <span style="color:{hex_color}; font-size:18px;">■</span>
                <b>{roi.get('label', f"ROI {roi['id']+1}")}</b><br>
                <small>({x1},{y1})→({x2},{y2}) &nbsp; {x2-x1}×{y2-y1}px</small>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# メインコンテンツ: タブ構成
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📁 ① 動画アップロード",
    "🖱️ ② ROI設定（マウスドロー）",
    "🚀 ③ トラッキング実行",
    "📊 ④ 結果確認"
])

# ============================================================
# タブ1: 動画アップロード
# ============================================================
with tab1:
    st.markdown("### 📁 動画ファイルをアップロード")

    col_upload, col_info = st.columns([3, 2])

    with col_upload:
        uploaded_file = st.file_uploader(
            "動画ファイルを選択 (MP4/AVI/MOV/M4V)",
            type=["mp4", "avi", "mov", "m4v"],
            help="POPラベルが映った動画をアップロードしてください"
        )

        use_demo = st.button(
            "🎬 サンプル動画を使用（メディア3.mp4）",
            use_container_width=True,
            type="secondary"
        )

    with col_info:
        st.markdown("""
        <div class="step-card">
            <b>📋 対応フォーマット</b><br>
            MP4, AVI, MOV, M4V<br><br>
            <b>⚡ 推奨スペック</b><br>
            • 解像度: 1920×1080 以下<br>
            • 長さ: 30秒以内<br>
            • フレームレート: 30fps以下<br><br>
            <b>🎯 最適な撮影条件</b><br>
            • POPラベルが鮮明に映っている<br>
            • ブレが少ない
        </div>
        """, unsafe_allow_html=True)

    video_source = None
    if uploaded_file is not None:
        video_source = uploaded_file
    elif use_demo:
        if os.path.exists(DEMO_VIDEO_PATH):
            video_source = DEMO_VIDEO_PATH
        else:
            st.warning("サンプル動画が見つかりません")

    if video_source is not None:
        with st.spinner("🔄 動画を読み込み中..."):
            if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
                shutil.rmtree(st.session_state.temp_dir)
            temp_dir = tempfile.mkdtemp(prefix="sam2_")
            st.session_state.temp_dir = temp_dir

            if isinstance(video_source, str):
                video_path = video_source
            else:
                video_path = os.path.join(temp_dir, "input_video.mp4")
                with open(video_path, "wb") as f:
                    f.write(video_source.getvalue())

            frames, fps, proc_w, proc_h, total, frames_dir, orig_w, orig_h = \
                extract_frames(video_path, temp_dir, scale=scale_factor)

            st.session_state.video_path = video_path
            st.session_state.video_frames = frames
            st.session_state.video_fps = fps
            st.session_state.video_w = proc_w
            st.session_state.video_h = proc_h
            st.session_state.orig_w = orig_w
            st.session_state.orig_h = orig_h
            st.session_state.total_frames = total
            st.session_state.frames_dir = frames_dir
            st.session_state.rois = []
            st.session_state.tracking_done = False
            st.session_state.tracking_results = None
            st.session_state.preview_frame_idx = 0

            # キャンバス表示サイズ計算
            disp_w = min(700, proc_w)
            disp_h = int(proc_h * disp_w / proc_w)
            st.session_state.canvas_display_w = disp_w
            st.session_state.canvas_display_h = disp_h
            st.session_state.scale_x = proc_w / disp_w
            st.session_state.scale_y = proc_h / disp_h

        st.markdown('<div class="success-banner">✅ 動画の読み込みが完了しました！</div>',
                    unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("元解像度", f"{orig_w}×{orig_h}")
        col2.metric("処理解像度", f"{proc_w}×{proc_h}")
        col3.metric("総フレーム数", f"{total}")
        col4.metric("FPS", f"{fps:.1f}")

        st.markdown("#### 🖼️ 動画プレビュー（先頭フレーム）")
        preview_col, _ = st.columns([3, 1])
        with preview_col:
            st.image(frames[0], caption="先頭フレーム", use_container_width=True)

        st.info("➡️ **「② ROI設定（マウスドロー）」タブに進んでROIを設定してください**")

    elif st.session_state.video_frames is not None:
        st.success("✅ 動画は読み込み済みです。「② ROI設定」タブで作業を続けてください。")
        # サムネイル表示
        st.image(st.session_state.video_frames[0],
                 caption=f"読み込み済み動画: {st.session_state.video_w}×{st.session_state.video_h}",
                 width=500)

# ============================================================
# タブ2: ROI設定（マウスドロー + 手動入力）
# ============================================================
with tab2:
    if st.session_state.video_frames is None:
        st.warning("⚠️ まず「① 動画アップロード」タブで動画を読み込んでください。")
        st.stop()

    from streamlit_drawable_canvas import st_canvas

    frames = st.session_state.video_frames
    w = st.session_state.video_w
    h = st.session_state.video_h
    disp_w = st.session_state.canvas_display_w
    disp_h = st.session_state.canvas_display_h
    sx = st.session_state.scale_x
    sy = st.session_state.scale_y

    st.markdown("### 🖱️ ROI設定")

    col_canvas, col_ctrl = st.columns([3, 2])

    with col_ctrl:
        st.markdown("#### 📌 設定オプション")

        # 参照フレーム選択
        frame_idx = st.slider(
            "参照フレーム",
            0, len(frames) - 1,
            st.session_state.preview_frame_idx,
            key="roi_frame_slider"
        )
        st.session_state.preview_frame_idx = frame_idx

        # ROIラベル入力
        roi_label = st.text_input(
            "ROIラベル名",
            value=f"{default_label_prefix} {len(st.session_state.rois)+1}",
            key="roi_label_input"
        )

        st.markdown("---")
        st.markdown("**🖱️ キャンバスで矩形を描いてから「ROIを確定」**")
        confirm_btn = st.button("✅ 描画したROIを確定", use_container_width=True, type="primary")

        st.markdown("---")
        st.markdown("**📐 座標で直接入力する場合**")
        coord_col1, coord_col2 = st.columns(2)
        with coord_col1:
            x1m = st.number_input("左上X", 0, w-1, max(0, w//4), key="mx1")
            y1m = st.number_input("左上Y", 0, h-1, max(0, h//4), key="my1")
        with coord_col2:
            x2m = st.number_input("右下X", 0, w-1, min(w-1, w*3//4), key="mx2")
            y2m = st.number_input("右下Y", 0, h-1, min(h-1, h*3//4), key="my2")
        manual_add = st.button("➕ 座標でROI追加", use_container_width=True)

        if manual_add:
            if x1m < x2m and y1m < y2m:
                if len(st.session_state.rois) < MAX_ROIS:
                    new_id = len(st.session_state.rois)
                    st.session_state.rois.append({
                        "id": new_id,
                        "box": [int(x1m), int(y1m), int(x2m), int(y2m)],
                        "label": roi_label,
                    })
                    st.session_state.tracking_done = False
                    st.session_state.tracking_results = None
                    st.rerun()
                else:
                    st.error(f"最大{MAX_ROIS}個まで")
            else:
                st.error("X1<X2, Y1<Y2 になるよう入力")

        st.markdown("---")
        if st.session_state.rois:
            st.markdown("**🗑️ ROI削除**")
            roi_options = [r.get("label", f"ROI {r['id']+1}") for r in st.session_state.rois]
            del_sel = st.selectbox("削除するROI", roi_options, key="del_roi_select")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                if st.button("🗑️ 選択削除", use_container_width=True):
                    del_idx = roi_options.index(del_sel)
                    st.session_state.rois.pop(del_idx)
                    for i, roi in enumerate(st.session_state.rois):
                        roi["id"] = i
                    st.session_state.tracking_done = False
                    st.rerun()
            with col_d2:
                if st.button("🗑️ 全削除", use_container_width=True, type="secondary"):
                    st.session_state.rois = []
                    st.session_state.tracking_done = False
                    st.rerun()

    with col_canvas:
        # 参照フレームをキャンバス背景に使用
        ref_frame = frames[frame_idx].copy()
        # 既存ROIを描画
        if st.session_state.rois:
            ref_frame_with_rois = draw_rois_on_frame(ref_frame, st.session_state.rois)
        else:
            ref_frame_with_rois = ref_frame

        # 表示サイズにリサイズ
        bg_pil = Image.fromarray(
            cv2.resize(ref_frame_with_rois, (disp_w, disp_h))
        )

        st.markdown("""
        <div class="canvas-hint">
            🖱️ <b>使い方:</b> キャンバス上でドラッグして矩形を描いてください。
            描き終えたら右側の「ROIを確定」ボタンを押してROIとして追加されます。
        </div>
        """, unsafe_allow_html=True)

        canvas_result = st_canvas(
            fill_color="rgba(255, 100, 0, 0.15)",
            stroke_width=2,
            stroke_color="#FF5050",
            background_image=bg_pil,
            update_streamlit=True,
            height=disp_h,
            width=disp_w,
            drawing_mode="rect",
            key=f"canvas_{frame_idx}",
        )

        # 描画結果からROIを取得
        if confirm_btn and canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            rects = [o for o in objects if o.get("type") == "rect"]
            if rects:
                # 最後に描かれた矩形を使用
                rect = rects[-1]
                left = rect.get("left", 0)
                top = rect.get("top", 0)
                rw = rect.get("width", 0)
                rh = rect.get("height", 0)
                scaleX = rect.get("scaleX", 1.0)
                scaleY = rect.get("scaleY", 1.0)

                # キャンバス座標 → 処理解像度座標に変換
                x1c = int(left * sx)
                y1c = int(top * sy)
                x2c = int((left + rw * scaleX) * sx)
                y2c = int((top + rh * scaleY) * sy)

                # クリップ
                x1c = max(0, min(x1c, w - 1))
                y1c = max(0, min(y1c, h - 1))
                x2c = max(0, min(x2c, w - 1))
                y2c = max(0, min(y2c, h - 1))

                if x2c > x1c + 5 and y2c > y1c + 5:
                    if len(st.session_state.rois) < MAX_ROIS:
                        new_id = len(st.session_state.rois)
                        st.session_state.rois.append({
                            "id": new_id,
                            "box": [x1c, y1c, x2c, y2c],
                            "label": roi_label,
                        })
                        st.session_state.tracking_done = False
                        st.session_state.tracking_results = None
                        st.success(f"✅ ROI「{roi_label}」を追加しました ({x1c},{y1c})-({x2c},{y2c})")
                        st.rerun()
                    else:
                        st.error(f"最大{MAX_ROIS}個まで")
                else:
                    st.warning("矩形が小さすぎます。もう少し大きく描いてください。")
            else:
                st.warning("矩形が描かれていません。キャンバス上でドラッグしてROIを描いてください。")

    # ============================================================
    # ROI一覧表示
    # ============================================================
    if st.session_state.rois:
        st.markdown("#### 📋 設定済みROI一覧")
        n_cols = min(len(st.session_state.rois), 4)
        roi_cols = st.columns(n_cols)
        for i, roi in enumerate(st.session_state.rois):
            with roi_cols[i % n_cols]:
                color = COLORS[roi["id"] % len(COLORS)]
                hex_c = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                x1, y1, x2, y2 = roi["box"]
                st.markdown(f"""
                <div style="background:{hex_c}22; border:2px solid {hex_c};
                            padding:10px; border-radius:8px; text-align:center;">
                    <span style="color:{hex_c}; font-size:20px;">■</span><br>
                    <b>{roi.get('label', f"ROI {roi['id']+1}")}</b><br>
                    <small>({x1},{y1})→({x2},{y2})<br>{x2-x1}×{y2-y1}px</small>
                </div>
                """, unsafe_allow_html=True)

        st.info(f"✅ {len(st.session_state.rois)}個のROIが設定されました。「③ トラッキング実行」タブに進んでください。")
    else:
        st.markdown("""
        <div class="warning-banner">
            💡 <b>ヒント:</b> 上のキャンバスでドラッグして矩形を描き、「ROIを確定」を押してください。
            または右パネルで座標を入力して「座標でROI追加」を押してください。
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# タブ3: トラッキング実行
# ============================================================
with tab3:
    if st.session_state.video_frames is None:
        st.warning("⚠️ まず「① 動画アップロード」タブで動画を読み込んでください。")
        st.stop()

    frames = st.session_state.video_frames
    fps = st.session_state.video_fps
    w = st.session_state.video_w
    h = st.session_state.video_h

    st.markdown("### 🚀 SAM2 トラッキング実行")

    if not st.session_state.rois:
        st.warning("⚠️ ROIが設定されていません。「② ROI設定」タブでROIを設定してください。")
    else:
        # サマリー表示
        run_col, info_col = st.columns([2, 3])

        with info_col:
            st.markdown(f"""
            <div class="step-card">
                📊 <b>トラッキング設定サマリー</b><br>
                • ROI数: <b>{len(st.session_state.rois)}個</b><br>
                • 処理フレーム数: <b>{len(frames)}フレーム</b><br>
                • 処理解像度: <b>{w}×{h}</b><br>
                • 推定処理時間: <b>~{max(30, len(frames) // 3 * len(st.session_state.rois))}秒</b> (CPU)
            </div>
            """, unsafe_allow_html=True)

            # ROI一覧
            for roi in st.session_state.rois:
                color = COLORS[roi["id"] % len(COLORS)]
                hex_c = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                x1, y1, x2, y2 = roi["box"]
                st.markdown(f"""
                <div class="roi-info-card">
                    <span style="color:{hex_c};">■</span>
                    <b>{roi.get('label', f"ROI {roi['id']+1}")}</b>
                    <small>({x1},{y1})-({x2},{y2})</small>
                </div>
                """, unsafe_allow_html=True)

        with run_col:
            run_tracking = st.button(
                "🎯 SAM2 トラッキング開始",
                use_container_width=True,
                type="primary",
                disabled=not os.path.exists(CHECKPOINT_PATH),
                help="SAM2モデルで全フレームのトラッキングを実行します"
            )

            if st.session_state.tracking_done:
                st.markdown('<div class="success-banner">✅ トラッキング完了済み</div>',
                            unsafe_allow_html=True)
                st.info("「④ 結果確認」タブで結果を確認できます")

        if run_tracking:
            prog_bar = st.progress(0, "初期化中...")
            status_box = st.empty()
            time_box = st.empty()

            try:
                t0 = time.time()
                prog_bar.progress(5, "SAM2モデルを準備中...")
                status_box.info("🔄 SAM2モデルをロード中... (初回は30秒程度かかります)")

                predictor = load_sam2_predictor()
                prog_bar.progress(20, "モデルロード完了")

                status_box.info("🔄 SAM2推論を実行中... しばらくお待ちください")

                def _update_prog(pct):
                    prog_bar.progress(20 + int(pct * 0.6), f"推論中... {pct}%")
                    elapsed = time.time() - t0
                    time_box.caption(f"⏱️ 経過時間: {elapsed:.1f}秒")

                tracking_results = run_sam2_tracking(
                    st.session_state.frames_dir,
                    st.session_state.rois,
                    progress_callback=_update_prog
                )

                prog_bar.progress(85, "動画生成中...")
                status_box.info("🔄 出力動画を生成中...")

                output_path = os.path.join(st.session_state.temp_dir, "tracked_output.mp4")
                create_output_video(
                    frames, tracking_results, st.session_state.rois,
                    fps, output_path, mask_opacity=mask_opacity
                )

                elapsed = time.time() - t0
                prog_bar.progress(100, "完了！")

                st.session_state.tracking_results = tracking_results
                st.session_state.tracking_done = True
                st.session_state.output_video_path = output_path

                status_box.empty()
                time_box.empty()
                prog_bar.empty()

                st.markdown(f"""
                <div class="success-banner">
                    🎉 トラッキング完了！ 処理時間: {elapsed:.1f}秒<br>
                    「④ 結果確認」タブで結果を確認・ダウンロードできます
                </div>
                """, unsafe_allow_html=True)
                time.sleep(1)
                st.rerun()

            except Exception as e:
                prog_bar.empty()
                status_box.error(f"❌ エラーが発生しました: {str(e)}")
                st.exception(e)

    # ============================================================
    # デモ結果を表示するオプション
    # ============================================================
    st.divider()
    st.markdown("### 🎬 デモ: 事前計算済みトラッキング結果")
    st.markdown("サンプル動画（メディア3.mp4）のトラッキング結果をプレビューできます。")

    if os.path.exists(DEMO_RESULT_VIDEO):
        demo_col1, demo_col2 = st.columns([2, 1])
        with demo_col1:
            with open(DEMO_RESULT_VIDEO, "rb") as f:
                demo_video_bytes = f.read()
            st.video(demo_video_bytes)
        with demo_col2:
            st.markdown("""
            <div class="step-card">
                <b>📊 デモ結果の詳細</b><br><br>
                • 動画: メディア3.mp4<br>
                • 解像度: 596×349 (50%スケール)<br>
                • フレーム数: 116<br>
                • FPS: 29.97<br>
                • ROI数: 7個<br><br>
                <b>検出したPOPラベル:</b><br>
                • 1段目左/右<br>
                • 2段目<br>
                • 3段目左/右<br>
                • 4段目左/右<br>
            </div>
            """, unsafe_allow_html=True)
            st.download_button(
                "⬇️ デモ動画をダウンロード",
                data=demo_video_bytes,
                file_name="pop_tracking_demo.mp4",
                mime="video/mp4",
                use_container_width=True
            )
    else:
        st.info("デモ動画が見つかりません。")

# ============================================================
# タブ4: 結果確認
# ============================================================
with tab4:
    # デモ結果も表示できるように分岐
    has_custom = (st.session_state.tracking_done and
                  st.session_state.tracking_results is not None)
    has_demo = os.path.exists(DEMO_RESULT_VIDEO)

    if not has_custom and not has_demo:
        st.info("⏳ トラッキングがまだ実行されていません。「③ トラッキング実行」タブでトラッキングを実行してください。")
        st.stop()

    st.markdown("### 📊 トラッキング結果")

    if has_custom:
        frames = st.session_state.video_frames
        rois = st.session_state.rois
        tracking_results = st.session_state.tracking_results

        # ============================================================
        # 統計情報
        # ============================================================
        st.markdown("#### 📈 トラッキング統計")
        stat_cols = st.columns(4)

        total_masks = sum(
            sum(1 for mask in masks.values() if mask is not None and mask.sum() > 0)
            for masks in tracking_results.values()
        )
        tracked_frames = sum(
            1 for masks in tracking_results.values()
            if any(m is not None and m.sum() > 0 for m in masks.values())
        )
        stat_cols[0].metric("追跡オブジェクト数", len(rois))
        stat_cols[1].metric("総フレーム数", len(frames))
        stat_cols[2].metric("検出フレーム数", tracked_frames)
        stat_cols[3].metric("総マスク数", total_masks)

        st.divider()

        # ============================================================
        # フレームビューア
        # ============================================================
        st.markdown("#### 🎬 フレーム別結果ビューア")
        view_col, info_col = st.columns([3, 2])

        with info_col:
            result_frame_idx = st.slider(
                "確認フレーム", 0, len(frames)-1, 0,
                key="result_frame_slider"
            )
            show_comparison = st.checkbox("原画との比較表示", value=False)

            if result_frame_idx in tracking_results:
                frame_masks = tracking_results[result_frame_idx]
                st.markdown("**🔍 このフレームの検出結果**")
                for obj_id, mask in frame_masks.items():
                    roi = next((r for r in rois if r["id"] == obj_id), None)
                    label = roi.get("label", f"ROI {obj_id+1}") if roi else f"Obj {obj_id}"
                    color = COLORS[obj_id % len(COLORS)]
                    hex_c = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

                    if mask is not None and mask.sum() > 0:
                        area = int(mask.sum())
                        ys, xs = np.where(mask)
                        cx, cy = int(xs.mean()), int(ys.mean())
                        st.markdown(f"""
                        <div style="background:{hex_c}22; border-left:4px solid {hex_c};
                                    padding:10px; border-radius:6px; margin:5px 0;">
                            <span style="color:{hex_c};">■</span> <b>{label}</b><br>
                            <small>
                            面積: {area:,} px²<br>
                            重心: ({cx}, {cy})<br>
                            範囲: ({xs.min()},{ys.min()})-({xs.max()},{ys.max()})
                            </small>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background:#f5f5f5; border-left:4px solid #aaa;
                                    padding:8px; border-radius:6px; margin:5px 0;">
                            <span style="color:#aaa;">■</span> {label}: 未検出
                        </div>
                        """, unsafe_allow_html=True)

        with view_col:
            result_frame = frames[result_frame_idx].copy()
            if result_frame_idx in tracking_results:
                vis_frame = apply_masks_to_frame(
                    result_frame,
                    tracking_results[result_frame_idx],
                    rois, mask_opacity
                )
            else:
                vis_frame = result_frame

            if show_comparison:
                comp_c1, comp_c2 = st.columns(2)
                with comp_c1:
                    st.image(result_frame, caption="原画", use_container_width=True)
                with comp_c2:
                    st.image(vis_frame, caption="SAM2結果", use_container_width=True)
            else:
                st.image(vis_frame,
                         caption=f"フレーム {result_frame_idx+1}/{len(frames)} - SAM2トラッキング",
                         use_container_width=True)

        # ============================================================
        # 面積グラフ
        # ============================================================
        st.divider()
        st.markdown("#### 📉 オブジェクト面積の時系列")
        try:
            import pandas as pd
            chart_data = {}
            for roi in rois:
                obj_id = roi["id"]
                label = roi.get("label", f"ROI {obj_id+1}")
                areas = []
                for fi in range(len(frames)):
                    if fi in tracking_results and obj_id in tracking_results[fi]:
                        mask = tracking_results[fi][obj_id]
                        area = int(mask.sum()) if mask is not None else 0
                    else:
                        area = 0
                    areas.append(area)
                chart_data[label] = areas
            df = pd.DataFrame(chart_data, index=range(len(frames)))
            df.index.name = "フレーム"
            st.line_chart(df, height=200)
        except Exception as e:
            st.warning(f"グラフ表示エラー: {e}")

        # ============================================================
        # ダウンロード
        # ============================================================
        st.divider()
        st.markdown("#### 💾 結果のダウンロード")
        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            if (st.session_state.output_video_path and
                    os.path.exists(st.session_state.output_video_path)):
                with open(st.session_state.output_video_path, "rb") as f:
                    video_bytes = f.read()
                st.download_button(
                    "⬇️ トラッキング動画 (MP4)",
                    data=video_bytes,
                    file_name="sam2_tracked_video.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    type="primary"
                )
                sz_mb = len(video_bytes) / 1024 / 1024
                st.caption(f"ファイルサイズ: {sz_mb:.2f} MB")
            else:
                if st.button("🎬 動画を再生成", use_container_width=True):
                    with st.spinner("動画を生成中..."):
                        output_path = os.path.join(
                            st.session_state.temp_dir, "tracked_output.mp4"
                        )
                        create_output_video(frames, tracking_results, rois,
                                            st.session_state.video_fps,
                                            output_path, mask_opacity)
                        st.session_state.output_video_path = output_path
                    st.rerun()

        with dl_col2:
            # JSON エクスポート
            export_data = {
                "video_info": {
                    "width": st.session_state.video_w,
                    "height": st.session_state.video_h,
                    "fps": st.session_state.video_fps,
                    "total_frames": st.session_state.total_frames,
                },
                "rois": st.session_state.rois,
                "tracking_summary": {
                    str(fi): {
                        str(oid): {
                            "detected": bool(mask is not None and mask.sum() > 0),
                            "area_pixels": int(mask.sum()) if mask is not None else 0,
                        }
                        for oid, mask in masks.items()
                    }
                    for fi, masks in tracking_results.items()
                }
            }
            json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
            st.download_button(
                "⬇️ トラッキングデータ (JSON)",
                data=json_str.encode("utf-8"),
                file_name="tracking_data.json",
                mime="application/json",
                use_container_width=True,
            )

        # ============================================================
        # サムネイルグリッド
        # ============================================================
        st.divider()
        st.markdown("#### 🗂️ フレームサムネイル")
        with st.expander("サムネイル一覧を表示"):
            thumb_cols = 6
            rows = (len(frames) + thumb_cols - 1) // thumb_cols
            for row in range(min(rows, 4)):
                cols = st.columns(thumb_cols)
                for col_idx in range(thumb_cols):
                    fi = row * thumb_cols + col_idx
                    if fi < len(frames):
                        with cols[col_idx]:
                            thumb = frames[fi].copy()
                            if fi in tracking_results:
                                thumb = apply_masks_to_frame(
                                    thumb, tracking_results[fi], rois, mask_opacity
                                )
                            thumb_small = cv2.resize(thumb, (200, 120))
                            st.image(thumb_small, caption=f"F{fi+1}",
                                     use_container_width=True)

    else:
        # カスタムトラッキングがない場合はデモを表示
        st.info("📌 カスタムトラッキング結果がありません。デモ結果を表示しています。")
        st.markdown("#### 🎬 デモトラッキング動画（メディア3.mp4）")
        with open(DEMO_RESULT_VIDEO, "rb") as f:
            demo_bytes = f.read()
        st.video(demo_bytes)
        st.download_button(
            "⬇️ デモ動画をダウンロード",
            data=demo_bytes,
            file_name="pop_tracking_demo.mp4",
            mime="video/mp4",
        )

        # デモフレーム画像表示
        st.divider()
        st.markdown("#### 🖼️ デモ結果フレーム")
        demo_frames = sorted(
            [f for f in os.listdir(os.path.dirname(__file__))
             if f.startswith("result_frame_") and f.endswith(".jpg")]
        )
        if demo_frames:
            cols = st.columns(len(demo_frames))
            for i, fname in enumerate(demo_frames):
                fpath = os.path.join(os.path.dirname(__file__), fname)
                with cols[i]:
                    st.image(fpath, caption=fname.replace("result_frame_", "F").replace(".jpg", ""),
                             use_container_width=True)
