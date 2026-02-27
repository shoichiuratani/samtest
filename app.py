"""
SAM2 POPラベル トラッキングアプリ
Streamlit + SAM2 を使用した高精度動画オブジェクトトラッキング
"""

import streamlit as st
import numpy as np
import cv2
import os
import json
import tempfile
import shutil
import time
from pathlib import Path
from PIL import Image
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
    .tracking-stats {
        background: #263238;
        color: #eceff1;
        padding: 15px;
        border-radius: 8px;
        font-family: monospace;
        font-size: 13px;
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
</style>
""", unsafe_allow_html=True)

# ============================================================
# 定数設定
# ============================================================
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "sam2.1_hiera_tiny.pt")
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
MAX_ROIS = 8
COLORS = [
    (255, 50, 50),    # 赤
    (50, 255, 50),    # 緑
    (50, 100, 255),   # 青
    (255, 200, 0),    # 黄
    (255, 100, 0),    # オレンジ
    (180, 0, 255),    # 紫
    (0, 220, 220),    # シアン
    (255, 50, 200),   # ピンク
]
COLOR_NAMES = ["赤", "緑", "青", "黄", "オレンジ", "紫", "シアン", "ピンク"]

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
        "rois": [],               # [{"id": int, "box": [x1,y1,x2,y2], "label": str}]
        "drawing": False,
        "draw_start": None,
        "preview_frame_idx": 0,
        "tracking_done": False,
        "tracking_results": None, # {frame_idx: {obj_id: mask}}
        "output_video_path": None,
        "temp_dir": None,
        "current_tool": "rect",
        "selected_roi_color": 0,
        "roi_labels": {},
        "tracking_progress": 0,
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


def extract_frames(video_path: str, temp_dir: str) -> tuple:
    """動画からフレームを抽出してJPEGとして保存"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    frames_rgb = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_rgb.append(frame_rgb)
        # SAM2用にJPEG保存（ゼロパディング5桁）
        cv2.imwrite(os.path.join(frames_dir, f"{idx:05d}.jpg"), frame)
        idx += 1
    cap.release()
    return frames_rgb, fps, w, h, total, frames_dir


def draw_rois_on_frame(frame: np.ndarray, rois: list, alpha: float = 0.25) -> np.ndarray:
    """フレームにROIを描画"""
    result = frame.copy()
    overlay = frame.copy()
    for roi in rois:
        x1, y1, x2, y2 = roi["box"]
        color = COLORS[roi["id"] % len(COLORS)]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        label = roi.get("label", f"ROI {roi['id']+1}")
        # ラベル背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(result, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    return result


def apply_masks_to_frame(frame: np.ndarray, masks_dict: dict, rois: list) -> np.ndarray:
    """SAM2マスクをフレームに適用して可視化"""
    result = frame.copy()
    overlay = frame.copy()

    for obj_id, mask in masks_dict.items():
        color = COLORS[obj_id % len(COLORS)]
        roi = next((r for r in rois if r["id"] == obj_id), None)
        label = roi.get("label", f"ROI {obj_id+1}") if roi else f"Obj {obj_id}"

        if mask is None or mask.sum() == 0:
            continue

        mask_bool = mask.astype(bool)

        # 半透明マスク
        overlay[mask_bool] = color

        # 輪郭線
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2)

        # ラベルをマスク重心に表示
        if mask_bool.sum() > 0:
            ys, xs = np.where(mask_bool)
            cx, cy = int(xs.mean()), int(ys.mean())
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result, (cx - 3, cy - th - 8), (cx + tw + 3, cy), color, -1)
            cv2.putText(result, label, (cx, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.addWeighted(overlay, 0.35, result, 0.65, 0, result)
    return result


def run_sam2_tracking(frames_dir: str, rois: list) -> dict:
    """SAM2でトラッキング実行"""
    predictor = load_sam2_predictor()

    with torch.inference_mode():
        inference_state = predictor.init_state(video_path=frames_dir)
        predictor.reset_state(inference_state)

        # ROIをSAM2に登録（フレーム0）
        for roi in rois:
            x1, y1, x2, y2 = roi["box"]
            box = np.array([x1, y1, x2, y2], dtype=np.float32)
            obj_id = roi["id"]
            _, _, _ = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                box=box,
            )

        # 全フレームで伝播
        results = {}
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            results[frame_idx] = {}
            for i, obj_id in enumerate(obj_ids):
                mask = (mask_logits[i] > 0.0).squeeze().cpu().numpy()
                results[frame_idx][obj_id] = mask

    return results


def create_output_video(frames: list, tracking_results: dict, rois: list,
                        fps: float, output_path: str) -> None:
    """トラッキング結果を動画に書き出し"""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for i, frame in enumerate(frames):
        vis_frame = frame.copy()
        if i in tracking_results:
            vis_frame = apply_masks_to_frame(vis_frame, tracking_results[i], rois)

        # フレーム番号を表示
        cv2.putText(vis_frame, f"Frame: {i+1}/{len(frames)}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        out.write(cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))

    out.release()


# ============================================================
# UI: ヘッダー
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🎯 SAM2 POPラベル トラッキングシステム</h1>
    <p style="margin:0; opacity:0.85; font-size:16px;">
        Segment Anything Model 2 による高精度マルチオブジェクトトラッキング
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

    st.markdown("### 📋 使い方")
    st.markdown("""
    <div class="step-card">
        <span class="step-badge">1</span><b>動画アップロード</b><br>
        <small>MP4/AVI/MOV 形式に対応</small>
    </div>
    <div class="step-card">
        <span class="step-badge">2</span><b>ROI設定</b><br>
        <small>トラッキングしたい領域をドラッグで描画</small>
    </div>
    <div class="step-card">
        <span class="step-badge">3</span><b>トラッキング実行</b><br>
        <small>SAM2が全フレームを自動解析</small>
    </div>
    <div class="step-card">
        <span class="step-badge">4</span><b>結果確認・ダウンロード</b><br>
        <small>フレーム閲覧と動画ダウンロード</small>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🎨 ROI設定オプション")
    default_label_prefix = st.text_input("ラベルプレフィックス", value="POPラベル")
    show_mask_boundary = st.checkbox("マスク境界線を強調", value=True)
    mask_opacity = st.slider("マスク透明度", 0.1, 0.7, 0.35, 0.05)

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
                <small>位置: ({x1},{y1}) - ({x2},{y2})<br>
                サイズ: {x2-x1}×{y2-y1}px</small>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# メインコンテンツ: タブ構成
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "📁 ① 動画アップロード",
    "🎯 ② ROI設定・トラッキング",
    "📊 ③ 結果確認"
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

        # デモ動画ボタン
        use_demo = st.button(
            "🎬 サンプル動画を使用（アップロード済み動画）",
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
            • 十分な明るさ<br>
            • POPラベルが鮮明に映っている<br>
            • ブレが少ない
        </div>
        """, unsafe_allow_html=True)

    # ファイル処理
    video_source = None
    if uploaded_file is not None:
        video_source = uploaded_file
    elif use_demo:
        demo_path = "/home/user/uploaded_files/メディア3.mp4"
        if os.path.exists(demo_path):
            video_source = demo_path
        else:
            st.warning("サンプル動画が見つかりません")

    if video_source is not None:
        with st.spinner("🔄 動画を読み込み中..."):
            # 一時ディレクトリ作成
            if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
                shutil.rmtree(st.session_state.temp_dir)
            temp_dir = tempfile.mkdtemp(prefix="sam2_")
            st.session_state.temp_dir = temp_dir

            # 動画パスを設定
            if isinstance(video_source, str):
                video_path = video_source
            else:
                video_path = os.path.join(temp_dir, "input_video.mp4")
                with open(video_path, "wb") as f:
                    f.write(video_source.getvalue())

            # フレーム抽出
            frames, fps, w, h, total, frames_dir = extract_frames(video_path, temp_dir)

            # セッション状態を更新
            st.session_state.video_path = video_path
            st.session_state.video_frames = frames
            st.session_state.video_fps = fps
            st.session_state.video_w = w
            st.session_state.video_h = h
            st.session_state.total_frames = total
            st.session_state.frames_dir = frames_dir
            st.session_state.rois = []
            st.session_state.tracking_done = False
            st.session_state.tracking_results = None
            st.session_state.preview_frame_idx = 0

        st.markdown('<div class="success-banner">✅ 動画の読み込みが完了しました！</div>',
                    unsafe_allow_html=True)

        # 動画情報表示
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("解像度", f"{w} × {h}")
        col2.metric("フレームレート", f"{fps:.1f} fps")
        col3.metric("総フレーム数", f"{total}")
        col4.metric("動画尺", f"{total/fps:.2f} 秒")

        # プレビュー
        st.markdown("#### 🖼️ 動画プレビュー（先頭フレーム）")
        preview_col, _ = st.columns([3, 1])
        with preview_col:
            st.image(frames[0], caption="先頭フレーム", use_container_width=True)

        st.info("➡️ **「② ROI設定・トラッキング」タブに進んでROI（追跡領域）を設定してください**")

    elif st.session_state.video_frames is not None:
        st.success("✅ 動画は読み込み済みです。「② ROI設定・トラッキング」タブで作業を続けてください。")

# ============================================================
# タブ2: ROI設定・トラッキング
# ============================================================
with tab2:
    if st.session_state.video_frames is None:
        st.warning("⚠️ まず「① 動画アップロード」タブで動画を読み込んでください。")
        st.stop()

    frames = st.session_state.video_frames
    fps = st.session_state.video_fps
    w = st.session_state.video_w
    h = st.session_state.video_h

    st.markdown("### 🎯 ROI（追跡領域）の設定")

    # ============================================================
    # ROI設定セクション
    # ============================================================
    col_main, col_ctrl = st.columns([3, 2])

    with col_ctrl:
        st.markdown("#### 📌 ROI追加")

        st.markdown("**フレームを選択（ROI設定基準フレーム）**")
        frame_idx = st.slider(
            "参照フレーム",
            0, len(frames) - 1,
            st.session_state.preview_frame_idx,
            key="roi_frame_slider"
        )
        st.session_state.preview_frame_idx = frame_idx

        st.markdown("**ROI名を入力**")
        roi_label = st.text_input(
            "ラベル名",
            value=f"{default_label_prefix} {len(st.session_state.rois)+1}",
            key="roi_label_input"
        )

        st.markdown("**ROI座標を手動入力**")
        coord_col1, coord_col2 = st.columns(2)
        with coord_col1:
            x1 = st.number_input("左上X", 0, w - 1, max(0, w // 4), key="roi_x1")
            y1 = st.number_input("左上Y", 0, h - 1, max(0, h // 4), key="roi_y1")
        with coord_col2:
            x2 = st.number_input("右下X", 0, w - 1, min(w - 1, w * 3 // 4), key="roi_x2")
            y2 = st.number_input("右下Y", 0, h - 1, min(h - 1, h * 3 // 4), key="roi_y2")

        # ROI追加ボタン
        if st.button("➕ ROIを追加", use_container_width=True, type="primary"):
            if x1 < x2 and y1 < y2:
                if len(st.session_state.rois) < MAX_ROIS:
                    new_id = len(st.session_state.rois)
                    st.session_state.rois.append({
                        "id": new_id,
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "label": roi_label,
                    })
                    st.session_state.tracking_done = False
                    st.session_state.tracking_results = None
                    st.rerun()
                else:
                    st.error(f"最大{MAX_ROIS}個までROIを設定できます")
            else:
                st.error("X1 < X2、Y1 < Y2 となるよう入力してください")

        st.divider()

        # ROI削除
        if st.session_state.rois:
            st.markdown("**ROIの削除**")
            roi_options = [r.get('label', f"ROI {r['id']+1}") for r in st.session_state.rois]
            selected_del = st.selectbox("削除するROIを選択", roi_options, key="del_roi_select")
            if st.button("🗑️ 選択したROIを削除", use_container_width=True):
                del_idx = roi_options.index(selected_del)
                st.session_state.rois.pop(del_idx)
                # IDを再割り当て
                for i, roi in enumerate(st.session_state.rois):
                    roi["id"] = i
                st.session_state.tracking_done = False
                st.session_state.tracking_results = None
                st.rerun()

            if st.button("🗑️ すべてのROIをクリア", use_container_width=True, type="secondary"):
                st.session_state.rois = []
                st.session_state.tracking_done = False
                st.session_state.tracking_results = None
                st.rerun()

    with col_main:
        # 現在のフレームにROIを描画して表示
        current_frame = frames[frame_idx].copy()
        if st.session_state.rois:
            display_frame = draw_rois_on_frame(current_frame, st.session_state.rois)
        else:
            display_frame = current_frame

        st.image(display_frame,
                 caption=f"フレーム {frame_idx+1}/{len(frames)} - ROI設定プレビュー",
                 use_container_width=True)

        if not st.session_state.rois:
            st.markdown("""
            <div class="warning-banner">
                💡 <b>ヒント:</b> 右側のパネルでROI座標を入力し「ROIを追加」ボタンを押してください。<br>
                フレームスライダーで参照フレームを選択してから、
                追跡したいPOPラベルの領域を指定してください。
            </div>
            """, unsafe_allow_html=True)

    # ============================================================
    # ROI一覧表示
    # ============================================================
    if st.session_state.rois:
        st.markdown("#### 📋 設定済みROI一覧")
        roi_cols = st.columns(min(len(st.session_state.rois), 4))
        for i, roi in enumerate(st.session_state.rois):
            with roi_cols[i % 4]:
                color = COLORS[roi["id"] % len(COLORS)]
                hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                x1, y1, x2, y2 = roi["box"]
                st.markdown(f"""
                <div style="background:{hex_color}22; border:2px solid {hex_color};
                            padding:10px; border-radius:8px; text-align:center;">
                    <span style="color:{hex_color}; font-size:20px;">■</span><br>
                    <b>{roi.get('label', f"ROI {roi['id']+1}")}</b><br>
                    <small>({x1},{y1})-({x2},{y2})<br>{x2-x1}×{y2-y1}px</small>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # ============================================================
        # トラッキング実行
        # ============================================================
        st.markdown("### 🚀 SAM2トラッキング実行")

        run_col, info_col = st.columns([2, 3])
        with info_col:
            st.markdown(f"""
            <div class="step-card">
                📊 <b>トラッキング設定サマリー</b><br>
                • ROI数: <b>{len(st.session_state.rois)}個</b><br>
                • 対象フレーム: <b>{len(frames)}フレーム</b><br>
                • 解像度: <b>{w}×{h}</b><br>
                • 推定時間: <b>~{len(frames) * len(st.session_state.rois) // 10 + 5}秒</b> (CPU)
            </div>
            """, unsafe_allow_html=True)

        with run_col:
            run_tracking = st.button(
                "🎯 SAM2トラッキング開始",
                use_container_width=True,
                type="primary",
                disabled=not os.path.exists(CHECKPOINT_PATH)
            )

        if run_tracking:
            prog_container = st.empty()
            status_container = st.empty()
            time_container = st.empty()

            try:
                start_time = time.time()
                prog_container.progress(0, "SAM2モデルを準備中...")

                status_container.info("🔄 SAM2モデルをロード中...")
                predictor = load_sam2_predictor()
                prog_container.progress(20, "モデルロード完了")

                status_container.info("🔄 SAM2推論を実行中...")
                prog_container.progress(40, "フレーム解析中...")

                # トラッキング実行
                tracking_results = run_sam2_tracking(
                    st.session_state.frames_dir,
                    st.session_state.rois
                )

                prog_container.progress(80, "動画生成中...")
                status_container.info("🔄 出力動画を生成中...")

                # 出力動画作成
                output_path = os.path.join(st.session_state.temp_dir, "tracked_output.mp4")
                create_output_video(frames, tracking_results, st.session_state.rois,
                                    fps, output_path)

                prog_container.progress(100, "完了!")
                elapsed = time.time() - start_time

                st.session_state.tracking_results = tracking_results
                st.session_state.tracking_done = True
                st.session_state.output_video_path = output_path

                status_container.empty()
                time_container.empty()
                prog_container.empty()

                st.markdown(f"""
                <div class="success-banner">
                    🎉 トラッキング完了！ 処理時間: {elapsed:.1f}秒<br>
                    「③ 結果確認」タブで結果を確認・ダウンロードできます
                </div>
                """, unsafe_allow_html=True)

                time.sleep(1)
                st.rerun()

            except Exception as e:
                prog_container.empty()
                status_container.error(f"❌ エラーが発生しました: {str(e)}")
                st.exception(e)

        if st.session_state.tracking_done:
            st.markdown('<div class="success-banner">✅ トラッキング済み - 「③ 結果確認」タブで確認できます</div>',
                        unsafe_allow_html=True)

# ============================================================
# タブ3: 結果確認
# ============================================================
with tab3:
    if not st.session_state.tracking_done or st.session_state.tracking_results is None:
        st.info("⏳ トラッキングがまだ実行されていません。「② ROI設定・トラッキング」タブでトラッキングを実行してください。")
        st.stop()

    frames = st.session_state.video_frames
    rois = st.session_state.rois
    tracking_results = st.session_state.tracking_results

    st.markdown("### 📊 トラッキング結果")

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
    # フレーム別結果ビューア
    # ============================================================
    st.markdown("#### 🎬 フレーム別結果ビューア")

    view_col, info_col = st.columns([3, 2])

    with info_col:
        result_frame_idx = st.slider(
            "確認フレーム",
            0, len(frames) - 1, 0,
            key="result_frame_slider"
        )

        # 現フレームの統計
        if result_frame_idx in tracking_results:
            frame_masks = tracking_results[result_frame_idx]
            st.markdown("**🔍 このフレームの検出結果**")
            for obj_id, mask in frame_masks.items():
                roi = next((r for r in rois if r["id"] == obj_id), None)
                label = roi.get("label", f"ROI {obj_id+1}") if roi else f"Obj {obj_id}"
                color = COLORS[obj_id % len(COLORS)]
                hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

                if mask is not None and mask.sum() > 0:
                    area = mask.sum()
                    ys, xs = np.where(mask)
                    cx, cy = int(xs.mean()), int(ys.mean())
                    st.markdown(f"""
                    <div style="background:{hex_color}22; border-left:4px solid {hex_color};
                                padding:10px; border-radius:6px; margin:5px 0;">
                        <span style="color:{hex_color};">■</span> <b>{label}</b><br>
                        <small>
                        面積: {area:,} px<br>
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

        # アニメーション再生ボタン
        st.divider()
        show_comparison = st.checkbox("原画との比較表示", value=False)

    with view_col:
        result_frame = frames[result_frame_idx].copy()
        if result_frame_idx in tracking_results:
            vis_frame = apply_masks_to_frame(
                result_frame,
                tracking_results[result_frame_idx],
                rois
            )
        else:
            vis_frame = result_frame

        if show_comparison:
            # 横並び比較
            comp_cols = st.columns(2)
            with comp_cols[0]:
                st.image(result_frame, caption="原画", use_container_width=True)
            with comp_cols[1]:
                st.image(vis_frame, caption="SAM2トラッキング結果", use_container_width=True)
        else:
            st.image(vis_frame,
                     caption=f"フレーム {result_frame_idx+1}/{len(frames)} - SAM2トラッキング結果",
                     use_container_width=True)

    # ============================================================
    # オブジェクト別面積グラフ
    # ============================================================
    st.divider()
    st.markdown("#### 📉 オブジェクト面積の時系列グラフ")

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
        st.warning(f"グラフ表示中にエラー: {e}")

    # ============================================================
    # 動画ダウンロード
    # ============================================================
    st.divider()
    st.markdown("#### 💾 結果動画のダウンロード")

    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        if (st.session_state.output_video_path and
                os.path.exists(st.session_state.output_video_path)):
            with open(st.session_state.output_video_path, "rb") as f:
                video_bytes = f.read()
            st.download_button(
                label="⬇️ トラッキング動画をダウンロード (MP4)",
                data=video_bytes,
                file_name="sam2_tracked_video.mp4",
                mime="video/mp4",
                use_container_width=True,
                type="primary"
            )
            size_mb = len(video_bytes) / (1024 * 1024)
            st.caption(f"ファイルサイズ: {size_mb:.2f} MB")
        else:
            if st.button("🎬 動画を再生成", use_container_width=True):
                with st.spinner("動画を生成中..."):
                    output_path = os.path.join(st.session_state.temp_dir, "tracked_output.mp4")
                    create_output_video(frames, tracking_results, rois,
                                        fps, output_path)
                    st.session_state.output_video_path = output_path
                st.rerun()

    with dl_col2:
        # JSON形式でトラッキングデータをエクスポート
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
                        "area_pixels": int(mask.sum()) if (mask is not None) else 0,
                    }
                    for oid, mask in masks.items()
                }
                for fi, masks in tracking_results.items()
            }
        }
        json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
        st.download_button(
            label="⬇️ トラッキングデータ (JSON)",
            data=json_str.encode("utf-8"),
            file_name="tracking_data.json",
            mime="application/json",
            use_container_width=True,
        )

    # ============================================================
    # フレームグリッド（サムネイル一覧）
    # ============================================================
    st.divider()
    st.markdown("#### 🗂️ フレームサムネイル（全フレーム結果）")

    with st.expander("サムネイル一覧を表示"):
        thumb_cols = 6
        rows = (len(frames) + thumb_cols - 1) // thumb_cols
        for row in range(min(rows, 5)):  # 最大5行 = 30フレーム
            cols = st.columns(thumb_cols)
            for col_idx in range(thumb_cols):
                fi = row * thumb_cols + col_idx
                if fi < len(frames):
                    with cols[col_idx]:
                        thumb = frames[fi].copy()
                        if fi in tracking_results:
                            thumb = apply_masks_to_frame(thumb, tracking_results[fi], rois)
                        thumb_small = cv2.resize(thumb, (200, 120))
                        st.image(thumb_small, caption=f"F{fi+1}", use_container_width=True)
