"""
POPトラッキング結果ビューア
"""
import streamlit as st
import cv2, numpy as np, os
from PIL import Image

st.set_page_config(page_title="POP Tracking Result", layout="wide")
st.title("🎯 SAM2 POPラベル トラッキング結果")

VIDEO_PATH = "/home/user/webapp/pop_tracking_h264.mp4"
FRAMES_DIR = "/home/user/webapp"

# ----- サイドバー -----
with st.sidebar:
    st.markdown("## 📊 トラッキング概要")
    st.success("✅ SAM2 初回セグメンテーション完了")
    st.info("🎯 7個のPOPラベルを追跡")
    
    colors = {
        "50yen-POP 1F-L": "#3230FF",
        "50yen-POP 1F-R": "#32C832",
        "50yen-POP 2F"  : "#E6A000",
        "50yen-POP 3F-L": "#00A0E6",
        "50yen-POP 3F-R": "#00C8C8",
        "50yen-POP 4F-L": "#DC32DC",
        "50yen-POP 4F-R": "#32DC32",
    }
    st.markdown("### 🏷️ POPラベル一覧")
    for name, color in colors.items():
        st.markdown(f'<div style="background:{color}33;border-left:4px solid {color};'
                    f'padding:6px;border-radius:4px;margin:3px 0;">'
                    f'<b style="color:{color}">■</b> {name}</div>',
                    unsafe_allow_html=True)

# ----- メイン -----
tab1, tab2 = st.tabs(["🎬 動画", "🖼️ フレーム比較"])

with tab1:
    st.markdown("### SAM2+MIL トラッキング動画")
    if os.path.exists(VIDEO_PATH):
        with open(VIDEO_PATH, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
        st.download_button("⬇️ 動画をダウンロード", video_bytes,
                          "pop_tracking.mp4", "video/mp4",
                          use_container_width=True, type="primary")
    else:
        st.error("動画ファイルが見つかりません")

with tab2:
    st.markdown("### フレーム比較")
    frame_files = sorted([f for f in os.listdir(FRAMES_DIR)
                          if f.startswith("result_frame_") and f.endswith(".jpg")])
    sam2_mask = "/home/user/webapp/sam2_mask_frame0.jpg"
    
    if frame_files:
        col_names = ["Frame 0 (開始)", "Frame 29 (1秒)", "Frame 58 (中間)",
                     "Frame 87 (後半)", "Frame 115 (末尾)"]
        
        st.markdown("#### SAM2 初回セグメンテーション")
        if os.path.exists(sam2_mask):
            st.image(sam2_mask, caption="SAM2マスク（フレーム0）", use_container_width=True)
        
        st.markdown("#### 全フレームトラッキング結果")
        cols = st.columns(len(frame_files))
        for i, (col, fname) in enumerate(zip(cols, frame_files)):
            with col:
                img_path = os.path.join(FRAMES_DIR, fname)
                cap_name = col_names[i] if i < len(col_names) else fname
                st.image(img_path, caption=cap_name, use_container_width=True)
