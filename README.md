# SAM2 POPラベル トラッキングシステム

> Streamlit × SAM2 (Segment Anything Model 2) による高精度マルチオブジェクト動画トラッキング

---

## 🎯 概要

POPラベルや店頭ディスプレイの動画をアップロードし、マウスでROI（関心領域）を複数設定。  
Meta AI の **SAM2.1** を使って全フレームにわたる高精度なセグメンテーション＆トラッキングを行います。

---

## 🚀 主な機能

| 機能 | 説明 |
|------|------|
| 📁 動画アップロード | MP4/AVI/MOV 対応、最大500MB |
| 🎯 マルチROI設定 | 最大8個のROIをフレーム上で指定 |
| 🤖 SAM2トラッキング | SAM2.1 Tinyモデルによる高精度追跡 |
| 📊 結果可視化 | フレーム別マスク表示 + 面積グラフ |
| 💾 動画ダウンロード | トラッキング結果をMP4で出力 |
| 📋 JSONエクスポート | 座標・面積データをJSON形式で出力 |

---

## 🛠️ セットアップ

### 1. 依存パッケージのインストール

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install streamlit opencv-python-headless pillow numpy pandas
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### 2. SAM2.1 モデルのダウンロード

```bash
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
     -O checkpoints/sam2.1_hiera_tiny.pt
```

### 3. アプリ起動

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` にアクセス

---

## 📋 使い方

### Step 1: 動画アップロード
- 「① 動画アップロード」タブを開く
- MP4/AVI/MOV ファイルをドラッグ＆ドロップ
- サンプル動画ボタンでテストも可能

### Step 2: ROI設定
- 「② ROI設定・トラッキング」タブを開く
- フレームスライダーで対象フレームを選択
- ラベル名を入力し、座標 (X1,Y1)-(X2,Y2) を指定
- 「ROIを追加」ボタンで登録（最大8個）

### Step 3: トラッキング実行
- ROI設定後、「SAM2トラッキング開始」ボタンをクリック
- CPUで処理（3秒動画で約10〜30秒）

### Step 4: 結果確認
- 「③ 結果確認」タブでフレームごとの結果を確認
- 面積グラフでトラッキング精度を確認
- 動画/JSONをダウンロード

---

## ⚙️ 技術仕様

- **モデル**: SAM2.1 Hiera Tiny (149MB)
- **設定**: `configs/sam2.1/sam2.1_hiera_t.yaml`
- **デバイス**: CPU (CUDA利用可能時は自動でGPU使用)
- **フレームワーク**: Streamlit 1.54+, PyTorch 2.10+

---

## 📁 ファイル構成

```
webapp/
├── app.py                  # メインアプリ
├── checkpoints/
│   └── sam2.1_hiera_tiny.pt  # SAM2モデル（別途ダウンロード）
├── .streamlit/
│   └── config.toml         # Streamlit設定
├── .gitignore
└── README.md
```

---

## 🔧 カスタマイズ

### 高精度モデルへの変更（GPU推奨）

`app.py` の定数を変更:
```python
CHECKPOINT_PATH = "checkpoints/sam2.1_hiera_large.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
```

ダウンロード:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
     -O checkpoints/sam2.1_hiera_large.pt
```
