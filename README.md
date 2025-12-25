# Realtime Webcam 3D Gaussian Splatting

Webカメラの画像をキャプチャし、AppleのSHARPモデルとMetalSplatterレンダラーを使用してリアルタイムで3D Gaussian Splatsを生成するmacOSアプリケーションです。

![デモ](image/main_app_camera.gif)

![デモ2](image/main_app_static.gif)

## 必要条件

### ハードウェア
- Apple Silicon Mac（M1以降）
- Webカメラ（内蔵または外付け）
- 16GB RAM（最小）、32GB RAM（推奨）

### ソフトウェア
- macOS 14.0（Sonoma）以降
- Xcode 15.0以降
- Python 3.10-3.13（PyTorch MPSサポート付き）

## セットアップ

### 1. Python依存パッケージのインストール

```bash
# 仮想環境の作成と有効化
python3 -m venv venv
source venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt

# ml-sharpのインストール（別途クローンが必要）
git clone https://github.com/apple/ml-sharp.git
cd ml-sharp/ml-sharp
pip install -e .
cd ../..
```

### 2. アプリケーションのビルドと実行

**重要:** Swift Package Managerのリソースバンドルの制限により、このアプリは**Xcode**から実行する必要があります（`swift run`は使用不可）。

1. Xcodeで`RealtimeWebcam3DGS/RealtimeWebcam3DGS.xcodeproj`を開く
2. RealtimeWebcam3DGSスキームを選択
3. ビルドして実行（Cmd+R）

### 3. SHARPサーバーの起動

アプリを使用する前に、別のターミナルでSHARP Pythonサーバーを起動してください：

```bash
source venv/bin/activate
python3 RealtimeWebcam3DGS/sharp_server.py
```

初回実行時にSHARPモデル（約2.6GB）がダウンロードされます。

#### Core MLバックエンド（オプション - Apple Siliconで高速化）

Apple Neural Engineを使用した高速推論のために、モデルをCore MLに変換できます：

```bash
# coremltoolsのインストール
pip install coremltools

# モデルの変換（数分かかります、初回のみ必要）
python3 RealtimeWebcam3DGS/convert_to_coreml.py --output RealtimeWebcam3DGS/models/sharp.mlpackage

# Core MLバックエンドでサーバーを実行
python3 RealtimeWebcam3DGS/sharp_server.py --coreml
```

Core MLはApple Neural Engineを活用し、約10〜30%高速な推論を提供します。

## 使い方

1. **カメラの選択**: コントロールパネルのドロップダウンからWebカメラを選択します。

2. **サーバーの起動**: まだ起動していない場合は、「Start Server」をクリックしてSHARP Pythonサーバーを起動します。

3. **キャプチャの開始**: 「Start」をクリックしてキャプチャパイプラインを開始します：
   - アプリは5秒ごとにフレームをキャプチャします（設定可能）
   - 各フレームは3DGS生成のためにSHARPモデルに送信されます
   - 生成されたPLYはリアルタイムで読み込まれ、レンダリングされます

4. **ビューコントロール**:
   - **ドラッグ**: 3Dビューを回転
   - **ピンチ/スクロール**: ズームイン/アウト

5. **手動キャプチャ**: 「Capture Now」をクリックして即座にキャプチャを実行します。

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                      Swiftアプリケーション                    │
├─────────────────────────────────────────────────────────────┤
│  CameraCaptureManager     │  AppCoordinator                 │
│  (AVFoundation)           │  (パイプライン制御)               │
├───────────────────────────┼─────────────────────────────────┤
│  SHARPClient              │  SplatRenderManager             │
│  (Unixソケットクライアント)  │  (MetalSplatter統合)            │
└───────────────────────────┴─────────────────────────────────┘
                │                           │
                │ Unixドメインソケット         │ PLYファイル
                ▼                           │
┌─────────────────────────────────────────────────────────────┐
│                     SHARP Pythonサーバー                     │
│  (ml-sharp / PyTorch / MPS)                                 │
└─────────────────────────────────────────────────────────────┘
```

## 設定

| 設定項目 | デフォルト値 | 説明 |
|---------|------------|------|
| キャプチャ間隔 | 5秒 | 自動キャプチャの間隔 |
| 自動キャプチャ | オン | 自動キャプチャの有効/無効 |

## トラブルシューティング

### サーバーが起動しない
- Python環境が有効化されていることを確認してください
- PyTorch MPSが利用可能か確認: `python -c "import torch; print(torch.backends.mps.is_available())"`
- ml-sharpがインストールされているか確認: `python -c "from sharp.models import create_predictor"`

### カメラが検出されない
- システム環境設定 > プライバシーとセキュリティ > カメラでカメラのアクセス許可を付与してください
- 「Refresh」ボタンでカメラリストを更新してみてください

### 生成が遅い
- MPS（Metal Performance Shaders）が使用されていることを確認してください（CPUではなく）
- 初回の生成はモデルの読み込みのため遅くなる場合があります（約10〜20秒）
- 2回目以降の生成は1.5〜2.5秒程度になります
