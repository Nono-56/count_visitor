# count_visitor
訪問者の通過方向をリアルタイムで可視化・集計するためのフルスタック構成です。Web カメラ映像に対して YOLOv8 による人物検出とIDトラッキングを行い、任意に設定したラインを左右どちらから横切ったかをカウントします。React 製フロントエンドからライン位置をドラッグで更新でき、結果は WebSocket またはポーリングで表示されます。

## プロジェクト構成
- `backend/` : FastAPI + OpenCV + Ultralytics (YOLOv8) による検出・ストリーム配信・状態管理。
- `frontend/` : Vite(React + TypeScript) で構築した UI。MJPEG 映像の表示とライン編集、カウント表示を担当します。
- `requirements.txt` : バックエンド用 Python 依存関係。

## 動作要件
- Python 3.10 以上（3.11 推奨）
- Node.js 20 以上、および npm
- Web カメラデバイス（`cv2.VideoCapture(0)` が利用可能なこと）
- 初回起動時に YOLOv8 モデル（`yolov8n.pt`）が自動ダウンロードされます。ネットワーク接続が必要です。

## セットアップと起動

### バックエンド (FastAPI)
1. 仮想環境を作成してアクティブ化します。
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. 依存パッケージをインストールします。
   ```powershell
   pip install -r requirements.txt
   ```
3. サーバーを起動します（デフォルトで `http://127.0.0.1:8000`）。
   ```powershell
   python -m backend.app
   ```
   または、開発用途でホットリロードを使う場合は次のコマンドも利用できます。
   ```powershell
   uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
   ```

主要エンドポイント:
- `GET /stream` : MJPEG 形式の映像ストリーム。
- `GET /status` : ライン座標と集計値（左右カウント）を返却。
- `PUT /line` : ライン座標を更新。ボディに `{ "start": {"x": 0.5, "y": 0.2}, "end": {"x": 0.5, "y": 0.8} }` のような正規化座標を渡します。
- `WEBSOCKET /ws/status` : ライン変更やカウント更新をリアルタイム配信。

ライン設定とカウンタは `backend/state.json` に永続化され、アプリ再起動後も前回の値を復元します。

### フロントエンド (React + Vite)
1. 依存関係をインストールします。
   ```powershell
   cd frontend
   npm install
   ```
2. 開発サーバーを起動します（`http://localhost:5423`）。
   ```powershell
   npm run dev
   ```
3. ブラウザで `http://localhost:5423` を開き、映像とカウントを確認します。

ビルド:
```powershell
npm run build
```
生成物は `frontend/dist` 配下に出力されます。

> **バックエンド URL の変更**
> フロントエンドは既定で `http://localhost:8000` を参照します。別ホスト／ポートでバックエンドを起動する場合は、環境変数 `VITE_BACKEND_URL` を設定してから `npm run dev` または `npm run build` を実行してください。
>
> 例: `VITE_BACKEND_URL=http://192.168.0.10:8000 npm run dev`

## フロントエンドの主な機能
- `/stream` を `<video>` 要素で表示し、同サイズの SVG オーバーレイを重ねてライン位置を視覚化。
- ライン端点をドラッグして移動、ドロップ時に `/line` API へ PUT リクエストを送信して永続化。
- `/ws/status` に自動接続し、カウントやライン変更をリアルタイム反映。WebSocket が利用できない場合は `/status` をポーリング。
- 右→左 / 左→右 のカウントをカード形式で表示、同期状態やバックエンドの URL も確認可能。

## テストとトラブルシューティング
- カメラが認識されない場合は `backend/video.py` 内の `source` 引数を変更して適切なデバイス番号を指定してください。
- GPU が利用できない環境では CPU 推論になります。必要に応じて `YOLO('yolov8n.pt')` を軽量モデルに変更可能です。
- WebSocket 接続に失敗した場合でも自動で 5 秒ごとに `/status` のポーリングへフォールバックします。
- `backend/state.json` を削除することでカウントとライン設定をリセットできます。
