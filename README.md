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

ライン設定とカウンタは既定で `backend/state.json` に永続化され、アプリ再起動後も前回の値を復元します。環境変数 `LINE_COUNTER_STATE_PATH` を指定すると保存先を任意に切り替えられます。

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

## Docker での実行
1. Docker と Docker Compose が利用できる環境を用意します。Linux で USB カメラを利用する場合は、ホストの `/dev/video0` などデバイスファイルにアクセスできることを確認してください。
2. ルートディレクトリで以下を実行し、バックエンド・フロントエンドを同時に起動します。
   ```bash
   docker compose up --build
   ```
3. フロントエンドは `http://localhost:5423`、バックエンド API は `http://localhost:8000` に公開されます。

### コンテナ構成
- `backend` サービス: `python:3.11-slim` ベース。`uvicorn` で FastAPI アプリを 8000 番ポートで起動します。環境変数 `LINE_COUNTER_STATE_PATH=/data/state.json` を介して状態ファイルをボリューム `backend_state` に保存します。
- `frontend` サービス: Node.js でビルドした静的ファイルを Nginx (5423 番ポート) から配信します。ビルド時に `VITE_BACKEND_URL` ビルド引数でバックエンド URL を埋め込みます（既定は `http://localhost:8000`）。

### オプション設定
- 他のホスト・ポートでバックエンドを公開する場合は、`docker-compose.yml` の `frontend.build.args.VITE_BACKEND_URL` を変更してください。
- カメラデバイスが `/dev/video0` 以外の場合や Windows/macOS で USB カメラを共有する場合は、`devices` やプラットフォーム固有の共有設定を調整してください。カメラを利用しない開発用途では `devices` 行をコメントアウトしても動作します。
- 状態ファイルを初期化したい場合は、コンテナ停止後に `docker volume rm count_visitor_backend_state` を実行するとリセットできます。

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
