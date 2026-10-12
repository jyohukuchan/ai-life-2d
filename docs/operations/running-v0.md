# v0の実行と観察

この手順は実装済みのv0を対象にする。[次の1世界1体の計画](../experiments/2026-10-12-single-agent-foraging/plan.md)は未実装・未実行。以下の「現在」「最新」は各記録の作成時点を含むので、研究の現在地は[文書案内](../README.md)を参照する。設定パスは2026年10月12日の配置へ更新済み。

2026年10月11日。専用2D世界、L0〜L3、出生・死亡・遺伝、観察用Web画面を実装した。外的な課題報酬と事前学習は使わない。[実験計画](../experiments/2026-10-11-v0-implementation/plan-v0.md)と[更新仕様](../protocols/learning-protocol-v0.md)を基準とする。睡眠、Dreaming、狩り、建築はv1の範囲である。

## このマシンですぐ使う

作業ディレクトリは `/home/homelab1/coding-local/ai-life-2d`。構築済みの `.venv` を使う。

```bash
cd ~/coding-local/ai-life-2d
# R9700をこのプロセスからの唯一のGPUとして使う。このマシンで確認した番号。
ROCR_VISIBLE_DEVICES=2 .venv/bin/ailife run \
  --config configs/baselines/pilot.yaml --device cuda:0
```

既定の試行IDは時刻・方式・設定ハッシュを含む。明示する場合は `--output runs/my-trial` を付ける。既存の出力ディレクトリへの上書きは拒否する。GPUを使う場合は **`--device cuda:0` を指定する**。省略するとCPUになる。

パイロット設定の初期32個体、容量128、幅64は変更可能な基準値で、実験全体の固定条件ではない。幅64では一個体65,497パラメータ。この数には予測・価値ヘッド等も含む。各個体の重み、遺伝基底、正規化、Adamのm/v/更新回数、RLトレースは独立している。親からは出生時の遺伝基底と指定した遺伝変数を継承し、獲得重みやoptimizer状態は継承しない。

新規世界には内部タイルの約4%に固定障害物を置く。一辺1〜4タイルの小さな塊や短い壁を散らし、歩ける場所全体がつながる配置にする。移動・出生・初期配置・食料生成は通行可能なタイルで行い、障害物は局所RGB視覚と観察画面に灰色で表示する。視野は局所正方形、音は距離減衰というv0の感覚仕様を用いる。

密度はYAMLの `obstacle_density` または `--obstacle-density` で0〜0.2の範囲で指定できる。`--obstacle-density 0` は障害物なしの比較条件。地形は世界シードから独立した乱数で実行前に生成し、保存・再開およびリプレイにも保持する。障害物対応前のcheckpointやリプレイは障害物なしとして読み込む。障害物を加える新規試行と以前の測定値は、別の環境条件として扱う。

```bash
# 小さいCPU試行
.venv/bin/ailife run --mode L3 --worlds 2 --initial 4 \
  --capacity 16 --hidden 16 --ticks 100

# 方式を変える
ROCR_VISIBLE_DEVICES=2 .venv/bin/ailife run \
  --config configs/baselines/pilot.yaml --device cuda:0 --mode L1

# 同じ設定・状態から期間だけ延ばして再開（出力先は新しくする）
ROCR_VISIBLE_DEVICES=2 .venv/bin/ailife run \
  --resume runs/my-trial/checkpoint.pt --ticks 20000 \
  --device cuda:0 --output runs/my-trial-continued
```

SIGINT / SIGTERMは、一遷移を完了してから停止し、再開可能なcheckpointを保存する。SIGKILL、電源断、ドライバの異常にはこの終了処理は働かない。定期checkpointから再開する。tickの期間や観察頻度は変更できるが、モデルや物理設定が違うcheckpointの再開は拒否する。

再開は同じ世界の続きであり、独立した新しい標本ではない。`metadata.json` に再開元と開始tickを記録する。出生・死亡の累計は再開前からの値を引き継ぐ。

## 絶対方角を入力するL3

[実験結果](../experiments/2026-10-11-compass/results.md)と[方位入力の設計・実行方法](../protocols/compass-input-design.md)。`--compass absolute` で常時one-hotの方位入力、`--compass masked` で同じ入力枠をゼロ、`--compass none` で従来形式。操作は相対的なままで、視野は半径5を使用した。

## 採食を較正した新条件でのL3比較

[最新のL3とRANDOMの結果](../experiments/2026-10-11-feeding-calibrated-l3/results.md)。各3世界、10,000tick。[比較画面](http://192.168.0.66:8765/?compare=1&study=feeding-calibrated-10k-study)は上段L3・下段RANDOM。設定は `configs/experiments/2026-10-11-feeding-calibrated-l3/feeding-calibrated-L3.yaml`。

## 採食を必須にした現在のランダム基準

[較正結果](../experiments/2026-10-11-random-feeding-calibration/results.md)、設定 `configs/experiments/2026-10-11-random-feeding-calibration/random-feeding-window.yaml`。縦96×横32、基礎代謝0.016、再生率0.00001。各世界の採食1回以上・2,000tick存続・10,000tickまでの全滅を必須にした。旧条件のL3比較は下に履歴として残す。

## 基礎代謝0.016のL3とRANDOM

[実行結果・評価器の記録](../experiments/2026-10-11-migration-met016/results.md)と[比較画面](http://192.168.0.66:8765/?compare=1&study=migration-met016-10k-study)。設定は `configs/experiments/2026-10-11-migration-met016/migration-met016.yaml`、各群3世界、10,000tick。

## ランダム群の生存期間の探索

[探索結果](../experiments/2026-10-11-random-survival/results.md)に2,000tick存続・10,000tickまでの全滅を調べた30シード×3条件と、別の30シードでの確認を記載した。再利用用の `configs/experiments/2026-10-11-random-survival/random-survival-window.yaml` は基礎代謝0.016のRANDOM・3世界。観察画面は `?run=実験名` で直接指定できる。

## 横長マップと評価器診断

[出生場所と食料供給場所を分離する条件](../experiments/2026-10-11-regional-migration/plan.md)を追加した。縦28×横112、出生は左25%、食料は右25%、供給の期待値は従来の57×57・0.5倍条件と同じ。`height`・`width` が未指定なら従来の `size` による正方形になる。

`log_reward_components: true` または `--log-reward-components` で、L3の実際の評価器入力・成分寄与を観察用に保存する。L0・RANDOMでは `--no-log-reward-components` を使う。

## ゲーム画面を見る

L3・3世界と完全ランダム動作・3世界の対照を[比較表示](http://192.168.0.66:8765/?compare=1)で観察できる。条件と解析手順は[ランダム対照群](../experiments/2026-10-11-population-control/plan.md)を参照。完全ランダムは `--mode RANDOM` で指定し、固定ネットワークのL0と区別する。

現在は `ai-life-2d-viewer.service` というユーザーサービスで観察サーバーを起動している。

- このマシン：[http://localhost:8765](http://localhost:8765)
- 同じLAN：[http://192.168.0.66:8765](http://192.168.0.66:8765)

ライブ、世界一覧、個体のクリック追跡と局所視覚、ランダムな連続区間のリプレイを使える。リプレイには再生・一時停止・速度・位置の操作がある。ライブは0.5秒程度に一度公開する最新状態であり、途中の全tickの動画ではない。

```bash
systemctl --user status ai-life-2d-viewer
journalctl --user -u ai-life-2d-viewer -n 30
systemctl --user stop ai-life-2d-viewer

# サービスを使わず手動で起動する場合
.venv/bin/ailife serve --root runs --host 0.0.0.0 --port 8765
```

インターネットへの公開URLは設定していない。外部からこのホストへSSH接続できる場合は、閲覧者のマシンで次の転送を行い、`http://localhost:8765` を開ける。接続先はその閲覧者が到達できるSSHホスト名へ置き換える。

```bash
ssh -N -L 8765:127.0.0.1:8765 homelab1@SSH_HOST
```

SSHを使わず一時的な外部URLを発行する場合は、Cloudflareの[Quick Tunnel](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)が候補になる。`cloudflared tunnel --url http://localhost:8765` で発行できる。Cloudflareのアカウントは不要だが、URLを知る人が閲覧でき、プロセス停止でURLは失効する。本作業では公開トンネルは起動していない。

ブラウザは観察ファイルを読む独立したプロセスへ接続する。世界や学習を操作するHTTP APIはない。閲覧者が停止しても、読み込みが遅くても、シミュレーション側は待たない。公開用のキューは長さ1とし、たまったライブ表示は捨てる。記録とCPUへの状態転送には計算コストがあるため、速度表で観察あり・なしを分けて測る。

## リプレイは最初はランダムに選ぶ

実行前に、全世界×開始tickの組から重複なく一様に選び、固定長の区間を保存する。選択には独立した観察用乱数を使う。選択表は `replay-selection.json` に先に書く。画面の区間一覧は開始tick順で表示する。結果を見て選び直さず、早期絶滅した世界や何も起きなかった区間も残す。再開した実験では、再開以降の期間から観察区間を新たに選び、前の実験とは別に記録する。

全試行には軽い統計と出生・死亡の記録を残す。イベントを条件にした追加リプレイは将来の別枠とし、ランダム標本に混ぜない。観察用リプレイを個体の学習へ戻す仕組みはない。

## 保存内容

| ファイル | 内容 |
| --- | --- |
| `config.json`, `metadata.json` | 設定、設定ハッシュ、環境、更新対象、乱数設計、コードのハッシュ |
| `metrics.jsonl`, `summary.json` | 全世界の生存数、食料、出生・死亡、世代、打切り・全滅、実測速度 |
| `events.jsonl` | 全ての出生ID・親IDと死亡ID・原因 |
| `checkpoint.pt` | 世界、遺伝基底、生涯重み、全個体状態・optimizer・乱数カウンタ |
| `reward-components/schema.json`, `reward-components/*.npz` | L3診断を有効にした場合の全生存遷移の評価器入力・成分・正規化・イベント |
| `latest.json` | Web表示用の最新世界状態 |
| `replay-selection.json`, `replays/*.npz`, `replays/*.json` | 実行前の選択表、選んだ連続区間、メタデータ |

自然な絶滅と、NaNや実行障害は別の状態として扱う。数値の有限性、出生時リセット、エネルギー収支、衝突、保存再開等はテストで検査する。全滅後も予定期間まで記録を進める。個体の自動補充やリスポーンはない。

## 速度を測る

```bash
ROCR_VISIBLE_DEVICES=2 .venv/bin/ailife bench --device cuda:0 \
  --worlds 3 --initial 32 --capacity 128 --hidden 64 --mode L3 \
  --warmup 10 --steps 100 --output runs/my-bench.jsonl

# 観察の処理も含める
ROCR_VISIBLE_DEVICES=2 .venv/bin/ailife bench --device cuda:0 \
  --worlds 3 --initial 32 --capacity 128 --hidden 64 --mode L3 --observe
```

`world_ticks_per_second`は全世界のtickの合計、`agent_decisions_per_second`は実際に生存していた個体の判断数の合計。3世界で150世界tick/秒なら、各世界はおよそ50tick/秒で進む。容量128を全員生存128体と数えない。測定前後でGPUを同期し、ウォームアップを分ける。予測メモリが大きすぎる条件は、割当て前に未実行として記録する。

## 環境の再構築とテスト

```bash
scripts/setup.sh rocm
# GPUを使わない別環境なら scripts/setup.sh cpu
.venv/bin/playwright install chromium
.venv/bin/python -m pytest
# 観察サーバーと実行中の試行がある状態で、実ブラウザを検査する
.venv/bin/python scripts/verify_browser.py --run RUN_NAME
```

ROCm環境はAMDのmulti-arch wheelとPyPIを使い、確認した依存バージョンを `requirements-rocm.lock` に保存している。このホストではPyTorchからのGPU順序はV620、V620、R9700だった。`rocm-smi`の表示番号とは一致しない。別ホストではGPU名を確認して選ぶ。一つのシミュレーションプロセスは一つのGPUを使い、複数GPUには独立プロセスを割り当てる。

詳しい測定結果は[検証記録](../experiments/2026-10-11-v0-implementation/results.md)、障害物の追加と確認結果は[障害物の検証記録](../experiments/2026-10-11-v0-implementation/obstacles.md)に記載する。

## GPU最適化と混合精度

[検証結果と実行例](../experiments/2026-10-11-acceleration/results.md)。`--execution eager`は従来処理、`kernels`は学習部分の融合、`compiled`は世界・観測も含む融合。`--precision fp32/bf16/fp16`で行列演算の精度を選び、基準重み・統計・物理はFP32を保つ。最初の推奨入口は`configs/experiments/2026-10-11-acceleration/acceleration-fp32.yaml`。R9700/gfx1201の当Torch/ROCmビルドでは通常実行のFP16/BF16学習にエラーを再現したため、その経路は拒否している。
