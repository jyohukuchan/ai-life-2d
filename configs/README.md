# 実行設定

各YAMLはプロジェクトルートから指定する。既存ファイルは場所だけを変更し、設定値は変更していない。

- `baselines/`：共通動作確認。現在の [pilot.yaml](baselines/pilot.yaml) は従来の32体・容量128のv0設定で、次の1体実験ではない。
- `experiments/<実験ID>/`：個別実験の設定。実験IDは[文書側の一覧](../docs/experiments/README.md)と対応する。

主な再利用先は次のとおり。

| 用途 | 設定 |
| --- | --- |
| 実装済みv0の動作確認 | [pilot](baselines/pilot.yaml) |
| 直近の比較の基準環境とL3 | [絶対方位・96×32](experiments/2026-10-11-compass/compass-absolute.yaml) |
| 採食を含めて較正したランダム | [採食較正](experiments/2026-10-11-random-feeding-calibration/random-feeding-window.yaml) |
| 最適化FP32の既存確認 | [FP32 compiled](experiments/2026-10-11-acceleration/acceleration-fp32.yaml) |

```bash
.venv/bin/ailife run --config configs/baselines/pilot.yaml --device cpu
```

新実験の設定は、機能と数値条件を確認してから追加する。1世界1体の採食計画には、現実装にない繁殖・寿命死の明示的な無効化等があるため、まだ実行用YAMLを置いていない。

過去の文書・manifestにある旧パスとの対応は[移動記録](../docs/overview/path-migrations-2026-10-12.json)を参照。保存済みコード内の旧パスは変更していない。
