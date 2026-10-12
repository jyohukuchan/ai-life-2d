# 研究文書の案内

更新日：2026年10月12日。長期方針、実装仕様、個別実験、先行研究を分けて管理する。

## 現在地

幅・セッション数の比較は完了した。予測誤差は低下したが、安定した採食・存続は未達。次は **1世界1体の小さい環境で、経験による行動改善と内的評価の選抜を切り分ける**。[次の実験計画](experiments/2026-10-12-single-agent-foraging/plan.md)は未実装・未実行。

| 区分 | 入口 | 役割 |
| --- | --- | --- |
| 長期方針 | [研究目標](overview/research-goals.md)、[段階計画](overview/roadmap.md) | 何を目指し、どの証拠で次へ進むか |
| 判断の履歴 | [方針と変更理由](overview/decisions.md) | 決定、候補、未決定事項を区別する |
| 個別実験 | [実験一覧](experiments/README.md) | 仮説・条件・結果・データ・図を実験IDでまとめる |
| 実装仕様 | [学習v0](protocols/learning-protocol-v0.md)、[方位入力](protocols/compass-input-design.md) | 実際に動く更新則と入力条件 |
| 運用 | [実行・観察](operations/running-v0.md) | 実行、保存、再開、閲覧の手順 |
| 先行研究 | [調査一覧](research/README.md)、[未採用候補](research/not-adopted.md) | 根拠、相違点、採用範囲を整理する |
| 保存した旧案 | [改訂前の構想](archive/concept-before-curriculum-2026-10-12.md)、[Minecraft案](archive/minecraft-plan-2026-10-09.md) | 方針の変遷を残す |
| 文書の追加方法 | [構成規約](overview/repository-layout.md)、[計画テンプレート](templates/experiment-plan.md)、[結果テンプレート](templates/experiment-results.md) | 今後の研究を同じ形式で蓄積する |

長期目標を変更するときは `overview/`、特定実験の条件や結果を追加するときは `experiments/<実験ID>/` を更新する。実装前の案で既存仕様を書き換えない。実験開始前の計画を残し、開始後の条件変更は別版・別試行として記録する。

2026年10月12日の移動対応と数値データ等の内容ハッシュは[移動記録](overview/path-migrations-2026-10-12.json)に保存した。
