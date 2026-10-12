# 実験一覧

更新日：2026年10月12日。新しい計画・結果は同じ実験IDの下へ追加する。日付は文書・実験の識別日で、すべての実行日時を表すものではない。

| 実験ID | 状態・役割 | 入口 |
| --- | --- | --- |
| `2026-10-12-single-agent-foraging` | 次の計画、未実装・未実行 | [1世界1体の採食と学習経路](2026-10-12-single-agent-foraging/plan.md) |
| `2026-10-12-width-session-replication` | 完了。幅・セッション数の明確な存続改善は未確認 | [計画](2026-10-12-width-session-replication/plan.md)、[結果](2026-10-12-width-session-replication/results.md) |
| `2026-10-11-model-session-scaling` | 完了。3系列の探索比較 | [計画](2026-10-11-model-session-scaling/plan.md)、[結果](2026-10-11-model-session-scaling/results.md) |
| `2026-10-11-gpu-dispatch` | 完了。V620とR9700の速度差の診断 | [結果](2026-10-11-gpu-dispatch/results.md) |
| `2026-10-11-acceleration` | 完了。演算融合・低精度と学習傾向 | [結果](2026-10-11-acceleration/results.md) |
| `2026-10-11-session-inheritance` | 終了。毎回未学習の群は一部停止、安定維持は未達 | [方法・結果](2026-10-11-session-inheritance/results.md) |
| `2026-10-11-compass` | 完了。相対操作を維持した絶対方位入力 | [結果](2026-10-11-compass/results.md)、[入力仕様](../protocols/compass-input-design.md) |
| `2026-10-11-feeding-calibrated-l3` | 完了。採食を較正した条件でL3とランダムを比較 | [結果](2026-10-11-feeding-calibrated-l3/results.md) |
| `2026-10-11-random-feeding-calibration` | 完了。採食・短期存続・長期全滅で環境を較正 | [結果](2026-10-11-random-feeding-calibration/results.md) |
| `2026-10-11-random-survival` | 完了。採食を条件にしなかった旧較正 | [結果と限界](2026-10-11-random-survival/results.md) |
| `2026-10-11-migration-met016` | 完了。旧較正でのL3・ランダム比較 | [結果](2026-10-11-migration-met016/results.md) |
| `2026-10-11-regional-migration` | 完了。出生地域・食料地域の分離と評価器診断 | [設計](2026-10-11-regional-migration/plan.md)、[結果](2026-10-11-regional-migration/results.md) |
| `2026-10-11-half-food` | 完了。食料供給0.5倍 | [結果](2026-10-11-half-food/results.md) |
| `2026-10-11-population-control` | 完了。初期条件のL3・ランダム個体数比較 | [対照設計](2026-10-11-population-control/plan.md)、[結果](2026-10-11-population-control/results.md) |
| `2026-10-11-v0-implementation` | 基盤の実装・検証済み。旧探索計画には未実施項目も含む | [旧計画](2026-10-11-v0-implementation/plan-v0.md)、[検証](2026-10-11-v0-implementation/results.md)、[障害物](2026-10-11-v0-implementation/obstacles.md) |

既存の実行IDは維持する。幅・セッション反復は `runs/rep20-20261012-*`、40セッション比較は `runs/scale40-20261011-*`、その他は各結果文書に元データ・観察先を記す。`runs/` の自動集計は実行成果物、ここにある結果文書はその解釈と限界を含む研究記録。
