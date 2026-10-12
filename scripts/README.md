# 実行・解析・検証の入口

プロジェクトルートを作業ディレクトリとして呼び出す。既存ファイル名は保存済みコードや過去の手順との互換性のため維持し、設定パスだけ新配置に更新した。

| 用途 | スクリプト |
| --- | --- |
| 環境構築・パイロット | [setup.sh](setup.sh)、[pilots.sh](pilots.sh) |
| 方位条件の準備 | [prepare_compass_experiment.py](prepare_compass_experiment.py) |
| セッション間継承 | [run_session_selection.py](run_session_selection.py) |
| 規模・セッション比較 | [run_scaling_study.py](run_scaling_study.py)、[execute_scaling_study.py](execute_scaling_study.py) |
| 多数系列の準備・実行 | [prepare_replicate_study.py](prepare_replicate_study.py)、[execute_replicate_study.py](execute_replicate_study.py) |
| 個体数・ランダム較正の解析 | [analyze_population.py](analyze_population.py)、[compare_population.py](compare_population.py)、[analyze_random_survival.py](analyze_random_survival.py) |
| 評価器・継承の解析 | [analyze_reward_components.py](analyze_reward_components.py)、[analyze_session_selection.py](analyze_session_selection.py)、[audit_session_selection.py](audit_session_selection.py) |
| 規模・時間の解析 | [analyze_scaling_study.py](analyze_scaling_study.py)、[analyze_scaling_relations.py](analyze_scaling_relations.py) |
| 多数系列の集計・最終監査 | [analyze_replicate_study.py](analyze_replicate_study.py)、[analyze_replicate_results.py](analyze_replicate_results.py) |
| 性能計測・プロファイル | [benchmark_acceleration.py](benchmark_acceleration.py)、[run_acceleration_benchmarks.py](run_acceleration_benchmarks.py)、[profile_acceleration.py](profile_acceleration.py)、[analyze_acceleration.py](analyze_acceleration.py) |
| 学習・物理の検証 | [validate_acceleration_learning.py](validate_acceleration_learning.py)、[check_compiled_physics.py](check_compiled_physics.py)、[smoke_scaling_study.py](smoke_scaling_study.py) |
| 観察画面の検証 | [verify_browser.py](verify_browser.py)、[verify_regions_browser.py](verify_regions_browser.py) |

解析の出力は既定で `runs/`。レポートに採用する集計・図を対応する `docs/experiments/<実験ID>/data/`・`figures/` へ保存し、元データと再計算コマンドを結果に記す。

`runs/*-source/` 内のスクリプトは実行当時の保存コード。現在のスクリプトで置き換えない。新しい共通機能は `src/ailife/` に実装し、試行固有の設定やseedは計画・設定・manifestへ置く。
