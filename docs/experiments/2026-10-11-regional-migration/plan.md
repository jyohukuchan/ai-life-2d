# 出生場所と食料供給場所を分離する条件

2026年10月11日。従来の正方形マップでの評価器診断と並行し、横長の環境を追加する。学習方式・内的評価・出生コストは従来どおりで、移動に外的報酬を付けない。

| 項目 | 横長条件 |
| --- | --- |
| 外壁を含む縦×横 | 28×112（1:4、従来57×57に近い面積） |
| 初期個体・容量 | 各世界32・128 |
| 世界シード | 42・1051・2060 |
| 初期配置・全ての出生 | 左側25%、x=1〜27の通行可能な空き床 |
| 食料の初期分布・再生 | 右側25%、x=84〜110の通行可能な床 |
| 障害物 | 内側の4%。全ての床は連結 |
| 食料の基準 | 直近の0.5倍条件、再生率0.00005 |
| 実行期間 | 各方式10,000tick |
| 比較方式 | L3・L0・完全ランダムを各3世界 |

子は親の位置に関係なく左側へ即座に現れる。クローンと二親繁殖の両方に適用する。二親繁殖には、従来どおり隣接する成熟した両親の同意が必要。左側の空き床を候補ごとに一様抽選し、同じ出生地点への競合と個体容量による制限を処理する。不成立時に出生コストは取らない。出生地点の移動に追加の時間・エネルギーコストは設定していない。

総供給量を保つため、世界シードごとに従来の57×57マップの `Σfertility` を参照し、右側の分布を拡大して合計を一致させる。初期食料の期待数 `initial_food × Σfertility` と、全供給床が空いている場合の再生期待数 `regrowth × Σfertility` が従来と一致する。各セルの確率が1を超えない範囲で調整し、地域が狭すぎて予算を収容できない設定は拒否する。

一致させるのは**潜在的な供給の期待値**である。実際の食料数は抽選、食料が既にある床、個体による占有、全滅後の再生停止によって変わる。各tickの実現値を強制的に合わせたり、食べられていない食料を無限に積み上げたりはしない。

| 世界 | 初期食料の期待数 | 食料の再生期待数/tick | 食料エネルギーの供給期待値/tick |
| --- | ---: | ---: | ---: |
| 0 | 71.7950 | 0.0299146 | 0.299146 |
| 1 | 71.3132 | 0.0297138 | 0.297138 |
| 2 | 68.7031 | 0.0286263 | 0.286263 |

初期個体の体力、出生時体力、コスト、寿命、観測半径、音、遺伝方式は0.5倍条件と同じ。出生後の獲得重み・記憶は継承しないため、右側へ移動できた親の子も左側から独立して活動・学習する。右側に食料を集める設定は、総供給量に加えて資源の局所密度と移動距離を変える条件である。

全体マップの淡い青色は出生領域、淡い黄緑色は食料領域を示す観察者向けの表示。個体のRGB入力ではどちらも通常の床で、座標・領域名・移動方向を直接渡さない。

## 評価器の成分別記録

[正方形の診断設定](../../../configs/experiments/2026-10-11-regional-migration/evaluator-diagnostics-square.yaml)は、従来の0.5倍・L3と同じ世界をtick 0から実行する。横長L3でも同じ診断を記録する。

`reward-components/` に、旧個体が生存していた全遷移について、tick・世界・ID、評価器入力 `phi`、出生時に定まる係数との積、tanh後の内的評価、実際の正規化スケール、採食・移動・成功した繁殖・死亡のフラグを保存する。死亡した遷移も含み、新生児へのスロット再利用でIDを取り違えない。1,000tick単位でGPUから取り出し、圧縮NPZにする。個体へ戻す入力や経験再生には使わない。

加算できる寄与はtanh**前**の値。成分を除いた効果 `tanh(total) − tanh(total − group)` は他成分を固定した局所的な差で、互いに加算できない。正規化後の差を計算する際も、実際に記録したスケールを固定する。別の報酬で学習をやり直した結果ではない。

解析では平均・絶対平均・RMSと、個体ごとの平均を引いた変動のRMSを残す。RMSの大きさだけで方策への影響や学習の有用性を判定しない。採食した遷移の評価平均も記述統計であり、採食の因果効果ではない。

記録あり・なしで、CPUとR9700上の世界・学習状態が一致することを確認した。設定とログの変更は保存再開時に許可し、物理やネットワーク設定の変更は引き続き拒否する。

## 実行・確認

```bash
# 同じ正方形での診断
ROCR_VISIBLE_DEVICES=2 .venv/bin/ailife run \
  --config configs/experiments/2026-10-11-regional-migration/evaluator-diagnostics-square.yaml --device cuda:0 \
  --output runs/MY-SQUARE-DIAGNOSTIC

# 横長の学習群
ROCR_VISIBLE_DEVICES=0 .venv/bin/ailife run \
  --config configs/experiments/2026-10-11-regional-migration/rectangular-migration.yaml --device cuda:0 \
  --output runs/MY-MIGRATION-L3

# 対照群。L0への置き換えも可能
ROCR_VISIBLE_DEVICES=1 .venv/bin/ailife run \
  --config configs/experiments/2026-10-11-regional-migration/rectangular-migration.yaml --mode RANDOM \
  --no-log-reward-components --device cuda:0 --output runs/MY-MIGRATION-RANDOM

# 解析は学習用Pythonとは独立して実行できる
python3 scripts/analyze_reward_components.py runs/evaluator-square-L3-10k \
  --output runs/evaluator-square-analysis
python3 scripts/compare_population.py --study runs/migration-10k-study.json \
  --output runs/migration-analysis/comparison
```

[横長マップの比較画面](http://192.168.0.66:8765/?compare=1&study=migration-10k-study)は、左列L3・右列完全ランダム、上から世界0・1・2。保存区間の選択方法は従来どおり実行前の一様ランダムで、全滅後の区間も含む。旧正方形のリプレイと固定URLも維持する。
