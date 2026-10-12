# 完全ランダム動作との比較

2026年10月11日、障害物ありのL3・3世界の継続実行に対して、RANDOM・3世界を追加した。世界シードは両群とも42・1051・2060。地形・食料・初期位置とエネルギー、寿命、成熟、出生条件、容量をそろえ、ゲーム内tickで比較する。

両群とも100,000tickまで完走した。[個体数の推移と比較結果](results.md)に最終人数・平均・図をまとめている。

RANDOMは毎tick、移動4択、口の操作2択、発信5択、繁殖3択の各成分を独立した一様分布から選ぶ。体調や観測に応じて選択肢を制限しない。不成立の操作は他の群と同じ世界ルールで解決する。ネットワーク・学習・遺伝による行動確率の変化はなく、子も同じ一様ランダム則を使う。独立な個体ID・tick・用途別の疑似乱数により保存再開も再現する。固定ネットワークを使うL0とは別の対照条件である。

L3は `obstacles-demo-20261011` の1,800tickから `obstacles-100k` に継続し、RANDOMは同じ初期条件のtick 0から `random-control-100k` を新規実行する。両群の上限は100,000tick。処理速度の差で世代交代が変わらないよう、寿命や物理は全てゲーム内tickに従う。初期配置が一致し、出生・死亡の記録から復元した全tickの個体数が計測ログと一致することを確認する。

[6世界の観察画面](http://192.168.0.66:8765/?compare=1&study=comparison-original-100k)は上段L3、下段RANDOM。各枠はその群の最新のゲーム内tickを表示する。右の個体数表は両群に共通する最新の集計tickを使う。リプレイは実験選択から各群の保存区間を開ける。原データは各試行の `metrics.jsonl` と `events.jsonl`。

```bash
# RANDOMだけを別の独立試行として再実行する場合
ROCR_VISIBLE_DEVICES=0 .venv/bin/ailife run \
  --config configs/experiments/2026-10-11-population-control/random-control.yaml --device cuda:0

# 現在の両群を出生・死亡記録から比較する（ホストPythonのNumPy・matplotlib）
python3 scripts/compare_population.py
```

現在の実行は `ai-life-obstacles-100k` と `ai-life-random-control-100k` のユーザーサービスで管理する。統計・checkpoint・実行前にランダム選択した観察区間を保存する。比較表・図・CSVは `runs/population-comparison/` に生成する。

35件のテストが成功。ランダム分布・観測に依存しない選択・初期世界の一致・再開・学習状態がないこと・同tickでの比較を検査した。Chromiumでも6世界の表示、同tickの表、各群への切替、モバイル表示とJavaScriptエラーなしを確認した。

この比較は神経ネットワーク・生涯学習・遺伝を組み合わせたL3全体と、一様ランダム則との差を見る。生涯学習だけの寄与を分けるには、固定ネットワークL0等との比較も必要になる。
