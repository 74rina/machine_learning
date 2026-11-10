# MLP ベースラインの評価結果

- 実行日時（日本時間）: 2026-11-10T13:53:38+09:00
- 目的変数: DIC
- データ件数: 1454（学習 1163 / 検証 291）
- 特徴量数: 16
- モデル: Linear(16, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 1)
- 損失関数: nn.MSELoss()（標準化した DIC に対して計算）
- 最適化: Adam / 学習率 0.001 / バッチサイズ 32
- エポック: 200（Early stopping なし）
- 分割: ランダム 80% / 20%、seed=42
- 前処理: ID・Unnamed 列の除外、TA1.x → TA1、学習側だけで中央値補完と入力・目的変数の標準化
- 実行環境: CPU / threads=1 / Python 3.13.11

| 評価項目                                    |            結果 |
| ------------------------------------------- | --------------: |
| 最良エポック                                |             129 |
| **Validation MSE（DIC の元の尺度）**        | **28.03951193** |
| Validation RMSE                             |      5.29523483 |
| 学習平均を常に予測した場合の Validation MSE |  13417.32066221 |
| 最終エポックの Validation MSE               |     36.38151861 |

全エポックのうち Validation MSE が最小の重みを `model.pt` に保存した。
このスコアはモデル選択に使ったホールドアウトの値であり、独立テストや Kaggle のスコアではない。
座標が重なるサンプルを含むランダム分割のため、未知の海域への汎化性能は別途検証する。

## 保存ファイル

- `model.pt`: 重み・前処理統計量・分割行番号と ID・学習設定・データ SHA-256
- `metrics.json`: 評価値・設定・パッケージバージョン
- `history.csv`: 各エポックの損失・学習 MSE・検証 MSE
- `validation_predictions.csv`: 検証データの ID・正解 DIC・予測 DIC
- `learning_curve.png`: DIC の元の尺度での学習・検証 MSE

## 再評価

プロジェクト直下で実行する。

```bash
python -m src.evaluate --checkpoint /Users/rinayoneyama/rina/Keio/machine_learning/checkpoints/baseline/model.pt --train-csv /Users/rinayoneyama/rina/Keio/machine_learning/dataset/train.csv
```
