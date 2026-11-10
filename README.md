# CalCOFI DIC 予測

海洋観測データから溶存無機炭素（`DIC`）を予測する、PyTorch の MLP。

## 環境構築

Pythonの環境構築を行う。

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 学習

`dataset/train.csv` を使い、学習結果を `checkpoints/baseline/` に保存する。

```bash
python -m src.train
python -m src.train --epochs 100 --output-dir checkpoints/baseline_100
```

## 評価

保存した最良モデルの検証結果を再計算するには、次を実行します。

```bash
python -m src.evaluate
```

実測の評価結果と実行条件は [checkpoints/baseline/evaluation.md](checkpoints/baseline/evaluation.md) に記録します。

| ファイル                     | 内容                                                                 |
| ---------------------------- | -------------------------------------------------------------------- |
| `model.pt`                   | 最良モデルの重み、前処理の統計量、行分割、学習設定、データの SHA-256 |
| `metrics.json`               | 検証指標と実行情報                                                   |
| `history.csv`                | エポックごとの学習・検証の記録                                       |
| `validation_predictions.csv` | 検証データの正解値と予測値                                           |
| `learning_curve.png`         | 学習曲線                                                             |
| `evaluation.md`              | 評価結果のメモ                                                       |

## Kaggleへの提出

保存済みの最良モデルで `dataset/test.csv` を予測する。

```bash
python -m src.predict
```

`submissions/baseline.csv` に、サンプル提出ファイルと同じ ID 順・列構成（`id,DIC`、485 行、インデックスなし）で保存する。
