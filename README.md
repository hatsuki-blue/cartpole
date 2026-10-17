# 棒立てを学ぶAI(CartPole × DQN)

学習は Python(PyTorch + Gymnasium)、展示はブラウザだけで動きます。

```
fes2026/
  .venv/              Python 3.12 の仮想環境(PyTorch GPU 版入り)
  cartpole/
    train.py          学習スクリプト(GPU があれば自動で使う)
    requirements.txt
    web/
      index.html      展示ページ
      models.js       学習結果(train.py が書き出す)
    checkpoints/      学習済みモデル(PyTorch 形式・参考用)
```

---

## 1. この PC で学習する

準備(仮想環境と PyTorch のインストール)は済んでいます。PowerShell で次を実行します。

```powershell
cd C:\Users\hatsuki\Documents\fes2026\cartpole
..\.venv\Scripts\python.exe train.py
```

> `python train.py` だけだと MSYS2 版の Python(PyTorch なし)が動くので使わないでください。

### 鍛えたAI(いじわるあり)も作る

普通の完成AIは、つつかれるとすぐ倒れます。練習中に棒の長さ・重力をランダムに変え、ときどき棒をつついて鍛えたAIも作ると、展示で並べて比べられます。

```powershell
..\.venv\Scripts\python.exe train.py --robust --episodes 1000
```

`web\models_robust.js` ができ、展示画面に「鍛えたAI」の枠が増えます(なければ枠は出ません)。

時間で止めることもできます(例: 4時間)。10分ごとに途中保存され、`Ctrl+C` で止めてもそこまでの結果が残ります。

```powershell
..\.venv\Scripts\python.exe train.py --robust --minutes 240
```

ノートPCで長時間回すときは、電源につなぎ、スリープしない設定にしておいてください。
いじわるありは成績がばらつくので、`eval` が 500 に届かなくても大丈夫です。展示で実際につついて比べてください。

### 表示の見かた(数字は例)

```
device: cuda (NVIDIA GeForce RTX 3050 Ti Laptop GPU)   ← GPU で学習している
ep    10  return    15  avg10   20.3  eval   9.3  eps 0.81    3.2s
...
ep   600  return   500  avg10  480.0  eval 500.0  eps 0.05  250.0s
best: episode 420, eval(5) 500.0
wrote ...\web\models.js (2.0 MB)                      ← 完了
```

| 項目 | 意味 |
| --- | --- |
| `ep` | 練習した回数 |
| `return` | その回に立てていたステップ数(最大 500 = 10 秒) |
| `avg10` | 直近 10 回の平均 |
| `eval` | ランダム行動なしでテストしたときの平均 |
| `eps` | ランダムに動く確率(だんだん下がる) |

`eval` が 500 に届けば完成です。

### うまくいかないとき

- 最後まで `eval` が低い → 乱数を変えてやり直す: `..\.venv\Scripts\python.exe train.py --seed 1`
- 練習回数を増やす: `--episodes 1000`
- `device: cpu` と出る → GPU 版 PyTorch が入っていない(下の 2. の手順で入れ直す)

---

## 2. 別の PC(GPU が強い学習用PC)で学習する

1. `cartpole` フォルダをその PC にコピーする(`.venv` はコピーしない)
2. Python 3.12 を入れる(<https://www.python.org/downloads/>)。3.14 は PyTorch 未対応のことがある
3. `cartpole` フォルダで PowerShell を開いて実行:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

最後が `True` なら GPU が使えます。`False` なら NVIDIA ドライバを更新してください。

4. 学習する:

```powershell
.\.venv\Scripts\python.exe train.py
```

5. できた `web\models.js` を展示用PCに持っていく

---

## 3. 展示用PC

1. `web` フォルダ(`index.html`・`models.js`・`models_robust.js`)をコピーする
2. `index.html` をブラウザ(Chrome / Edge)で開く
3. `F11` で全画面にする

Python・GPU・ネット接続は不要です。`models.js` がないと「学習データがありません」と表示されます。

### 展示の見どころ

- 「学習前」「練習中」「完成」のAIが同時に棒を立てる。プルダウンで別の練習回数のAIにも切り替えられる
- 棒をクリック(または上のボタン)でつついて邪魔できる
- 棒の長さ・重力を変えると、練習していない状況でAIがどうなるか試せる
- 「あなた」の枠でキーボードの ← → を使って人間も挑戦できる
- 下のグラフで、練習するほど上手になる様子が見える
