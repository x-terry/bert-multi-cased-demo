# bert-multi-cased-demo

基于 **BERT（multilingual-cased）** 的多分类文本分类 Demo，使用 Hugging Face `transformers` + `Trainer` 完成微调与推理。

适合快速上手：本地加载预训练模型 → 用自有 CSV 数据微调 → 用 checkpoint 做分类预测。

---

## 功能概览


| 能力    | 说明                                                  |
| ----- | --------------------------------------------------- |
| 多分类微调 | 默认对 20 类水果名称做分类（可改为任意标签数）                           |
| 本地模型  | 从 `./model` 加载分词器与预训练权重，无需联网                        |
| 训练监控  | 支持 TensorBoard（`report_to="tensorboard"`）           |
| 推理    | `pipeline('sentiment-analysis')` 加载训练产出的 checkpoint |


项目内另附外卖评论数据集 `waimai_10k.csv`（二分类情感），可按同样流程切换训练。

---

## 目录结构

```
bert-multi-cased-demo/
├── core/
│   ├── train.py          # 训练入口
│   └── main.py           # 推理入口
├── data/
│   ├── custom_fruit.csv  # 水果多分类示例（20 类）
│   └── waimai_10k.csv    # 外卖评论情感数据（约 1.2 万条）
├── model/                # 本地 BERT 预训练模型与词表
│   ├── config.json
│   ├── vocab.txt
│   └── tokenizer_config.json
├── cache/                # 训练输出（checkpoint，运行后生成）
├── logs/                 # TensorBoard 日志（运行后生成）
├── pip.txt               # 依赖版本
├── LICENSE
└── README.md
```

> 说明：`model/` 目录需包含完整权重文件（如 `pytorch_model.bin` / `model.safetensors`）。若仓库中未带权重，请自行放入与 `config.json` 匹配的 BERT multilingual-cased 权重。

---

## 环境准备

建议使用 Python 3.8+，并优先使用 GPU（CUDA）加速训练。

```bash
# 创建虚拟环境（可选）
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
# source venv/bin/activate

# 安装依赖
pip install -r pip.txt
pip install pandas scikit-learn tensorboard
```

`pip.txt` 中的核心依赖：

- `torch==2.2.0`
- `transformers==4.32.1`
- `datasets==2.16.1`

---

## 数据格式

训练数据为 CSV，**必须包含两列**：


| 列名      | 含义              |
| ------- | --------------- |
| `label` | 类别编号（整数，从 0 开始） |
| `text`  | 待分类文本           |


示例（`data/custom_fruit.csv`）：

```csv
label,text
0,apple
1,banana
2,orange
...
19,nectarine
```

切换数据集时，修改 `core/train.py` 中的路径即可：

```python
train_data = "./data/custom_fruit.csv"
valid_data = "./data/custom_fruit.csv"
```

同时把 `num_labels` 改成与标签数一致：

```python
model = AutoModelForSequenceClassification.from_pretrained('./model', num_labels=20)
```

---

## 训练

在项目根目录执行：

```bash
python core/train.py
```

主要流程：

1. 从 `./model` 加载 Tokenizer 与分类模型
2. 读取 CSV → `datasets` → 分词（`max_length=300`）
3. 使用 `Trainer` 微调，指标含 accuracy / precision / recall / f1
4. checkpoint 保存到 `./cache`，日志写入 `./logs`

### 关键超参（`core/train.py`）


| 参数                            | 默认值   | 说明         |
| ----------------------------- | ----- | ---------- |
| `per_device_train_batch_size` | 32    | 训练 batch   |
| `per_device_eval_batch_size`  | 32    | 验证 batch   |
| `learning_rate`               | 5e-5  | 学习率        |
| `warmup_ratio`                | 0.2   | warmup 比例  |
| `max_steps`                   | 900   | 最大训练步数     |
| `save_steps`                  | 300   | 每隔多少步保存    |
| `evaluation_strategy`         | epoch | 按 epoch 评估 |
| `num_labels`                  | 20    | 分类类别数      |


查看训练曲线：

```bash
tensorboard --logdir ./logs
```

---

## 推理

训练完成后，用 checkpoint 做预测。编辑 `core/main.py`：

1. 将 `pipeline` 中的路径改成实际 checkpoint（如 `./cache/checkpoint-900`）
2. 修改待预测文本 `sequence`
3. 运行：

```bash
python core/main.py
```

输出示例：

```text
[{'label': 'LABEL_19', 'score': 0.98}]
```

`LABEL_N` 对应训练时的类别编号 `N`。

---

## 快速对照：水果标签


| label | text       |
| ----- | ---------- |
| 0     | apple      |
| 1     | banana     |
| 2     | orange     |
| 3     | grape      |
| 4     | watermelon |
| 5     | pear       |
| 6     | kiwi       |
| 7     | peach      |
| 8     | lemon      |
| 9     | blueberry  |
| 10    | strawberry |
| 11    | raspberry  |
| 12    | mango      |
| 13    | pineapple  |
| 14    | papaya     |
| 15    | coconut    |
| 16    | grapefruit |
| 17    | plum       |
| 18    | apricot    |
| 19    | nectarine  |


---

## 注意事项

1. **工作目录**：请在仓库根目录运行脚本，路径均为相对路径（`./model`、`./data`、`./cache`）。
2. **类别数一致**：`num_labels` 必须与数据中的标签数量一致，否则训练或推理会出错。
3. **训练 / 验证集**：当前 Demo 训练与验证使用同一文件，仅作演示；正式实验请拆分独立验证集。
4. `main.py` **中的** `num_labels`：推理脚本里写的是 `2`，若你训练的是 20 类水果，请与训练配置保持一致，或以 checkpoint 内配置为准。

---

## License

[MIT](./LICENSE) © 2024 x-terry