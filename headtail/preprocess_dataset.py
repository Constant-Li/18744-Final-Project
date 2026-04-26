import json
import random
from collections import defaultdict

# ---- 参数 ----
INPUT_JSON = "transfer_labels.json"        # 上一步生成的 json
TRAIN_JSON = "train.json"              # 输出的训练集 json
TEST_JSON  = "test.json"               # 输出的测试集 json
TEST_RATIO = 0.1                       # 测试集占比
SEED       = 42                        # 随机种子，保证可复现

# ---- 读取数据 ----
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"总数据量: {len(data)}")

# ---- 按 label 分组 ----
by_label = defaultdict(list)
for item in data:
    by_label[item["label"]].append(item)

for lbl, items in by_label.items():
    print(f"  label={lbl}: {len(items)} 条")

# ---- 计算测试集每类数量 ----
# test 总数 = 总数 * 0.2，其中 label0 和 label1 各占一半
total = len(data)
test_total = int(round(total * TEST_RATIO))
per_class_test = test_total // 2           # 每类一半

# 安全检查：避免某一类样本不够
max_per_class = min(len(by_label[0]), len(by_label[1]))
if per_class_test > max_per_class:
    print(f"[警告] 每类需要 {per_class_test} 条，但样本最少的一类只有 {max_per_class} 条，"
          f"自动缩减为 {max_per_class}")
    per_class_test = max_per_class

print(f"测试集每类取 {per_class_test} 条，共 {per_class_test * 2} 条")

# ---- 随机划分 ----
rng = random.Random(SEED)

test_set, train_set = [], []

for lbl in (0, 1):
    items = by_label[lbl][:]      # 拷贝一份
    rng.shuffle(items)
    test_set.extend(items[:per_class_test])
    train_set.extend(items[per_class_test:])

# 如果除 0/1 以外还有其他 label（理论上没有，保险起见），都扔进训练集
for lbl, items in by_label.items():
    if lbl not in (0, 1):
        train_set.extend(items)

# 最后再整体 shuffle 一次，避免顺序上的偏置
rng.shuffle(train_set)
rng.shuffle(test_set)

# ---- 保存 ----
with open(TRAIN_JSON, "w", encoding="utf-8") as f:
    json.dump(train_set, f, ensure_ascii=False, indent=4)

with open(TEST_JSON, "w", encoding="utf-8") as f:
    json.dump(test_set, f, ensure_ascii=False, indent=4)

print(f"训练集: {len(train_set)} 条 -> {TRAIN_JSON}")
print(f"测试集: {len(test_set)} 条 -> {TEST_JSON}")
print(f"  测试集中 label=0: {sum(1 for x in test_set if x['label']==0)}")
print(f"  测试集中 label=1: {sum(1 for x in test_set if x['label']==1)}")