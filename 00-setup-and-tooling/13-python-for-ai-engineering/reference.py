"""
Phase 0 Lesson 13: Python for AI Engineering
只用標準函式庫(沒有numpy)寫出AI專案常用的小工具:
CSV/JSON讀寫、固定種子的訓練/測試切分、mini-batch產生器、
純Python向量運算、標準化、最近中心點分類器,以及速度與記憶體比較。
執行:python3 reference.py;測試:python3 -m unittest -v test_reference
"""

from __future__ import annotations

import csv
import json
import math
import operator
import random
import tempfile
import time
import tracemalloc
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

# 資料表中「答案欄位」的名稱
LABEL_FIELD = "label"


# 一筆資料:features是特徵(不可變tuple),label是答案;frozen=True讓它可以放進set
@dataclass(frozen=True)
class Record:
    features: tuple[float, ...]
    label: int


# 造假資料:兩類各自以不同中心與散布產生高斯分布,用固定seed讓結果可重現
def make_dataset(n: int, seed: int = 0) -> list[Record]:
    rng = random.Random(seed)
    centers = {0: (2.0, 50.0), 1: (4.0, 70.0)}
    spreads = (1.0, 10.0)
    records = []
    for i in range(n):
        label = i % 2
        features = tuple(rng.gauss(c, s) for c, s in zip(centers[label], spreads))
        records.append(Record(features, label))
    return records


# Record轉成可寫入檔案的一列(x1, x2, ..., label)
def record_to_row(record: Record) -> dict[str, float | int]:
    row: dict[str, float | int] = {f"x{i + 1}": v for i, v in enumerate(record.features)}
    row[LABEL_FIELD] = record.label
    return row


# 讀回一列並檢查欄位,不符就丟錯,避免默默讀錯資料
def row_to_record(row: dict, names: Sequence[str], label_field: str = LABEL_FIELD) -> Record:
    if label_field not in row:
        raise ValueError(f"row has no {label_field!r} field: {row}")
    if set(row) != {*names, label_field}:
        raise ValueError(f"row fields {list(row)} do not match {[*names, label_field]}")
    return Record(tuple(float(row[name]) for name in names), int(row[label_field]))


# 依副檔名存成json或csv
def save_records(records: Sequence[Record], path: str | Path) -> Path:
    path = Path(path)
    rows = [record_to_row(r) for r in records]
    if not rows:
        raise ValueError("no records to save")
    if path.suffix == ".json":
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    elif path.suffix == ".csv":
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError(f"unsupported file type: {path.suffix!r}")
    return path


# 依副檔名讀回json或csv;csv讀出來都是字串,所以要轉型
def load_records(path: str | Path, label_field: str = LABEL_FIELD) -> list[Record]:
    path = Path(path)
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    else:
        raise ValueError(f"unsupported file type: {path.suffix!r}")
    if not rows:
        return []
    names = [key for key in rows[0] if key != label_field]
    return [row_to_record(row, names, label_field) for row in rows]


# 先洗牌再切,固定seed;至少留1筆在測試與訓練兩邊
def train_test_split(items: Sequence, test_fraction: float = 0.2, seed: int = 0) -> tuple[list, list]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be between 0 and 1, got {test_fraction}")
    if len(items) < 2:
        raise ValueError(f"need at least 2 items to split, got {len(items)}")
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    n_test = min(max(1, round(len(items) * test_fraction)), len(items) - 1)
    test = [items[i] for i in order[:n_test]]
    train = [items[i] for i in order[n_test:]]
    return train, test


# 把資料一批一批吐出來(mini-batch);drop_last決定最後不足一批要不要丟掉
def batches(items: Iterable, batch_size: int, drop_last: bool = False) -> Iterator[list]:
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    return _batch_iter(iter(items), batch_size, drop_last)


# batches的實作:用islice每次從迭代器取batch_size個,取不到就結束
def _batch_iter(iterator: Iterator, batch_size: int, drop_last: bool) -> Iterator[list]:
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch or (drop_last and len(batch) < batch_size):
            return
        yield batch


# 內積;strict=True讓長度不同時直接報錯,不會默默少算
def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


# 平均數
def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean() needs at least one value")
    return sum(values) / len(values)


# 標準差(除以n的母體版本)
def std(values: Sequence[float]) -> float:
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


# 逐欄算平均與標準差;zip(*rows)把「列」轉成「欄」
def column_stats(rows: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    columns = list(zip(*rows))
    return [mean(c) for c in columns], [std(c) for c in columns]


# 用訓練資料的平均與標準差,回傳一個標準化函式(測試資料也用同一組數字,避免資料洩漏)
def make_standardizer(rows: Sequence[Sequence[float]]) -> Callable[[Sequence[float]], list[float]]:
    means, stds = column_stats(rows)
    scales = [s if s > 0 else 1.0 for s in stds]

    def standardize(row: Sequence[float]) -> list[float]:
        return [(v - m) / s for v, m, s in zip(row, means, scales, strict=True)]

    return standardize


# 一批一批累加,算出每個類別的中心點(平均),不用一次載入全部資料
def fit_centroids(labelled_batches: Iterable[Sequence[tuple[Sequence[float], int]]]) -> dict[int, list[float]]:
    sums: dict[int, list[float]] = {}
    counts: Counter[int] = Counter()
    for batch in labelled_batches:
        for row, label in batch:
            previous = sums.get(label, [0.0] * len(row))
            sums[label] = [s + v for s, v in zip(previous, row, strict=True)]
            counts[label] += 1
    return {label: [s / counts[label] for s in sums[label]] for label in sorted(sums)}


# 兩點的平方距離
def squared_distance(a: Sequence[float], b: Sequence[float]) -> float:
    diff = [x - y for x, y in zip(a, b, strict=True)]
    return dot(diff, diff)


# 預測:離哪個類別的中心點最近就是哪一類
def predict(centroids: dict[int, list[float]], row: Sequence[float]) -> int:
    return min(centroids, key=lambda label: squared_distance(row, centroids[label]))


# 準確率:預測對的比例
def accuracy(centroids: dict[int, list[float]], pairs: Sequence[tuple[Sequence[float], int]]) -> float:
    return sum(predict(centroids, row) == label for row, label in pairs) / len(pairs)


# 內積(索引迴圈版);長度不同時不會報錯
def dot_index_loop(a: Sequence[float], b: Sequence[float]) -> float:
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


# 內積(map 版);長度不同時只算到較短的那個
def dot_map(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(map(operator.mul, a, b))


# 跑repeat次取最快的時間,減少雜訊
def best_time(fn: Callable, *args, repeat: int = 5) -> float:
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - start)
    return min(times)


# 量一個函式執行期間記憶體用量的最高點
def peak_bytes(fn: Callable[[], object]) -> int:
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


# 先建完整list再加總(佔記憶體)
def sum_squares_list(n: int) -> int:
    return sum([i * i for i in range(n)])


# 用generator邊算邊加(省記憶體)
def sum_squares_generator(n: int) -> int:
    return sum(i * i for i in range(n))


# 把上面所有工具串起來跑一遍,回傳結果摘要
def run_demo(n: int = 400, seed: int = 7, batch_size: int = 48, vector_size: int = 200_000) -> dict:
    records = make_dataset(n, seed)
    with tempfile.TemporaryDirectory() as tmp:
        from_csv = load_records(save_records(records, Path(tmp) / "data.csv"))
        from_json = load_records(save_records(records, Path(tmp) / "data.json"))

    train, test = train_test_split(from_csv, test_fraction=0.2, seed=seed)
    raw_train = [(r.features, r.label) for r in train]
    raw_test = [(r.features, r.label) for r in test]
    standardize = make_standardizer([r.features for r in train])
    scaled_train = [(standardize(r.features), r.label) for r in train]
    scaled_test = [(standardize(r.features), r.label) for r in test]

    rng = random.Random(seed)
    a = [rng.random() for _ in range(vector_size)]
    b = [rng.random() for _ in range(vector_size)]
    timings = {
        "index loop": best_time(dot_index_loop, a, b),
        "zip generator": best_time(dot, a, b),
        "map + mul": best_time(dot_map, a, b),
    }
    peaks = {
        "list": peak_bytes(lambda: sum_squares_list(100_000)),
        "generator": peak_bytes(lambda: sum_squares_generator(100_000)),
    }
    return {
        "n_records": len(records),
        "roundtrip_equal": from_csv == records and from_json == records,
        "n_train": len(train),
        "n_test": len(test),
        "overlap": len(set(train) & set(test)),
        "batch_size": batch_size,
        "batch_sizes": [len(batch) for batch in batches(scaled_train, batch_size)],
        "scaled_means": [round(m, 6) + 0.0 for m in column_stats([row for row, _ in scaled_train])[0]],
        "accuracy_raw": accuracy(fit_centroids(batches(raw_train, batch_size)), raw_test),
        "accuracy_scaled": accuracy(fit_centroids(batches(scaled_train, batch_size)), scaled_test),
        "vector_size": vector_size,
        "timings": timings,
        "peaks": peaks,
    }


# 把摘要排成要印出來的文字
def format_report(summary: dict) -> str:
    lines = [
        "Python for AI engineering: a standard-library toolkit",
        f"data     {summary['n_records']} records, CSV and JSON load back unchanged: {summary['roundtrip_equal']}",
        f"split    {summary['n_train']} train, {summary['n_test']} test, records in both: {summary['overlap']}",
        f"batches  size {summary['batch_size']}: {summary['batch_sizes']}",
        f"scale    train feature means after standardizing: {summary['scaled_means']}",
        f"model    nearest-centroid test accuracy: raw {summary['accuracy_raw']:.2f}, "
        f"standardized {summary['accuracy_scaled']:.2f}",
        f"speed    dot product of {summary['vector_size']:,} floats, best of 5:",
    ]
    for name, seconds in summary["timings"].items():
        lines.append(f"           {name:15} {seconds * 1000:7.2f} ms")
    lines.append("memory   peak bytes to sum 100,000 squares:")
    for name, size in summary["peaks"].items():
        lines.append(f"           {name:15} {size:>11,}")
    return "\n".join(lines)


# 進入點
def main() -> None:
    print(format_report(run_demo()))


if __name__ == "__main__":
    main()
