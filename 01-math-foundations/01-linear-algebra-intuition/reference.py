# Phase 1 / Lesson 1: Linear Algebra Intuition
# 完整參考版本，對照官方教材 docs/en.md + code/vectors.py
# 先讀懂、跑起來看結果，之後你會憑印象重打

import math


class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        # 向量加法：對應位置相加
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        # 向量減法：對應位置相減
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def __mul__(self, scalar):
        # 純量乘法：每個分量都乘上同一個數字，例如 [1,2,3] * 3 = [3,6,9]
        return Vector([x * scalar for x in self.components])

    def dot(self, other):
        # 內積
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        # 向量長度：sqrt(x1^2 + x2^2 + ...)
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        # 正規化：縮放成長度為 1 的向量，方向不變
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        # 餘弦相似度：內積除以兩個長度相乘，AI 領域超常用的相似度指標
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def angle_between(self, other):
        # 算兩個向量的夾角（度數）。cosine_similarity 算出來的是 cos(角度)，
        # 這裡用 acos（反餘弦）把 cos 值換算回實際角度，degrees 再把弧度轉成度數。
        # ⚠️ 這一項這堂課只帶著跑過計算結果，沒真的講清楚為什麼 cos_theta 能換算成角度。
        cos_theta = self.cosine_similarity(other)
        cos_theta = max(-1.0, min(1.0, cos_theta))  # 防止浮點數誤差超出 [-1,1] 範圍
        return math.degrees(math.acos(cos_theta))

    def project_onto(self, other):
        # 把 self 投影到 other 方向上：想成 self 在太陽正上方照下來，落在 other
        # 這條線上的影子有多長、指向哪。scalar 算的是「影子佔 other 長度的比例」，
        # 乘回 other 的每個分量就是影子這個向量本身。
        # ⚠️ 這堂課只講過怎麼算、沒有真的講清楚幾何上「影子」這個直覺，是 Gram-Schmidt 的內部工具。
        scalar = self.dot(other) / other.dot(other)
        return Vector([scalar * x for x in other.components])

    def __repr__(self):
        return f"Vector({self.components})"


def is_independent(vectors):
    # 判斷一組向量是否「線性獨立」:沒有任何一個向量可以用其他向量加加減減、
    # 乘個倍數湊出來(例如 [2,1,0] = 2*[1,0,0] + [0,1,0]，這樣就不獨立)。
    # 做法是列運算(高斯消去法):把向量當矩陣的列，一路消去，最後還剩幾個
    # 「非零列」(rank，矩陣的秩)，如果 rank 等於向量的數量，就是線性獨立。
    # ⚠️ 這堂課只提過名詞，沒有帶著手算過列運算的過程，測驗這題也答錯過。
    n = len(vectors)
    if n == 0:
        return True
    dim = vectors[0].dim
    rows = [v.components[:] for v in vectors]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n


def gram_schmidt(vectors):
    # 把一組線性獨立的向量，轉換成「正交歸一基底」:每個向量互相垂直(正交)、
    # 長度都是1(歸一)。做法是一個一個處理，每個新向量都先扣掉它在前面
    # 已處理好的向量方向上的投影(project_onto)，扣掉之後剩下的部分保證
    # 跟前面的都垂直，再正規化(normalize)成長度1。
    # ⚠️ 這堂課只理解邏輯、沒手打，用途是數值方法/QR分解，不是日常AI開發常直接手刻的東西。
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = w.project_onto(u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal


class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))  # (列數, 行數)

    def __matmul__(self, other):
        # @ 符號：矩陣乘法專用運算子
        if isinstance(other, Vector):
            # 矩陣 x 向量
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        # 矩陣 x 矩陣
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        # 轉置：行列互換
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def rank(self):
        # 矩陣的秩(rank):矩陣裡「真正獨立」的列數量，也就是把每一列當向量，
        # 有幾個是線性獨立的(邏輯跟上面的 is_independent 一樣，用列運算算)。
        # rank 越低代表列跟列之間重複、有共線的資訊越多。
        # ⚠️ 這堂課測驗答錯的就是這題，講得不夠清楚；這個概念之後LoRA會用到
        # (LoRA用低rank矩陣去逼近一個大矩陣，減少要訓練的參數量)，優先度不低，值得回頭補。
        rows = [row[:] for row in self.rows]
        m, n = self.shape
        r = 0
        for col in range(n):
            pivot = None
            for row in range(r, m):
                if abs(rows[row][col]) > 1e-10:
                    pivot = row
                    break
            if pivot is None:
                continue
            rows[r], rows[pivot] = rows[pivot], rows[r]
            scale = rows[r][col]
            rows[r] = [x / scale for x in rows[r]]
            for row in range(m):
                if row != r and abs(rows[row][col]) > 1e-10:
                    factor = rows[row][col]
                    rows[row] = [rows[row][j] - factor * rows[r][j] for j in range(n)]
            r += 1
        return r

    def __repr__(self):
        return f"Matrix({self.rows})"


# --- 底下是測試，直接跑這個檔案就會看到結果 ---
if __name__ == "__main__":
    print("=== Vector 基本運算 ===")
    a = Vector([1, 2, 3])
    b = Vector([4, 5, 6])
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"a * 3 = {a * 3}")
    print(f"a.dot(b) = {a.dot(b)}")
    print(f"|a| = {a.magnitude():.4f}")
    print(f"a normalize = {a.normalize()}")
    print(f"cosine_similarity(a, b) = {a.cosine_similarity(b):.4f}")
    print(f"angle_between(a, b) = {a.angle_between(b):.2f} 度")

    print("\n=== 投影 ===")
    p = Vector([3, 4])
    q = Vector([1, 0])
    print(f"project {p} onto {q} = {p.project_onto(q)}")

    print("\n=== 線性獨立 ===")
    e1 = Vector([1, 0, 0])
    e2 = Vector([0, 1, 0])
    dep = Vector([2, 1, 0])
    print(f"{{e1, e2}} independent: {is_independent([e1, e2])}")
    print(f"{{e1, e2, 2*e1+e2}} independent: {is_independent([e1, e2, dep])}")

    print("\n=== Gram-Schmidt 正交化 ===")
    u1 = Vector([1, 1, 0])
    u2 = Vector([1, 0, 1])
    basis = gram_schmidt([u1, u2])
    for i, u in enumerate(basis):
        print(f"u{i+1} = {u}, |u{i+1}| = {u.magnitude():.6f}")

    print("\n=== Matrix 運算 ===")
    rotation_90 = Matrix([[0, -1], [1, 0]])
    point = Vector([3, 1])
    print(f"rotation_90 @ point = {rotation_90 @ point}")

    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[5, 6], [7, 8]])
    print(f"m1 @ m2 = {m1 @ m2}")
    print(f"m1.transpose() = {m1.transpose()}")
    print(f"m1.rank() = {m1.rank()}")

    rank_deficient = Matrix([[1, 2], [2, 4]])
    print(f"rank_deficient.rank() = {rank_deficient.rank()}  (應該是 1，因為第二列是第一列的兩倍)")
