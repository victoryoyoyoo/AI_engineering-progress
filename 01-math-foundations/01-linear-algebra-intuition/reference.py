# Phase 1 / Lesson 1: Linear Algebra Intuition
# 完整參考版本，對照官方教材 docs/en.md + code/vectors.py
# 先讀懂、跑起來看結果，之後憑印象重打
#
# 整份檔案的結構:
#   Vector class : 用純 Python list 實作向量(加減、純量乘、內積、長度、正規化、餘弦相似度、投影)
#   Matrix class : 用「list 的 list」實作矩陣(矩陣乘向量、矩陣乘矩陣、轉置、rank)
#   is_independent / gram_schmidt : 線性獨立判斷、正交歸一化
#   最底下的 if __name__ == "__main__" : 跑一遍所有功能，印出結果
# 這裡刻意不用 NumPy，目的是看清楚每個運算底層實際在做什麼(NumPy 版本見 numpy_version.py)

import math  # 標準函式庫:提供 acos(反餘弦)、degrees(弧度轉度數)


# === 🔴 ===
class Vector:
    """向量:一串數字,同時代表「方向」與「長度」。例如 [3, 4] 是往右3、往上4的箭頭。"""

    def __init__(self, components):
        # components:傳進來的數字序列(list/tuple 都行),用 list() 複製成新的 list,
        # 避免外面的 list 之後被修改時,連帶影響到這個向量
        self.components = list(components)
        # dim:維度數,也就是分量個數。[1,2,3] 的 dim 是 3
        self.dim = len(self.components)

    def __add__(self, other):
        # __add__ 是 Python 的「魔術方法」:定義後可以直接寫 a + b
        # 向量加法：對應位置相加,例如 [1,2,3] + [4,5,6] = [5,7,9]
        # zip(A, B) 把兩個 list 同一位置的元素配成一對:(1,4), (2,5), (3,6)
        # 中括號裡的 for 寫法叫 list comprehension(串列生成式),一行建立新 list
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        # 向量減法：對應位置相減,[4,5,6] - [1,2,3] = [3,3,3]
        # 幾何意義:從 other 的箭頭尖端指向 self 的箭頭尖端的那支箭頭
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def __mul__(self, scalar):
        # 純量乘法：每個分量都乘上同一個數字，例如 [1,2,3] * 3 = [3,6,9]
        # 幾何意義:只改變長度(放大或縮小),方向不變;scalar 為負數時方向會反過來
        return Vector([x * scalar for x in self.components])

    def dot(self, other):
        # 內積(dot product):對應位置相乘後全部加總,結果是「一個數字」,不是向量
        # [1,2,3]·[4,5,6] = 1*4 + 2*5 + 3*6 = 32
        # 意義:衡量兩個向量「指向多一致」,同方向為正、垂直為 0、反方向為負
        # sum(...) 裡面是 generator expression(不加中括號,不會先建立整個 list,省記憶體)
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        # 向量長度：sqrt(x1^2 + x2^2 + ...)(畢氏定理推廣到任意維度)
        # [3,4] 的長度 = sqrt(9+16) = 5
        # ** 是次方運算子,x**2 是平方;** 0.5 是開平方根
        return sum(x**2 for x in self.components) ** 0.5

    # === 🟡 ===
    def normalize(self):
        # 正規化：縮放成長度為 1 的向量，方向不變
        # 做法:每個分量除以自己的長度。[3,4] / 5 = [0.6, 0.8],新長度 = 1
        # 用途:只想比較「方向」、不想被長度干擾時使用(例如 embedding 比較)
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        # 餘弦相似度：內積除以兩個長度相乘，AI 領域超常用的相似度指標
        # 公式:cos(θ) = (a·b) / (|a| × |b|),結果落在 [-1, 1]
        # 1 = 方向完全相同,0 = 垂直(無關),-1 = 方向完全相反
        # 除以長度乘積的目的:讓結果只反映方向,跟向量本身長短無關
        return self.dot(other) / (self.magnitude() * other.magnitude())

    # === 🟡 ===
    def angle_between(self, other):
        # 算兩個向量的夾角（度數）。cosine_similarity 算出來的是 cos(角度)，
        # 這裡用 acos（反餘弦）把 cos 值換算回實際角度，degrees 再把弧度轉成度數。
        # ⚠️ 這一項這堂課只帶著跑過計算結果，沒真的講清楚為什麼 cos_theta 能換算成角度。
        cos_theta = self.cosine_similarity(other)
        # min/max 夾擠:浮點數運算會有微小誤差,cos 可能算成 1.0000000000000002,
        # acos 收到超出 [-1, 1] 的值會直接丟 ValueError,所以先壓回合法範圍
        cos_theta = max(-1.0, min(1.0, cos_theta))  # 防止浮點數誤差超出 [-1,1] 範圍
        # acos 回傳弧度(radian),math.degrees 轉成大家習慣的度數(π 弧度 = 180 度)
        return math.degrees(math.acos(cos_theta))

    # === 🟡 ===
    def project_onto(self, other):
        # 把 self 投影到 other 方向上：想成 self 在太陽正上方照下來，落在 other
        # 這條線上的影子有多長、指向哪。scalar 算的是「影子佔 other 長度的比例」，
        # 乘回 other 的每個分量就是影子這個向量本身。
        # 例:self=[3,4], other=[1,0] → scalar = (3*1+4*0)/(1*1+0*0) = 3 → 影子 = [3,0]
        # ⚠️ 這堂課只講過怎麼算、沒有真的講清楚幾何上「影子」這個直覺，是 Gram-Schmidt 的內部工具。
        scalar = self.dot(other) / other.dot(other)
        return Vector([scalar * x for x in other.components])

    def __repr__(self):
        # __repr__ 決定 print(向量) 時顯示的字串;沒寫的話只會印出記憶體位址
        return f"Vector({self.components})"


# === 🔴 ===
class Matrix:
    """矩陣:一個「機器」,吃進 n 維向量、吐出 m 維向量。內部用 list 的 list 存,每個小 list 是一列。"""

    def __init__(self, rows):
        # rows:例如 [[1,2],[3,4]],外層每個元素是矩陣的一「列」
        # [list(row) for row in rows] 把每一列都複製一份,避免與外部共用同一個物件
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))  # (列數, 行數)

    def __matmul__(self, other):
        # @ 符號：矩陣乘法專用運算子(寫 A @ B 時 Python 會呼叫這個方法)
        # 規則:輸出的第 (i, j) 格 = A 的第 i 列 與 B 的第 j 欄 做內積
        if isinstance(other, Vector):
            # 矩陣 x 向量
            # 輸出向量的第 i 個數字 = 矩陣第 i 列 與輸入向量 做內積
            # 例:[[0,-1],[1,0]] @ [3,1] = [0*3+(-1)*1, 1*3+0*1] = [-1, 3](逆時針轉 90 度)
            # 輸入向量維度必須等於矩陣的行數(shape[1]),輸出維度等於矩陣的列數(shape[0])
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        # 矩陣 x 矩陣
        # 三層迴圈:i 走過左矩陣的列、j 走過右矩陣的欄、k 是內積時加總的位置
        # 形狀規則:(m×n) @ (n×p) = (m×p),中間的 n 必須相同
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

    # === 🟢 ===
    # transpose(矩陣轉置),Matrix class簡化時整個拿掉,完全沒教
    def transpose(self):
        # 轉置：行列互換,原本的第 i 列第 j 欄搬到第 j 列第 i 欄
        # [[1,2],[3,4]] 轉置後是 [[1,3],[2,4]];shape (m, n) 變成 (n, m)
        # 外層迴圈走過「原矩陣的欄」(變成新矩陣的列),內層走過「原矩陣的列」
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    # === 🟢 ===
    # rank(矩陣的秩),只提過名詞、測驗答錯,LoRA會用到,優先度不低
    def rank(self):
        # 矩陣的秩(rank):矩陣裡「真正獨立」的列數量，也就是把每一列當向量，
        # 有幾個是線性獨立的(邏輯跟上面的 is_independent 一樣，用列運算算)。
        # rank 越低代表列跟列之間重複、有共線的資訊越多。
        # ⚠️ 這堂課測驗答錯的就是這題，講得不夠清楚；這個概念之後LoRA會用到
        # (LoRA用低rank矩陣去逼近一個大矩陣，減少要訓練的參數量)，優先度不低，值得回頭補。
        #
        # 演算法是高斯消去法(Gaussian elimination):
        #   逐欄尋找「主元(pivot)」→ 把主元列換到目前位置 → 主元列縮成 1 →
        #   用它去消掉其他列在這一欄的數字。每成功找到一個主元,rank 就加 1。
        rows = [row[:] for row in self.rows]  # row[:] 是複製一份,避免改到原矩陣
        m, n = self.shape
        r = 0  # r:目前已經找到的主元數,也就是 rank 的累計值
        for col in range(n):
            pivot = None
            # 從第 r 列往下找這一欄「不是 0」的列;1e-10 是浮點數的容忍值,避免把 1e-17 當成非零
            for row in range(r, m):
                if abs(rows[row][col]) > 1e-10:
                    pivot = row
                    break
            if pivot is None:
                # 這一欄從第 r 列往下全是 0,沒有主元,換下一欄
                continue
            # 列交換:把找到主元的那一列換到第 r 列(Python 可以一行同時交換兩個值)
            rows[r], rows[pivot] = rows[pivot], rows[r]
            scale = rows[r][col]
            # 主元列整列除以主元,讓主元變成 1
            rows[r] = [x / scale for x in rows[r]]
            # 用主元列消掉其他所有列在這一欄的值:列 -= factor × 主元列
            for row in range(m):
                if row != r and abs(rows[row][col]) > 1e-10:
                    factor = rows[row][col]
                    rows[row] = [rows[row][j] - factor * rows[r][j] for j in range(n)]
            r += 1
        return r

    def __repr__(self):
        return f"Matrix({self.rows})"


# === 🟡 ===
def is_independent(vectors):
    # 判斷一組向量是否「線性獨立」:沒有任何一個向量可以用其他向量加加減減、
    # 乘個倍數湊出來(例如 [2,1,0] = 2*[1,0,0] + [0,1,0]，這樣就不獨立)。
    # 做法是列運算(高斯消去法):把向量當矩陣的列，一路消去，最後還剩幾個
    # 「非零列」(rank，矩陣的秩)，如果 rank 等於向量的數量，就是線性獨立。
    # ⚠️ 這堂課只提過名詞，沒有帶著手算過列運算的過程，測驗這題也答錯過。
    n = len(vectors)  # n:向量個數,rank 必須等於它才算獨立
    if n == 0:
        # 空集合視為獨立(數學慣例)
        return True
    dim = vectors[0].dim  # 每個向量的維度,同時也是消去時要走過的「欄數」
    rows = [v.components[:] for v in vectors]  # 每個向量複製成一列,組成矩陣
    rank = 0
    # 以下與 Matrix.rank() 完全相同的高斯消去流程,差別只在輸入是向量列表
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
    # rank 小於向量數,代表有向量被別人「吃掉」(能被湊出來),就是線性相依
    return rank == n


# === 🟡 ===
def gram_schmidt(vectors):
    # 把一組線性獨立的向量，轉換成「正交歸一基底」:每個向量互相垂直(正交)、
    # 長度都是1(歸一)。做法是一個一個處理，每個新向量都先扣掉它在前面
    # 已處理好的向量方向上的投影(project_onto)，扣掉之後剩下的部分保證
    # 跟前面的都垂直，再正規化(normalize)成長度1。
    # ⚠️ 這堂課只理解邏輯、沒手打，用途是數值方法/QR分解，不是日常AI開發常直接手刻的東西。
    orthonormal = []  # 已經處理好的正交歸一向量,會一路累積
    for v in vectors:
        w = v
        # 扣掉 w 在每個已處理向量方向上的影子,剩下的部分與它們都垂直
        for u in orthonormal:
            proj = w.project_onto(u)
            w = w - proj
        # 扣完長度趨近 0,代表這個向量本來就能被前面的湊出來(線性相依),略過
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())  # 縮成長度 1 再放進結果
    return orthonormal


# === 🟡 ===
# --- 底下是測試，直接跑這個檔案就會看到結果 ---
# if __name__ == "__main__":只有「直接執行這個檔案」時才會跑,被別的檔案 import 時不會跑
if __name__ == "__main__":
    print("=== Vector 基本運算 ===")
    a = Vector([1, 2, 3])
    b = Vector([4, 5, 6])
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")  # 預期 [5, 7, 9]
    print(f"a - b = {a - b}")  # 預期 [-3, -3, -3]
    print(f"a * 3 = {a * 3}")  # 預期 [3, 6, 9]
    print(f"a.dot(b) = {a.dot(b)}")  # 預期 32
    print(f"|a| = {a.magnitude():.4f}")  # :.4f 是格式化,取小數點後 4 位;sqrt(14)=3.7417
    print(f"a normalize = {a.normalize()}")
    print(f"cosine_similarity(a, b) = {a.cosine_similarity(b):.4f}")  # 兩者方向很接近,約 0.9746
    print(f"angle_between(a, b) = {a.angle_between(b):.2f} 度")

    print("\n=== 投影 ===")
    p = Vector([3, 4])
    q = Vector([1, 0])  # x 軸方向的單位向量,投影到它上面就等於取 x 分量
    print(f"project {p} onto {q} = {p.project_onto(q)}")  # 預期 [3, 0]

    print("\n=== 線性獨立 ===")
    e1 = Vector([1, 0, 0])
    e2 = Vector([0, 1, 0])
    dep = Vector([2, 1, 0])  # = 2*e1 + e2,可以被前兩個湊出來,所以相依
    # f-string 裡要印出大括號本身時要寫兩個 {{ }}
    print(f"{{e1, e2}} independent: {is_independent([e1, e2])}")  # 預期 True
    print(f"{{e1, e2, 2*e1+e2}} independent: {is_independent([e1, e2, dep])}")  # 預期 False

    print("\n=== Gram-Schmidt 正交化 ===")
    u1 = Vector([1, 1, 0])
    u2 = Vector([1, 0, 1])  # 與 u1 不垂直,經過處理後會被轉成垂直的
    basis = gram_schmidt([u1, u2])
    # enumerate 同時給出「編號」與「元素」;輸出的每個向量長度應該都是 1
    for i, u in enumerate(basis):
        print(f"u{i+1} = {u}, |u{i+1}| = {u.magnitude():.6f}")

    print("\n=== Matrix 運算 ===")
    rotation_90 = Matrix([[0, -1], [1, 0]])  # 逆時針旋轉 90 度的矩陣(Lesson 3 會細講)
    point = Vector([3, 1])
    print(f"rotation_90 @ point = {rotation_90 @ point}")  # 預期 [-1, 3]

    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[5, 6], [7, 8]])
    print(f"m1 @ m2 = {m1 @ m2}")  # 預期 [[19, 22], [43, 50]]
    print(f"m1.transpose() = {m1.transpose()}")  # 預期 [[1, 3], [2, 4]]
    print(f"m1.rank() = {m1.rank()}")  # 兩列互相獨立,預期 2

    rank_deficient = Matrix([[1, 2], [2, 4]])
    print(f"rank_deficient.rank() = {rank_deficient.rank()}  (應該是 1，因為第二列是第一列的兩倍)")
