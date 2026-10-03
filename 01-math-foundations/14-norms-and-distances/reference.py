"""
Lesson 14: Norms and Distances(範數與距離)——完整參考版

這份檔案是「課程原版程式碼 + 詳細中文註解」,依重要程度由上到下排列:
    🔴 核心:L1/L2/Lp/L-inf範數與距離、內積、cosine相似度、向量正規化
    🟡 理解層:Mahalanobis、Jaccard、edit distance、KL、Wasserstein、embedding搜尋、KNN
    🟢 之後有空再補:矩陣求逆與共變異數的實作細節、找前k名、L1/L2正則化模擬

一句話主軸:「距離函式決定了什麼叫『相似』」。
    d(a, b) = ||a - b||,所有距離都是「兩個向量相減之後,量那個差的大小」,
    所以懂了範數(量向量大小的方法)就懂了距離。
"""
import math
import random


# 🔴 norms, distances, dot product, cosine similarity

def l1_norm(x):
    """L1範數(Manhattan距離):所有分量取絕對值再相加。像在城市方格裡走路,只能沿著街道走、不能斜穿。"""
    # abs(xi)取絕對值;sum(... for xi in x)是generator expression,邊走邊加總,不必先建立list
    return sum(abs(xi) for xi in x)


def l2_norm(x):
    """L2範數(Euclidean距離):各分量平方後相加再開根號。就是國中學的畢氏定理,推廣到n維。"""
    # xi ** 2是平方;math.sqrt開根號。例:(3,4) -> sqrt(9+16) = 5
    return math.sqrt(sum(xi ** 2 for xi in x))


def lp_norm(x, p):
    """Lp範數:L1和L2的一般化。p=1就是L1、p=2就是L2、p趨近無限大就是取最大分量。"""
    # p是無限大時,公式的極限是「最大的那個絕對值」,不能直接代入公式(會算出inf),要特別處理
    if p == float('inf'):
        return max(abs(xi) for xi in x)
    # 每個分量取絕對值、做p次方、加總,最後開p次方根(** (1 / p))
    return sum(abs(xi) ** p for xi in x) ** (1 / p)


def linf_norm(x):
    """L-infinity範數(Chebyshev距離):只看差最多的那一個維度,其他維度完全忽略。"""
    # 例:西洋棋的國王,往任何方向(含斜線)走一步都算1,用的就是這種距離
    return max(abs(xi) for xi in x)


def l1_distance(a, b):
    """兩點的L1距離 = 兩個向量相減後的L1範數。"""
    # zip(a, b)把兩個list同位置的元素配成一對,一次走一維
    return sum(abs(ai - bi) for ai, bi in zip(a, b))


def l2_distance(a, b):
    """兩點的L2距離 = 兩個向量相減後的L2範數。"""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def lp_distance(a, b, p):
    """兩點的Lp距離:先算出差向量,再交給lp_norm量大小。"""
    # 先逐維相減得到差向量,再量它的大小,對應「距離就是差的範數」這個主軸
    diff = [ai - bi for ai, bi in zip(a, b)]
    return lp_norm(diff, p)


def linf_distance(a, b):
    """兩點的L-infinity距離:各維度差的絕對值,取最大的那個。"""
    return max(abs(ai - bi) for ai, bi in zip(a, b))


def dot_product(a, b):
    """內積:對應位置相乘再加總。同時包含「方向」跟「長度」兩種資訊。"""
    # 幾何意義:a . b = ||a|| * ||b|| * cos(夾角)
    return sum(ai * bi for ai, bi in zip(a, b))


def cosine_similarity(a, b):
    """cosine相似度:只看兩個向量的夾角,完全不管長度。範圍從-1(反方向)到+1(同方向),垂直是0。"""
    # 公式:內積 / (a的長度 * b的長度),等於把兩個向量都縮成長度1之後再做內積
    dot = dot_product(a, b)
    # 分母要用的兩個長度
    norm_a = l2_norm(a)
    norm_b = l2_norm(b)
    # 零向量沒有方向,分母是0會除以0;這裡約定回傳0.0(沒有相似度可言)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    # 內積 ÷ 兩個長度相乘:長度被約掉,只剩方向
    return dot / (norm_a * norm_b)


def cosine_distance(a, b):
    """cosine距離 = 1 - cosine相似度,把「越像越大」翻成「越像越小」,範圍0到2。"""
    # 翻轉成「越像越小」,跟其他距離同一個方向
    return 1.0 - cosine_similarity(a, b)


def normalize_vector(v):
    """L2正規化:把向量縮成長度1,方向不變。正規化之後,內積和cosine相似度就完全相同。"""
    norm = l2_norm(v)
    # 零向量沒辦法縮放(除以0),原樣回傳一份複製
    if norm == 0:
        return v[:]
    return [vi / norm for vi in v]


def find_nearest_neighbor(query, dataset, distance_fn, **kwargs):
    """在資料集裡找離query最近的一個點,回傳(索引, 距離)。distance_fn可以換成任何距離函式。"""
    # 先假設最近距離是無限大,任何真實距離都會比它小
    best_idx = 0
    best_dist = float('inf')
    # enumerate同時給索引i跟資料點point
    for i, point in enumerate(dataset):
        # **kwargs把額外參數(例如Mahalanobis需要的cov_matrix)原封不動傳給距離函式
        d = distance_fn(query, point, **kwargs)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx, best_dist


def demo_norms():
    """列出幾個向量的L1、L2、L3、L-inf範數,觀察大小關係:L-inf <= L2 <= L1。"""
    print("=" * 65)
    print("NORMS: MEASURING VECTOR SIZE")
    print("=" * 65)

    # 每個元素是(顯示名稱, 向量),涵蓋均勻分佈、集中在單一維度等不同形狀
    vectors = [
        ("(3, 4)", [3, 4]),
        ("(1, 1, 1, 1)", [1, 1, 1, 1]),
        ("(5, 0, 0)", [5, 0, 0]),
        ("(1, 2, 3, 4, 5)", [1, 2, 3, 4, 5]),
    ]

    # {'L1':>8s}是靠右對齊、佔8格寬的字串格式,讓表格欄位整齊
    print(f"  {'Vector':<20s} {'L1':>8s} {'L2':>8s} {'L3':>8s} {'L-inf':>8s}")
    print(f"  {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    # 逐個向量印出四種範數,對照大小關係
    for name, v in vectors:
        print(f"  {name:<20s} {l1_norm(v):>8.3f} {l2_norm(v):>8.3f} "
              f"{lp_norm(v, 3):>8.3f} {linf_norm(v):>8.3f}")

    print()
    # 觀察上表:每一列都符合 L∞ ≤ L2 ≤ L1;(5,0,0)只有一格有值,所以三種相等
    print("  Note: L-inf <= L2 <= L1 always holds.")
    print()


def demo_distances():
    """對同一對點(A, B)算出各種距離與相似度,看它們各自給出什麼數字。"""
    print("=" * 65)
    print("DISTANCES BETWEEN TWO POINTS")
    print("=" * 65)

    a = [1, 2, 3]
    # 差向量是 (3, -2, 3):三種量法會給出不同的數字
    b = [4, 0, 6]

    print(f"  A = {a}")
    print(f"  B = {b}")
    print()
    # 走格子:|3| + |-2| + |3| = 8
    print(f"  L1 (Manhattan):   {l1_distance(a, b):.4f}")
    # 直線:√(9+4+9) = √22,約 4.69
    print(f"  L2 (Euclidean):   {l2_distance(a, b):.4f}")
    print(f"  L3:               {lp_distance(a, b, 3):.4f}")
    # 最大差:取 3、2、3 裡最大的,是 3
    print(f"  L-inf (Chebyshev):{linf_distance(a, b):.4f}")
    # cosine 只看夾角,跟上面長度型的距離是不同種類的量法
    print(f"  Cosine distance:  {cosine_distance(a, b):.4f}")
    print(f"  Cosine similarity:{cosine_similarity(a, b):.4f}")
    # 內積沒有除以長度,向量越長數字越大
    print(f"  Dot product:      {dot_product(a, b):.4f}")
    print()


def demo_cosine_vs_dot():
    """比較cosine與內積:B是A放大2倍(同方向),cosine認為兩者一樣,內積卻把長度也算進去。"""
    print("=" * 65)
    print("COSINE SIMILARITY vs DOT PRODUCT")
    print("=" * 65)

    a = [1, 2, 3]
    b = [2, 4, 6]
    # B 是 A 放大 2 倍(同方向);C 的方向跟 A 不同
    c = [3, 1, 0]

    print(f"  A = {a}")
    print(f"  B = {b}  (A scaled by 2)")
    print(f"  C = {c}  (different direction)")
    print()
    print(f"  {'Pair':<10s} {'Cosine':>10s} {'Dot':>10s}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 10}")
    # A 和 B 同方向:cosine 是 1,內積 28 卻比 A 對 C 的 5 大很多,因為 B 比較長
    print(f"  {'A vs B':<10s} {cosine_similarity(a, b):>10.4f} {dot_product(a, b):>10.4f}")
    print(f"  {'A vs C':<10s} {cosine_similarity(a, c):>10.4f} {dot_product(a, c):>10.4f}")
    print(f"  {'B vs C':<10s} {cosine_similarity(b, c):>10.4f} {dot_product(b, c):>10.4f}")
    print()
    print("  Cosine says A and B are identical (same direction).")
    print("  Dot product says B is more similar because of larger magnitude.")
    print()

    # 先把三個向量都縮成長度1,再重新比較
    a_norm = normalize_vector(a)
    b_norm = normalize_vector(b)
    c_norm = normalize_vector(c)

    # 縮成長度 1 之後,cosine 和內積的數字完全一樣
    print("  After L2 normalization:")
    print(f"  {'Pair':<10s} {'Cosine':>10s} {'Dot':>10s}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 10}")
    print(f"  {'A vs B':<10s} {cosine_similarity(a_norm, b_norm):>10.4f} {dot_product(a_norm, b_norm):>10.4f}")
    print(f"  {'A vs C':<10s} {cosine_similarity(a_norm, c_norm):>10.4f} {dot_product(a_norm, c_norm):>10.4f}")
    print()
    print("  After normalization, cosine and dot product are identical.")
    print()


def demo_norm_ordering():
    """隨機產生多組點,驗證不管維度多高,永遠是 L-inf <= L2 <= L1。"""
    print("=" * 65)
    print("NORM ORDERING: L-inf <= L2 <= L1 (always)")
    print("=" * 65)

    # 固定亂數種子,每次執行結果都一樣,方便對照
    random.seed(55)
    for trial in range(5):
        dim = random.randint(2, 10)
        # random.gauss(0, 5):平均0、標準差5的常態分佈亂數
        a = [random.gauss(0, 5) for _ in range(dim)]
        b = [random.gauss(0, 5) for _ in range(dim)]

        # 同一對隨機點,分別量三種距離,驗證大小順序
        d1 = l1_distance(a, b)
        d2 = l2_distance(a, b)
        dinf = linf_distance(a, b)

        # Python允許連續比較 dinf <= d2 <= d1,等於 (dinf <= d2) and (d2 <= d1)
        holds = dinf <= d2 <= d1
        print(f"  dim={dim:>2d}  L1={d1:>8.3f}  L2={d2:>8.3f}  L-inf={dinf:>8.3f}  ordering holds: {holds}")

    print()
    print("  For any p1 < p2: ||x||_p2 <= ||x||_p1")
    print("  Higher p values focus on fewer (larger) components.")
    print()


def demo_different_neighbors():
    """同一份資料、同一個查詢點,換不同距離函式,最近的鄰居就可能不同。"""
    print("=" * 65)
    print("SAME DATA, DIFFERENT METRICS, DIFFERENT NEAREST NEIGHBORS")
    print("=" * 65)

    # 固定亂數種子,每次結果一樣
    random.seed(123)
    # 8 個候選點,每個點 5 維
    n_points = 8
    # 每個點是 5 維向量
    dim = 5

    # 造出三種不同形狀的點:前3個是一般的圓形分佈、中間3個沿第0維被拉長、最後2個離原點很遠
    dataset = []
    for i in range(n_points):
        # 前 3 個點:一般的圓形分佈
        if i < 3:
            point = [random.gauss(0, 1) for _ in range(dim)]
        # 中間 3 個點:沿第 0 維被拉長
        elif i < 6:
            base = [random.gauss(0, 0.5) for _ in range(dim)]
            # 把第0維放大5倍,讓這組點在該方向特別長
            base[0] *= 5
            point = base
        else:
            point = [random.gauss(3, 0.3) for _ in range(dim)]
        dataset.append(point)

    # 查詢點:要找離它最近的候選點
    query = [1.0, 0.5, -0.5, 1.0, 0.2]

    print(f"  Query: {[round(x, 2) for x in query]}")
    print()
    print(f"  {'Point':<8s} {'L1':>8s} {'L2':>8s} {'Cosine':>8s} {'L-inf':>8s}")
    print(f"  {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")

    # 每種距離各存一份(點的索引, 距離)清單,最後找各自的最小值
    results = {"L1": [], "L2": [], "Cosine": [], "L-inf": []}

    for i, point in enumerate(dataset):
        # 同一個查詢點對同一個候選點,四種距離各算一次
        d_l1 = l1_distance(query, point)
        d_l2 = l2_distance(query, point)
        d_cos = cosine_distance(query, point)
        d_linf = linf_distance(query, point)

        results["L1"].append((i, d_l1))
        results["L2"].append((i, d_l2))
        results["Cosine"].append((i, d_cos))
        results["L-inf"].append((i, d_linf))

        print(f"  P{i:<6d} {d_l1:>8.3f} {d_l2:>8.3f} {d_cos:>8.4f} {d_linf:>8.3f}")

    print()
    # 每種距離各自選出最近的鄰居(這組資料四種選的是同一個點,差別在後面的排名)
    print("  Nearest neighbor by metric:")
    for metric_name, dists in results.items():
        # key=lambda x: x[1]:以tuple的第2個元素(距離)當比較依據
        best = min(dists, key=lambda x: x[1])
        print(f"    {metric_name:<8s}: Point {best[0]} (distance = {best[1]:.4f})")

    l1_best = min(results["L1"], key=lambda x: x[1])[0]
    l2_best = min(results["L2"], key=lambda x: x[1])[0]
    cos_best = min(results["Cosine"], key=lambda x: x[1])[0]
    linf_best = min(results["L-inf"], key=lambda x: x[1])[0]

    # 四個都選同一個點才算一致;只要有不同就代表「距離的選擇改變了答案」
    all_same = (l1_best == l2_best == cos_best == linf_best)
    if not all_same:
        print()
        print("  The metrics DISAGREE on which point is nearest.")
        print("  Your distance function defines your notion of similarity.")
    print()


# 🟡 Mahalanobis, Jaccard, edit distance, KL, Wasserstein, embedding search, KNN

def mahalanobis_distance(x, y, cov_matrix):
    """Mahalanobis距離:考慮資料的共變異數(各特徵的尺度與相關性),先「白化」再量L2。
        公式:sqrt((x-y)^T * S^-1 * (x-y)),S是共變異數矩陣。S是單位矩陣時退化成一般L2。"""
    n = len(x)
    # 先算差向量
    diff = [xi - yi for xi, yi in zip(x, y)]

    # 共變異數矩陣的反矩陣(實作細節在🟢區的invert_matrix)
    inv_cov = invert_matrix(cov_matrix)

    # temp = diff^T * S^-1:一個向量乘一個矩陣,得到另一個向量(三層迴圈的手刻矩陣乘法)
    temp = [0.0] * n
    for i in range(n):
        for j in range(n):
            temp[i] += diff[j] * inv_cov[j][i]

    # 再跟diff做內積,得到 diff^T * S^-1 * diff 這個純量
    result = sum(temp[i] * diff[i] for i in range(n))
    # max(0, result):浮點誤差可能讓結果變成極小的負數,開根號會出錯,先夾到0以上
    return math.sqrt(max(0, result))


def jaccard_similarity(set_a, set_b):
    """Jaccard相似度:交集大小 / 聯集大小。專門比較「集合」的重疊程度,範圍0到1。"""
    # 兩個都是空集合:約定視為完全相同(1.0),避免0/0
    if not set_a and not set_b:
        return 1.0
    # & 是集合交集、| 是集合聯集
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def jaccard_distance(set_a, set_b):
    """Jaccard距離 = 1 - Jaccard相似度。"""
    return 1.0 - jaccard_similarity(set_a, set_b)


def edit_distance(s1, s2):
    """編輯距離(Levenshtein):把s1變成s2最少需要幾次「插入、刪除、替換」單一字元。用動態規劃(DP)填表。"""
    # m、n 是兩個字串的長度
    m, n = len(s1), len(s2)
    # dp[i][j] = s1前i個字元 變成 s2前j個字元 的最少操作次數;建成(m+1)x(n+1)的表,多的一格是「空字串」
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 邊界:空字串變成長度j的字串,要插入j次;長度i的字串變成空字串,要刪除i次
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # 這兩個字元剛好相同:不用花操作,直接繼承左上角的答案
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # 不同就要花1次操作,從三種方式挑最便宜的:
                # 上方(刪除s1的字元)、左方(插入s2的字元)、左上(替換成另一個字元)
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1]
                )

    # 右下角就是整串字串的編輯距離
    return dp[m][n]


def kl_divergence(p, q):
    """KL散度:用Q來近似P會損失多少資訊。不對稱(KL(P||Q) != KL(Q||P)),所以不算真正的距離。(Lesson 9教過)"""
    # 累加每一格的 p × log(p ÷ q)
    total = 0.0
    for pi, qi in zip(p, q):
        # pi是0的項貢獻為0(0 * log0 約定為0),直接跳過
        if pi > 0:
            # P有機率的地方Q卻是0:log(pi/0)是無限大,整個KL就是無限大
            if qi <= 0:
                return float('inf')
            total += pi * math.log(pi / qi)
    return total


def wasserstein_1d(p, q):
    """一維Wasserstein距離(Earth Mover's Distance):把分布P的「土堆」搬成Q要花多少工。
        一維時等於兩個累積分布(CDF)的差取絕對值後加總。是真正的距離,分布完全不重疊時也有意義。"""
    assert len(p) == len(q), "Distributions must have the same number of bins"
    # n 是格子(位置)的數量
    n = len(p)
    cdf_p = [0.0] * n
    cdf_q = [0.0] * n

    # 累積分布:第i格 = 前i格機率的總和
    cdf_p[0] = p[0]
    cdf_q[0] = q[0]
    for i in range(1, n):
        cdf_p[i] = cdf_p[i - 1] + p[i]
        cdf_q[i] = cdf_q[i - 1] + q[i]

    # 每道門檻要搬過去的土量,全部加起來就是總工
    return sum(abs(cdf_p[i] - cdf_q[i]) for i in range(n))


def demo_mahalanobis():
    """兩個點跟平均的L2距離幾乎一樣,但一個順著資料的相關方向、一個垂直於它;Mahalanobis能分辨出後者比較「不尋常」。"""
    print("=" * 65)
    print("MAHALANOBIS DISTANCE")
    print("=" * 65)

    # 固定亂數種子,每次結果一樣
    random.seed(42)
    n = 200
    data = []
    # 造出相關的兩個特徵:y約等於0.8x再加一點雜訊,所以點會沿著一條斜線分佈
    for _ in range(n):
        x = random.gauss(0, 3)
        y = 0.8 * x + random.gauss(0, 1)
        # 把這一筆資料(兩個特徵)存起來
        data.append([x, y])

    # 共變異數矩陣記錄兩個特徵一起變動的程度(Lesson 10)
    cov = compute_covariance(data)
    mean = [sum(d[0] for d in data) / n, sum(d[1] for d in data) / n]

    # 順著相關軸的點(沿資料本來就會散開的方向) vs 垂直於相關軸的點(違反資料的相關結構)
    point_along = [mean[0] + 3, mean[1] + 0.8 * 3]
    # 橫過資料方向的點:直線距離不大,但很不尋常
    point_perp = [mean[0] + 1, mean[1] - 3]

    # 兩種距離對兩個點各量一次,對照結論
    l2_along = l2_distance(mean, point_along)
    l2_perp = l2_distance(mean, point_perp)
    mah_along = mahalanobis_distance(mean, point_along, cov)
    mah_perp = mahalanobis_distance(mean, point_perp, cov)

    # 註:0.8 是造資料時的係數,實際相關係數約 0.92
    print(f"  Data: {n} points with correlated features (r ~ 0.8)")
    print(f"  Mean: ({mean[0]:.2f}, {mean[1]:.2f})")
    print(f"  Covariance: [[{cov[0][0]:.2f}, {cov[0][1]:.2f}], [{cov[1][0]:.2f}, {cov[1][1]:.2f}]]")
    print()
    print(f"  Point along correlation axis:  {[round(x, 2) for x in point_along]}")
    print(f"    L2 distance from mean:       {l2_along:.4f}")
    print(f"    Mahalanobis distance:         {mah_along:.4f}")
    print()
    print(f"  Point perpendicular to axis:   {[round(x, 2) for x in point_perp]}")
    print(f"    L2 distance from mean:       {l2_perp:.4f}")
    print(f"    Mahalanobis distance:         {mah_perp:.4f}")
    print()
    # 結論:直線距離分不出兩點的差別,Mahalanobis 分得出來
    print("  L2 says both points are similar distances from the mean.")
    print("  Mahalanobis correctly identifies the perpendicular point as")
    print("  more unusual given the correlation structure of the data.")
    print()


def demo_jaccard():
    """四組集合的Jaccard相似度:部分重疊、完全相同、完全不重疊、一個包含另一個。"""
    print("=" * 65)
    print("JACCARD SIMILARITY (SETS)")
    print("=" * 65)

    pairs = [
        ({"cat", "dog", "fish"}, {"cat", "bird", "fish", "snake"}),
        ({"python", "java", "rust"}, {"python", "java", "rust"}),
        ({"a", "b", "c"}, {"d", "e", "f"}),
        ({"ml", "ai", "data"}, {"ml", "ai", "data", "ops", "cloud"}),
    ]

    # 逐組印出兩個集合與它們的相似度、距離
    for a, b in pairs:
        # 交集大小 ÷ 聯集大小
        j = jaccard_similarity(a, b)
        # sorted(a)讓集合(本來無順序)印出來的順序固定,方便閱讀
        print(f"  A = {sorted(a)}")
        print(f"  B = {sorted(b)}")
        print(f"  Jaccard similarity: {j:.4f}")
        # 距離 = 1 − 相似度
        print(f"  Jaccard distance:   {1 - j:.4f}")
        print()


def demo_edit_distance():
    """幾組字串的編輯距離,包含經典的kitten -> sitting(答案是3)與空字串的邊界情況。"""
    print("=" * 65)
    print("EDIT DISTANCE (LEVENSHTEIN)")
    print("=" * 65)

    pairs = [
        ("kitten", "sitting"),
        ("sunday", "saturday"),
        ("hello", "hello"),
        ("", "abc"),
        ("algorithm", "altruistic"),
        ("python", "pytorch"),
    ]

    # 逐組算編輯距離
    for s1, s2 in pairs:
        # 回傳最少需要幾次插入、刪除、替換
        d = edit_distance(s1, s2)
        print(f"  '{s1}' -> '{s2}':  distance = {d}")

    print()


def demo_kl_divergence():
    """驗證KL散度不對稱:KL(P||Q)和KL(Q||P)算出來的數字不一樣。"""
    print("=" * 65)
    print("KL DIVERGENCE (NOT SYMMETRIC)")
    print("=" * 65)

    # P 是真實分布(90% 正面),Q 是你以為的分布(公平硬幣)
    p = [0.9, 0.1]
    q = [0.5, 0.5]

    # 兩個方向各算一次,驗證 KL 不對稱
    kl_pq = kl_divergence(p, q)
    kl_qp = kl_divergence(q, p)

    print(f"  P = {p}")
    print(f"  Q = {q}")
    print(f"  KL(P || Q) = {kl_pq:.4f} nats")
    print(f"  KL(Q || P) = {kl_qp:.4f} nats")
    print(f"  Difference: {abs(kl_pq - kl_qp):.4f}")
    print(f"  KL divergence is NOT a distance metric.")
    print()

    # 第二組:四個選項的分布
    p2 = [0.25, 0.25, 0.25, 0.25]
    q2 = [0.1, 0.1, 0.1, 0.7]

    print(f"  P = {p2}")
    print(f"  Q = {q2}")
    print(f"  KL(P || Q) = {kl_divergence(p2, q2):.4f} nats")
    print(f"  KL(Q || P) = {kl_divergence(q2, p2):.4f} nats")
    print()


def demo_wasserstein():
    """五種情境比較Wasserstein與KL:分布不重疊時KL變無限大,Wasserstein仍給出有意義的有限數字。"""
    print("=" * 65)
    print("WASSERSTEIN DISTANCE (EARTH MOVER'S DISTANCE)")
    print("=" * 65)

    cases = [
        ("Identical",
         [0.25, 0.25, 0.25, 0.25],
         [0.25, 0.25, 0.25, 0.25]),
        ("Shifted right by 1",
         [0.5, 0.5, 0.0, 0.0],
         [0.0, 0.5, 0.5, 0.0]),
        ("Shifted right by 2",
         [0.5, 0.5, 0.0, 0.0],
         [0.0, 0.0, 0.5, 0.5]),
        ("Opposite ends",
         [1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]),
        ("Spread vs concentrated",
         [0.25, 0.25, 0.25, 0.25],
         [0.0, 0.0, 0.0, 1.0]),
    ]

    # 每個情境同時算搬土距離和 KL,對照沒重疊時的差別
    for name, p, q in cases:
        # 土量 × 搬的距離
        w = wasserstein_1d(p, q)
        # 兩邊沒重疊時 KL 會是無限大
        kl = kl_divergence(p, q)
        # KL是無限大時印成文字"inf",不用數字格式
        kl_str = f"{kl:.4f}" if kl != float('inf') else "inf"
        print(f"  {name}")
        print(f"    P = {p}")
        print(f"    Q = {q}")
        print(f"    Wasserstein: {w:.4f}    KL: {kl_str}")
        print()

    # 結論:沒重疊時 KL 全是無限大,搬土距離仍然分得出遠近
    print("  Wasserstein provides finite, meaningful distances even when")
    print("  distributions do not overlap (where KL goes to infinity).")
    print()


def demo_embedding_search():
    """模擬「文件embedding搜尋」:用cosine、L2、內積三種方式替同一個query排名,看排名會不會不同。"""
    print("=" * 65)
    print("EMBEDDING SIMILARITY SEARCH")
    print("=" * 65)

    # 固定亂數種子,每次執行得到同樣的假 embedding
    random.seed(77)
    # 假裝每篇文件被模型轉成 64 維的向量
    dim = 64

    # 8 篇文件:前 5 篇是 AI 相關、後 3 篇是系統相關
    documents = [
        "machine learning algorithms",
        "deep neural networks",
        "natural language processing",
        "computer vision models",
        "reinforcement learning agents",
        "database query optimization",
        "web server configuration",
        "network security protocols",
    ]

    # 造假的embedding:前5篇(AI相關)在前10維加上2.0的「主題訊號」,後3篇(系統相關)在第10到19維加訊號
    # 前2篇再多加一段共同訊號,代表它們特別接近
    embeddings = []
    for i, doc in enumerate(documents):
        # 先造一個隨機向量當底,再依主題加訊號
        base = [random.gauss(0, 1) for _ in range(dim)]
        # 前 5 篇(AI 相關)
        if i < 5:
            for j in range(10):
                base[j] += 2.0
        else:
            for j in range(10, 20):
                base[j] += 2.0
        # 第 0、1 篇再多一段共同訊號
        if i in [0, 1]:
            for j in range(20, 25):
                base[j] += 1.5
        # 存起來,之後拿來跟 query 比
        embeddings.append(base)

    # query是第0篇加上一點雜訊,理論上最像它自己
    query_embedding = embeddings[0][:]
    # 小雜訊:標準差 0.3,比底的 1 小很多
    noise = [random.gauss(0, 0.3) for _ in range(dim)]
    # 把雜訊加到每一維上,模擬「查詢跟原文件不完全一樣」
    query_embedding = [q + n for q, n in zip(query_embedding, noise)]

    print(f"  Query: '{documents[0]}' (with noise)")
    print(f"  Embedding dimension: {dim}")
    print()

    # 三種方式各存一份(文件索引, 分數)
    cosine_scores = []
    l2_scores = []
    dot_scores = []

    for i in range(len(documents)):
        # cosine 相似度:越大越像
        cos = cosine_similarity(query_embedding, embeddings[i])
        # L2 距離:越小越像
        l2 = l2_distance(query_embedding, embeddings[i])
        # 內積:越大越像,而且向量越長分數越高
        dp = dot_product(query_embedding, embeddings[i])
        # 把這篇文件的 cosine 分數存起來
        cosine_scores.append((i, cos))
        l2_scores.append((i, l2))
        dot_scores.append((i, dp))

    # 相似度越大越像(用負號由大到小排);距離越小越像(由小到大排)
    cosine_ranked = sorted(cosine_scores, key=lambda x: -x[1])
    l2_ranked = sorted(l2_scores, key=lambda x: x[1])
    dot_ranked = sorted(dot_scores, key=lambda x: -x[1])

    print(f"  {'Rank':<6s} {'Cosine':<35s} {'L2':<35s} {'Dot Product':<35s}")
    print(f"  {'-' * 6} {'-' * 35} {'-' * 35} {'-' * 35}")
    # 逐名次印出三種方式各自排出的文件
    for rank in range(len(documents)):
        # 第 rank 名的(文件索引, 分數),三種排名各取一次
        ci, cs = cosine_ranked[rank]
        li, ls = l2_ranked[rank]
        di, ds = dot_ranked[rank]
        # 只取文件名前 25 個字,對齊成固定寬度,方便並排比較
        cos_str = f"{documents[ci][:25]:<25s} ({cs:.3f})"
        l2_str = f"{documents[li][:25]:<25s} ({ls:.1f})"
        dot_str = f"{documents[di][:25]:<25s} ({ds:.1f})"
        print(f"  {rank + 1:<6d} {cos_str:<35s} {l2_str:<35s} {dot_str:<35s}")

    print()
    # 結論:cosine 看主題方向、L2 受長度差影響、內積兩者混合
    print("  Cosine similarity focuses on direction (topic similarity).")
    print("  L2 distance is sensitive to magnitude differences.")
    print("  Dot product blends direction and magnitude.")
    print()


def demo_knn_classification():
    """KNN分類:同一個query、同一份訓練資料,換不同距離函式,投票結果可能預測出不同類別。"""
    print("=" * 65)
    print("KNN CLASSIFICATION: DISTANCE METRIC CHANGES THE PREDICTION")
    print("=" * 65)

    random.seed(99)

    # 每筆是(座標, 類別標籤);三個類別各3個點
    training_data = [
        ([1.0, 5.0], "A"),
        ([1.5, 4.5], "A"),
        ([2.0, 4.0], "A"),
        ([5.0, 1.0], "B"),
        ([4.5, 1.5], "B"),
        ([4.0, 2.0], "B"),
        ([3.0, 3.0], "C"),
        ([3.5, 2.5], "C"),
        ([2.5, 3.5], "C"),
    ]

    # 要預測類別的新點
    query = [2.8, 2.8]

    print(f"  Query: {query}")
    print(f"  Training set: {len(training_data)} points, 3 classes")
    print()

    # 看最近的 3 個鄰居
    k = 3
    # 四種距離各跑一次 KNN 分類
    for metric_name, dist_fn in [("L1", l1_distance), ("L2", l2_distance),
                                   ("Cosine", cosine_distance), ("L-inf", linf_distance)]:
        # 存每個訓練點的(距離, 類別, 座標)
        distances = []
        for point, label in training_data:
            d = dist_fn(query, point)
            # 記下(距離, 類別, 座標)
            distances.append((d, label, point))
        # 由近到遠排序,取前k個當鄰居
        distances.sort(key=lambda x: x[0])

        # 取最近的 k 個
        neighbors = distances[:k]
        # 統計k個鄰居各類別的票數
        votes = {}
        for d, label, point in neighbors:
            votes[label] = votes.get(label, 0) + 1
        # max(votes, key=votes.get):找票數最多的那個類別
        prediction = max(votes, key=votes.get)

        print(f"  Metric: {metric_name}")
        for d, label, point in neighbors:
            print(f"    Neighbor: {point}  class={label}  dist={d:.4f}")
        print(f"    Prediction (k={k}): {prediction}")
        print()


# 🟢 matrix inverse, covariance, k nearest, L1/L2 regularization simulation

def invert_matrix(matrix):
    """用Gauss-Jordan消去法求反矩陣:把[A | I]用列運算變成[I | A^-1]。這裡是Mahalanobis用到的實作細節。"""
    n = len(matrix)
    # 增廣矩陣:原矩陣右邊接上單位矩陣(對角線1、其他0)
    augmented = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]

    for col in range(n):
        # 部分選主元:在這一欄往下找絕對值最大的列換上來,避免除以很小的數造成誤差放大
        max_row = col
        for row in range(col + 1, n):
            if abs(augmented[row][col]) > abs(augmented[max_row][col]):
                max_row = row
        # 交換兩列
        augmented[col], augmented[max_row] = augmented[max_row], augmented[col]

        pivot = augmented[col][col]
        # 主元接近0代表矩陣是奇異矩陣(沒有反矩陣),無法繼續
        if abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular or near-singular")
        # 整列除以主元,讓主元變成1
        for j in range(2 * n):
            augmented[col][j] /= pivot

        # 用主元列把其他列這一欄的值消成0
        for row in range(n):
            if row != col:
                factor = augmented[row][col]
                for j in range(2 * n):
                    augmented[row][j] -= factor * augmented[col][j]

    # 右半邊就是反矩陣
    return [row[n:] for row in augmented]


def compute_covariance(data):
    """算共變異數矩陣:第(i, j)格代表「第i個特徵」與「第j個特徵」一起變動的程度。"""
    n = len(data)
    d = len(data[0])
    # 每個特徵(每一欄)的平均值
    means = [sum(data[i][j] for i in range(n)) / n for j in range(d)]
    # 置中:每個值減掉該欄平均
    centered = [[data[i][j] - means[j] for j in range(d)] for i in range(n)]
    cov = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            # 除以(n-1)是樣本共變異數(無偏估計),不是除以n
            cov[i][j] = sum(centered[k][i] * centered[k][j] for k in range(n)) / (n - 1)
    # cov[i][j] 代表特徵 i 和特徵 j 一起變動的程度
    return cov


def find_k_nearest(query, dataset, distance_fn, k=5, **kwargs):
    """回傳離query最近的前k個點,格式是[(索引, 距離), ...],由近到遠。"""
    # 存(索引, 距離)
    distances = []
    for i, point in enumerate(dataset):
        d = distance_fn(query, point, **kwargs)
        distances.append((i, d))
    # 依距離(tuple第2個元素)由小到大排序,取前k個
    distances.sort(key=lambda x: x[1])
    return distances[:k]


def demo_regularization():
    """模擬L1與L2正則化對權重的影響:L1把小權重推到剛好0(稀疏),L2讓所有權重縮小但不會是0。"""
    print("=" * 65)
    print("L1 vs L2 REGULARIZATION EFFECT ON WEIGHTS")
    print("=" * 65)

    random.seed(42)
    # 假裝模型有 10 個權重
    n_features = 10
    # 隨機產生 10 個初始權重(平均 0、標準差 2)
    weights = [random.gauss(0, 2) for _ in range(n_features)]

    print(f"  Original weights: {[round(w, 3) for w in weights]}")
    print(f"  L1 norm: {l1_norm(weights):.4f}")
    print(f"  L2 norm: {l2_norm(weights):.4f}")
    print()

    # lr 是每一步處罰的力道
    lr = 0.1

    # L1:每一步固定往0的方向走lr那麼遠(L1的梯度是符號函數,大小固定),小於lr的權重會直接跨到0並停住
    w_l1 = weights[:]
    for step in range(50):
        for i in range(n_features):
            grad = lr * (1 if w_l1[i] > 0 else (-1 if w_l1[i] < 0 else 0))
            # 往 0 的方向走固定的一步
            w_l1[i] -= grad
            if abs(w_l1[i]) < lr:
                w_l1[i] = 0.0

    # L2:每一步往0的方向走「跟自己成正比」的距離(L2平方的梯度是2w),越接近0走得越慢,所以永遠到不了0
    w_l2 = weights[:]
    for step in range(50):
        for i in range(n_features):
            # L2 的處罰力道跟權重本身成正比:權重越小,力道越小
            grad = lr * 2 * w_l2[i]
            w_l2[i] -= grad

    # 註:這裡只有處罰、沒有「答對」那一項,所以跑 50 步兩邊最後都幾乎全 0;要看差異請看前面幾步
    print(f"  After L1 regularization (50 steps):")
    print(f"    Weights: {[round(w, 3) for w in w_l1]}")
    print(f"    Zeros:   {sum(1 for w in w_l1 if w == 0.0)}/{n_features}")
    print(f"    L1 norm: {l1_norm(w_l1):.4f}")
    print()
    print(f"  After L2 regularization (50 steps):")
    print(f"    Weights: {[round(w, 3) for w in w_l2]}")
    print(f"    Zeros:   {sum(1 for w in w_l2 if abs(w) < 1e-10)}/{n_features}")
    print(f"    L2 norm: {l2_norm(w_l2):.4f}")
    print()
    print("  L1 drives 'small' weights to exactly zero (sparsity).")
    print("  L2 shrinks all weights but none reach exactly zero.")
    print()


if __name__ == "__main__":
    demo_norms()
    demo_distances()
    demo_cosine_vs_dot()
    demo_mahalanobis()
    demo_jaccard()
    demo_edit_distance()
    demo_kl_divergence()
    demo_wasserstein()
    demo_norm_ordering()
    demo_different_neighbors()
    demo_embedding_search()
    demo_knn_classification()
    demo_regularization()
