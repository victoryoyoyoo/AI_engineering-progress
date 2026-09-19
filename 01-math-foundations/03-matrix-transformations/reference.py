import math  # 標準函式庫:cos、sin、pi 用來建立旋轉矩陣

# 這是這堂課(Matrix Transformations)官方課程的完整參考程式碼。
# Top-Down 策略:看懂邏輯、逐段講解過,不強制手刻。
# 這份檔案會被 git 追蹤、推上 GitHub。
#
# 檔案結構(依重要程度排序):
#   🔴 特徵值/特徵向量  : 這堂課真正的核心,推導過一次並用程式驗證 A@v == λ*v
#   🟡 變換矩陣         : 旋轉/縮放/切斜/鏡射,各自是「固定數字排法」的 2x2 矩陣
#   🟡 矩陣運算工具     : 上一課的矩陣乘向量、矩陣乘矩陣,這堂課拿來做複合變換
#   🟡 demo_*           : 各功能示範,最底下 __main__ 依序執行
# 矩陣一律用「list 的 list」表示,例如 [[1, 2], [3, 4]] 是 2x2 矩陣,外層每個元素是一列。
# 所有矩陣都只處理 2x2(這堂課只教 2D);3D 旋轉在 numpy_version.py 裡順便看過。


# === 🔴 ===
# --- 特徵值 / 特徵向量(這堂課的核心) ---
# 直覺:一個矩陣對空間做變換時,大部分向量的方向會被歪掉,
# 但有少數「特殊方向」上的向量只會被拉長或縮短,方向完全不變。
# 這些特殊方向叫特徵向量(eigenvector),被拉長縮短的倍數叫特徵值(eigenvalue)。
# 定義式:A v = λ v(左邊是矩陣乘向量,右邊只是把向量乘一個數字 λ)

def eigenvalues_2x2(matrix):
    # 解 characteristic equation: λ² - trace·λ + det = 0
    # 用一元二次方程式公式解:λ = (trace ± sqrt(trace²-4det)) / 2
    # 推導路線:A v = λ v → (A - λI) v = 0 → 要有非零解 v,det(A - λI) 必須等於 0
    #           → 展開 2x2 行列式,得到上面那條關於 λ 的二次方程式
    # trace(跡)= 主對角線相加 a+d,det(行列式)= a*d - b*c
    a, b = matrix[0]  # 第 0 列的兩個數字拆給 a、b(解構賦值)
    c, d = matrix[1]  # 第 1 列的兩個數字拆給 c、d
    trace = a + d
    det = a * d - b * c
    # 判別式 = 二次公式根號裡面的東西,正負決定有沒有實數解
    discriminant = trace ** 2 - 4 * det
    if discriminant < 0:
        # 判別式是負的,代表沒有實數解,eigenvalue是複數(旋轉矩陣就是這種情況)
        # 幾何意義:旋轉會改變每個向量的方向,不存在「方向不變」的特殊向量
        real = trace / 2                     # 複數的實部
        imag = (-discriminant) ** 0.5 / 2    # 複數的虛部,兩個解互為共軛
        return (complex(real, imag), complex(real, -imag))
    sqrt_disc = discriminant ** 0.5
    # 兩個解:加號的是較大的 eigenvalue,減號的是較小的
    # 例:[[2,1],[1,2]] → trace=4, det=3, 判別式=4 → λ = (4±2)/2 = 3 或 1
    return ((trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2)

def eigenvector_2x2(matrix, eigenvalue):
    # 給定一個 eigenvalue,反推對應的 eigenvector(解 (A-λI)v=0)。
    # 這裡用的是簡化技巧,不同情況取不同的湊法,最後正規化成長度1方便比較
    # (A-λI)v=0 的第一列是 (a-λ)*v0 + b*v1 = 0,所以取 v = [b, λ-a] 就會滿足
    # 第二列是 c*v0 + (d-λ)*v1 = 0,所以取 v = [λ-d, c] 就會滿足
    a, b = matrix[0]
    c, d = matrix[1]
    if abs(b) > 1e-10:
        # b 不是 0:用第一列的關係湊出 v(1e-10 是浮點數「幾乎等於 0」的容忍值)
        v = [b, eigenvalue - a]
    elif abs(c) > 1e-10:
        # b 是 0 但 c 不是:改用第二列的關係湊
        v = [eigenvalue - d, c]
    else:
        # b、c 都是 0(對角矩陣):特徵向量就是座標軸方向,看 eigenvalue 對應哪一個軸
        if abs(a - eigenvalue) < 1e-10:
            v = [1, 0]
        else:
            v = [0, 1]
    # 正規化:除以自己的長度,讓 eigenvector 長度為 1(方向才是重點,長度可以任意)
    mag = (v[0] ** 2 + v[1] ** 2) ** 0.5
    return [v[0] / mag, v[1] / mag]


# === 🟡 ===
# --- 變換矩陣(全部回傳 2x2 的 list of list) ---
# 每種變換都是「固定數字排法」的矩陣:矩陣 @ 點 = 變換後的點
# 記法:矩陣的第 0 欄是 (1,0) 變換後的落點,第 1 欄是 (0,1) 變換後的落點

def rotation_2d(theta):
    # 旋轉矩陣:[[cosθ,-sinθ],[sinθ,cosθ]],乘上任何點會讓它繞原點轉 theta 弧度
    # theta 用「弧度」不是度數(π 弧度 = 180 度);逆時針為正
    # 例:theta=π/2 → cos=0, sin=1 → [[0,-1],[1,0]],把 (1,0) 轉成 (0,1)
    # 旋轉只改方向、不改長度,所以 det=1
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]

def scaling_2d(sx, sy):
    # 縮放矩陣:對角線放 x/y 方向各自的縮放倍數，其餘是0
    # 例:scaling_2d(2, 3) @ (1,1) = (2,3),x 放大 2 倍、y 放大 3 倍
    return [[sx, 0], [0, sy]]

def shearing_2d(kx, ky):
    # 切斜矩陣:kx 讓 x 被 y 拖著偏移，ky 讓 y 被 x 拖著偏移
    # 例:shearing_2d(1, 0) @ (1,1) = (1*1+1*1, 1) = (2,1),y 越大 x 被推得越遠
    # 切斜會把正方形推成平行四邊形,面積不變,所以 det=1
    return [[1, kx], [ky, 1]]

def reflection_x():
    # 對 x 軸鏡射:x 不變、y 變號
    return [[1, 0], [0, -1]]

def reflection_y():
    # 對 y 軸鏡射:x 變號、y 不變
    # 鏡射的 det=-1,負號代表「翻面」(空間的左右手方向被反轉)
    return [[-1, 0], [0, 1]]


# === 🟡 ===
# --- 上一課學過的矩陣運算,這堂課會重複用到 ---

def mat_vec_mul(matrix, vector):
    # 矩陣乘向量:每一列跟輸入向量做內積
    # 輸出向量的第 i 個數字 = 矩陣第 i 列 與 vector 的內積
    # 內層 sum(... for j ...) 就是內積,外層 for i 走過每一列
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]

def mat_mul(a, b):
    # 矩陣乘矩陣,用來做複合變換(先套用哪個、後套用哪個)
    # 結果第 (i,j) 格 = a 的第 i 列 與 b 的第 j 欄 的內積
    # 順序很重要:mat_mul(S, R) 代表「先 R、後 S」,因為對點來說是 S @ (R @ 點),從右邊先算
    rows_a, cols_b = len(a), len(b[0])  # 結果的列數 = a 的列數,行數 = b 的行數
    cols_a = len(a[0])                  # a 的行數,也就是內積時要加總的長度 k
    return [
        [sum(a[i][k] * b[k][j] for k in range(cols_a)) for j in range(cols_b)]
        for i in range(rows_a)
    ]


def det_2x2(matrix):
    # 2x2 行列式,上一課學過的公式,這堂課拿來驗證「det=空間縮放倍數」這個直覺
    # 公式 ad - bc;絕對值是面積放大倍數,det=0 代表空間被壓成一條線(資訊遺失)
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


# === 🟡 ===
def demo_transformations():
    # 示範四種基本變換分別對一個點做了什麼
    print("=" * 60)
    print("BASIC TRANSFORMATIONS")
    print("=" * 60)

    point = [1.0, 0.0]
    angle = math.pi / 4  # 45 度

    # (1,0) 逆時針轉 45 度,落在 (cos45, sin45) ≈ (0.7071, 0.7071)
    rotated = mat_vec_mul(rotation_2d(angle), point)
    print(f"Rotate (1,0) by 45 deg: ({rotated[0]:.4f}, {rotated[1]:.4f})")

    # (1,1) x 放大 2 倍、y 放大 3 倍 → (2, 3)
    scaled = mat_vec_mul(scaling_2d(2, 3), [1.0, 1.0])
    print(f"Scale (1,1) by (2,3): ({scaled[0]:.1f}, {scaled[1]:.1f})")

    # (1,1) 切斜:x' = 1*1 + 1*1 = 2,y' = 0*1 + 1*1 = 1 → (2, 1)
    sheared = mat_vec_mul(shearing_2d(1, 0), [1.0, 1.0])
    print(f"Shear (1,1) kx=1: ({sheared[0]:.1f}, {sheared[1]:.1f})")

    # (2,1) 對 y 軸鏡射:x 變號 → (-2, 1)
    reflected = mat_vec_mul(reflection_y(), [2.0, 1.0])
    print(f"Reflect (2,1) across y: ({reflected[0]:.1f}, {reflected[1]:.1f})")


def demo_composition():
    # 示範複合變換的順序會不會影響結果(答案:會)
    print("\n" + "=" * 60)
    print("COMPOSITION: ORDER MATTERS")
    print("=" * 60)

    R = rotation_2d(math.pi / 2)  # 逆時針轉 90 度
    S = scaling_2d(2, 0.5)        # x 放大 2 倍、y 縮成一半

    rotate_then_scale = mat_mul(S, R)  # 先R後S,寫成 S@R(從右邊先算)
    scale_then_rotate = mat_mul(R, S)  # 先S後R,寫成 R@S

    point = [1.0, 0.0]
    # 先轉:(1,0)→(0,1),再縮放:(0,1)→(0,0.5)
    result1 = mat_vec_mul(rotate_then_scale, point)
    # 先縮放:(1,0)→(2,0),再轉:(2,0)→(0,2)
    result2 = mat_vec_mul(scale_then_rotate, point)

    print(f"Rotate 90 then scale: ({result1[0]:.2f}, {result1[1]:.2f})")
    print(f"Scale then rotate 90: ({result2[0]:.2f}, {result2[1]:.2f})")
    # 兩個 list 用 == 比較內容;結果是 False,證明矩陣乘法不符合交換律
    print(f"Same? {result1 == result2}")


def demo_eigenvalues():
    # 示範:算出A的eigenvalue跟eigenvector,再驗證 A@v == λ*v
    print("\n" + "=" * 60)
    print("EIGENVALUES AND EIGENVECTORS")
    print("=" * 60)

    A = [[2, 1], [1, 2]]
    vals = eigenvalues_2x2(A)  # 預期 3 與 1
    print(f"Matrix: {A}")
    print(f"Eigenvalues: {vals[0]:.4f}, {vals[1]:.4f}")

    for val in vals:
        vec = eigenvector_2x2(A, val)
        result = mat_vec_mul(A, vec)          # 左邊:真正的矩陣乘向量
        scaled = [val * vec[0], val * vec[1]]  # 右邊:單純乘一個倍數
        # round(x, 4) 四捨五入到小數 4 位,避免浮點數尾巴讓輸出很雜
        print(f"  lambda={val:.1f}, v={[round(x,4) for x in vec]}")
        # 兩行輸出的數字要一模一樣,才代表這個 (λ, v) 真的滿足 A v = λ v
        print(f"    A@v = {[round(x,4) for x in result]}")
        print(f"    l*v = {[round(x,4) for x in scaled]}")


def demo_determinant_as_scaling():
    # 示範:det的絕對值就是這個變換把面積放大/縮小的倍數
    print("\n" + "=" * 60)
    print("DETERMINANT AS VOLUME SCALING")
    print("=" * 60)

    print(f"det(rotation 45) = {det_2x2(rotation_2d(math.pi/4)):.4f}")  # 1:旋轉不改面積
    print(f"det(scale 2,3)   = {det_2x2(scaling_2d(2, 3)):.1f}")        # 6:面積放大 2*3=6 倍
    print(f"det(shear kx=1)  = {det_2x2(shearing_2d(1, 0)):.1f}")       # 1:切斜不改面積
    print(f"det(reflect y)   = {det_2x2(reflection_y()):.1f}")          # -1:面積不變但翻面

    singular = [[1, 2], [2, 4]]  # 第二列是第一列的兩倍,兩欄成比例
    print(f"det(singular)     = {det_2x2(singular):.1f}")  # 0:整個平面被壓成一條線
    print("Singular: columns are proportional, space collapses to a line.")


if __name__ == "__main__":
    # 直接執行這個檔案時才會跑;被 import 時不會跑
    demo_transformations()
    demo_composition()
    demo_eigenvalues()
    demo_determinant_as_scaling()
