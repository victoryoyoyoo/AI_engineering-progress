import math

# 這是這堂課(Matrix Transformations)官方課程的完整參考程式碼。
# Top-Down 策略:看懂邏輯、逐段講解過,不強制手刻。
# 這份檔案會被 git 追蹤、推上 GitHub。


# --- 變換矩陣(全部回傳 2x2 的 list of list) ---

def rotation_2d(theta):
    # 旋轉矩陣:[[cosθ,-sinθ],[sinθ,cosθ]],乘上任何點會讓它繞原點轉 theta 弧度
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]

def scaling_2d(sx, sy):
    # 縮放矩陣:對角線放 x/y 方向各自的縮放倍數，其餘是0
    return [[sx, 0], [0, sy]]

def shearing_2d(kx, ky):
    # 切斜矩陣:kx 讓 x 被 y 拖著偏移，ky 讓 y 被 x 拖著偏移
    return [[1, kx], [ky, 1]]

def reflection_x():
    # 對 x 軸鏡射:x 不變、y 變號
    return [[1, 0], [0, -1]]

def reflection_y():
    # 對 y 軸鏡射:x 變號、y 不變
    return [[-1, 0], [0, 1]]


# --- 上一課學過的矩陣運算,這堂課會重複用到 ---

def mat_vec_mul(matrix, vector):
    # 矩陣乘向量:每一列跟輸入向量做內積
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]

def mat_mul(a, b):
    # 矩陣乘矩陣,用來做複合變換(先套用哪個、後套用哪個)
    rows_a, cols_b = len(a), len(b[0])
    cols_a = len(a[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(cols_a)) for j in range(cols_b)]
        for i in range(rows_a)
    ]


# --- 特徵值 / 特徵向量(這堂課的核心) ---

def eigenvalues_2x2(matrix):
    # 解 characteristic equation: λ² - trace·λ + det = 0
    # 用一元二次方程式公式解:λ = (trace ± sqrt(trace²-4det)) / 2
    a, b = matrix[0]
    c, d = matrix[1]
    trace = a + d
    det = a * d - b * c
    discriminant = trace ** 2 - 4 * det
    if discriminant < 0:
        # 判別式是負的,代表沒有實數解,eigenvalue是複數(旋轉矩陣就是這種情況)
        real = trace / 2
        imag = (-discriminant) ** 0.5 / 2
        return (complex(real, imag), complex(real, -imag))
    sqrt_disc = discriminant ** 0.5
    return ((trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2)

def eigenvector_2x2(matrix, eigenvalue):
    # 給定一個 eigenvalue,反推對應的 eigenvector(解 (A-λI)v=0)。
    # 這裡用的是簡化技巧,不同情況取不同的湊法,最後正規化成長度1方便比較
    a, b = matrix[0]
    c, d = matrix[1]
    if abs(b) > 1e-10:
        v = [b, eigenvalue - a]
    elif abs(c) > 1e-10:
        v = [eigenvalue - d, c]
    else:
        if abs(a - eigenvalue) < 1e-10:
            v = [1, 0]
        else:
            v = [0, 1]
    mag = (v[0] ** 2 + v[1] ** 2) ** 0.5
    return [v[0] / mag, v[1] / mag]


def det_2x2(matrix):
    # 2x2 行列式,上一課學過的公式,這堂課拿來驗證「det=空間縮放倍數」這個直覺
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def demo_transformations():
    # 示範四種基本變換分別對一個點做了什麼
    print("=" * 60)
    print("BASIC TRANSFORMATIONS")
    print("=" * 60)

    point = [1.0, 0.0]
    angle = math.pi / 4

    rotated = mat_vec_mul(rotation_2d(angle), point)
    print(f"Rotate (1,0) by 45 deg: ({rotated[0]:.4f}, {rotated[1]:.4f})")

    scaled = mat_vec_mul(scaling_2d(2, 3), [1.0, 1.0])
    print(f"Scale (1,1) by (2,3): ({scaled[0]:.1f}, {scaled[1]:.1f})")

    sheared = mat_vec_mul(shearing_2d(1, 0), [1.0, 1.0])
    print(f"Shear (1,1) kx=1: ({sheared[0]:.1f}, {sheared[1]:.1f})")

    reflected = mat_vec_mul(reflection_y(), [2.0, 1.0])
    print(f"Reflect (2,1) across y: ({reflected[0]:.1f}, {reflected[1]:.1f})")


def demo_composition():
    # 示範複合變換的順序會不會影響結果(答案:會)
    print("\n" + "=" * 60)
    print("COMPOSITION: ORDER MATTERS")
    print("=" * 60)

    R = rotation_2d(math.pi / 2)
    S = scaling_2d(2, 0.5)

    rotate_then_scale = mat_mul(S, R)  # 先R後S,寫成 S@R(從右邊先算)
    scale_then_rotate = mat_mul(R, S)  # 先S後R,寫成 R@S

    point = [1.0, 0.0]
    result1 = mat_vec_mul(rotate_then_scale, point)
    result2 = mat_vec_mul(scale_then_rotate, point)

    print(f"Rotate 90 then scale: ({result1[0]:.2f}, {result1[1]:.2f})")
    print(f"Scale then rotate 90: ({result2[0]:.2f}, {result2[1]:.2f})")
    print(f"Same? {result1 == result2}")


def demo_eigenvalues():
    # 示範:算出A的eigenvalue跟eigenvector,再驗證 A@v == λ*v
    print("\n" + "=" * 60)
    print("EIGENVALUES AND EIGENVECTORS")
    print("=" * 60)

    A = [[2, 1], [1, 2]]
    vals = eigenvalues_2x2(A)
    print(f"Matrix: {A}")
    print(f"Eigenvalues: {vals[0]:.4f}, {vals[1]:.4f}")

    for val in vals:
        vec = eigenvector_2x2(A, val)
        result = mat_vec_mul(A, vec)          # 左邊:真正的矩陣乘向量
        scaled = [val * vec[0], val * vec[1]]  # 右邊:單純乘一個倍數
        print(f"  lambda={val:.1f}, v={[round(x,4) for x in vec]}")
        print(f"    A@v = {[round(x,4) for x in result]}")
        print(f"    l*v = {[round(x,4) for x in scaled]}")


def demo_determinant_as_scaling():
    # 示範:det的絕對值就是這個變換把面積放大/縮小的倍數
    print("\n" + "=" * 60)
    print("DETERMINANT AS VOLUME SCALING")
    print("=" * 60)

    print(f"det(rotation 45) = {det_2x2(rotation_2d(math.pi/4)):.4f}")
    print(f"det(scale 2,3)   = {det_2x2(scaling_2d(2, 3)):.1f}")
    print(f"det(shear kx=1)  = {det_2x2(shearing_2d(1, 0)):.1f}")
    print(f"det(reflect y)   = {det_2x2(reflection_y()):.1f}")

    singular = [[1, 2], [2, 4]]
    print(f"det(singular)     = {det_2x2(singular):.1f}")
    print("Singular: columns are proportional, space collapses to a line.")


if __name__ == "__main__":
    demo_transformations()
    demo_composition()
    demo_eigenvalues()
    demo_determinant_as_scaling()
