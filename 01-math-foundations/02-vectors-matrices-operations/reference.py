import random  # 標準函式庫:Matrix.random 用它產生隨機小數,random.seed 用來固定結果

# 這是這堂課(Vectors, Matrices & Operations)官方課程的完整參考程式碼。
# 逐段講解過邏輯,但依 Top-Down 策略沒有手打進 practice.py,只手打了
# numpy_version.py 裡那一行核心邏輯。
#
# 檔案結構:
#   Matrix class : 純 Python 實作矩陣(matmul、加減、逐項乘、轉置、行列式、逆矩陣、單位矩陣)
#   Vector class : 純 Python 實作向量(加減、純量乘、內積、長度、正規化)
#   relu_matrix  : ReLU 激活函數
#   demo_*       : 各功能的示範,最底下 __main__ 依序執行
# 矩陣資料一律存成「list 的 list」,外層一個元素是一列(row)。


# === 🔴 ===
class Matrix:
    """矩陣:m 列 n 行的數字表格,可以當成一台「吃 n 維向量、吐 m 維向量」的機器。"""

    def __init__(self, data):
        # data 是「一列一列」的資料(list of list),rows=幾列、cols=幾行
        # list(row) 複製每一列,避免跟傳進來的原始資料共用同一個物件(改動時互相影響)
        self.data = [list(row) for row in data]
        self.rows = len(self.data)        # 外層 list 長度 = 列數
        self.cols = len(self.data[0])     # 第一列的長度 = 行數(假設每列一樣長)
        self.shape = (self.rows, self.cols)  # 慣例寫法:(列數, 行數),跟 NumPy 的 .shape 一致

    # === 🔴 ===
    def matmul(self, other):
        # 矩陣乘法(真正的matrix multiply,不是逐項相乘)
        # 形狀規則:(m,n) @ (n,p) = (m,p),中間的n要對上
        # 對照C++三層迴圈:i是外層列、j是中層行、k是內層做內積加總
        # for(i) for(j) { sum=0; for(k) sum += A[i][k]*B[k][j]; result[i][j]=sum; }
        # 結果第 (i,j) 格 = 左矩陣第 i 列 與 右矩陣第 j 行 的內積
        # 例:[[1,2],[3,4]] @ [[5,6],[7,8]] 的第 (0,0) 格 = 1*5 + 2*7 = 19
        if self.cols != other.rows:
            # 中間維度對不上就無法相乘,直接丟出清楚的錯誤訊息(方便 debug shape 錯誤)
            raise ValueError(
                f"Cannot multiply shapes {self.shape} and {other.shape}: "
                f"inner dimensions {self.cols} != {other.rows}"
            )
        # 最外層 for i 產生每一列,內層 for j 產生該列的每個數字,sum(... for k ...) 就是內積
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def __matmul__(self, other):
        # 讓 @ 這個運算子自動呼叫上面的 matmul,語法糖
        # 沒定義 __matmul__ 的話,寫 A @ B 會報 TypeError
        return self.matmul(other)

    # === 🟡 ===
    def __repr__(self):
        # 純粹是印出來好看用的排版邏輯(對齊欄寬、加中括號),跟矩陣運算邏輯無關
        # 第一步:算出每一欄最寬的字串長度,讓所有數字靠右對齊
        col_widths = []
        for j in range(self.cols):
            # f"{數字:.4f}" 是格式化字串:固定顯示小數點後 4 位
            width = max(len(f"{self.data[i][j]:.4f}") for i in range(self.rows))
            col_widths.append(width)
        lines = []
        for i in range(self.rows):
            # f"{x:{寬度}.4f}":寬度本身也可以用變數代入,不足的位數補空白
            row_str = "  ".join(
                f"{self.data[i][j]:{col_widths[j]}.4f}" for j in range(self.cols)
            )
            # 用 / | \ 拼出大括號的形狀:第一列左邊用 /、最後一列用 \、中間用 |
            bracket_l = "|" if 0 < i < self.rows - 1 else ("/" if i == 0 else "\\")
            bracket_r = "|" if 0 < i < self.rows - 1 else ("\\" if i == 0 else "/")
            lines.append(f"  {bracket_l} {row_str} {bracket_r}")
        header = f"Matrix {self.rows}x{self.cols}:"
        return header + "\n" + "\n".join(lines)

    def __add__(self, other):
        # 矩陣加法,同時處理三種情況:
        # 1. 形狀完全一樣 -> 直接逐項相加
        # 2. other 只有一列(1, cols) -> broadcasting,把這一列複製到每一列再加
        # 3. other 只有一行(rows, 1) -> broadcasting,把這一行複製到每一行再加
        # 這就是 bias 為什麼能加到 batch 資料上的底層邏輯
        if isinstance(other, Matrix):
            # 先檢查形狀是否完全相同,是的話不需要 broadcasting
            if other.shape == self.shape:
                # 情況 1:同位置相加
                return Matrix([
                    [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
                    for i in range(self.rows)
                ])
            if other.rows == 1 and other.cols == self.cols:
                # 情況 2:other 永遠只用第 0 列(other.data[0][j]),等於每一列都加同一組 bias
                return Matrix([
                    [self.data[i][j] + other.data[0][j] for j in range(self.cols)]
                    for i in range(self.rows)
                ])
            if other.cols == 1 and other.rows == self.rows:
                # 情況 3:other 永遠只用第 0 行(other.data[i][0]),每一行都加同一欄數字
                return Matrix([
                    [self.data[i][j] + other.data[i][0] for j in range(self.cols)]
                    for i in range(self.rows)
                ])
        # 形狀對不上、也不能 broadcasting,就主動丟出錯誤(raise,對照C++的throw)
        raise ValueError(f"Cannot add shapes {self.shape} and {other.shape}")

    def __sub__(self, other):
        # 矩陣減法,同位置相減(這裡沒處理broadcasting,只有加法有處理)
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        # 純量乘法:每個元素都乘上同一個數字
        # 例:[[1,2],[3,4]] * 3 = [[3,6],[9,12]],shape 不變
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        # 逐項相乘(element-wise):同位置的數字互乘,形狀不變。
        # 跟下面的 matmul 是完全不同的運算,不要搞混
        # 例:[[1,2],[3,4]] 逐項乘 [[5,6],[7,8]] = [[5,12],[21,32]];matmul 則是 [[19,22],[43,50]]
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def transpose(self):
        # 轉置:把行跟列互換,(m,n) 變成 (n,m),原本[i][j]的數字換到[j][i]
        # 外層 for i 走過「原矩陣的行」(變成新矩陣的列),內層 for j 走過「原矩陣的列」
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    @property
    def T(self):
        # 讓 矩陣.T 就直接拿到轉置結果(不用打矩陣.transpose()),
        # @property 讓一個方法看起來像屬性一樣,呼叫時不用加括號
        return self.transpose()

    def determinant(self):
        # 行列式:只有正方形矩陣才有意義,代表這個矩陣把空間放大/縮小幾倍。
        # 1x1、2x2 有公式直接算;更大的矩陣用「餘因子展開」遞迴算
        # (拆成一堆更小的矩陣的行列式加總,是 determinant 呼叫自己,遞迴)
        if self.rows != self.cols:
            raise ValueError("Determinant only defined for square matrices")
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            # 2x2 公式:ad - bc,例 [[4,7],[2,6]] → 4*6 - 7*2 = 10
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        # 沿著第 0 列展開:對每個 j,挖掉第 0 列與第 j 行,剩下的小矩陣叫 minor(子矩陣)
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            # (-1)**j 讓正負號交替(+ - + - ...),乘上該位置的數字再乘子矩陣的行列式
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        # 2x2矩陣專用的逆矩陣公式:先算 determinant,是0就代表沒有逆矩陣(丟錯誤)。
        # abs(det) < 1e-10 是「浮點數幾乎等於0」的判斷方式,因為電腦算出來的0.0000001
        # 也該當成0處理,直接比較 det == 0 在浮點數運算裡不可靠
        # 公式:[[a,b],[c,d]] 的逆 = (1/det) × [[d,-b],[-c,a]]
        if self.shape != (2, 2):
            raise ValueError("This method only works for 2x2 matrices")
        det = self.determinant()
        if abs(det) < 1e-10:
            # det=0 代表矩陣把空間壓扁(資訊遺失),無法還原,所以沒有逆矩陣
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        # 單位矩陣:對角線是1、其餘是0,乘上任何東西都不改變它。
        # @staticmethod 代表這個方法不需要「先有一個矩陣物件」才能呼叫,
        # 直接用 Matrix.identity(3) 就能建出一個新矩陣,跟C++的static method概念一樣
        # 條件運算式 `1 if i == j else 0`:行列編號相同(對角線)放 1,否則放 0
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])

    @staticmethod
    def zeros(rows, cols):
        # 建一個指定形狀、內容全部是0的矩陣
        # [0] * cols 產生 cols 個 0 的 list;外層迴圈每次都建新 list(不能用 [[0]*cols]*rows,
        # 那樣每一列會是同一個物件,改一列全部跟著變)
        return Matrix([[0] * cols for _ in range(rows)])

    @staticmethod
    def random(rows, cols, low=-1.0, high=1.0):
        # 建一個指定形狀、內容是隨機小數的矩陣,常用來初始化神經網路的weights
        # random.uniform(low, high) 在 [low, high] 之間均勻抽一個小數;_ 代表不使用的迴圈變數
        return Matrix([
            [random.uniform(low, high) for _ in range(cols)]
            for _ in range(rows)
        ])


# === 🟡 ===
class Vector:
    """向量:一串數字。這份檔案的向量只用在最後的 demo_vectors,神經網路示範全部用 Matrix。"""

    def __init__(self, data):
        # data 是一串數字(list),size 記錄這個向量有幾維
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        # 讓 print(向量) 印出好看的樣子,不加這個會印出記憶體位置那種亂碼
        return f"Vector({self.data})"

    def __add__(self, other):
        # 對應位置相加(頭尾相接的幾何意義),zip 把兩邊同位置的數字配成一對
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        # 對應位置相減
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        # 純量乘法:每個元素都乘上同一個數字,只改變長度(方向不變,除非scalar是負的)
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        # 內積:對應位置相乘再全部加總,衡量兩個向量指向同一方向的程度
        # [3,4]·[1,2] = 3*1 + 4*2 = 11
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        # 長度:自己跟自己內積再開根號,畢氏定理的推廣
        # [3,4] → sqrt(9+16) = 5.0
        return sum(x ** 2 for x in self.data) ** 0.5

    def normalize(self):
        # 正規化:除以自己的長度,變成長度=1、方向不變的向量(這課沒手打的項目)
        # [3,4] / 5 = [0.6, 0.8]
        mag = self.magnitude()
        return Vector([x / mag for x in self.data])


# === 🟡 ===
def relu_matrix(m):
    # ReLU 激活函數:負數變0、正數維持原樣,逐元素進行。
    # 跟 numpy_version.py 裡手打的 np.maximum(0, x) 是同一件事,只是這裡是純Python版本
    # max(0, val):val 是負數時回傳 0,否則回傳 val 本身;巢狀 list comprehension 逐列逐格處理
    # 為什麼需要 ReLU:沒有它,多層矩陣相乘等價於一層矩陣,網路學不到非線性關係
    return Matrix([[max(0, val) for val in row] for row in m.data])


def demo_basic_operations():
    # 示範:矩陣加減法、純量乘法、element-wise乘法、matrix multiply、轉置
    # "=" * 60 是字串重複,產生 60 個等號當分隔線
    print("=" * 60)
    print("BASIC MATRIX OPERATIONS")
    print("=" * 60)

    # 建立兩個 2x2 矩陣,後面每個運算都用同一組 A、B,方便比較結果
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])

    print("\nA =")
    print(A)
    print("\nB =")
    print(B)

    print("\nA + B =")
    print(A + B)  # 預期 [[6,8],[10,12]]

    print("\nA - B =")
    print(A - B)  # 預期 [[-4,-4],[-4,-4]]

    print("\nA * 3 (scalar) =")
    print(A.scalar_multiply(3))  # 預期 [[3,6],[9,12]]

    print("\nA * B (element-wise) =")
    print(A.element_wise_multiply(B))  # 預期 [[5,12],[21,32]]

    print("\nA @ B (matrix multiply) =")
    print(A @ B)  # 預期 [[19,22],[43,50]],跟上面逐項乘的結果完全不同

    print("\nA^T =")
    print(A.T)  # 預期 [[1,3],[2,4]]


def demo_determinant_inverse():
    # 示範:算行列式、算逆矩陣、驗證 A @ A的逆矩陣 = identity matrix
    print("\n" + "=" * 60)
    print("DETERMINANT AND INVERSE")
    print("=" * 60)

    # det = 4*6 - 7*2 = 10,不是 0,所以有逆矩陣
    A = Matrix([[4, 7], [2, 6]])
    print("\nA =")
    print(A)
    print(f"\ndet(A) = {A.determinant()}")  # 4*6 - 7*2 = 10

    A_inv = A.inverse_2x2()
    print("\nA^-1 =")
    print(A_inv)

    # 矩陣乘上自己的逆矩陣,結果應該是單位矩陣(浮點數誤差內)
    print("\nA @ A^-1 (should be identity) =")
    print(A @ A_inv)

    I = Matrix.identity(3)
    print("\nIdentity 3x3 =")
    print(I)


def demo_broadcasting():
    # 示範 broadcasting:(2,3) 的矩陣加上 (1,3) 的 bias,bias那一列自動延伸複製兩次
    print("\n" + "=" * 60)
    print("BROADCASTING")
    print("=" * 60)

    # output 代表一層網路對 2 筆資料的輸出,每筆 3 個數字;bias 只有 1 列,需要 broadcasting
    output = Matrix([[1, 2, 3], [4, 5, 6]])
    bias = Matrix([[10, 20, 30]])  # shape (1,3):只有一列,加到 output 的每一列上

    print("\nOutput =")
    print(output)
    print("\nBias =")
    print(bias)
    print("\nOutput + Bias (broadcast) =")
    print(output + bias)  # 預期 [[11,22,33],[14,25,36]]


def demo_neural_network_layer():
    # 示範完整的兩層神經網路前向傳播(forward pass):
    # x -> (W1 @ x + b1) -> ReLU -> h1 -> (W2 @ h1 + b2) -> z2
    # 這就是 relu(W @ x + b) 這個模式疊兩層的樣子
    print("\n" + "=" * 60)
    print("NEURAL NETWORK FORWARD PASS")
    print("=" * 60)

    random.seed(42)  # 固定隨機種子,確保每次執行結果一樣,方便對照

    # 三層的大小決定權重矩陣形狀:W1 是 (hidden, input),W2 是 (output, hidden)
    input_size = 3    # 輸入 3 個特徵
    hidden_size = 4   # 隱藏層 4 個神經元
    output_size = 2   # 輸出 2 個數字

    # 每個向量都用「直立的 (n,1) 矩陣」表示,這樣才能跟權重矩陣做 @
    x = Matrix([[0.5], [0.8], [0.2]])                # shape (3,1)
    W1 = Matrix.random(hidden_size, input_size)      # shape (4,3):吃 3 維、吐 4 維
    b1 = Matrix([[0.0]] * hidden_size)               # shape (4,1):每個隱藏神經元一個 bias
    W2 = Matrix.random(output_size, hidden_size)     # shape (2,4):吃 4 維、吐 2 維
    b2 = Matrix([[0.0]] * output_size)               # shape (2,1)

    # 先印形狀,確認每個矩陣的 shape 跟上面推論的一致再進行運算
    print(f"\nInput x: {x.shape}")
    print(f"W1: {W1.shape}")
    print(f"W2: {W2.shape}")

    # 第一層:(4,3) @ (3,1) = (4,1),再加 (4,1) 的 bias,然後過 ReLU
    z1 = (W1 @ x) + b1
    h1 = relu_matrix(z1)
    print(f"\nHidden layer pre-activation z1: {z1.shape}")
    print(z1)
    print(f"\nHidden layer post-ReLU h1: {h1.shape}")
    print(h1)

    # 第二層:(2,4) @ (4,1) = (2,1),再加 bias;輸出層這裡沒有再過 ReLU
    z2 = (W2 @ h1) + b2
    print(f"\nOutput z2: {z2.shape}")
    print(z2)

    # 最後兩行印出各層 shape 的變化,追蹤 shape 是除錯神經網路最常用的方法
    print("\nThis is a complete 2-layer neural network forward pass.")
    print("Layer 1: (4x3) @ (3x1) + (4x1) -> (4x1) -> ReLU -> (4x1)")
    print("Layer 2: (2x4) @ (4x1) + (2x1) -> (2x1)")


def demo_vectors():
    # 示範:Vector class 的加減法、純量乘法、內積、長度、正規化
    print("\n" + "=" * 60)
    print("VECTOR OPERATIONS")
    print("=" * 60)

    # v 的長度剛好是 5(3-4-5 直角三角形),方便肉眼驗證
    v = Vector([3, 4])
    w = Vector([1, 2])

    print(f"\nv = {v}")
    print(f"w = {w}")
    print(f"v + w = {v + w}")        # [4, 6]
    print(f"v - w = {v - w}")        # [2, 2]
    print(f"v * 2 = {v * 2}")        # [6, 8]
    print(f"v . w = {v.dot(w)}")     # 11
    print(f"|v| = {v.magnitude()}")  # 5.0
    print(f"v normalized = {v.normalize()}")  # [0.6, 0.8]
    print(f"|v normalized| = {v.normalize().magnitude()}")  # 應該印出接近1.0


def demo_weight_matrix_intuition():
    # 示範:weight matrix 的每一列在做什麼——這裡刻意設計成
    # 第0列複製輸入特徵0、第1列複製輸入特徵1、第2列取兩者平均,
    # 用來直觀感受「一列 = 一種從輸入抽取資訊的方式」
    print("\n" + "=" * 60)
    print("WEIGHT MATRIX INTUITION")
    print("=" * 60)

    print("\nA weight matrix transforms input features into output features.")
    print("Each row extracts one pattern from the input.\n")

    # 矩陣乘向量時,輸出的每個數字 = 該列與輸入向量的內積,所以「列」就是一個偵測器
    # W 的每一列是一個偵測器,矩陣乘向量時每個偵測器各自跟輸入做內積
    W = Matrix([
        [1.0, 0.0, 0.0],   # 只看特徵 0
        [0.0, 1.0, 0.0],   # 只看特徵 1
        [0.5, 0.5, 0.0],   # 特徵 0 與 1 各取一半,等於平均
    ])
    x = Matrix([[0.8], [0.6], [0.1]])

    print("Weight matrix W (3 detectors, 3 inputs):")
    print(W)
    print("\nInput x:")
    print(x)
    print("\nW @ x =")
    result = W @ x  # 預期 [[0.8],[0.6],[0.7]]
    print(result)
    print("\nRow 0 of W = [1, 0, 0]: copies input feature 0")
    print("Row 1 of W = [0, 1, 0]: copies input feature 1")
    print("Row 2 of W = [0.5, 0.5, 0]: averages features 0 and 1")


if __name__ == "__main__":
    # 依序跑過每個demo,__main__ 判斷是C++裡沒有的東西:
    # 這個檔案被直接執行時這段才會跑,被別的檔案import進去時不會跑
    demo_vectors()
    demo_basic_operations()
    demo_determinant_inverse()
    demo_broadcasting()
    demo_weight_matrix_intuition()
    demo_neural_network_layer()
