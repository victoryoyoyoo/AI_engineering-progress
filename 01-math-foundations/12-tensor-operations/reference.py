"""
Lesson 12: Tensor Operations(張量運算)
Tensor就是維度不限的陣列。這支程式先用NumPy示範shape/reshape/transpose/broadcasting,
再看記憶體排列(strides)、歸約、einsum、常見AI tensor shape、multi-head attention的shape追蹤,
最後從零手寫一個有shape與strides的Tensor class,看底層怎麼運作。

檔案結構:
    🔴 shape、reshape、squeeze/unsqueeze、transpose、broadcasting
    🟡 記憶體排列與strides、歸約(sum/mean/max)、einsum、AI常見shape、attention shape追蹤
    🟢 從零實作的Tensor class(純Python list,不用NumPy的資料結構)
名詞:shape=每個軸的大小;axis=軸;stride=沿某軸走一步要在記憶體跳幾個元素;
      contiguous=元素在記憶體裡照邏輯順序排;broadcasting=自動把大小為1的軸拉大去對齊
"""
import numpy as np
from functools import reduce
from itertools import product as iterproduct


# 🔴 shape, reshape, transpose, broadcasting

def demo_reshape_numpy():
    """示範reshape(-1)、squeeze、unsqueeze(np.newaxis)、transpose、permute、flatten,全部用NumPy寫。"""
    print("=" * 60)
    print("RESHAPE / SQUEEZE / UNSQUEEZE / TRANSPOSE (NumPy)")
    print("=" * 60)

    # arange(12)產生0到11共12個元素的一維陣列,reshape(2, 6)切成2列6欄
    # reshape只換切法、不動資料,規則:新shape各軸相乘必須等於元素總數12
    data = np.arange(12).reshape(2, 6)
    print(f"Original: shape={data.shape}")

    # 同樣12個元素可以切成(3,4)、(2,2,3)、(12,)等任何乘積為12的shape
    print(f"reshape(3, 4):  {data.reshape(3, 4).shape}")
    # -1代表這個軸的大小交給NumPy算:總元素數 / 其他軸大小的乘積(12 / 3 = 4)
    print(f"reshape(-1, 3): {data.reshape(-1, 3).shape}")

    # shape (1,3,1,2):第0軸與第2軸大小是1,squeeze的目標就是把這種軸移除
    t = np.arange(6).reshape(1, 3, 1, 2)
    # squeeze()移除所有大小為1的軸:(1,3,1,2) -> (3,2);squeeze(0)只移除第0軸:(1,3,1,2) -> (3,1,2)
    print(f"squeeze():      {t.shape} -> {t.squeeze().shape}")
    print(f"squeeze(0):     {t.shape} -> {t.squeeze(0).shape}")

    # 一維向量shape (3,),沒有橫直之分;unsqueeze用來指定要多出一個大小為1的軸
    v = np.array([1, 2, 3])
    # np.newaxis(等於None)寫在方括號裡,會在該位置插入一個大小為1的軸,是NumPy版的unsqueeze
    print(f"unsqueeze(0):   {v.shape} -> {v[np.newaxis, :].shape}")
    print(f"unsqueeze(1):   {v.shape} -> {v[:, np.newaxis].shape}")

    # 2維的transpose:列變欄、欄變列,(2,3) -> (3,2);.T等同於反轉所有軸的順序
    mat = np.arange(6).reshape(2, 3)
    print(f"transpose:      {mat.shape} -> {mat.T.shape}")

    # transpose(0, 2, 3, 1):新的第0、1、2、3位,依序取舊的第0、2、3、1軸
    # shape (1,2,3,4) -> (1,3,4,2),這是NCHW轉NHWC的寫法
    t4d = np.arange(24).reshape(1, 2, 3, 4)
    print(f"permute (0,2,3,1): {t4d.shape} -> {t4d.transpose(0, 2, 3, 1).shape}")

    # 連續兩次reshape:先變成(2,4,4,2),再用reshape(2, -1)把第1軸以後全部攤平成(2,32)
    # 這就是把一批圖片從每張(4,4,2)攤平成每張一排32個數字
    flat = np.arange(2 * 4 * 4 * 2).reshape(2, 4, 4, 2).reshape(2, -1)
    print(f"flatten from axis 1: {flat.shape}")
    print()


def demo_broadcasting_numpy():
    """示範broadcasting:bias加到一批資料、逐通道縮放、外積、成對距離,以及規則檢查表。
        規則:靠右對齊 -> 左邊補1 -> 每個位置必須相等或有一邊是1 -> 大小為1的軸被拉大。
        拉大只是概念上的複製,NumPy實際上不會複製資料。"""
    print("=" * 60)
    print("BROADCASTING (NumPy)")
    print("=" * 60)

    print("\n--- Adding bias to batch ---")
    # 情境一:神經網路每層輸出 + bias
    # randn是標準常態分布隨機數;activations shape (4,3) = 4筆資料x3個特徵
    activations = np.random.randn(4, 3)
    # bias shape (3,):每個特徵一個偏移量
    bias = np.array([0.1, 0.2, 0.3])
    # (4,3)+(3,):bias左邊補1變(1,3),沿第0軸拉大成4列,每一筆資料都加同一組bias,結果(4,3)
    result = activations + bias
    print(f"activations: {activations.shape}")
    print(f"bias:        {bias.shape}")
    print(f"result:      {result.shape}")

    print("\n--- Channel-wise scaling ---")
    # 情境二:逐通道縮放
    # images shape (2,3,4,4) = (批次, 通道, 高, 寬)
    images = np.random.randn(2, 3, 4, 4)
    # 三個通道各有一個縮放倍率;reshape(1,3,1,1)把倍率放在第1軸(通道軸)
    # 其他軸都是1,才能沿批次、高、寬三個方向自動拉大
    scale = np.array([0.5, 1.0, 1.5]).reshape(1, 3, 1, 1)
    # (2,3,4,4)*(1,3,1,1):每個位置比較,3對3相等、其餘是1被拉大,結果(2,3,4,4)
    result = images * scale
    print(f"images: {images.shape}")
    print(f"scale:  {scale.shape}")
    print(f"result: {result.shape}")

    print("\n--- Outer product via broadcasting ---")
    # 情境三:外積(乘法表)
    # reshape(-1, 1):一維(3,)變成直排(3,1),-1由NumPy算出是3
    a = np.array([1, 2, 3]).reshape(-1, 1)
    # reshape(1, -1):一維(4,)變成橫排(1,4)
    b = np.array([10, 20, 30, 40]).reshape(1, -1)
    # (3,1)*(1,4):兩邊各有一個軸被拉大,結果(3,4),第(i,j)格 = a[i] * b[j]
    outer = a * b
    print(f"a: {a.shape}, b: {b.shape}")
    print(f"outer product: {outer.shape}")
    print(outer)

    print("\n--- Pairwise distances via broadcasting ---")
    # 情境四:5個點與3個點,兩兩之間的歐氏距離,不用寫雙層迴圈
    # 每個點是2維座標,points_a shape (5,2)、points_b shape (3,2)
    points_a = np.random.randn(5, 2)
    points_b = np.random.randn(3, 2)
    # points_a[:, np.newaxis, :]:(5,2) -> (5,1,2);points_b[np.newaxis, :, :]:(3,2) -> (1,3,2)
    # 相減時互相拉大成(5,3,2),diff[i,j]就是「第i個a點 - 第j個b點」的座標差
    diff = points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :]
    # 差平方後沿最後一軸(x、y座標)加總,再開根號,得到距離表 (5,3)
    # axis=-1代表最後一個軸,不需要記軸是第幾個
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))
    print(f"points_a: {points_a.shape}")
    print(f"points_b: {points_b.shape}")
    print(f"diff:     {diff.shape}")
    print(f"distances: {distances.shape}")

    print("\n--- Broadcasting rules check ---")
    # 一組shape配對,逐一測試能不能相加:
    #   (8,1,6,1)+(7,1,5) -> 補成(1,7,1,5),結果(8,7,6,5)
    #   (3,4)+(4,)        -> 補成(1,4),結果(3,4)
    #   (2,1,3)+(1,4,3)   -> 結果(2,4,3)
    #   (3,1)+(1,4)       -> 結果(3,4)
    shapes_to_test = [
        ((8, 1, 6, 1), (7, 1, 5)),
        ((3, 4), (4,)),
        ((2, 1, 3), (1, 4, 3)),
        ((3, 1), (1, 4)),
    ]
    # 每一項是一對shape的tuple,迴圈時同時解構成sa、sb
    for sa, sb in shapes_to_test:
        # zeros只是用來造出指定shape的陣列,數值不重要,重點是shape能不能對齊
        a = np.zeros(sa)
        b = np.zeros(sb)
        try:
            result = a + b
            print(f"  {sa} + {sb} -> {result.shape}")
        # 對不齊時NumPy丟出ValueError,訊息會說明哪兩個shape無法廣播
        except ValueError as e:
            print(f"  {sa} + {sb} -> ERROR: {e}")

    print()


# 🟡 memory layout, reductions, einsum, common AI shapes, attention shape trace

def demo_memory_layout():
    """示範記憶體排列:strides、連續性(contiguous)、轉置只對調strides、切片產生view。"""
    print("=" * 60)
    print("MEMORY LAYOUT")
    print("=" * 60)

    # 這個2x3陣列在記憶體裡是連續一排:1 2 3 4 5 6(row-major,C order)
    a = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"Array shape: {a.shape}")
    # a.strides以「位元組」為單位:int64每個元素8位元組,往下一列跳3個元素=24位元組,往右一欄跳1個=8位元組
    # 除以itemsize換算成「元素個數」:(3, 1)
    print(f"Strides (bytes): {a.strides}")
    print(f"Strides (elements): {tuple(s // a.itemsize for s in a.strides)}")
    # C-contiguous:照row-major順序連續排列;F-contiguous:照column-major順序連續排列
    print(f"C-contiguous: {a.flags['C_CONTIGUOUS']}")
    print(f"F-contiguous: {a.flags['F_CONTIGUOUS']}")
    print(f"Memory layout: {a.ravel()}")

    print("\n--- After transpose ---")
    # 轉置不搬資料,只把shape與strides順序對調:shape (2,3)->(3,2)、strides (24,8)->(8,24)
    # 元素變成不再按邏輯順序連續排列,所以不是C-contiguous
    b = a.T
    print(f"Transposed shape: {b.shape}")
    print(f"Strides (bytes): {b.strides}")
    print(f"C-contiguous: {b.flags['C_CONTIGUOUS']}")
    print(f"F-contiguous: {b.flags['F_CONTIGUOUS']}")
    print(f"Note: transpose swapped strides without moving data")

    print("\n--- Contiguous copy ---")
    # ascontiguousarray真的重新排一份資料,讓轉置後的陣列在記憶體裡也是連續的
    c = np.ascontiguousarray(b)
    print(f"After ascontiguousarray:")
    print(f"  C-contiguous: {c.flags['C_CONTIGUOUS']}")
    print(f"  Strides: {c.strides}")

    print("\n--- Row-major vs Column-major ---")
    # order="C"是列優先(預設),order="F"是欄優先(Fortran/MATLAB的排法)
    # 同樣的邏輯內容,在記憶體裡的排列順序不同
    row_major = np.array([[1, 2, 3], [4, 5, 6]], order='C')
    col_major = np.array([[1, 2, 3], [4, 5, 6]], order='F')
    print(f"Row-major (C) flat: {row_major.ravel(order='K')}")
    print(f"Col-major (F) flat: {col_major.ravel(order='K')}")
    print(f"Row-major strides: {row_major.strides}")
    print(f"Col-major strides: {col_major.strides}")

    print("\n--- Stride tricks: creating a view ---")
    # 切片也是不複製資料的view,只是換了起點與strides
    x = np.arange(12).reshape(3, 4)
    print(f"Original:\n{x}")
    print(f"Strides: {x.strides}")
    # x[:, ::2]:所有列,每隔一欄取一個(第0、2欄)。::2是「每隔2個取1個」的切片寫法
    # 結果shape (3,2),欄方向的stride變成原本的2倍,因此不連續
    sliced = x[:, ::2]
    print(f"Every other column (x[:, ::2]):\n{sliced}")
    print(f"Sliced strides: {sliced.strides}")
    print(f"Sliced is contiguous: {sliced.flags['C_CONTIGUOUS']}")
    print()


def demo_reduction_operations():
    """示範歸約(reduction):sum、mean、max、argmax沿指定的軸把該軸「收掉」,shape少一個軸。"""
    print("=" * 60)
    print("REDUCTION OPERATIONS")
    print("=" * 60)

    # 3個軸的tensor (2,3,4);axis=k代表沿第k軸做運算,該軸消失
    x = np.random.randn(2, 3, 4)
    print(f"Input shape: {x.shape}")

    # sum()沒有指定axis:所有元素加成一個純量
    # sum(axis=0):(2,3,4) -> (3,4);axis=1 -> (2,4);axis=2 -> (2,3)
    # axis=(1,2):同時收掉兩個軸 -> (2,)
    print(f"\n  sum():           {x.sum().shape if hasattr(x.sum(), 'shape') else 'scalar'}")
    print(f"  sum(axis=0):     {x.sum(axis=0).shape}")
    print(f"  sum(axis=1):     {x.sum(axis=1).shape}")
    print(f"  sum(axis=2):     {x.sum(axis=2).shape}")
    print(f"  sum(axis=(1,2)): {x.sum(axis=(1,2)).shape}")

    # max(axis=-1)取每一列最大值;argmax(axis=-1)回傳最大值所在的位置編號,shape同樣少最後一軸
    print(f"\n  mean(axis=0):    {x.mean(axis=0).shape}")
    print(f"  max(axis=-1):    {x.max(axis=-1).shape}")
    print(f"  argmax(axis=-1): {x.argmax(axis=-1).shape}")

    print("\n--- Global Average Pooling (vision) ---")
    # Global Average Pooling(CNN最後一層常用):特徵圖 (批次2, 通道64, 高7, 寬7)
    feature_map = np.random.randn(2, 64, 7, 7)
    # 沿高、寬兩個軸取平均,每個通道壓成一個數字:(2,64,7,7) -> (2,64)
    pooled = feature_map.mean(axis=(2, 3))
    print(f"  Feature map: {feature_map.shape}")
    print(f"  After GAP:   {pooled.shape}")

    print("\n--- Sequence mean pooling (NLP) ---")
    # 序列平均池化(NLP):4句、每句128個token、每個token 768維
    hidden_states = np.random.randn(4, 128, 768)
    # mask標記哪些token是真的(1)、哪些是補齊用的padding(0);最後一軸設成1,方便廣播成768維
    mask = np.ones((4, 128, 1))
    # 切片賦值:每句從第100個token之後都當作padding,設成0
    mask[:, 100:, :] = 0
    # 先用mask把padding位置歸零,沿token軸(axis=1)加總,再除以真正的token數
    # (4,128,768)*(4,128,1) -> 廣播 -> 加總 (4,768);mask.sum(axis=1)是(4,1),除法時再廣播
    pooled = (hidden_states * mask).sum(axis=1) / mask.sum(axis=1)
    print(f"  Hidden states: {hidden_states.shape}")
    print(f"  Mask:          {mask.shape}")
    print(f"  Pooled:        {pooled.shape}")

    print()


def demo_einsum():
    """示範einsum基本用法。規則:每個軸用一個字母命名,箭頭左邊有、右邊沒有的字母,該軸相乘後加總;
        右邊有的字母保留。每個例子都跟NumPy內建函式的結果對照。"""
    print("=" * 60)
    print("EINSUM NOTATION")
    print("=" * 60)

    print("\n--- Dot product: i,i-> ---")
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    # "i,i->":i在右邊沒出現,沿i相乘再加總,等於內積 1*4+2*5+3*6 = 32
    result = np.einsum("i,i->", a, b)
    verify = np.dot(a, b)
    print(f"  einsum: {result}, np.dot: {verify}")

    print("\n--- Outer product: i,j->ij ---")
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0])
    # "i,j->ij":i、j都保留,沒有軸被加總,等於外積,結果shape (3,2)
    result = np.einsum("i,j->ij", a, b)
    verify = np.outer(a, b)
    print(f"  einsum:\n{result}")
    print(f"  np.outer:\n{verify}")

    print("\n--- Matrix multiply: ik,kj->ij ---")
    A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    B = np.array([[7, 8, 9], [10, 11, 12]], dtype=float)
    # "ik,kj->ij":k兩邊同名且不在輸出,沿k相乘加總,等於矩陣乘法 (3,2)@(2,3) = (3,3)
    result = np.einsum("ik,kj->ij", A, B)
    verify = A @ B
    print(f"  A: {A.shape}, B: {B.shape}")
    print(f"  einsum result:\n{result}")
    print(f"  matmul result:\n{verify}")

    print("\n--- Trace: ii-> ---")
    M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    # "ii->":同一個字母出現兩次代表只取對角線,輸出沒有軸則全部加總 = trace(主對角線總和)
    result = np.einsum("ii->", M)
    verify = np.trace(M)
    print(f"  einsum: {result}, np.trace: {verify}")

    print("\n--- Transpose: ij->ji ---")
    # "ij->ji":沒有字母消失,只是i、j換位置 = 轉置
    result = np.einsum("ij->ji", A)
    verify = A.T
    print(f"  einsum:\n{result}")

    print("\n--- Diagonal: ii->i ---")
    # "ii->i":取對角線但保留成一個軸,得到對角線上的向量
    result = np.einsum("ii->i", M)
    verify = np.diag(M)
    print(f"  einsum: {result}, np.diag: {verify}")

    print("\n--- Sum over axis: ij->i (row sums) ---")
    # "ij->i":j不在輸出,沿j加總,得到每一列的總和
    result = np.einsum("ij->i", A)
    verify = A.sum(axis=1)
    print(f"  einsum: {result}, sum(axis=1): {verify}")

    print("\n--- Batch matrix multiply: bij,bjk->bik ---")
    batch_A = np.random.randn(4, 3, 5)
    batch_B = np.random.randn(4, 5, 2)
    # "bij,bjk->bik":b是批次軸(保留),沿j相乘加總,等於批次矩陣乘法
    # (4,3,5)與(4,5,2) -> (4,3,2)
    result = np.einsum("bij,bjk->bik", batch_A, batch_B)
    verify = np.matmul(batch_A, batch_B)
    print(f"  batch_A: {batch_A.shape}, batch_B: {batch_B.shape}")
    print(f"  einsum result: {result.shape}")
    print(f"  matmul result: {verify.shape}")
    # allclose用容許誤差比較兩個浮點陣列,不用==(浮點運算順序不同會有微小差異)
    print(f"  match: {np.allclose(result, verify)}")

    print("\n--- Hadamard (element-wise) product: ij,ij->ij ---")
    C = np.array([[1, 2], [3, 4]], dtype=float)
    D = np.array([[5, 6], [7, 8]], dtype=float)
    # "ij,ij->ij":三邊字母都一樣,沒有加總,等於逐元素相乘(Hadamard product)
    result = np.einsum("ij,ij->ij", C, D)
    verify = C * D
    print(f"  einsum:\n{result}")
    print(f"  element-wise:\n{verify}")

    print()


def demo_einsum_gallery():
    """用一張表列出常見einsum寫法,並印出每一種的輸入shape與輸出shape。"""
    print("=" * 60)
    print("EINSUM GALLERY: ALL COMMON PATTERNS")
    print("=" * 60)

    # 每一項:(名稱, einsum字串, 第一個tensor的shape, 第二個tensor的shape)
    two_operand_ops = [
        ("Vector dot product",      "i,i->",     (4,),       (4,)),
        ("Outer product",           "i,j->ij",   (3,),       (4,)),
        ("Matrix-vector product",   "ij,j->i",   (3, 4),     (4,)),
        ("Matrix multiply",         "ij,jk->ik", (3, 4),     (4, 5)),
        ("Batch matmul",            "bij,bjk->bik", (2, 3, 4), (2, 4, 5)),
        ("Batch outer product",     "bi,bj->bij", (2, 3),    (2, 4)),
        ("Frobenius norm squared",  "ij,ij->",   (3, 4),     (3, 4)),
        ("Tensor contraction",      "ijk,jkl->il", (2, 3, 4), (3, 4, 5)),
    ]

    # 只有一個輸入的運算:(名稱, einsum字串, shape)
    single_operand_ops = [
        ("Trace",                   "ii->",      (4, 4)),
        ("Diagonal",                "ii->i",     (4, 4)),
        ("Row sum",                 "ij->i",     (3, 4)),
        ("Column sum",              "ij->j",     (3, 4)),
        ("Transpose",               "ij->ji",    (3, 4)),
    ]

    # 固定亂數種子,結果可重現
    np.random.seed(0)
    for name, subscripts, shape_a, shape_b in two_operand_ops:
        # *shape_a把tuple拆開當成多個參數傳入:randn(*(3,4)) 等於 randn(3, 4)
        a = np.random.randn(*shape_a)
        b = np.random.randn(*shape_b)
        # subscripts是變數存放的einsum字串,例如"ik,kj->ij"
        result = np.einsum(subscripts, a, b)
        # 結果若是純量(例如內積)沒有有意義的shape,就顯示"scalar";hasattr檢查物件有沒有shape屬性
        result_shape = result.shape if hasattr(result, 'shape') and result.shape else 'scalar'
        print(f"  {name:30s}  {subscripts:15s}  "
              f"{shape_a} x {shape_b} -> {result_shape}")

    for name, subscripts, shape_a in single_operand_ops:
        a = np.random.randn(*shape_a)
        result = np.einsum(subscripts, a)
        result_shape = result.shape if hasattr(result, 'shape') and result.shape else 'scalar'
        print(f"  {name:30s}  {subscripts:15s}  "
              f"{shape_a} -> {result_shape}")

    print()
    print("--- Bilinear form (3-operand einsum): i,ij,j-> ---")
    # 三個運算元的einsum:x^T W y = sum_i sum_j x[i]*W[i][j]*y[j],結果是一個純量(雙線性形式)
    x = np.array([1.0, 2.0, 3.0])
    W = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    y = np.array([1.0, 1.0, 1.0])
    result = np.einsum("i,ij,j->", x, W, y)
    # 用矩陣乘法連乘算同一件事,對照einsum結果
    manual = x @ W @ y
    print(f"  x: {x.shape}, W: {W.shape}, y: {y.shape}")
    print(f"  x^T W y = einsum: {result}, manual: {manual}")

    print()


def demo_ai_tensor_shapes():
    """列出深度學習裡常見的tensor shape與佔用記憶體,以及NCHW/NHWC互轉、切頭與合併頭的reshape。"""
    print("=" * 60)
    print("COMMON AI TENSOR SHAPES")
    print("=" * 60)

    print("\n--- Vision: (B, C, H, W) ---")
    # 視覺:一批圖片 (批次B=32, 通道C=3(RGB), 高H=224, 寬W=224)
    B, C, H, W = 32, 3, 224, 224
    # astype(np.float32):轉成32位元浮點數,每個元素4位元組(NumPy預設是64位元、8位元組)
    images = np.random.randn(B, C, H, W).astype(np.float32)
    print(f"Image batch: {images.shape}")
    # .size是元素總數;格式化的:,加千分位逗號;.nbytes是總位元組數,除以1024*1024換算成MB
    print(f"  Total elements: {images.size:,}")
    print(f"  Memory (float32): {images.nbytes / 1024 / 1024:.1f} MB")

    # 卷積核心shape (輸出通道64, 輸入通道3, 核心高3, 核心寬3)
    kernel = np.random.randn(64, 3, 3, 3).astype(np.float32)
    print(f"Conv2D kernel (64 filters, 3x3): {kernel.shape}")

    print("\n--- NLP: (B, T, D) ---")
    # 語言:一批句子的token向量 (批次16, 序列長度T=512, 每個token的維度D=768)
    B, T, D = 16, 512, 768
    embeddings = np.random.randn(B, T, D).astype(np.float32)
    print(f"Token embeddings: {embeddings.shape}")
    print(f"  Total elements: {embeddings.size:,}")
    print(f"  Memory (float32): {embeddings.nbytes / 1024 / 1024:.1f} MB")

    # embedding表 (詞彙數, 維度):查表時用token編號取出對應的一列;50257是GPT-2的詞彙數
    vocab_size = 50257
    embed_table = np.random.randn(vocab_size, D).astype(np.float32)
    print(f"Embedding table (GPT-2): {embed_table.shape}")
    print(f"  Memory: {embed_table.nbytes / 1024 / 1024:.1f} MB")

    print("\n--- Attention: (B, H, T, D_head) ---")
    # attention:12個頭,每個頭的維度 D_head = 768 // 12 = 64
    H = 12
    D_head = D // H
    # Query shape (批次, 頭數, 序列長度, 每頭維度)
    Q = np.random.randn(B, H, T, D_head).astype(np.float32)
    print(f"Query tensor: {Q.shape}")
    print(f"  Head dim: {D_head}")
    # 注意力分數是「每個位置對每個位置」,所以有兩個T軸;序列長度T增加,記憶體以T的平方成長
    attn_scores = np.random.randn(B, H, T, T).astype(np.float32)
    print(f"Attention scores: {attn_scores.shape}")
    print(f"  Memory: {attn_scores.nbytes / 1024 / 1024:.1f} MB")

    print("\n--- Weight shapes ---")
    # 各種層的權重shape;Linear的shape是 (輸出維度, 輸入維度)
    shapes = {
        "Linear (768 -> 3072)": (3072, 768),
        "Linear (3072 -> 768)": (768, 3072),
        "Conv2D (3->64, 7x7)": (64, 3, 7, 7),
        "Conv2D (64->128, 3x3)": (128, 64, 3, 3),
        "LayerNorm (768)": (768,),
        "Embedding (50257, 768)": (50257, 768),
        "Positional (1024, 768)": (1024, 768),
    }
    for name, shape in shapes.items():
        # reduce(f, 序列, 初始值):把序列從左到右用f合併成一個值;這裡是把shape各軸連乘 = 參數量
        params = reduce(lambda a, b: a * b, shape, 1)
        print(f"  {name}: {shape} -> {params:,} params")

    print("\n--- Layout conversion: NCHW <-> NHWC ---")
    # NCHW(通道在前,PyTorch)與NHWC(通道在後,TensorFlow)互轉
    nchw = np.random.randn(2, 3, 4, 4)
    # np.transpose(x, 順序):新的第0、1、2、3位取舊的第0、2、3、1軸;(2,3,4,4) -> (2,4,4,3)
    nhwc = np.transpose(nchw, (0, 2, 3, 1))
    # 反方向:新順序取舊的第0、3、1、2軸,回到(2,3,4,4);兩個順序互為反向
    back = np.transpose(nhwc, (0, 3, 1, 2))
    print(f"NCHW: {nchw.shape}")
    print(f"NHWC: {nhwc.shape}")
    print(f"Back to NCHW: {back.shape}")
    print(f"Round-trip match: {np.allclose(nchw, back)}")

    print("\n--- Reshaping for multi-head attention ---")
    B, T, D = 4, 128, 768
    H = 12
    D_head = D // H
    # multi-head attention切頭:(批次, 序列, 維度)整理成(批次, 頭, 序列, 每頭維度)
    x = np.random.randn(B, T, D)
    print(f"Input: {x.shape}")

    # 第一步reshape:把最後一軸768拆成12個頭x每頭64,(4,128,768) -> (4,128,12,64)
    step1 = x.reshape(B, T, H, D_head)
    print(f"After reshape to (B,T,H,D_head): {step1.shape}")

    # 第二步transpose:把「頭」的軸搬到第1位,(4,128,12,64) -> (4,12,128,64),之後每個頭獨立運算
    step2 = step1.transpose(0, 2, 1, 3)
    print(f"After transpose to (B,H,T,D_head): {step2.shape}")

    # 合併頭:先transpose換回去,再reshape把頭與每頭維度接回768;round-trip驗證資料沒有被弄亂
    step3 = step2.transpose(0, 2, 1, 3).reshape(B, T, D)
    print(f"Merge heads back: {step3.shape}")
    print(f"Round-trip match: {np.allclose(x, step3)}")

    print()


def demo_attention_einsum():
    """用einsum走一遍multi-head self-attention,印出每一步的shape。這裡只追蹤shape,attention的數學意義在Phase 7。"""
    print("=" * 60)
    print("ATTENTION MECHANISM via EINSUM")
    print("=" * 60)

    # B批次、H頭數、T序列長度、D每個頭的維度
    B, H, T, D = 2, 4, 8, 16
    # E=embedding維度=頭數x每頭維度=64
    E = H * D

    # 固定亂數種子,輸出可重現
    np.random.seed(42)

    # 輸入X (2,8,64):2句話、每句8個token、每個token 64維
    X = np.random.randn(B, T, E)
    print(f"Input X: {X.shape}  (batch, seq_len, embed_dim)")

    # 三個投影權重矩陣 (64,64);乘0.02讓初始值很小,模擬常見的權重初始化
    W_q = np.random.randn(E, E) * 0.02
    W_k = np.random.randn(E, E) * 0.02
    W_v = np.random.randn(E, E) * 0.02

    # "bte,ek->btk":e被加總,等於每個token向量乘上權重矩陣;(2,8,64)@(64,64) -> (2,8,64)
    Q = np.einsum("bte,ek->btk", X, W_q)
    K = np.einsum("bte,ek->btk", X, W_k)
    V = np.einsum("bte,ek->btk", X, W_v)
    print(f"Q, K, V: {Q.shape}")

    # 切頭:reshape把最後一軸64拆成(4,16),得到(2,8,4,16);transpose(0,2,1,3)把頭移到第1位,得到(2,4,8,16)
    Q = Q.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    K = K.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    V = V.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    print(f"After split heads: Q={Q.shape}, K={K.shape}, V={V.shape}")

    # "bhtd,bhsd->bhts":b、h同名保留,d被加總,t、s保留
    # = 每個頭裡,每個查詢位置t與每個鍵位置s做內積,得到 (2,4,8,8) 的分數表
    # 除以sqrt(D)避免內積數值隨維度變大而過大,讓softmax保持穩定
    scores = np.einsum("bhtd,bhsd->bhts", Q, K) / np.sqrt(D)
    print(f"Attention scores: {scores.shape}")

    # 內部定義的softmax,沿指定軸把分數轉成機率(總和為1)
    def softmax(x, axis=-1):
        # 先減去該軸的最大值再exp,避免exp溢位(數值穩定性,Lesson 13細講)
        # keepdims=True保留被收掉的軸(大小變1),才能跟x廣播相減
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)

    # 沿最後一軸(被查詢的位置s)做softmax:每個查詢位置對所有位置的權重總和為1
    weights = softmax(scores, axis=-1)
    print(f"Attention weights: {weights.shape}")
    print(f"  weights sum per query (should be 1.0): {weights[0, 0, 0].sum():.6f}")

    # "bhts,bhsd->bhtd":沿s相乘加總 = 用注意力權重對V做加權平均,回到(2,4,8,16)
    attn_output = np.einsum("bhts,bhsd->bhtd", weights, V)
    print(f"Attention output: {attn_output.shape}")

    # 合併頭:transpose把頭移回第2位得到(2,8,4,16),reshape把(4,16)接成64,得到(2,8,64)
    concat = attn_output.transpose(0, 2, 1, 3).reshape(B, T, E)
    print(f"Concatenated heads: {concat.shape}")

    # 輸出投影權重;最後輸出shape與輸入相同 (2,8,64),可以接到下一層
    W_o = np.random.randn(E, E) * 0.02
    output = np.einsum("bte,ek->btk", concat, W_o)
    print(f"Final output: {output.shape}")
    print()


# 🟢 Tensor class from scratch (strides, reshape, permute, sum, element-wise)

class Tensor:
    """從零實作的Tensor:用一個攤平的Python list存資料,加上shape與strides兩個描述資料的metadata。
        多維索引透過strides換算成攤平list的位置,reshape只需要改shape與strides,不必搬動資料。"""
    def __init__(self, data, shape=None):
        """接受巢狀list/tuple、NumPy陣列或單一數字;shape參數可另外指定切法(元素總數必須吻合)。"""
        # 巢狀list:遞迴攤平成一維,並同時推算出shape
        if isinstance(data, (list, tuple)):
            self._data, self._shape = self._flatten_nested(data)
        # NumPy陣列:flatten()攤平成一維,tolist()轉成Python list,shape直接取自陣列
        elif isinstance(data, np.ndarray):
            self._data = data.flatten().tolist()
            self._shape = tuple(data.shape)
        # 單一數字:當成0維的純量,shape是空tuple ()
        else:
            self._data = [data]
            self._shape = ()

        # 呼叫端指定shape時,元素總數必須相等,否則無法切成該shape
        # reduce(f, 序列, 1):把shape各軸連乘,初始值1
        if shape is not None:
            total = reduce(lambda a, b: a * b, shape, 1)
            if total != len(self._data):
                raise ValueError(
                    f"Cannot reshape {len(self._data)} elements into shape {shape}"
                )
            self._shape = tuple(shape)

        # 最後由shape算出strides
        self._strides = self._compute_strides(self._shape)

    def _flatten_nested(self, data):
        """遞迴把巢狀list攤平,回傳 (攤平後的list, shape)。同一層每個子項的shape必須一致。"""
        # 遞迴終止條件:遇到單一數字,回傳只有它自己的list,shape為()
        if not isinstance(data, (list, tuple)):
            return [data], ()
        # 空list:0個元素,shape (0,)
        if len(data) == 0:
            return [], (0,)

        # 對每個子項遞迴,得到(子項的攤平資料, 子項的shape)
        sub_results = [self._flatten_nested(item) for item in data]
        # 以第一個子項的shape為標準,其餘子項必須相同,否則不是規則的矩形陣列
        sub_shape = sub_results[0][1]
        # enumerate給出(編號, 內容);(_, s)解構出子項的shape,底線_表示不使用資料部分
        for i, (_, s) in enumerate(sub_results):
            if s != sub_shape:
                raise ValueError(
                    f"Inconsistent shapes at index {i}: {s} vs {sub_shape}"
                )

        # 把所有子項的攤平資料依序接成一個list;extend會把list內的元素逐一加入
        flat = []
        for sub_data, _ in sub_results:
            flat.extend(sub_data)

        # 目前這一層的shape = (這層的長度,) + 子項的shape
        # 例:[[1,2,3],[4,5,6]] -> 這層長度2,子項shape (3,),結果(2,3);(len(data),)的逗號讓它是tuple
        return flat, (len(data),) + sub_shape

    @staticmethod
    def _compute_strides(shape):
        """由shape算strides(row-major):最後一軸的stride是1,往前每一軸的stride = 後一軸stride x 後一軸大小。
            例:shape (2,3,4) -> strides (12,4,1)。"""
        # 純量沒有軸,strides是空tuple
        if len(shape) == 0:
            return ()
        # 先全部填1,最後一軸的stride固定是1
        strides = [1] * len(shape)
        # range(倒數第二軸, -1, -1):從倒數第二軸一路往前算到第0軸,-1是每次減1
        for i in range(len(shape) - 2, -1, -1):
            # 沿第i軸走一步,要跳過後面所有軸的元素總數
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)

    @property
    def shape(self):
        """shape(每個軸的大小)。@property讓它用t.shape取值,不需要加括號。"""
        return self._shape

    @property
    def rank(self):
        """rank(軸的數量) = shape的長度;跟線性代數的「矩陣的秩」是不同的概念。"""
        return len(self._shape)

    @property
    def size(self):
        """元素總數。"""
        return len(self._data)

    @property
    def strides(self):
        """strides(沿每個軸走一步要跳的元素數)。"""
        return self._strides

    def _flat_index(self, indices):
        """把多維索引轉成攤平list的位置:sum(索引 x stride)。例:shape (2,3)、索引(1,2) -> 1*3 + 2*1 = 5。"""
        # 索引個數必須等於軸的數量
        if len(indices) != len(self._shape):
            raise IndexError(
                f"Expected {len(self._shape)} indices, got {len(indices)}"
            )
        # 累加器,最後就是攤平list裡的位置
        idx = 0
        # zip把「每個軸的索引」與「該軸的stride」配對,enumerate再加上軸編號i
        for i, (ind, stride) in enumerate(zip(indices, self._strides)):
            # 範圍檢查:索引必須在 [0, 該軸大小) 之內
            if ind < 0 or ind >= self._shape[i]:
                raise IndexError(
                    f"Index {ind} out of range for axis {i} with size {self._shape[i]}"
                )
            # 沿該軸走ind步,每步跳stride個元素
            idx += ind * stride
        return idx

    def __getitem__(self, indices):
        """t[i, j, k]取值。方括號裡的多個索引會自動打包成tuple,只單一數字時補成長度1的tuple。這個簡化版必須給滿所有軸的索引。"""
        # t[3]傳進來的是整數3,不是tuple,先包成(3,)統一處理
        if not isinstance(indices, tuple):
            indices = (indices,)
        # 索引個數等於軸數,才能定位到單一元素
        if len(indices) == len(self._shape):
            return self._data[self._flat_index(indices)]
        raise IndexError("Partial indexing not supported in this basic implementation")

    def __setitem__(self, indices, value):
        """t[i, j, k] = value 賦值,同樣先換算成攤平list的位置。"""
        if not isinstance(indices, tuple):
            indices = (indices,)
        self._data[self._flat_index(indices)] = value

    def reshape(self, new_shape):
        """換shape,元素順序與資料都不變。new_shape裡最多可有一個-1,由元素總數自動推算。"""
        # 轉成list才能修改其中的-1
        new_shape = list(new_shape)
        # neg_idx記錄-1出現的位置(-1代表還沒出現);known_product累計其他軸的乘積
        neg_idx = -1
        known_product = 1
        for i, s in enumerate(new_shape):
            # 遇到-1:記下位置;第二次遇到代表有兩個-1,無法推算,報錯
            if s == -1:
                if neg_idx != -1:
                    raise ValueError("Only one dimension can be -1")
                neg_idx = i
            else:
                known_product *= s

        if neg_idx != -1:
            # //是整數除法:-1那個軸的大小 = 元素總數 / 其他軸乘積
            new_shape[neg_idx] = self.size // known_product

        # 驗證新shape的乘積等於元素總數
        total = reduce(lambda a, b: a * b, new_shape, 1)
        if total != self.size:
            raise ValueError(
                f"Cannot reshape {self.size} elements into shape {tuple(new_shape)}"
            )

        # __new__只建立空物件、不執行__init__,再手動填入欄位,避免__init__重新推算shape
        result = Tensor.__new__(Tensor)
        # [:]複製一份list;新shape要配新的strides
        result._data = self._data[:]
        result._shape = tuple(new_shape)
        result._strides = self._compute_strides(result._shape)
        return result

    def squeeze(self, dim=None):
        """移除大小為1的軸;dim有指定時只移除那一軸(該軸大小不是1則什麼都不做)。"""
        # 指定軸:大小為1才移除,pop(dim)把該軸從shape的list拿掉
        if dim is not None:
            if self._shape[dim] != 1:
                return self.reshape(self._shape)
            new_shape = list(self._shape)
            new_shape.pop(dim)
            return self.reshape(tuple(new_shape) if new_shape else ())
        # 沒指定軸:保留所有大小不是1的軸
        new_shape = tuple(s for s in self._shape if s != 1)
        if not new_shape:
            new_shape = ()
        return self.reshape(new_shape)

    def unsqueeze(self, dim):
        """在dim位置插入一個大小為1的軸。dim可以是負數,-1代表插在最後面。"""
        # 負數索引換算:插入後總軸數是原本+1,所以 -1 對應 len+1-1 = 最後一個位置
        if dim < 0:
            dim = len(self._shape) + 1 + dim
        new_shape = list(self._shape)
        # list.insert(位置, 值):在該位置插入1
        new_shape.insert(dim, 1)
        return self.reshape(tuple(new_shape))

    def transpose(self, dim0, dim1):
        """交換兩個軸:建立一個「只有這兩個位置對調」的順序,交給permute處理。"""
        # 先是恆等順序[0,1,2,...],再把dim0與dim1兩個位置互換
        perm = list(range(self.rank))
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return self.permute(perm)

    def permute(self, dims):
        """按dims重排所有軸:新的第k軸 = 舊的dims[k]軸。這個簡化版會真的重新排列資料(真實框架只改strides)。"""
        # dims必須是0到rank-1每個數字恰好出現一次的排列
        if sorted(dims) != list(range(self.rank)):
            raise ValueError(f"Invalid permutation {dims} for rank {self.rank}")

        # 新shape:按dims的順序挑選舊shape的各軸大小
        new_shape = tuple(self._shape[d] for d in dims)
        result = Tensor.__new__(Tensor)
        result._shape = new_shape
        result._strides = self._compute_strides(new_shape)
        # 先建立同樣長度的空位,之後逐一填入
        result._data = [0] * self.size

        old_strides = self._strides
        # iterproduct(itertools.product):產生所有多維索引組合,等於多層巢狀迴圈走過每一個元素
        # 例:shape (2,3) -> (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
        # 星號*把generator裡的多個range拆成多個參數
        for old_indices in iterproduct(*(range(s) for s in self._shape)):
            # 同一個元素在新tensor裡的索引:按dims重排舊索引
            new_indices = tuple(old_indices[d] for d in dims)
            # 舊位置 = 舊索引與舊strides逐項相乘再加總
            old_flat = sum(i * s for i, s in zip(old_indices, old_strides))
            # 新位置 = 新索引與新strides逐項相乘再加總
            new_flat = sum(
                i * s for i, s in zip(new_indices, result._strides)
            )
            # 把資料搬到新位置
            result._data[new_flat] = self._data[old_flat]

        return result

    def flatten(self, start_dim=0, end_dim=-1):
        """把start_dim到end_dim之間的軸合併成一個軸(預設整個攤平成一維)。"""
        # 負數的軸編號換算成正數
        if end_dim < 0:
            end_dim = self.rank + end_dim
        # 新shape = start_dim之前的軸 + 被合併軸的大小乘積 + end_dim之後的軸
        new_shape = (
            list(self._shape[:start_dim])
            + [reduce(lambda a, b: a * b, self._shape[start_dim:end_dim + 1], 1)]
            + list(self._shape[end_dim + 1:])
        )
        return self.reshape(tuple(new_shape))

    def _elementwise_op(self, other, op):
        """逐元素運算的共用函式:op是兩個數的運算(例如加、乘),對應位置逐一計算,shape不變。"""
        # 與純量運算:每個元素都跟同一個數字做運算
        if isinstance(other, (int, float)):
            result_data = [op(x, other) for x in self._data]
            return Tensor(result_data, shape=self._shape)
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported type {type(other)}")
        # 兩個tensor運算時shape必須完全相同(這個簡化版沒有實作broadcasting)
        if self._shape != other._shape:
            raise ValueError(
                f"Shape mismatch: {self._shape} vs {other._shape}. "
                "Use broadcast() first."
            )
        # 因為shape相同,攤平後的兩個list位置一一對應,直接zip逐對運算
        result_data = [op(a, b) for a, b in zip(self._data, other._data)]
        return Tensor(result_data, shape=self._shape)

    def __add__(self, other):
        """定義 + 運算子:t1 + t2 或 t + 數字。lambda是沒有名字的小函式,這裡代表「兩數相加」。"""
        return self._elementwise_op(other, lambda a, b: a + b)

    def __mul__(self, other):
        """定義 * 運算子:逐元素相乘(不是矩陣乘法)。"""
        return self._elementwise_op(other, lambda a, b: a * b)

    def __sub__(self, other):
        """定義 - 運算子:逐元素相減。"""
        return self._elementwise_op(other, lambda a, b: a - b)

    def sum(self, axis=None):
        """沿axis加總,該軸消失;axis為None時所有元素加成一個數字。"""
        # 沒指定軸:整個攤平list加總
        if axis is None:
            return sum(self._data)
        # 負數軸編號換成正數
        if axis < 0:
            axis = self.rank + axis
        new_shape = list(self._shape)
        # 從shape拿掉被加總的那個軸,剩下的就是結果的shape
        axis_size = new_shape.pop(axis)

        # 結果的元素個數與strides
        result_size = reduce(lambda a, b: a * b, new_shape, 1)
        # 累加器,初始值全部是0
        result_data = [0.0] * result_size
        result_strides = self._compute_strides(tuple(new_shape))

        # 走過原tensor的每一個元素
        for indices in iterproduct(*(range(s) for s in self._shape)):
            old_flat = sum(i * s for i, s in zip(indices, self._strides))
            # 把被加總的軸的索引拿掉,剩下的索引就是這個元素該加進結果的哪一格
            # indices[:axis]是axis之前的部分,indices[axis + 1:]是axis之後的部分
            new_indices = indices[:axis] + indices[axis + 1:]
            if new_indices:
                new_flat = sum(
                    i * s for i, s in zip(new_indices, result_strides)
                )
            else:
                new_flat = 0
            # 把原元素累加到結果對應的位置
            result_data[new_flat] += self._data[old_flat]

        if not new_shape:
            return result_data[0]
        return Tensor(result_data, shape=tuple(new_shape))

    def to_list(self):
        """轉回巢狀list,方便印出與檢查。"""
        if self.rank == 0:
            return self._data[0]
        return self._build_nested(self._data, self._shape, 0)

    def _build_nested(self, data, shape, offset):
        """遞迴依shape把攤平的data切回巢狀list。offset是目前這一層在攤平list裡的起點。"""
        if len(shape) == 1:
            return data[offset:offset + shape[0]]
        result = []
        stride = reduce(lambda a, b: a * b, shape[1:], 1)
        for i in range(shape[0]):
            result.append(self._build_nested(data, shape[1:], offset + i * stride))
        return result

    def __repr__(self):
        """print(t)時顯示的字串,格式是Tensor(shape=..., data=...)。"""
        return f"Tensor(shape={self._shape}, data={self.to_list()})"

    def to_numpy(self):
        """轉成NumPy陣列:攤平的list先變一維陣列,再按shape切回多維。"""
        return np.array(self._data).reshape(self._shape)


def demo_basic_tensor():
    """示範建立純量、向量、矩陣、3維tensor,讀取rank、shape、索引與strides。"""
    print("=" * 60)
    print("BASIC TENSOR OPERATIONS")
    print("=" * 60)

    scalar = Tensor(3.14)
    print(f"Scalar: shape={scalar.shape}, rank={scalar.rank}, value={scalar.to_list()}")

    vector = Tensor([1, 2, 3, 4, 5])
    print(f"Vector: shape={vector.shape}, rank={vector.rank}")

    matrix = Tensor([[1, 2, 3], [4, 5, 6]])
    print(f"Matrix: shape={matrix.shape}, rank={matrix.rank}")
    print(f"  matrix[0, 1] = {matrix[0, 1]}")
    print(f"  matrix[1, 2] = {matrix[1, 2]}")

    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(f"3D Tensor: shape={tensor_3d.shape}, rank={tensor_3d.rank}")
    print(f"  tensor[1, 0, 1] = {tensor_3d[1, 0, 1]}")

    print(f"\nStrides for shape {matrix.shape}: {matrix.strides}")
    print(f"Strides for shape {tensor_3d.shape}: {tensor_3d.strides}")
    print()


def demo_reshape_operations():
    """示範自製Tensor的reshape、-1推算、squeeze、unsqueeze、transpose、permute、flatten。"""
    print("=" * 60)
    print("RESHAPE OPERATIONS")
    print("=" * 60)

    data = Tensor(list(range(12)), shape=(2, 6))
    print(f"Original: shape={data.shape}")
    print(f"  {data.to_list()}")

    r1 = data.reshape((3, 4))
    print(f"\nReshaped to (3, 4): {r1.to_list()}")

    r2 = data.reshape((2, 2, 3))
    print(f"Reshaped to (2, 2, 3): {r2.to_list()}")

    r3 = data.reshape((-1, 3))
    print(f"Reshaped to (-1, 3): shape={r3.shape}, {r3.to_list()}")

    t = Tensor(list(range(6)), shape=(1, 3, 1, 2))
    print(f"\nBefore squeeze: shape={t.shape}")
    s = t.squeeze()
    print(f"After squeeze():  shape={s.shape}")
    s0 = t.squeeze(dim=0)
    print(f"After squeeze(0): shape={s0.shape}")

    v = Tensor([1, 2, 3])
    print(f"\nVector shape: {v.shape}")
    print(f"unsqueeze(0): {v.unsqueeze(0).shape}")
    print(f"unsqueeze(1): {v.unsqueeze(1).shape}")
    print(f"unsqueeze(-1): {v.unsqueeze(-1).shape}")

    mat = Tensor(list(range(6)), shape=(2, 3))
    print(f"\nOriginal: shape={mat.shape}, {mat.to_list()}")
    tr = mat.transpose(0, 1)
    print(f"Transpose(0,1): shape={tr.shape}, {tr.to_list()}")

    t4d = Tensor(list(range(24)), shape=(1, 2, 3, 4))
    perm = t4d.permute((0, 2, 3, 1))
    print(f"\nPermute (1,2,3,4) -> (0,2,3,1): {t4d.shape} -> {perm.shape}")

    batch_conv = Tensor(list(range(2 * 4 * 4 * 2)), shape=(2, 4, 4, 2))
    flat = batch_conv.flatten(start_dim=1)
    print(f"\nFlatten (2,4,4,2) from dim 1: shape={flat.shape}")
    print()


def demo_custom_tensor_class():
    """綜合示範自製Tensor:建立、reshape、逐元素運算、sum、轉NumPy、permute、flatten。"""
    print("=" * 60)
    print("CUSTOM TENSOR CLASS DEMO")
    print("=" * 60)

    t = Tensor([[1, 2, 3], [4, 5, 6]])
    print(f"Created: {t}")
    print(f"Shape: {t.shape}, Rank: {t.rank}, Size: {t.size}")
    print(f"Strides: {t.strides}")
    print(f"Element [1,2]: {t[1, 2]}")

    r = t.reshape((3, 2))
    print(f"\nReshaped to (3,2): {r}")

    r2 = t.reshape((-1,))
    print(f"Flattened: {r2}")

    u = t.unsqueeze(0)
    print(f"\nUnsqueeze(0): shape={u.shape}")
    s = u.squeeze(0)
    print(f"Squeeze(0):   shape={s.shape}")

    tr = t.transpose(0, 1)
    print(f"\nTranspose: {tr}")

    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[10, 20], [30, 40]])
    print(f"\na + b: {(a + b).to_list()}")
    print(f"a * b: {(a * b).to_list()}")
    print(f"a * 2: {(a * 2).to_list()}")

    print(f"\nSum all: {a.sum()}")
    print(f"Sum axis 0: {a.sum(axis=0).to_list()}")
    print(f"Sum axis 1: {a.sum(axis=1).to_list()}")

    np_arr = t.to_numpy()
    print(f"\nConverted to numpy: {np_arr.shape}")

    t3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    perm = t3d.permute((2, 0, 1))
    print(f"\n3D tensor {t3d.shape} permuted (2,0,1): {perm.shape}")
    print(f"  {perm.to_list()}")

    flat = t3d.flatten(start_dim=1)
    print(f"Flatten from dim 1: {flat.shape} -> {flat.to_list()}")

    print()


if __name__ == "__main__":
    demo_reshape_numpy()
    demo_broadcasting_numpy()
    demo_memory_layout()
    demo_reduction_operations()
    demo_einsum()
    demo_einsum_gallery()
    demo_ai_tensor_shapes()
    demo_attention_einsum()
    demo_custom_tensor_class()
    demo_basic_tensor()
    demo_reshape_operations()
