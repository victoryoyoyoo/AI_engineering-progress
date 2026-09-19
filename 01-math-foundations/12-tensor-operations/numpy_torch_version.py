"""
Lesson 12: Tensor Operations——NumPy與PyTorch對照
同樣的shape、stride、view、broadcasting、einsum概念,在兩個套件裡的寫法與差異。
重點差異:
    NumPy   strides單位是「位元組」,轉置後通常自動處理非連續記憶體
    PyTorch strides單位是「元素個數」,view()對非連續tensor會報錯
"""
import numpy as np
import torch


def demo_shape_and_stride():
    """同一個2x3陣列,在NumPy跟PyTorch裡讀出shape與strides"""
    # dtype=np.float32:每個元素佔4位元組,下面的strides才會是(12, 4)
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

    # a.strides = (12, 4):往下一列跳12位元組(3個元素x4位元組),往右一欄跳4位元組
    print(f"NumPy  shape={a.shape}  strides(bytes)={a.strides}")
    # NumPy strides單位是位元組;PyTorch strides單位是元素個數
    # t.stride() = (3, 1):往下一列跳3個元素,往右一欄跳1個元素,兩者意思相同
    # tuple(t.shape)把torch.Size轉成一般tuple,列印起來比較乾淨
    print(f"PyTorch shape={tuple(t.shape)}  strides(elements)={t.stride()}")


def demo_view_vs_reshape():
    """transpose之後tensor變成非連續,view()失敗、reshape()與contiguous().view()成功"""
    # arange(6)是0到5,reshape成2x3;記憶體裡實際排列是 0 1 2 3 4 5
    t = torch.arange(6).reshape(2, 3)
    # transpose只交換shape與stride,資料沒有搬動 -> 共用同一塊記憶體,但變成非連續
    tr = t.transpose(0, 1)

    print(f"t contiguous:  {t.is_contiguous()}")
    print(f"tr contiguous: {tr.is_contiguous()}")

    try:
        # view()只換shape的看法、不複製資料,前提是元素在記憶體裡要照順序排
        # tr的邏輯順序是 0 3 1 4 2 5,跟記憶體順序 0 1 2 3 4 5 不同,所以view失敗
        tr.view(6)
    except RuntimeError as e:
        # 只印錯誤訊息前60個字元,避免整段冗長訊息洗版
        print(f"view on non-contiguous fails: {str(e)[:60]}...")

    # reshape()寬容:能不複製就當view用,不行就自動複製一份再換shape
    print(f"reshape works (copies when needed): {tr.reshape(6)}")
    # contiguous()先真的照邏輯順序重新排一份資料,之後view()就能成功
    print(f"contiguous().view works: {tr.contiguous().view(6)}")


def demo_unsqueeze_and_broadcast():
    """unsqueeze插入大小為1的軸,配合broadcasting做逐元素運算"""
    # randn是標準常態分布的隨機數,shape (4,3):4筆資料、每筆3個特徵
    x = torch.randn(4, 3)
    bias = torch.tensor([0.1, 0.2, 0.3])
    # (4,3)+(3,):bias左邊補1變(1,3),再沿第0軸拉大成4列,結果(4,3)
    print(f"(4,3) + (3,) -> {tuple((x + bias).shape)}")

    # unsqueeze(1):在第1軸插入大小為1的軸,(3,)變(3,1),直排
    col = torch.tensor([1, 2, 3]).unsqueeze(1)
    # unsqueeze(0):在第0軸插入,(4,)變(1,4),橫排
    row = torch.tensor([10, 20, 30, 40]).unsqueeze(0)
    # (3,1)x(1,4):兩個方向各自拉大,得到3x4的乘法表(外積)
    print(f"(3,1) * (1,4) -> {tuple((col * row).shape)}")


def demo_einsum():
    """torch.einsum的寫法與NumPy相同:字母命名軸,箭頭右邊沒出現的軸被加總"""
    A = torch.randn(3, 4)
    B = torch.randn(4, 5)
    # "ik,kj->ij":k同名且不在輸出,沿k相乘再加總,就是矩陣乘法(3x4)@(4x5)=(3x5)
    out = torch.einsum("ik,kj->ij", A, B)
    # allclose用容許誤差比較浮點數,不用==(浮點運算順序不同會有微小差)
    print(f"einsum matmul matches @: {torch.allclose(out, A @ B)}")

    # attention分數的寫法:b=批次、h=頭、t/s=位置、d=每頭維度
    # d被加總,留下t、s,得到每個位置對每個位置的分數表 (2,4,8,8)
    Q = torch.randn(2, 4, 8, 16)
    K = torch.randn(2, 4, 8, 16)
    scores = torch.einsum("bhtd,bhsd->bhts", Q, K)
    print(f"attention scores shape: {tuple(scores.shape)}")


def demo_numpy_torch_bridge():
    """NumPy陣列與PyTorch tensor互轉,以及共用記憶體的行為"""
    # 指定float32,避免預設float64跟PyTorch常用的float32不一致
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = torch.from_numpy(a)
    # from_numpy不複製資料,t與a指向同一塊記憶體
    # 所以改a[0,0],t[0,0]也跟著變成100
    a[0, 0] = 100
    print(f"shared memory: t[0,0]={t[0, 0].item()}")
    # .item()把單一元素的tensor取出成Python數字
    # t.numpy()反向轉換,同樣共用記憶體(只限CPU tensor)
    print(f"back to numpy: {t.numpy().shape}")


if __name__ == "__main__":
    demo_shape_and_stride()
    demo_view_vs_reshape()
    demo_unsqueeze_and_broadcast()
    demo_einsum()
    demo_numpy_torch_bridge()
