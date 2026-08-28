# Lesson 2 語法筆記:Vectors, Matrices & Operations

## Learning Objectives 打勾清單

- [ ] Build a Matrix class with element-wise operations, matrix multiplication, transpose, determinant, and inverse — ⚠️ 沒有手刻。這是這堂課開始套用新的 Top-Down 流程後第一個刻意跳過手刻的項目,不是教學疏漏:matmul/transpose/determinant/inverse 全部程式碼都看過、逐段講解過邏輯(matmul 那段還用 C++ 三層迴圈對照過),口頭能解釋每個方法在幹嘛,但沒有自己手打進 practice.py。完整程式碼在 `reference.py`。
- [x] Distinguish element-wise multiplication from matrix multiplication and explain when each applies — 測驗第1題答對,形狀規則、內積 vs 對應相乘的差異都講清楚了
- [x] Implement a single dense neural network layer (`relu(W @ x + b)`) using only the from-scratch Matrix class — 做到了,但用的是 NumPy 版本(`weights @ inputs + bias` 再套 `np.maximum(0,...)`),不是原本要求的「只用手刻 Matrix class」。這是 Top-Down 新流程下刻意的調整:核心邏輯這行親手打、親自驗證過 shape 跟數值都對,但工具用 NumPy 不是從零刻的 Matrix class。
- [x] Explain broadcasting rules and how bias addition works in neural network frameworks — 一開始不確定 broadcasting 實際用在哪裡(選對答案但講不出應用場景),後來用「一次丟一整批資料進去,bias 要延伸套用到每一筆」的例子講清楚,也親手驗證過純 Python list 的 `+`(是串接)跟 numpy array 的 `+`(是逐項相加)完全是兩回事

⚠️ 尚未完成:Matrix class 完整手刻(matmul/transpose/determinant/inverse),依新的 Top-Down 策略是刻意不做,不是忘記教。

## 這堂課用的新教學策略

從這堂課開始,Phase 1(純數學章節)改成 **Top-Down 模式**:核心直覺、幾何意義、shape 規則要懂,但不用每個 method 都手刻;真正的模型架構本體章節(Autograd、Neural Network、Transformer、Attention)才會回到全部手刻。這堂課手打的只有一行核心邏輯(`output = np.maximum(0, weights @ inputs + bias)`),其他 Matrix class 的程式碼都是「看過、講過邏輯、能解釋」的方式教完。

## 今天評分

理解程度:8/10,矩陣乘法、shape 規則、element-wise vs matrix multiply、determinant/inverse 的直覺都是真的懂,不是背的
效率:明顯比 Lesson 1 快很多,Top-Down 流程省下大量手刻時間
完成度:核心 4 個 Learning Objectives 裡 3 個做到、1 個依新流程刻意跳過(非疏漏)

---

(下面不重複講數學/AI概念,只整理「程式語法」本身,之後忘記可以回來查。)

## List comprehension 複習:matmul 那段對照 C++ 三層迴圈

Lesson 1 已經學過 list comprehension 的基本語法,這堂課遇到雙層巢狀的版本,對照 C++ 的三層迴圈矩陣乘法看最快搞懂:

```python
def matmul(self, other):
    return Matrix([
        [
            sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
            for j in range(other.cols)
        ]
        for i in range(self.rows)
    ])
```

對照的 C++ 版本(這是刷題刷過很多次的三層迴圈矩陣乘法):

```cpp
for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols_B; j++) {
        int sum = 0;
        for (int k = 0; k < cols_A; k++) {
            sum += A[i][k] * B[k][j];
        }
        result[i][j] = sum;
    }
}
```

讀 list comprehension 的訣竅是**從最裡面往外讀**:
1. 最內層 `self.data[i][k] * other.data[k][j] for k in range(self.cols)` = C++ 的 `for(k...) sum += A[i][k]*B[k][j];`,外面包 `sum(...)` 就是把這輪迴圈的加總結果收集起來
2. 中間層 `for j in range(other.cols)` = C++ 的 `for(j...)`,對每個 j 重複算一次上面那個內積
3. 最外層 `for i in range(self.rows)` = C++ 的 `for(i...)`,對每個 i 重複算一次

三層迴圈,三層 for,順序完全對應,只是 Python 把「宣告空陣列 + append」濃縮成用中括號包起來自動收集結果。

## NumPy 是什麼、跟純 Python 的差別

NumPy 是一個函式庫(library),裡面已經幫你把矩陣/陣列的運算都寫好、還用 C 語言優化過速度。差別最明顯的地方是運算子的行為完全不同:

```python
a = [1, 2, 3]
b = [10, 20, 30]
a + b   # 純 Python list:[1, 2, 3, 10, 20, 30] —— 串接,不是數學加法

import numpy as np
c = np.array([1, 2, 3])
d = np.array([10, 20, 30])
c + d   # numpy array:[11, 22, 33] —— 逐項相加,才是數學加法
```

**這不是 Python 內建的行為,是 numpy 這個函式庫自己重新定義了 `+`、`-`、`*`、`@` 這些運算子在它的 array 型態上該怎麼運作。** 只要 `import numpy as np`,而且操作的是 `np.array(...)` 建出來的東西,這整套行為(包括 broadcasting)就自動生效,不用額外開啟。

## Broadcasting 實際運作的時機

之前搞不清楚 broadcasting 實際用在哪裡,後來釐清:**只有形狀不一樣、又可以「延伸對齊」的情況才會觸發**;形狀本來就一樣的話,直接逐項運算,沒有 broadcasting 這回事。

```python
# 沒有觸發 broadcasting(形狀本來就一樣):
weights @ inputs  # 結果 (2,1)
bias              # 也是 (2,1)
weights @ inputs + bias   # 直接逐項相加

# 有觸發 broadcasting(一次丟一整批資料,batch=5):
weights @ inputs  # inputs 變成 (3,5),結果 (2,5)
bias              # 還是 (2,1)
weights @ inputs + bias   # (2,5) 跟 (2,1) 形狀不一樣,numpy 自動把 bias 那一欄複製5次湊成 (2,5) 再相加
```

判斷準則:兩個陣列做運算時,如果其中一邊某個維度是 1(或缺少這個維度),numpy 就會嘗試把那個維度「複製延伸」去對齊另一邊,對不齊就直接報錯。

## `np.maximum(0, x)`

拿 `x` 裡每一個元素跟 `0` 比,取比較大的那個,逐元素進行。負數會變 0,正數維持原樣,是 relu 這個激活函數最單純的實作方式。跟純 Python 的 `max(0, x)` 不同——`max()` 只能比較兩個單一數字,`np.maximum()` 可以整個陣列一次比完,不用寫迴圈。

## Shape tuple `(m, n)` 的判讀方式複習

矩陣形狀 `(m, n)` 讀作「m 列、n 行」,矩陣乘法規則 `(m,n) @ (n,p) = (m,p)` 中間的 `n` 要對上。判斷一個向量是「幾維」,就是數這個向量裡總共裝了幾個數字——`(128, 1)` 裝了 128 個數字直立擺放,所以是 128 維,跟橫著寫的 `(1, 128)` 或單純的 `[128個數字]` 是同一件事,只是排列方式不同。

## f-string 複習

```python
print(f"Output shape:{output.shape}")
```

`f"..."` 讓字串裡的 `{}` 可以直接放變數或運算式,執行時自動換成實際值。不加 `f` 的話 `{}` 只是普通文字,不會被替換,需要自己手動用 `+` 拼接字串(比較麻煩)。
