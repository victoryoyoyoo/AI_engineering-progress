# Lesson 2 語法筆記:Vectors, Matrices & Operations

## Learning Objectives 打勾清單

- [ ] Build a Matrix class with element-wise operations, matrix multiplication, transpose, determinant, and inverse — ⚠️ 沒有手刻。這是這堂課開始套用新的 Top-Down 流程後第一個刻意跳過手刻的項目,不是教學疏漏:matmul/transpose/determinant/inverse 全部程式碼都看過、逐段講解過邏輯(matmul 那段還用 C++ 三層迴圈對照過),口頭能解釋每個方法在幹嘛,但沒有自己手打進 practice.py。完整程式碼在 `reference.py`。
- [x] Distinguish element-wise multiplication from matrix multiplication and explain when each applies — 測驗第1題答對,形狀規則、內積 vs 對應相乘的差異都講清楚了
- [x] Implement a single dense neural network layer (`relu(W @ x + b)`) using only the from-scratch Matrix class — 做到了,但用的是 NumPy 版本(`weights @ inputs + bias` 再套 `np.maximum(0,...)`),不是原本要求的「只用手刻 Matrix class」。這是 Top-Down 新流程下刻意的調整:核心邏輯這行親手打、親自驗證過 shape 跟數值都對,但工具用 NumPy 不是從零刻的 Matrix class。
- [x] Explain broadcasting rules and how bias addition works in neural network frameworks — 一開始不確定 broadcasting 實際用在哪裡(選對答案但講不出應用場景),後來用「一次丟一整批資料進去,bias 要延伸套用到每一筆」的例子講清楚,也親手驗證過純 Python list 的 `+`(是串接)跟 numpy array 的 `+`(是逐項相加)完全是兩回事

⚠️ 尚未完成:Matrix class 完整手刻(matmul/transpose/determinant/inverse),依新的 Top-Down 策略是刻意不做,不是忘記教。

## 核心重點整理(這堂課在教什麼)

| 概念 | 是什麼 | 關鍵規則 / 幾何意義 |
|---|---|---|
| 矩陣 (m×n) | 一台機器:吃 n 維向量,吐出 m 維向量 | 每一列(row)對應輸出的其中一個數字,是「輸入向量」跟「這一列」做內積算出來的 |
| element-wise multiply | 兩個形狀完全一樣的東西,同位置的數字互乘 | numpy 用 `*`,結果形狀不變 |
| matrix multiply | 一列跟一行做內積,不是同位置相乘 | numpy 用 `@`,形狀規則 (m,n) @ (n,p) = (m,p),中間的 n 要對上 |
| transpose(轉置) | 把矩陣的行跟列互換 | (m,n) 轉置後變成 (n,m),原本 [i][j] 轉置後在 [j][i] |
| determinant(行列式) | 只有正方形矩陣才有 | 代表這個矩陣把空間放大/縮小幾倍。是 0 代表空間被壓扁(至少一個維度變成一條線或一點),資訊永久遺失、沒辦法逆推 |
| inverse(逆矩陣) | 「反著做」的矩陣 | A 把 X 搬到 Y,A⁻¹ 把 Y 搬回 X。只有 determinant ≠ 0 才存在,因為壓扁遺失的資訊沒有回頭路 |
| identity matrix(單位矩陣) | 對角線是 1、其餘是 0 | 乘上任何東西都不改變它,是矩陣世界的「1」;A 乘上 A⁻¹ 結果就是它 |
| `relu(W @ x + b)` | 神經網路一層的樣子,深度學習裡最常重複的一行 | W 決定輸入怎麼組合、b 是每個輸出的偏移量、relu 把負數砍成 0;輸出維度由 W 的形狀(輸出維度, 輸入維度)決定 |
| broadcasting | 形狀不完全一樣的陣列做運算時,numpy 自動延伸對齊 | 把 size 是 1 的維度複製延伸去對齊另一邊,對不齊才報錯。常見於一次丟一整批(batch)資料,bias 會被自動複製 batch 次 |

## 我自己手打的部分

這堂課依 Top-Down 策略,大部分程式碼是看過、逐段講解、口頭能解釋邏輯,不是手打驗證的。真正自己手打並驗證過的,只有 `numpy_version.py` 裡這一行核心邏輯:

```python
output = np.maximum(0, weights @ inputs + bias)
```

其餘的 import、inputs/weights/bias 的建立、print 陳述式,是先幫忙寫好的樣板碼。`practice.py` 這堂課是空的(刻意的,矩陣類別完整版沒有手刻,原因寫在上面 Learning Objectives 那一項)。

## 這堂課用的新教學策略

從這堂課開始,Phase 1(純數學章節)改成 **Top-Down 模式**:核心直覺、幾何意義、shape 規則要懂,但不用每個 method 都手刻;真正的模型架構本體章節(Autograd、Neural Network、Transformer、Attention)才會回到全部手刻。這堂課手打的只有一行核心邏輯(`output = np.maximum(0, weights @ inputs + bias)`),其他 Matrix class 的程式碼都是「看過、講過邏輯、能解釋」的方式教完。

## 今天花的時間

這堂課總共專注 2:16:30,其中 AI 相關部分就是這 2:16:30(整段都算)。比 Lesson 1 的 6:35:45 快非常多,主要是 Top-Down 策略省下大量手刻時間,加上環境設定的一次性成本這堂課不用再付。

## 今天評分

理解程度:8/10,矩陣乘法、shape 規則、element-wise vs matrix multiply、determinant/inverse 的直覺都是真的懂,不是背的
效率:明顯比 Lesson 1 快很多,Top-Down 流程省下大量手刻時間
完成度:核心 4 個 Learning Objectives 裡 3 個做到、1 個依新流程刻意跳過(非疏漏)

## 這堂課的總結

這堂課把 Lesson 1 的矩陣乘向量,擴展成「矩陣乘矩陣」跟矩陣的其他基本操作。核心比喻是**矩陣是一台機器:吃進 n 維向量,吐出 m 維向量**,理解這個之後,element-wise multiply(同位置相乘)跟 matrix multiply(內積式相乘)的差異、transpose(行列互換)、determinant(空間放大縮小幾倍,0代表壓扁資訊遺失)、inverse(反著做的矩陣,只有det≠0才存在)全部都可以用「這台機器對空間做了什麼」的角度理解,不用死背規則。

這堂課也第一次接觸神經網路一層的樣子:`relu(W @ x + b)`——W決定輸入怎麼組合、b是偏移量、relu把負數砍成0,這一行就是深度學習裡最常重複出現的運算。broadcasting則解釋了「形狀不完全一樣也能運算」背後 numpy 自動延伸對齊的規則,常見於一次丟一整批(batch)資料時。

**這堂課的關鍵公式/規則,複習時直接看這幾條就夠:**

```
矩陣乘法形狀規則:   (m,n) @ (n,p) = (m,p),中間的 n 必須對上
determinant=0:      矩陣把空間壓扁,資訊遺失,沒有逆矩陣
inverse存在條件:     determinant ≠ 0
神經網路一層:        output = relu(W @ x + b)
relu:               max(0, x),負數變0,正數不變
broadcasting觸發:    某個維度是1(或缺少),numpy自動複製延伸去對齊另一邊
```

這堂課開始正式套用 Top-Down 策略:大部分程式碼看懂邏輯就過關,不用每個都手刻,速度比 Lesson 1 快很多,理解深度沒有打折。

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

## `raise`:主動丟出錯誤

Matrix class 完整版裡出現很多次,例如:

```python
if self.shape != other.shape:
    raise ValueError(f"Cannot add shapes {self.shape} and {other.shape}")
```

`raise` 是主動讓程式停下來、丟出一個錯誤,通常搭配 `if` 用來檢查「這個情況不該發生」——例如兩個矩陣形狀對不上還硬要相加。`ValueError` 是錯誤的種類(這裡代表傳進來的值不對),後面括號裡是錯誤訊息,可以用 f-string 把實際的形狀塞進去,方便除錯時知道到底是哪裡對不上。

跟 C++ 的 `throw` 是同樣的概念:C++ `throw std::invalid_argument("...")`,Python 就是 `raise ValueError("...")`。差別是 Python 內建了一整套錯誤種類可以選(`ValueError`、`TypeError`、`ZeroDivisionError`...),挑最符合情況的那個用,不像 C++ 常常什麼都丟同一種 exception。

`raise` 執行到就會立刻中斷當下的函式(跟 `return` 一樣會離開,但離開的原因是「出錯了」不是「算完了」),如果沒有被 `try/except`接住,程式會直接崩潰並印出錯誤訊息跟發生位置。這堂課還沒教 `try/except`(怎麼接住這個錯誤、不讓程式崩潰),之後遇到再補。

## Shape tuple `(m, n)` 的判讀方式複習

矩陣形狀 `(m, n)` 讀作「m 列、n 行」,矩陣乘法規則 `(m,n) @ (n,p) = (m,p)` 中間的 `n` 要對上。判斷一個向量是「幾維」,就是數這個向量裡總共裝了幾個數字——`(128, 1)` 裝了 128 個數字直立擺放,所以是 128 維,跟橫著寫的 `(1, 128)` 或單純的 `[128個數字]` 是同一件事,只是排列方式不同。

## f-string 複習

```python
print(f"Output shape:{output.shape}")
```

`f"..."` 讓字串裡的 `{}` 可以直接放變數或運算式,執行時自動換成實際值。不加 `f` 的話 `{}` 只是普通文字,不會被替換,需要自己手動用 `+` 拼接字串(比較麻煩)。
