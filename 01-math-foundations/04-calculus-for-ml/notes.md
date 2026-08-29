# Lesson 4 筆記:Calculus for ML(微積分)

## Learning Objectives 打勾清單

- [~] Compute numerical and analytical derivatives for common ML functions (x^2, sigmoid, cross-entropy) — x² 的數值法(Numerical)跟解析法(Analytical)導數完整講過、程式碼跑過驗證(誤差接近0),也對照過 x³/sin(x)/eˣ/1/x 這幾個函數。⚠️ sigmoid、cross-entropy 這兩個 ML 專用函數的導數,課程教材有列出公式(`sigmoid'(x)=f(x)(1-f(x))`),但這堂課沒有真的展開講解或驗證,留到之後遇到(邏輯迴歸/分類問題)再回頭補。
- [x] Implement gradient descent from scratch to minimize a loss function in 1D and 2D — 1D(f(x)=x²,從x=5走到接近0)自己手打+跑過;2D(f(x,y)=x²+y²)完整逐行拆解過(zip、list comprehension、numerical_gradient的中央差分公式),用具體數字追蹤過整個流程。
- [x] Derive the gradient of a linear regression model and train it via manual weight updates — 手推公式(dw=mean(2*error*x)、db=mean(2*error))看過懂邏輯,親眼驗證w、b從隨機亂猜(loss=67)收斂到接近真實答案(y=2x+1)的過程。
- [x] Explain the Hessian matrix, Taylor series approximations, and their connection to optimization methods — Hessian(瞎子下山比喻、跟Lesson 3 eigenvalue的連結、鞍點判斷)、泰勒展開(一階=梯度下降、二階=牛頓法)、牛頓法都講清楚,程式碼驗證過saddle跟bowl兩種地形算出的Hessian eigenvalue確實對應理論。

⚠️ 尚未完成:sigmoid/cross-entropy 的導數沒有實際展開講解,留到之後分類問題章節再補。

## 核心重點整理(這堂課在教什麼)

| 概念 | 是什麼 | 關鍵規則 / 幾何意義 |
|---|---|---|
| 導數 Derivative | 函數在某點的變化速度/斜率 | 幾何上是切線斜率;x稍微變動一點,y大概變多少 |
| 偏導數 Partial Derivative | 只動一個變數,其他變數固定不動,算出的變化率 | 神經網路裡每個權重各自的偏導數,分開算才知道各自該怎麼調 |
| 梯度 Gradient | 把每個變數的偏導數收集成一個向量 | 指向「往上爬最快」的方向;要下降就走它的反方向 |
| 梯度下降 Gradient Descent | 更新規則:新值=舊值−學習率×梯度 | 只需要一階導數,重複很多次就能逼近函數最低點 |
| 學習率 Learning Rate | 控制每一步跨多大 | 太大會讓局部線性逼近失效、甚至發散;太小收斂很慢 |
| 數值法 Numerical | 動一點點、看輸出差多少,逼近導數定義本身 | `[f(x+h)-f(x-h)]/(2h)`,任何函數都能用,但是逼近值有極小誤差 |
| 解析法 Analytical | 手推公式,精確算出導數 | 快、準,但要先會推導這個函數的公式 |
| Hessian 矩陣 | 二階偏導數湊成的矩陣,告訴你「地形彎的程度」 | eigenvalue全正=真的最低點;全負=最高點;有正有負=鞍點(假的最低點) |
| 鞍點 Saddle Point | 梯度=0但不是真正最低點的地方 | 前後平、左右也平,但其實旁邊還有更深的地方,靠Hessian的eigenvalue戳破 |
| 牛頓法 Newton's Method | 用梯度+Hessian一起算更新方向 | 理論上一步跳到谷底,但Hessian太大(N²),深度學習用不起 |
| 泰勒展開 Taylor Series | 用多項式局部逼近任何函數 | 一階近似=梯度下降在做的事;二階近似=牛頓法在做的事 |
| 線性迴歸 Linear Regression | y=wx+b,用梯度下降訓練w、b | predict→算loss→算梯度→更新,是所有神經網路訓練迴圈的縮影 |

## 這堂課的總結

這堂課只圍繞一件事展開:**導數(Derivative)告訴你「稍微調整一個東西,結果會變多少」,把這個資訊反過來用,就能自動找到讓函數最小的位置**——這就是梯度下降(Gradient Descent)。整堂課的更新規則從頭到尾只有一條,反覆套用在不同情境:

```
新值 = 舊值 − 學習率 × 梯度
```

先套在1個變數(x²練習),再套在2個座標(x,y同時下降),最後套在真正的模型參數(w、b的線性迴歸)——變數數量不同,規則完全一樣。梯度本身是靠偏導數(Partial Derivative)湊出來的,偏導數又可以用數值法(動一點點看輸出差多少,`[f(x+h)-f(x-h)]/(2h)`)或解析法(手推公式)兩種方式算,兩者算出來的答案幾乎一致(誤差在1e-9等級)。

Hessian 矩陣(二階導數)是這堂課的進階延伸:它告訴你地形彎的程度,能戳破「梯度=0但其實不是真正最低點」的鞍點(Saddle Point)假象——判斷方式直接用上 Lesson 3 學的 eigenvalue(全正=最低點、有正有負=鞍點)。泰勒展開(Taylor Series)則把一階/二階近似的概念串起來:梯度下降是一階近似,牛頓法(用梯度+Hessian)是二階近似,但因為 Hessian 大小是參數量的平方,深度學習實務上幾乎只能用一階方法。

**這堂課的關鍵公式,複習時直接看這幾條就夠:**

```
數值導數(中央差分):    f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
梯度下降更新規則:        新值 = 舊值 − 學習率 × 梯度
線性迴歸的梯度:          dw = mean(2*error*x),db = mean(2*error)
Hessian(2變數):         [[d²f/dx², d²f/dxdy], [d²f/dydx, d²f/dy²]]
Hessian eigenvalue判斷:  全正=最低點,全負=最高點,有正有負=鞍點
泰勒展開(到二階):        f(x0+h) ≈ f(x0) + f'(x0)*h + 0.5*f''(x0)*h²
牛頓法更新規則:          新值 = 舊值 − (Hessian反矩陣) × 梯度
```

**PyTorch/NumPy 對應:** 這堂課手刻的整套「算梯度→更新」流程,PyTorch 用 `loss.backward()`(自動微分算出所有梯度)+ `optimizer.step()`(套用更新規則)兩行取代,`torch.optim.SGD`/`Adam` 就是不同的更新規則實作。

**商業/工程價值:** 這是所有神經網路訓練的最底層機制——不管模型多大(從線性迴歸到GPT),訓練迴圈的骨架都是「predict→算loss→算梯度→更新」,學習率設定得好不好,直接決定一個模型能不能訓練成功、要花多少時間跟算力成本。

## 我自己手打的部分

這堂課手打了 1D 梯度下降這段(`practice.py`):

```python
x = 5.0
lr = 0.1
for step in range(20):
    grad = 2 * x
    x = x - lr * grad
    print(f"step {step:2d}  x={x:8.4f}  f(x)={x**2:10.6f}")
```

其中 `for step int range(20)` 有個筆誤(`int` 應該是 `in`),自己抓出來後改對,跑出來確認 x 從 5 收斂到接近 0。print 那行格式化語法是我幫忙補的。2D 梯度下降跟線性迴歸那兩段,是看懂完整邏輯、逐行拆解過(包括 `zip`、list comprehension、`numerical_gradient`函式怎麼對應中央差分公式),但沒有另外手打進 `practice.py`,完整程式碼在 `reference.py`。

## 今天花的時間

計時器顯示 2:21:16。

## 今天評分

理解程度:8/10,梯度下降的核心規則(新值=舊值−學習率×梯度)不管套用在幾個變數上都能講清楚,Hessian/鞍點/牛頓法/泰勒展開的關聯也接得起來
效率:中途在 `numerical_gradient`、`zip`、list comprehension 這幾段程式碼卡了一段時間,後來改用「先講數學公式、程式碼對應公式」的方式重講才順,也發現了自己容易把「變數數量不同」跟「一階/二階導數不同」這兩件事搞混,已經釐清
完成度:核心4個 Learning Objectives 裡3個完全做到,1個部分做到(sigmoid/cross-entropy導數留待之後補),測驗3/3

---

（下面不重複講數學/AI概念，只整理「程式語法」本身，之後忘記可以回來查。）

## List comprehension + zip:數值梯度更新那行

```python
point = [p - lr * g for p, g in zip(point, grad)]
```

### 對應的數學公式

梯度下降更新規則:`新位置 = 舊位置 − 學習率 × 梯度`,對每個變數(x、y)各自套用:

```
新x = 舊x − 學習率 × (x方向的梯度)
新y = 舊y − 學習率 × (y方向的梯度)
```

`point` 跟 `grad` 是分開存放的兩個 list,`zip` 負責把它們按位置配對起來(x配x的梯度、y配y的梯度),確保不會配錯對。

### 完整追蹤一次(用具體數字)

起始:`point = [4.0, 3.0]`,`grad = [8.0, 6.0]`,`lr = 0.1`

1. **`zip(point, grad)` 先執行** —— 按位置配對成兩組:`(4.0, 8.0)`、`(3.0, 6.0)`
2. **迴圈開始跑,一組一組拆開處理**
   - 第1輪:`p=4.0, g=8.0` → 算 `p - lr*g` = `4.0 - 0.1*8.0` = `3.2`
   - 第2輪:`p=3.0, g=6.0` → 算 `p - lr*g` = `3.0 - 0.1*6.0` = `2.4`
3. **兩輪算出來的值收集成新 list**:`[3.2, 2.4]`
4. **蓋回 `point`**:`point = [3.2, 2.4]`

跟實際跑出來的輸出一致:`step 0  point=(3.2000, 2.4000)`。

### 展開版(不用 list comprehension 的寫法)

```python
new_point = []
for p, g in zip(point, grad):
    new_value = p - lr * g
    new_point.append(new_value)
point = new_point
```

跟濃縮版做的事完全一樣,只是拆開成「宣告空list → 跑迴圈 → 每次append」三個動作。Python 工程師平常習慣用濃縮版(list comprehension),但意思上 100% 等價。

對照 C++:等同於 `vector<double> new_point; for(int i=0;i<point.size();i++) new_point.push_back(point[i]-lr*grad[i]); point=new_point;`。`zip` 相當於幫你把「同一個 index 的東西自動湊一對」做掉,不用自己管 index。

## `numerical_gradient` 函式對應的公式(中央差分法)

```python
def numerical_gradient(f, point, h=1e-7):
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)
        point_plus[i] += h
        point_minus[i] -= h
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)
    return gradient
```

對應公式:對每個變數 i,`偏導數 ≈ [f(第i個變數多一點點) − f(第i個變數少一點點)] / (2×一點點的量)`。這就是導數定義本身,直接用數字逼近,不套用任何解析公式,對每個變數各做一次、收集成一個梯度向量回傳。

## f-string 格式化複習

```python
print(f"step {step:2d}  x={x:8.4f}  f(x)={x**2:10.6f}")
```

`{step:2d}` 整數至少佔2字元寬;`{x:8.4f}` 浮點數至少佔8字元寬、小數4位;`{x**2:10.6f}` 同樣道理、小數6位。純粹是排版對齊用,跟演算法邏輯無關。
