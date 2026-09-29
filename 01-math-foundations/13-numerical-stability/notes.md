# Numerical Stability

學習浮點數運算的極限與數值穩定性防護。掌握上溢（overflow）、下溢（underflow）、災難性抵消（catastrophic cancellation）、穩定版 Softmax、Log-Sum-Exp 與梯度裁剪。

## 名詞
| 名詞 | 意思 |
|---|---|
| IEEE 754（浮點數標準） | 定義二進位浮點數怎麼存、怎麼四捨五入、`inf`/`nan`怎麼來的國際標準,所有現代CPU/GPU都照這套走 |
| Overflow（上溢） | 數字大到超出能存的範圍,變成`inf` |
| Underflow（下溢） | 數字小到存不下,直接變成`0` |
| Catastrophic cancellation（災難性抵銷） | 兩個很接近的數相減,有效位數被吃光,剩下的都是誤差 |
| NaN（不是數字） | 從沒有意義的運算(`0/0`、`inf-inf`、`inf*0`)產生,之後任何運算碰到它都是NaN |
| Inf（無限大(記號)） | 超出範圍或除以0產生的特殊值,不是真正的「無限」,是「已經超出電腦能存的極限」的標記 |
| Log-sum-exp（log-sum-exp技巧） | 用先提出最大值再取log的方式算`log(Σexp(x))`,避免overflow/underflow |
| Stable softmax（穩定版softmax） | 先減去`max(logits)`再取exp,結果不變但不會溢位 |
| Machine epsilon（機器精度） | 讓`1.0+e != 1.0`成立的最小`e`;float32約`1.19e-7` |
| Gradient checking（梯度檢查） | 用數值微分(`f(x+h)`跟`f(x-h)`的差)跟反向傳播算出的梯度對照,抓實作bug(這堂未教) |
| Mixed precision（混合精度） | 部分運算用float16加速,部分關鍵運算留在float32,兼顧速度與穩定 |
| Loss scaling（損失縮放） | 反向傳播前把loss放大一個倍數,避免小梯度在float16下溢成0,更新前再除回來 |
| bfloat16（brain floating point） | Google的16位元格式,指數位跟float32一樣多(範圍大),尾數位少(精度差),訓練偏好用它 |
| Gradient clipping（梯度裁剪） | 把梯度向量的長度限制在某個上限以內,避免梯度爆炸把權重震壞 |
| Numerical gradient（數值梯度） | 用`(f(x+h)-f(x-h))/2h`這種暴力方式近似算出的梯度,慢但可靠,用來驗證 |
| Layer Normalization（層正規化） | 把一組數字強制標準化成平均0、標準差1,防止數值一層一層疊加爆炸 |


## 重點
| 概念 | 我的理解 |
|---|---|
| IEEE 754 浮點限制 | 二進位無法精確表示所有十進位小數（如 0.1+0.2 != 0.3），比對兩數不可用 `==`，應使用容許誤差 `abs(a - b) < eps`。 |
| Overflow, Underflow & NaN | 數值超過表示上限變 `inf`，小於最小正數變 `0`。`inf * 0` 無法判斷失真前大小而給出 `NaN`，NaN 會迅速污染整個計算圖。 |
| Stable Softmax & LogSumExp | 計算 exp 前先減去向量最大值（exp(0)=1），在數學上等價約分但完全避免 exp 溢位；LogSumExp 將 log 與 softmax 合併防止極小值產生負無窮。 |
| Catastrophic Cancellation | 兩極度接近的大數相減時有效數字流失。例如計算變異數若用展開式 `E[X²] - (E[X])²` 遇到大數會完全算錯，應使用定義式 `E[(X - μ)²]`。 |
| Gradient Clipping | clip by value 逐個數值截斷會扭曲梯度方向；clip by norm 依 L2 norm 超出比例整體縮小，能保證梯度方向不變。 |

## 公式／程式
```text
浮點比較: abs(a - b) < 1e-9
穩定 Softmax: softmax(z)_i = exp(z_i - c) / Σ_j exp(z_j - c)，其中 c = max(z)
穩定 LogSumExp: logsumexp(x) = c + log(Σ exp(x_i - c))，其中 c = max(x)
Clip by Norm: scale = min(1.0, max_norm / (||g||_2 + 1e-6)); g_clipped = g * scale
```

```python
# 穩定版 LogSumExp 實作
def logsumexp(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))
```

## 卡住的地方
- 疑惑為何 `inf * 0` 不是 0；電腦無法區分 0 是精確 0 還是下溢後的近似值，為避免產生誤導性的假正確答案，IEEE 754 規範回傳 `NaN`。
- naive 變異數公式 `mean(x²) - mean(x)²` 遇到 1 億級別資料時，平方項達 10^16 吃光 float64 的有效位數，相減後誤差吃掉真正差異算出 0.0（災難性抵消）。
- 梯度裁剪時若使用 clip by value，不同維度被削平的比例不一會打亂最佳化前進方向，實務訓練應使用 clip by norm。
