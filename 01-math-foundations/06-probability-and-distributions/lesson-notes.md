# Lesson 6 語法筆記

這堂課主要複習/加深了之前學過的語法,新東西不多,整理一下重點。

## `zip()` 配對兩個 list

```python
for v, p in zip(values, probabilities)
```

`zip()` 把兩個(或多個)list按照相同的索引位置配對起來,變成一串tuple。例如 `values=[1,2,3]`、`probabilities=[0.2,0.3,0.5]`,`zip()` 會依序產生 `(1,0.2)`、`(2,0.3)`、`(3,0.5)`。`for v, p in zip(...)` 這種寫法叫**tuple unpacking(元組解包)**,直接把每個tuple拆成兩個變數,不用寫 `pair[0]`、`pair[1]`。

C++對照:類似同時遍歷兩個vector,但C++通常要手動用index(`for(int i=0;i<n;i++)`)或者用 `std::views::zip`(C++23才有,更早版本沒有內建的zip)。

## Generator expression(生成器表達式)vs List comprehension(列表推導式)

```python
sum(v * p for v, p in zip(values, probabilities))
```

注意這裡**沒有方括號 `[ ]`**,只有小括號(其實連小括號都省了,直接傳進 `sum()` 裡)。這叫 generator expression,跟之前學過的 list comprehension(`[x for x in ...]`)不一樣:

- List comprehension `[x*2 for x in range(10)]` 會**先把整個list在記憶體裡建好**,才回傳。
- Generator expression `(x*2 for x in range(10))` 是**一邊要一邊算**,不會一次把整串資料都存進記憶體。

當你只是要把結果丟進 `sum()`、`max()` 這種「一次消耗掉」的函式,用generator比較省記憶體,因為根本不需要真的建一個完整的list出來。

## `math` 模組

```python
import math
math.exp(x)   # e的x次方
math.log(x)   # 自然對數(以e為底)
```

C++對照:對應 `<cmath>` 裡的 `std::exp`、`std::log`,用法邏輯一樣,只是Python要先 `import math` 才能用,而且呼叫時要加 `math.` 前綴(除非用 `from math import exp, log`)。

## List comprehension 疊代(nested list comprehension的簡單版)

```python
shifted = [z - max_logit for z in logits]
exps = [math.exp(z) for z in shifted]
```

這兩行分開寫(先算shifted、再算exps),沒有硬塞成一行巢狀的推導式(雖然技術上可以寫成 `[math.exp(z - max_logit) for z in logits]`)。分開寫可讀性更好,尤其是每一步都有明確的數學意義(先做數值穩定平移、再取指數),分開命名變數方便除錯跟理解。

## NumPy `np.average` 帶權重

```python
np.average(die_values, weights=die_probs)
```

`np.average` 預設是算普通平均(每個元素權重相等),但傳入 `weights=` 參數後,就變成**加權平均**——等同於我們手刻的 `expected_value`。這是NumPy函式常見的模式:同一個函式名稱,靠可選參數(optional argument)切換行為,不用另外寫一個新函式。

## SciPy `scipy.special.softmax` / `log_softmax`

```python
from scipy.special import softmax, log_softmax
```

這兩個函式內部已經內建了「減最大值做數值穩定」的邏輯,不需要自己再手動處理——這也是為什麼手刻理解底層邏輯很重要:知道函式庫幫你做了什麼,遇到數值不穩定的bug時才知道要往哪裡查。
