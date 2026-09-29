# Lesson 13 - Numerical Stability(數值穩定性)

## 目錄

- [Learning Objectives 打勾清單](#learning-objectives-打勾清單)
- [30秒抓重點](#30秒抓重點)
- [公式速查表](#公式速查表)
- [這堂課的名詞總表](#這堂課的名詞總表)
- [這堂課我卡住/搞混的地方(完整問答記錄,給複習用)](#這堂課我卡住搞混的地方完整問答記錄給複習用)
- [常見地雷](#常見地雷)
- [相關概念(跨堂連結)](#相關概念跨堂連結)
- [面試向問題](#面試向問題)
- [課程結尾理解確認題](#課程結尾理解確認題)
- [我自己手打的部分](#我自己手打的部分)
- [今天評分](#今天評分)
- [程式語法筆記](#程式語法筆記)

## Learning Objectives 打勾清單

- [x] 用max-subtraction技巧實作數值穩定的softmax與log-sum-exp
- [x] 認出floating-point運算裡的overflow、underflow、catastrophic cancellation
- [ ] 用centered finite difference驗證analytical gradient跟numerical gradient是否一致 ⚠️(這堂沒教,記review-queue)
- [x] 解釋為什麼訓練偏好bfloat16而不是float16,以及loss scaling怎麼避免梯度下溢

## 30秒抓重點

- 電腦的小數是有限位數的近似值,`0.1+0.2≠0.3`不是bug,是二進位存不完0.1這個數字的必然結果;浮點數不能用`==`比,要用`abs(a-b)<eps`
- float16/32/64位元數越多,範圍跟精度越大;數字超過範圍變`inf`(overflow)、小到存不下變`0`(underflow)
- `inf`和`0`都可能是「真的就是那樣」,也可能是「原本是個正常數字、失真後變成的」;所以`inf*0`電腦不敢給答案,回傳`NaN`;`NaN`不能用`==`比,要用`isnan()`,而且NaN會傳染,一個壞值毀全部
- softmax/log-sum-exp的核心技巧:算exp之前先減去最大值,數學上答案不變(分子分母同乘同一個常數),但保證最大的一項是`exp(0)=1`,不可能溢位
- log-sum-exp是cross-entropy loss裡「先做softmax再取log」這兩步的穩定合體寫法,不用真的算出可能爆炸的機率中間值
- catastrophic cancellation:兩個很接近的大數相減,有效位數被吃光,剩下的全是誤差(例:naive公式算變異數,1億級的數字相減會整個算錯)
- gradient clipping:clip by value逐個夾、會改變梯度方向;clip by norm整體等比例縮小、方向不變,訓練實務標準是用clip by norm
- bfloat16犧牲精度換範圍(跟float32一樣大),float16犧牲範圍換精度;訓練偏好bfloat16因為訓練中數值容易忽大忽小,寧可精度差一點也不能爆
- loss scaling:訓練前把loss放大一個倍數,讓小梯度不會下溢成0,算完梯度再除回來;動態版本會依有沒有溢位自動調整倍數
- layer normalization:每一層都把數值重新拉回「平均0、標準差1」的範圍,防止數值隨層數疊加、指數成長爆炸

## 公式速查表

<details>
<summary>展開:公式速查表</summary>

| 用途 | 寫法 |
|---|---|
| 浮點數比較(不用`==`) | `abs(a - b) < 1e-9` |
| 穩定softmax | 先減最大值:`exp(z_i - max(z)) / Σexp(z_j - max(z))` |
| 穩定log-sum-exp | `c = max(v)`,結果 `c + log(Σexp(v_i - c))` |
| cross-entropy(用logsumexp) | `loss = -x_i + logsumexp(x)`,`x_i`是正確類別的logit |
| 穩定sigmoid | `x>=0`:`1/(1+exp(-x))`;`x<0`:`exp(x)/(1+exp(x))` |
| naive變異數(會抵銷,別用) | `mean(x²) - mean(x)²` |
| clip by value | 逐元素夾在`[-max_val, max_val]` |
| clip by norm | `norm = sqrt(Σg²)`;若`norm > max_norm`則每個元素乘`max_norm/norm` |
| layer norm | `mean = avg(x)`,`std = sqrt(var(x) + epsilon)`,結果 `(x - mean)/std * gamma + beta` |
| float32上限 | 約`3.4e38`,`exp(89)`已接近上限,`exp(100)`溢位成`inf` |
| float16上限 | `65504`,超過變`inf` |

</details>

## 這堂課的名詞總表

<details>
<summary>展開:名詞總表</summary>

| 英文 | 中文 | 一句話定義 |
|---|---|---|
| IEEE 754 | 浮點數標準 | 定義二進位浮點數怎麼存、怎麼四捨五入、`inf`/`nan`怎麼來的國際標準,所有現代CPU/GPU都照這套走 |
| Overflow | 上溢 | 數字大到超出能存的範圍,變成`inf` |
| Underflow | 下溢 | 數字小到存不下,直接變成`0` |
| Catastrophic cancellation | 災難性抵銷 | 兩個很接近的數相減,有效位數被吃光,剩下的都是誤差 |
| NaN | 不是數字 | 從沒有意義的運算(`0/0`、`inf-inf`、`inf*0`)產生,之後任何運算碰到它都是NaN |
| Inf | 無限大(記號) | 超出範圍或除以0產生的特殊值,不是真正的「無限」,是「已經超出電腦能存的極限」的標記 |
| Log-sum-exp | log-sum-exp技巧 | 用先提出最大值再取log的方式算`log(Σexp(x))`,避免overflow/underflow |
| Stable softmax | 穩定版softmax | 先減去`max(logits)`再取exp,結果不變但不會溢位 |
| Machine epsilon | 機器精度 | 讓`1.0+e != 1.0`成立的最小`e`;float32約`1.19e-7` |
| Gradient checking | 梯度檢查 | 用數值微分(`f(x+h)`跟`f(x-h)`的差)跟反向傳播算出的梯度對照,抓實作bug(這堂未教) |
| Mixed precision | 混合精度 | 部分運算用float16加速,部分關鍵運算留在float32,兼顧速度與穩定 |
| Loss scaling | 損失縮放 | 反向傳播前把loss放大一個倍數,避免小梯度在float16下溢成0,更新前再除回來 |
| bfloat16 | brain floating point | Google的16位元格式,指數位跟float32一樣多(範圍大),尾數位少(精度差),訓練偏好用它 |
| Gradient clipping | 梯度裁剪 | 把梯度向量的長度限制在某個上限以內,避免梯度爆炸把權重震壞 |
| Numerical gradient | 數值梯度 | 用`(f(x+h)-f(x-h))/2h`這種暴力方式近似算出的梯度,慢但可靠,用來驗證 |
| Layer Normalization | 層正規化 | 把一組數字強制標準化成平均0、標準差1,防止數值一層一層疊加爆炸 |

</details>

## 這堂課我卡住/搞混的地方(完整問答記錄,給複習用)

<details>
<summary>為什麼0.1+0.2不等於0.3</summary>

先用十進位類比:如果計算機只能存4位小數,`1/3`存成`0.3333`,那`0.3333+0.3333+0.3333=0.9999`,不是1,因為`1/3`一開始就存不準。

電腦的`0.1`在二進位裡也是存不完的數(像十進位的`1/3`),所以電腦實際存的是:

```
0.1 → 0.10000000000000000555...(多一點點)
0.2 → 0.20000000000000001110...(多一點點)
0.3 → 0.29999999999999998890...(少一點點)
```

`0.1+0.2`存的兩個「多一點點」加起來,跟`0.3`存的「少一點點」差了約`5.5e-17`,所以`==`比較是`False`。這不是bug,是有限位數的必然結果。

**規則:** 浮點數不要用`==`比,要用`abs(a-b) < 1e-9`這種容許誤差的比法(NumPy的`np.isclose`、PyTorch的`torch.allclose`做的就是這件事)。

</details>

<details>
<summary>inf是什麼、跟NaN有什麼運算規則</summary>

`inf`是infinity(無限大)的縮寫,代表「這個數已經超過電腦能存的最大值」,不是真的數學上的無限。有正負兩種:`inf`、`-inf`。

運算規則:

| 運算 | 結果 |
|---|---|
| `inf + 1` | `inf` |
| `inf * 2` | `inf` |
| `inf - inf` | `NaN`(無意義) |
| `1 / inf` | `0.0` |

`NaN`是Not a Number的縮寫,代表「這個運算沒有合理答案」。`NaN`不能用`==`比較(`nan==nan`也是`False`,IEEE754規定的),要用`math.isnan()`/`np.isnan()`/`torch.isnan()`檢查。而且NaN會傳染:只要一個NaN,加總、平均全部變NaN,神經網路的權重只要有一個變NaN,整個訓練就報廢。

</details>

<details>
<summary>為什麼inf*0不是0,而5*0是0</summary>

「任何數乘0都是0」這條規則的前提是那個數必須是**真正、有限、沒有失真嫌疑**的數字。

`inf`不是一個真正的數字,是個特殊符號,代表「無止盡變大」這件事本身。用極限角度想:`inf*0`實際上是在問「一個一直變大的數,乘上一個一直趨近0的數,答案是什麼」,這答案取決於兩者變化的速度,可能是0、可能是無限大、也可能是任何一個普通數字——沒有唯一答案,這叫「不定型」(indeterminate form)。

更貼近電腦實際狀況的理由:電腦沒辦法分辨手上這個`0`是「真的就是0」還是「原本是個非常小、小到存不下才被迫變成0的數」(underflow);`inf`同理,可能是真的無限大,也可能是「原本是個很大的數字,大到存不下才變成inf」(overflow)。

具體例子:

```
p = 1e-200 * 1e-200 = 1e-400  → 太小存不下,underflow變成 0.0
q = 1e200  * 1e200  = 1e400   → 太大存不下,overflow變成 inf

p * q 真正數學答案 = 1e-400 * 1e400 = 1     ← 應該是1
但電腦看到的是 0.0 * inf                    ← 兩邊都已經失真

如果電腦傻傻回答0,答案會是錯的(該是1);
與其冒險給一個看起來正常卻是錯的答案,不如誠實回答「不知道」(NaN)
```

而`5*0`裡的5是正常、沒有失真疑慮的有限數字,所以照舊是0。乘法規則整理:

| 算式 | 結果 |
|---|---|
| `5 × 0`、`-3.7 × 0`、`0 × 0` | `0` |
| `inf × 0` | `NaN`(唯一例外) |
| `inf × 5`(有限非0) | `inf` |
| `inf × inf` | `inf` |

只要算式裡**沒有`inf`**,乘法都照「乘0得0」走;一旦有`inf`參與,才要另外檢查是不是`inf×0`這個特殊狀況。

</details>

<details>
<summary>softmax為什麼要先減最大值,答案為什麼不變</summary>

softmax把一串分數(logits)變成機率:每個先取`exp`變正數,再除以總和。問題是分數一大,`exp`就溢位成`inf`,`inf/inf`又變成`NaN`。

解法:每個分數都先減去這組數字裡最大的那個,再做`exp`。因為每個數都減同一個常數,等於分子分母同時乘上`exp(-max)`這個常數,數學上完全約掉,答案不變。減完之後最大的一項變成`exp(0)=1`,其他都小於1,所以總和不可能溢位,也不會出現0/0。

實測驗證(小logits `[2,1,0.1]`跟大logits `[1000,1001,1002]`):

```
naive([2,1,0.1])    = [0.659, 0.242, 0.099]
stable([2,1,0.1])   = [0.659, 0.242, 0.099]   一樣(安全範圍內)
naive([1000,1001,1002])  → OverflowError(直接爆掉)
stable([1000,1001,1002]) = [0.090, 0.245, 0.665]  正常算出來
```

stable版本跟naive版本唯二的差別:多算一行`max_logit=max(logits)`,以及`exp(z)`變成`exp(z-max_logit)`,其他都一樣。

</details>

<details>
<summary>logsumexp到底是什麼、為什麼要繞這一大圈</summary>

先講它在解決什麼:cross-entropy loss需要「對正確答案那個類別的softmax機率取log,再加負號」。這是兩步疊在一起(先除法算機率、再取log),容易出事:機率如果算成0,`log(0)`是`-inf`。

logsumexp把這兩步壓縮成一步,不用真的先算出可能爆炸的機率。用具體數字`x=[2.0,1.0,0.1]`,正確答案是第0類推導:

```
loss = -log(softmax(x)_0)                              定義,代入算出0.417
     = -log( exp(x_0) / Σexp(x_j) )                    只是把softmax公式攤開寫
     = -( log(exp(x_0)) - log(Σexp(x_j)) )              用 log(a/b)=log(a)-log(b) 這條規則
     = -( x_0 - log(Σexp(x_j)) )                        因為 log(exp(x_0))=x_0(log和exp互相抵銷)
     = -x_0 + logsumexp(x)                              分配律展開負號
     = -2.0 + 2.417 = 0.417                              跟第一行算出來的一樣
```

`logsumexp(x)`就是這串算式裡「分母那坨的log值」,只要算出這一個數字,loss就直接等於`logsumexp(x)-x_i`,完全繞過容易爆炸的「先除法、再取log」路線。它本身不是機率,是算loss時的一個中繼數字。

推導本身也有兩層:先用`log(a/b)=log(a)-log(b)`把除法拆成減法,再用`log(exp(x))=x`(log和exp互為反函數,套一個再套另一個等於沒套)把其中一項化簡。

穩定版寫法跟softmax同一招:`c=max(values)`,結果是`c + log(Σexp(v-c))`,提出最大值再加回來,答案不變但不會溢位。

</details>

<details>
<summary>naive/stable兩個版本,為什麼不乾脆只留stable</summary>

實務上就是應該永遠只用stable版本。naive版本存在的唯一理由是教學:讓你先看到「不防護會怎麼爆炸」,才能理解stable版本那個「先減最大值/分正負處理」的步驟在防什麼。

stable版本 = naive版本的答案(小範圍內完全一樣) + 額外的安全網,沒有任何代價。所以PyTorch/NumPy的`softmax`、`sigmoid`、`log_softmax`內建實作本身就是stable版本,呼叫時根本不會碰到naive版本,框架已經幫你選好了。

</details>

<details>
<summary>catastrophic cancellation:兩個很接近的大數相減</summary>

naive公式算變異數是`平均(x²) - 平均(x)²`。用`data=[100000001, 100000002, 100000003]`實測:

```
真正變異數(用安全公式算) ≈ 0.667
naive公式算出來           = 0.0     ← 完全錯誤
```

原因:這三個數本身在1億左右,平方後變成約`10^16`,float64大概只有15~16位有效數字。兩個「大約10^16」的數相減,想留下的差異只有個位數,但有效數字全被1億²那部分吃光,結果算出0。

類比:兩把只能量到公分的尺,量兩個都是100公尺出頭的東西,差距只有幾公分——這幾公分低於尺的精度極限,量出來是噪音不是真正差距。

避免方法:不要把數字先變大再相減,先減掉一個共同的大數再運算,跟softmax減最大值同一種精神——避免中間值不必要地變大。

</details>

<details>
<summary>gradient clipping:clip by value vs clip by norm</summary>

Clip by value:逐個數字檢查,超過上限就直接換成上限值。`[10,20,30]`、上限15 → `[10,15,15]`。問題:20被削5、30被削15,削掉的量不一樣,梯度向量的**方向**被改變了。

Clip by norm:先算整個梯度向量的長度(L2 norm,`sqrt(各元素平方和)`),超過上限就算出縮小倍率`上限/原長度`,每個元素都乘上**同一個**倍率。`[10,20,30]`長度37.4,上限5,倍率0.134,結果`[1.34,2.67,4.01]`——比例完全沒變,方向保住,只是整體縮小。

實測梯度爆炸模擬(每步乘3.5倍):原始值從3.5一路爆炸到22518.8,裁剪後(clip by norm,上限1.0)全部被壓在1.0以內。

訓練實務標準用clip by norm(PyTorch的`clip_grad_norm_`),因為只縮小大小、不改變方向;方向代表參數該往哪調整最有效,方向被打亂等於調整方向也錯了。

</details>

<details>
<summary>bfloat16 vs float16、loss scaling</summary>

位元分配決定取捨:

```
格式        位元  指數位  尾數位  最大值      適合場合
float32     32    8      23      3.4e38      預設訓練
float16     16    5      10      65,504      推論
bfloat16    16    8      7       3.4e38      GPU/TPU訓練
```

float16犧牲範圍(最大只到6.5萬)換精度;bfloat16犧牲精度(尾數位少)換範圍(跟float32一樣大)。實測`100000`超過float16上限直接變`INF`,bfloat16還存得下。訓練偏好bfloat16,因為訓練過程數值變化劇烈,寧可精度差一點也不能讓數字動不動就爆掉。

Loss scaling:如果真的要用float16訓練,梯度常常小到直接下溢成0(下溢後那個參數就學不到東西)。解法是訓練前把loss乘一個大倍數(比如1024),讓小梯度落在float16能表示的範圍,算完梯度後再除回來,數值不受影響。動態版本:訓練途中梯度爆了(變inf)就把倍數減半,連續一段時間沒爆就加倍,自動找到合適倍數。這整套在PyTorch裡叫`torch.cuda.amp`。

</details>

<details>
<summary>Layer Normalization是什麼</summary>

問題:神經網路一層一層疊,如果每層都稍微放大數字,疊十幾層下來會像複利一樣越滾越大,最後爆掉。實測:沒有處理的話,第0層最大值2.60,第8層變成4068.94,短短8層放大超過1500倍。

Layer norm做的事:每經過一層,就把這組數字重新校正成「平均0、標準差1」。三步:算平均值 → 算變異數再開根號得標準差 → 每個數字減平均再除標準差,乘上可學習的`gamma`、加上可學習的`beta`讓網路能自己調回需要的尺度。`epsilon`是防止變異數剛好是0時除以0出錯的保底值。

實測效果:有做layer norm,第0層最大值1.31、第8層2.00,8層下來幾乎沒有成長。這是為什麼Transformer架構裡幾乎每一層都會看到它。

</details>

## 常見地雷

- 浮點數用`==`比較會踩雷,必須用`abs(a-b)<eps`
- `NaN`不能用`==`檢查(`nan==nan`也是`False`),要用`isnan()`
- naive公式算變異數(`平均(x²)-平均(x)²`)在數字偏大時會抵銷出錯,改用Welford算法
- clip by value會改變梯度方向,實務上不用它,用clip by norm

## 相關概念(跨堂連結)

- Lesson 1的L2範數(向量長度)= 這堂clip by norm裡算梯度長度用的同一個公式
- Lesson 12的tensor/broadcasting = layer norm、softmax這些運算實際上都是對tensor做的,shape要對得上
- Phase 7的attention/cross-entropy會直接用到這堂的stable softmax、logsumexp

## 面試向問題

<details>
<summary>Q1: 為什麼softmax要先減最大值?這個技巧還能用在哪裡?</summary>

因為`exp`函數對大的輸入會迅速溢位成`inf`,而`inf/inf`會變成`NaN`。減去最大值後,數學上答案不變(分子分母同乘一個常數約掉),但保證最大的一項是`exp(0)=1`,其他都小於1,不可能溢位。同樣的技巧用在log-sum-exp(cross-entropy loss的穩定寫法)、stable sigmoid(分正負處理讓exp的參數不超過0)上,是同一種「先把數字平移到安全範圍,再做危險運算」的精神。

</details>

<details>
<summary>Q2: bfloat16跟float16的差異是什麼?為什麼訓練偏好bfloat16?</summary>

兩者都是16位元,但位元分配不同:float16是5位指數+10位尾數,範圍小(最大6.5萬)但精度較高;bfloat16是8位指數+7位尾數,範圍跟float32一樣大(3.4e38)但精度較差。訓練過程中數值(尤其是梯度、中間激活值)容易忽大忽小,寧可犧牲一點精度也不能讓數字動不動就overflow變成inf/NaN,所以GPU/TPU訓練現在大多用bfloat16;float16則因為精度較高,更適合推論階段。

</details>

<details>
<summary>Q3: gradient clipping為什麼實務上用clip by norm而不是clip by value?</summary>

Clip by value逐個數字獨立夾在上下限內,各元素被削掉的量不同,會改變梯度向量的方向(方向代表參數該往哪調整最有效)。Clip by norm是先算整個梯度向量的長度,超過上限就用同一個倍率等比例縮小所有元素,方向完全不變、只是整體變小聲。所以PyTorch的`clip_grad_norm_`(業界標準做法)用的是clip by norm。

</details>

## 課程結尾理解確認題

<details>
<summary>Q1: 一個shape是(4,3)的tensor,加上一個shape是(3,)的tensor,能不能加?結果的shape是什麼?</summary>

能加。三步:1. `(3,)`左邊補1變`(1,3)`。2. 從右往左比:`3`對`3`一樣大,`4`對`1`有一邊是1。3. 是1的那邊被拉大成4,結果shape是`(4,3)`。(答對)

</details>

<details>
<summary>Q2: 為什麼算softmax之前要「先減去最大值」?這樣做為什麼答案不會變?</summary>

因為分數一大,`exp`就會溢位成`inf`,`inf/inf`又會變成`NaN`。先把每個分數都減去這組數字裡最大的那個,數學上答案完全不變(分子分母同時乘上`exp(-max)`這個常數,可以約掉),但減完之後最大的一項變成`exp(0)=1`,其餘都小於1,總和至少是1,所以不可能溢位、也不會出現0/0的情況。

</details>

<details>
<summary>Q3: inf*0和5*0為什麼結果不一樣?</summary>

`5`是正常、沒有失真嫌疑的有限數字,`5*0=0`成立。但`inf`不是真正的數字,是個代表「無止盡變大」的特殊符號,而且電腦沒辦法分辨手上這個`0`是「真的就是0」還是「原本是個很小的數字,underflow之後才變成0」,`inf`也可能是「原本是個很大的數字,overflow之後才變成inf」。如果傻傻地讓`inf*0=0`,遇到`p=1e-200*1e-200`(underflow成0)、`q=1e200*1e200`(overflow成inf)這種情況,`p*q`真正答案應該是1,卻會被錯誤地算成0。與其冒險給出一個看起來正常卻是錯的答案,電腦選擇誠實回答「不知道」,也就是`NaN`。

</details>

<details>
<summary>Q4: clip by norm跟clip by value差在哪?為什麼訓練通常用clip by norm?</summary>

Clip by value逐個數字檢查,超過上限就直接換成上限值,各元素被削掉的量不一樣多,會改變梯度向量的方向。Clip by norm先算整個梯度向量的長度(L2 norm),超過上限就算出一個縮小倍率,讓每一個元素都乘上同一個倍率,方向完全不變、只是整體等比例縮小。訓練實務標準用clip by norm,因為梯度的方向代表參數該往哪個方向調整最有效,方向被打亂等於調整方向也跟著錯了。

</details>

## 我自己手打的部分

- `practice.py`:這堂改成直接複製講解時用到的程式碼進`practice.py`(依09-29起的新規則,不再要求手打),包含`softmax_naive`、`softmax_stable`、`logsumexp_naive`、`logsumexp_stable`四個函式,以及對照小logits/大logits的實際輸出

## 今天評分

- 花費時間:64分51秒(課程建議時間約120分鐘)
- 三題課程結尾理解確認題:3/3全對,回答內容完整、無需大量提示
- 中途有幾個地方需要用具體數字/類比重講(`0.1+0.2`、`inf*0`、`log(a/b)=log(a)-log(b)`推導步驟),都在第二次講解後確認懂了

## 程式語法筆記

<details>
<summary>展開:程式語法筆記</summary>

| 語法 | 意思 |
|---|---|
| `math.exp(z)` | 算`e`的`z`次方,`z`太大(Python約709以上)會直接丟`OverflowError` |
| `max(logits)` | 找出list裡最大的數字 |
| `[e / total for e in exps]` | list comprehension,對每個元素做同一件事、組成新list |
| `math.isnan(x)` / `math.isinf(x)` | 檢查是不是NaN/Inf,不能用`==` |
| `sum(1 for v in values if math.isnan(v))` | 計算符合條件的元素個數的慣用寫法 |
| `math.sqrt(sum(g ** 2 for g in gradients))` | 算L2範數(向量長度):平方、加總、開根號 |
| `struct.pack('f', x)` / `struct.pack('e', x)` | 把Python數字打包成指定精度(float32/float16)的位元組,用來模擬不同浮點格式 |

</details>
