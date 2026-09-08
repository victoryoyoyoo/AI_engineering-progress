# Lesson 9 - Information Theory(資訊理論)

## Learning Objectives 打勾清單
- [x] 從零算出entropy、cross-entropy、KL divergence,解釋三者的關係
- [x] 推導為什麼「最小化cross-entropy loss」等於「最大化log-likelihood」
- [x] 算出特徵與目標之間的mutual information,用來排序特徵重要性 ⚠️(核心MI公式對過+跑過demo,`feature_selection_mi_demo`那個完整排序範例沒有實際帶過,記review-queue)
- [x] 解釋perplexity是「模型實際上在幾個選項裡猶豫」

## 這堂課的名詞總表

| 英文 | 中文 | 一句話定義 |
|---|---|---|
| Information content | 資訊量/驚訝程度 | 單一事件的驚訝程度:-log(p(x)) |
| Entropy | 熵 | 整個分布的平均驚訝程度,不確定性的下限 |
| Cross-entropy | 交叉熵 | 用模型猜的Q去描述真實分布P,平均要花多少bit(=分類loss) |
| KL divergence | KL散度 | 用Q取代P多浪費的bit數,不對稱,不是真正的距離 |
| Mutual information | 互資訊 | 知道一個變數,能讓你對另一個變數的不確定性減少多少 |
| One-hot | 獨熱編碼 | 正確答案位置是1,其他全部是0的向量表示法 |
| Perplexity | 困惑度 | 交叉熵取指數,模型平均像在幾個選項間猶豫 |
| Softmax | — | 把任意實數向量轉成合法機率分布(總和=1) |
| Negative log-likelihood | 負對數概似 | 跟cross-entropy loss數學上完全相同 |
| Bits / Nats | — | log base 2 / log base e 的單位差別,PyTorch預設用nats |

---

### Information Content(驚訝程度)

```
I(x) = -log(p(x))
```

機率越低的事件,資訊量(驚訝程度)越大;必然發生的事(p=1)資訊量是0,因為你早就知道會發生,沒有帶來任何新資訊。

| 事件 | 機率 | 驚訝程度(bits) |
|---|---|---|
| 公平銅板正面 | 0.5 | 1.0 |
| 骰子擲到6 | 0.167 | 2.58 |
| 千分之一的事件 | 0.001 | 9.97 |
| 必然發生的事 | 1.0 | 0.0 |

### Entropy(熵)——整個分布的平均驚訝程度

```
H(P) = -sum( p(x) * log(p(x)) )   對所有x加總
```

把每個結果的驚訝程度(`-log(p)`)按機率加權平均。

```
公平銅板:    H = -(0.5*log2(0.5) + 0.5*log2(0.5)) = 1.0 bit
偏態銅板(99%正面): H = -(0.99*log2(0.99) + 0.01*log2(0.01)) = 0.08 bits
```

![Entropy示意圖](images/entropy_demo.png)

**規律**:分布越平(所有結果機率接近),熵越高;分布越集中(某個結果幾乎壟斷),熵越低。選項數越多,熵的上限也越高(6面骰子熵上限是log2(6)=2.58,比銅板的log2(2)=1還高)。熵衡量的是一個分布「本質上有多少不確定性」,是壓不掉的下限。

### Cross-Entropy(交叉熵)——你每天在用的loss function

```
H(P, Q) = -sum( p(x) * log(q(x)) )
```

P是真實分布(標籤),Q是模型猜的分布。Q跟P一樣時,交叉熵=熵本身;Q跟P差越多,交叉熵越大。

**手算例子**(true=[0.7,0.2,0.1], Q=[0.6,0.25,0.15]):

```
H(P,Q) = -[0.7*log2(0.6) + 0.2*log2(0.25) + 0.1*log2(0.15)]
       = -[0.7*(-0.737) + 0.2*(-2.0) + 0.1*(-2.737)]
       = 1.19 bits
```

**分類問題的簡化版**:P是one-hot向量(真實類別機率=1,其他=0)時,公式簡化成:

```
H(P, Q) = -log(q(true_class))
```

這就是`CrossEntropyLoss()`底層在算的東西。模型對正確答案猜得機率越高,loss越小;猜得越離譜,loss爆炸性變大。

![Cross-entropy對照圖](images/cross_entropy_demo.png)

上圖:好模型(橘色跟藍色柱子接近) → 交叉熵低(1.19 bits);壞模型(橘色跟藍色差很多) → 交叉熵高(3.02 bits)。**柱子越接近,交叉熵越低,代表模型猜得越準。**

### KL Divergence(KL散度)——多浪費了多少bit

```
D_KL(P || Q) = H(P, Q) - H(P)
```

`H(P)`是理論最低成本(用P自己編碼P的下限)。`H(P,Q)`是用不完美的Q去編碼P,實際要花的成本,一定≥H(P)。兩者的差,就是「猜不準多浪費的部分」。

```
H(true) = 1.1568 bits
H(true, good) = 1.1896 bits
KL(true || good) = 1.1896 - 1.1568 = 0.0328 bits   ← 猜得準,浪費很少
```

**KL散度不對稱**:`D_KL(P||Q) != D_KL(Q||P)`,驗證過(P=[0.9,0.1], Q=[0.5,0.5]時,KL(P‖Q)=0.531,KL(Q‖P)=0.737,兩個方向不一樣),所以KL不是真正的距離度量。

**訓練時的意義**:H(P)在訓練過程中是常數(標籤資料不變)。想成「總成本 = 固定成本 + 變動成本」——固定成本(H(P))不變,想壓低總成本(H(P,Q))就只能壓變動成本(KL)。所以「最小化交叉熵」跟「最小化KL散度」是同一個優化問題,本質上是把模型的Q推向真實的P。

### 六個概念怎麼串起來(整堂課最關鍵的一張圖)

![Information theory概念串連圖](images/info_theory_summary.png)

`H(P)`(理論下限) → `H(P,Q)`(用Q實際要付出的成本) → 兩者差是`D_KL(P‖Q)`(猜不準浪費掉的部分) → 因為H(P)是常數,最小化交叉熵=最小化KL散度 → `Perplexity`是把交叉熵換算成更直覺的「困惑程度」數字。互資訊`I(X;Y)`是獨立的一支,不在這條鏈上。

### Cross-Entropy = Negative Log-Likelihood(MLE推導)

對N筆訓練樣本(真實類別y_i),假設樣本間獨立:

```
概似(Likelihood)     = product( q(y_i) )         所有樣本機率的連乘
對數概似(Log-likelihood) = sum( log(q(y_i)) )       連乘取log變連加
負對數概似(NLL)       = -sum( log(q(y_i)) )        取負號、加總
```

而`-log(q(y_i))`剛好就是`cross_entropy_loss`每一筆的值。所以NLL加總起來(取平均)就是交叉熵loss。**最小化交叉熵 = 最小化NLL = 最大化訓練資料的概似**,三件事是同一件事的不同講法。

demo驗證(1000筆隨機樣本):
```
Cross-entropy loss:      1.400910 nats
Neg log-likelihood:      1.400910 nats
Difference:               0.00e+00   ← 完全相同
```

### Mutual Information(互資訊)——知道X能讓你對Y少猜多少

```
I(X;Y) = H(X) - H(X|Y)
```

`H(X)`是原本對X的不確定性,`H(X|Y)`是已知Y後對X剩下的不確定性,差值就是Y幫你消除掉的不確定性。

![Mutual information Venn圖](images/mutual_info_venn.png)

- X、Y獨立(圖左,兩圓不重疊) → `I(X;Y)=0`
- X、Y部分相關(圖中,部分重疊) → 重疊面積就是`I(X;Y)`
- X完全決定Y(圖右,完全重疊) → `I(X;Y) = H(X) = H(Y)`

demo驗證:
```
Independent:   MI = 0.0000 bits
Dependent:     MI = 0.5310 bits
```

**跟Pearson相關係數的差別**:Pearson(-1到1)只抓得到「線性關係」,遇到非線性(比如U型)關係會誤判成無關;互資訊能抓到任何形式的統計關聯,不管線性非線性都算得出來。在特徵選擇上,MI分數越高代表這個特徵越有預測力,MI趨近0代表基本上是雜訊。

### Perplexity(困惑度)——模型「實際上在幾個選項間猶豫」

```
Perplexity = e^(交叉熵)   (nats)   或   2^(交叉熵)   (bits)
```

perplexity=50代表模型平均起來的猶豫程度,像是要從**50個選項裡均勻亂猜**一樣困惑。perplexity越低,模型越有把握。

**跟accuracy的差別**:accuracy是非黑即白(對/錯);perplexity量的是「機率分布有多集中」,同樣答對的兩個模型,一個給正確答案0.9的機率(很篤定),一個只給0.34(矇對的),accuracy一樣但perplexity差很多。perplexity比accuracy更嚴格,能捕捉模型對整個機率分布的掌握程度。

demo:未訓練的隨機模型,vocab_size=50,實際跑出來perplexity=81.23(比50還高,代表**比隨機亂猜還爛**——驗證了「perplexity < vocab size才代表比隨機好」這個判準)。

## 這堂課的總結

資訊理論的六個概念,其實是同一套邏輯的不同切面:Information content量單一事件的驚訝程度,Entropy把它平均成整個分布的不確定性下限,Cross-entropy是「用不完美的模型去猜」實際要付出的成本(=loss function),KL divergence是這中間多浪費的部分,Perplexity把交叉熵換算成更直覺的「困惑選項數」。因為標籤的熵H(P)訓練時是常數,最小化交叉熵、最小化KL散度、最大化log-likelihood,三件事在數學上是同一個優化問題。Mutual information是獨立的一支,量兩個變數共享了多少資訊,在特徵選擇上比Pearson相關係數更全面(抓得到非線性關係)。

## 我自己手打的部分

這堂課走Top-down策略(數學/理論課,不強制手打)。核心8個函式(`information_content`、`entropy`、`cross_entropy`、`kl_divergence`、`mutual_information`、`softmax`、`cross_entropy_loss`、`perplexity`)都對照公式讀過、跑過demo驗證數字一致,放進`practice.py`。`conditional_entropy`、`joint_entropy`、`label_smoothing_demo`、`feature_selection_mi_demo`這幾個屬於延伸內容(教學目標沒有明確要求),沒有實際帶過,記進review-queue。

## 今天評分

理解程度:6個核心概念都逐一講解+確認過,post測驗3題用問答方式都答對(中間有補強one-hot、perplexity vs accuracy的差別、Pearson vs Poisson的混淆)。整體理解紮實,沒有像Lesson 8那樣退步
效率:延續「每教完一個概念就停下確認」的節奏,這次retention明顯比Lesson 8好,沒有出現連續好幾個「不知道」的狀況
完成度:4個Learning Objectives都完成,MI的完整特徵排序demo跳過記review-queue;新增了配圖(entropy/cross-entropy/概念串連圖/互資訊Venn圖),也把Lesson 1-8的舊筆記回頭補了圖並推上GitHub
花費時間:56分4秒
