# Lesson 10 - Dimensionality Reduction(降維)

## Learning Objectives 打勾清單
- [ ] 從零實作PCA:置中資料、算共變異數矩陣、特徵分解、投影
- [ ] 用explained variance ratio跟elbow method選要留幾個主成分
- [ ] 比較PCA、t-SNE、UMAP在視覺化MNIST digits(2D)上的差異跟取捨
- [ ] 用RBF kernel的Kernel PCA,分離標準PCA處理不了的非線性資料結構

(上課中,還在教PCA的`fit`方法,之後陸續補完剩下的objectives)

## 這堂課的名詞總表

| 英文 | 中文 | 一句話定義 |
|---|---|---|
| Dimensionality reduction | 降維 | 把高維資料壓縮成低維,同時盡量保留有用資訊 |
| Curse of dimensionality | 維度詛咒 | 維度越高,資料點之間的距離越趨於相同,「近」失去意義 |
| PCA (Principal Component Analysis) | 主成分分析 | 找出資料變異最大的方向,依變異量排序,只留前k個方向 |
| Eigenvector / Eigenvalue | 特徵向量/特徵值 | 特徵向量是矩陣變換後方向不變的特殊方向,特徵值是那個方向被拉長/壓縮的倍數;PCA裡特徵向量=變異方向,特徵值=那個方向的變異量大小 |
| Covariance | 共變異數 | 量化兩個特徵是不是傾向一起變大變小(正相關)、一個大一個小(負相關),還是無關(接近0) |
| Covariance matrix | 共變異數矩陣 | 每對特徵之間共變異數排成的表格,一定是對稱矩陣(cov(i,j)=cov(j,i)),對角線是各特徵自己的變異數 |
| Explained variance ratio | 解釋變異比例 | 每個主成分佔總變異量的比例,判斷保留k個維度夠不夠 |
| Reconstruction error | 還原誤差 | 降維後再還原回原始維度,跟原始資料差多少(MSE) |
| Kernel trick | 核技巧 | 不用真的算出高維座標,只算兩兩資料點的相似度(核矩陣)就能做到等效的高維運算 |
| Kernel PCA | 核PCA | 用kernel trick在隱含的高維空間做PCA,能處理非線性資料(如同心圓) |
| t-SNE | — | 保留「鄰居關係」的非線性降維法,常用來畫圖視覺化,有perplexity參數(跟Lesson 9的困惑度只是同名,概念不同) |
| UMAP | — | 類似t-SNE但更快、更保留全域結構,常用n_neighbors/min_dist |

## PCA的`fit`方法,一步一步拆解

對照`reference.py`第21-46行。

![PCA fit流水線](images/pca_fit_pipeline.png)

| 步驟 | 程式碼 | 在幹嘛 |
|---|---|---|
| 1. 置中 | `self.mean = np.mean(X, axis=0)`<br>`X_centered = X - self.mean` | 對每個特徵(每一欄)分別算平均值,再讓每個特徵減掉自己的平均。PCA只關心資料怎麼變化,不關心資料原本座落在哪 |
| 2. 算共變異數矩陣 | `cov_matrix = np.cov(X_centered, rowvar=False)` | 算出每對特徵之間共變異數排成的表格,一定對稱。`rowvar=False`是告訴numpy「每一欄才是一個特徵」(不加的話numpy會誤把每一列樣本當成變數,算出形狀完全錯誤的矩陣) |
| 3. 特徵分解 | `eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)` | 把共變異數矩陣拆解成「方向(特徵向量)」+「每個方向的變異量大小(特徵值)」。用`eigh`(而非`eig`)是因為共變異數矩陣是對稱矩陣,`eigh`是對稱矩陣專用、更快更穩定 |
| 4. 排序 | `sorted_idx = np.argsort(eigenvalues)[::-1]`<br>`eigenvalues = eigenvalues[sorted_idx]`<br>`eigenvectors = eigenvectors[:, sorted_idx]` | `eigh`預設由小到大排序,要反過來變大到小。`argsort`回傳的是「排序後的索引位置」不是排序後的值本身。`eigenvectors[:, sorted_idx]`裡`eigenvectors`每一欄才是一個完整方向,要整欄一起搬,不能拆散重排 |
| 5. 留前k個 | `self.components = eigenvectors[:, :self.n_components].T`<br>`self.explained_variance_ratio_ = eigenvalues[:k] / eigenvalues.sum()` | 只留變異量最大的前k個方向。`explained_variance_ratio_`是每個留下方向的變異量佔全部方向總變異量的比例,這是一個陣列(留k個主成分就有k個數字),把它們加總可以知道「留下這k維總共保留了多少原始資訊」 |

## 這堂課我卡住/搞混的地方(給review-queue參考)

- `axis=0` vs `axis=1`:一開始不清楚`np.mean(X, axis=0)`是在對「每一欄」分別算平均,不是把整個矩陣混在一起算。用身高體重的表格重講後理解。
- `X`為什麼可以裝下多筆多維資料:一開始沒意識到`X`是一個矩陣(類似C++的`vector<vector<double>>`),不是單一數字,靠`.shape`去理解它裝了幾筆、幾維。
- `rowvar=False`要不要加、怎麼判斷有沒有設反:一開始不知道怎麼判斷「對不對」。後來理解成:自己先從「有幾個特徵」推論出共變異數矩陣「應該」是幾乘幾,再用`.shape`印出來對照,不是用猜的。
- `eigenvectors[:, sorted_idx]`這種2D重排語法一開始看不懂,用「每一欄是一個方向,要整欄一起搬」配合小數字例子(3個特徵值+對應方向)重新帶過才清楚。
- 「PCA是多少」這種問法本身不完整:PCA是方法不是數值,真正有數值的是「保留了多少變異量(explained variance ratio加總)」「降到幾維」這些具體問題。

## 課程結尾理解確認題

**第1題:PCA前3個主成分的explained variance ratio是`[0.95, 0.03, 0.02]`,你會留幾維?為什麼?**

答案:留**1維**就夠了。關鍵判斷點不是「第2、3名彼此接近」,而是**第1名跟第2名之間的斷崖式落差**(elbow method的核心精神)。第1主成分自己就已經抓住95%的變異量,已經非常高;第2、3個主成分合計只貢獻5%,等級接近雜訊。除非應用場景明確要求保留到98%以上,才會考慮留2維(0.95+0.03=0.98)。

**第2題:同心圓資料,標準PCA為什麼分不開兩群?Kernel PCA靠什麼機制能分開?**

答案:PCA只能找**一條直線方向**上變異量最大化的方向,同心圓資料沒有明顯的直線主軸(任何方向的變異量都差不多),所以PCA找不到能區分兩群的方向。Kernel PCA透過RBF核函數,把「兩點的距離」轉換成「兩點在隱含高維空間裡的相似度」,靠這個相似度(核)矩陣重新做一次類似PCA的運算——本質上是用kernel trick把非線性問題轉成一個高維空間裡的線性問題去解,不用真的算出高維座標。

**第3題:維度詛咒具體是什麼現象?會讓哪類演算法失效,為什麼?**

答案:維度詛咒是指——維度越高,所有資料點彼此之間的距離會趨向「差不多遠」,「近」跟「遠」的差別逐漸消失。原因跟大數法則類似:距離公式是`sqrt(每個維度差的平方和)`,維度越多,等於加總越多個隨機的差值,每個點的總距離會被拉向同一個穩定值(好運跟壞運互相抵消)。這會讓**KNN、聚類**這類依賴「距離近=相似」為判斷基礎的演算法失效,因為維度一高,這個判斷基礎本身就消失了。

補充角度(跟上面是同一個根本原因的另一種呈現):要在d維空間裡維持跟1維一樣的「資料密度」,需要的樣本數會**隨維度指數成長**(1維10筆、2維100筆、3維1000筆、100維理論上要10^100筆)。現實中資料筆數不可能跟著指數成長,結果就是任何真實資料集,在高維空間裡永遠顯得極度稀疏——不管是「距離趨同」還是「密度趨近於零」,講的都是同一件事:高維空間比你以為的大太多太多,資料填不滿它。

## 我自己手打的部分

`fit`方法(核心層)逐行講解過,對照練習放進`practice.py`。`transform`/`inverse_transform`/`reconstruction_error`/`KernelPCA`(理解層)講邏輯+demo驗證過,沒有逐行手刻但已同步進`practice.py`。`reference.py`依3層優先度(核心→理解層→之後有空再補)重新排版過。curse of dimensionality用實際demo(1000個隨機點在不同維度下的距離比例)講解過。

t-SNE / UMAP的內部演算法細節、課程的3個Exercises,沒有實際帶過,記進review-queue。

## 今天評分

理解程度:PCA的`fit`5步驟(核心層)逐步拆解、反覆用小數字例子確認過,`axis`、矩陣形狀、`argsort`重排這幾個一開始卡住的地方後來都釐清;`transform`/`inverse_transform`/`reconstruction_error`/Kernel PCA(理解層)講邏輯+demo驗證過;curse of dimensionality用實際跑出來的距離比例數字帶過,3題理解確認題都答對(第1題判斷邏輯有小修正,第3題提示後自己想到大數法則的連結)
效率:過程中反覆用「拆更小的具體數字例子」處理卡住的地方(np.mean的axis、共變異數矩陣形狀判斷、eigenvectors重排語法),比起直接看程式碼更有效
完成度:4個Learning Objectives裡,PCA從零實作、explained variance ratio/elbow method、Kernel PCA分離非線性資料 這3個做到;t-SNE/UMAP的MNIST視覺化比較沒有實際帶過,記進review-queue。另外這堂課也順手把Lesson 1-9的`reference.py`補上3層優先度標記並依標記重新排版、`practice.py`同步workflow標準化、`reference.py`分層排序標準化,都存成往後每堂課固定會做的習慣
花費時間:1小時31分鐘
