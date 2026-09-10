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

## 我自己手打的部分

(待補,等Build It實際動手的部分完成後更新)

## 今天評分

(待補,課程結束時填寫)
