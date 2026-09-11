# Notes Index

## Phase 1 — Math Foundations

| # | 課程 | 這堂在教什麼(一句話) | 筆記 |
|---|---|---|---|
| 1 | Linear Algebra Intuition | vector就是一串數字,同時代表方向跟長度,加減法是箭頭頭尾相接,純量乘法只改變長度 | [notes.md](01-math-foundations/01-linear-algebra-intuition/notes.md) |
| 2 | Vectors, Matrices & Operations | 矩陣可以想成一台機器:吃進n維向量,吐出m維向量,每個輸出數字是輸入向量跟矩陣某一列做內積 | [notes.md](01-math-foundations/02-vectors-matrices-operations/notes.md) |
| 3 | Matrix Transformations | 旋轉/縮放/切斜/鏡射矩陣都是「對空間做某種固定操作」的說明書,各自固定的數字排法 | [notes.md](01-math-foundations/03-matrix-transformations/notes.md) |
| 4 | Calculus for ML | 導數告訴你「稍微調整一個東西,結果會變多少」,梯度是每個變數偏導數收集成一個向量,指向「往上爬最快」的方向 | [notes.md](01-math-foundations/04-calculus-for-ml/notes.md) |
| 5 | Chain Rule & Automatic Differentiation | 反向傳播每一站永遠只做同一件事:這一站的梯度 = 上游梯度 × 這一站自己的局部導數 | [notes.md](01-math-foundations/05-chain-rule-and-autodiff/notes.md) |
| 6 | Probability and Distributions | PDF某一點的值是密度不是機率,連續變數在單一點的機率永遠是0,要積分一段區間才有意義 | [notes.md](01-math-foundations/06-probability-and-distributions/notes.md) |
| 7 | Bayes' Theorem | 貝氏定理只是條件機率的兩種寫法互換角度而已:後驗 = 似然 × 先驗 / 證據 | [notes.md](01-math-foundations/07-bayes-theorem/notes.md) |
| 8 | Optimization | 三種optimizer是同一問題的三種答案:梯度下降(只看現在)、momentum(記住過去方向)、Adam(momentum+每個權重自己的步伐) | [notes.md](01-math-foundations/08-optimization/notes.md) |
| 9 | Information Theory | Information content量單一事件的驚訝程度,entropy是整個分布的平均驚訝程度,是不確定性的下限 | [notes.md](01-math-foundations/09-information-theory/notes.md) |
| 10 | Dimensionality Reduction | PCA的fit流程:置中資料→算共變異數矩陣→特徵分解→由大到小排序→留變異量最大的前k個方向 | [notes.md](01-math-foundations/10-dimensionality-reduction/notes.md) |

