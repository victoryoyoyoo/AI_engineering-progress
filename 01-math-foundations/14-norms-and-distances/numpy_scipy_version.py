"""
Lesson 14: Norms and Distances——NumPy與SciPy對照
同樣的範數、距離、cosine相似度,用套件一行寫完,並跟手刻版本對答案。
    NumPy  np.linalg.norm(x, ord=...)   算各種範數
    SciPy  scipy.spatial.distance        內建各種距離(cityblock=L1、euclidean=L2、cosine=cosine距離...)
"""
import numpy as np
from scipy.spatial import distance
from scipy.stats import wasserstein_distance


def demo_norms():
    """np.linalg.norm的ord參數決定要算哪一種範數。"""
    x = np.array([3.0, 4.0])
    # ord=1:絕對值相加;ord=2(預設):平方和開根號;ord=np.inf:取最大絕對值
    print(f"L1={np.linalg.norm(x, 1)}  L2={np.linalg.norm(x)}  L-inf={np.linalg.norm(x, np.inf)}")


def demo_distances():
    """scipy.spatial.distance的函式命名跟課程不同,對照表:cityblock=L1、euclidean=L2、chebyshev=L-inf、cosine=cosine距離(不是相似度)。"""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 0.0, 6.0])
    print(f"L1 (cityblock):  {distance.cityblock(a, b)}")
    print(f"L2 (euclidean):  {distance.euclidean(a, b):.4f}")
    print(f"L-inf (chebyshev): {distance.chebyshev(a, b)}")
    # scipy的cosine回傳的是「距離」= 1 - 相似度,所以要用1減回來才是相似度
    print(f"cosine similarity: {1 - distance.cosine(a, b):.4f}")


def demo_jaccard_and_edit():
    """scipy的jaccard吃的是「布林向量」(某個元素有沒有出現),不是Python的set。edit distance scipy沒有內建。"""
    # 把兩個集合轉成布林向量:位置代表{cat,dog,fish,bird,snake},True代表這個元素在集合裡
    a = np.array([True, True, True, False, False])
    b = np.array([True, False, True, True, True])
    # scipy的jaccard回傳距離,1減掉就是相似度;課程手算是2/5=0.4
    print(f"Jaccard similarity: {1 - distance.jaccard(a, b):.4f}")


def demo_mahalanobis():
    """套件版Mahalanobis要傳「共變異數矩陣的反矩陣」,不是共變異數矩陣本身。"""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 3, 200)
    data = np.column_stack([x, 0.8 * x + rng.normal(0, 1, 200)])
    # np.cov預設一列是一個變數,我們的資料一列是一筆資料,所以要rowvar=False
    cov = np.cov(data, rowvar=False)
    inv_cov = np.linalg.inv(cov)
    mean = data.mean(axis=0)
    print(f"Mahalanobis: {distance.mahalanobis(mean, mean + [3, 2.4], inv_cov):.4f}")


def demo_wasserstein():
    """scipy的wasserstein_distance吃「位置」跟「權重」:位置是各格的座標(0,1,2,3),權重是機率。"""
    positions = [0, 1, 2, 3]
    # 課程範例:[0.5,0.5,0,0] 搬成 [0,0,0.5,0.5],預期答案是2
    print(f"Wasserstein: {wasserstein_distance(positions, positions, [0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5])}")


def demo_cosine_matrix():
    """課程Use It的寫法:一次算出一整批向量彼此的cosine相似度矩陣。"""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 8))
    # keepdims=True讓norms保持(5,1)形狀,這樣才能廣播(broadcasting)整列除下去(Lesson 12)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = X / norms
    # 正規化之後,矩陣乘自己的轉置,每一格就是兩個向量的cosine相似度
    sim = X_normalized @ X_normalized.T
    print(f"similarity matrix shape: {sim.shape}, diagonal all 1: {np.allclose(np.diag(sim), 1.0)}")


if __name__ == "__main__":
    demo_norms()
    demo_distances()
    demo_jaccard_and_edit()
    demo_mahalanobis()
    demo_wasserstein()
    demo_cosine_matrix()
