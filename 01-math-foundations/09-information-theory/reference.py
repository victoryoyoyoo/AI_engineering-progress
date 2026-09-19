"""
Lesson 9: Information Theory
熵、交叉熵、KL散度、互資訊——loss function底層的數學語言

檔案結構:
    🔴 單一事件資訊量、熵、交叉熵、KL散度、互資訊、softmax、分類用的交叉熵損失、困惑度
    🟡 負對數概似、條件熵、聯合熵、label smoothing示範、用互資訊做特徵選擇
名詞對照:
    bit  = 以2為底的log算出的資訊單位
    nat  = 以e為底的log算出的資訊單位(深度學習框架預設用nat)
    1 nat = 1/ln(2) bit,兩者只差一個常數倍
"""
import math
import random


# === 🔴 ===
def information_content(p, base=2):
    """單一事件的資訊量(驚訝程度): I(x) = -log(p(x))
    機率越低越驚訝,機率為1(必然發生)資訊量為0。
    """
    # 機率<=0的事件不可能發生,驚訝程度定義成無限大;同時避開math.log(0)會丟出錯誤
    if p <= 0:
        return float('inf')
    # 必然發生的事(p=1)沒有任何資訊量;也避免-0.0這種負零的顯示問題
    if p >= 1:
        return 0.0
    # 換底公式:log_base(p) = ln(p) / ln(base)
    # 例:p=0.5,base=2 -> -ln(0.5)/ln(2) = 1 bit(丟公平硬幣得到正面,驚訝1 bit)
    #     p=0.001 -> 約9.97 bit(千分之一的事件很驚訝)
    return -math.log(p) / math.log(base)


def entropy(probs, base=2):
    """熵 = 整個分布的平均驚訝程度: H(P) = -sum(p(x) * log(p(x)))
    公平硬幣熵最大(1 bit),越偏態的分布熵越低(越好預測)。
    """
    # 每個事件的驚訝程度,乘上該事件發生的機率,再全部加總 = 期望的驚訝程度
    # generator expression放在sum()裡,不用先建立完整的list
    # 尾端的 "if p > 0" 濾掉機率0的事件:0*log(0)在極限下定義為0,直接略過
    return sum(
        p * information_content(p, base)
        for p in probs if p > 0
    )


def cross_entropy(p, q, base=2):
    """交叉熵: H(P,Q) = -sum(p(x) * log(q(x)))
    用「模型猜的分布Q」去編碼「真實分布P」的事件,平均要付出多少bit。
    Q跟P越像,交叉熵越接近H(P);Q跟P差越多,交叉熵越大。
    """
    total = 0.0
    # p是真實分布、q是模型預測的分布,兩個list同一位置對應同一個事件
    for pi, qi in zip(p, q):
        # 真實機率為0的事件對期望值沒有貢獻,略過
        if pi > 0:
            # 真實會發生、模型卻認為機率是0:log(0)是負無限大,交叉熵變成無限大
            # 這正是分類模型過度自信時loss暴增的原因
            if qi <= 0:
                return float('inf')
            # 權重是「真實機率」pi,驚訝程度是用「模型機率」qi算的:-log(qi)
            total += pi * (-math.log(qi) / math.log(base))
    return total


def kl_divergence(p, q, base=2):
    """KL散度: D_KL(P||Q) = H(P,Q) - H(P)
    用Q取代P「多付出」的bit數,不對稱(D_KL(P||Q) != D_KL(Q||P))。
    """
    # 交叉熵 = 熵 + KL散度 -> KL = 交叉熵 - 熵
    # H(P)是資料本身固定的下限,訓練時只有KL能被模型改善,所以最小化交叉熵等於最小化KL
    return cross_entropy(p, q, base) - entropy(p, base)


def mutual_information(joint_probs, base=2):
    """互資訊: I(X;Y) = sum sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    X、Y獨立時MI=0;越相關MI越大。特徵選擇時可以用來排序哪個特徵有用。
    """
    # joint_probs是二維list:joint_probs[i][j] = P(X=i, Y=j),全部加起來為1
    rows = len(joint_probs)
    cols = len(joint_probs[0])

    # 邊際分布:把不要的那個變數加總掉
    # margin_x[i] = P(X=i) = 第i列所有欄加總;margin_y[j] = P(Y=j) = 第j欄所有列加總
    margin_x = [sum(joint_probs[i][j] for j in range(cols)) for i in range(rows)]
    margin_y = [sum(joint_probs[i][j] for i in range(rows)) for j in range(cols)]

    mi = 0.0
    for i in range(rows):
        for j in range(cols):
            pxy = joint_probs[i][j]
            # 三個機率都必須>0才能取log、才能當分母
            if pxy > 0 and margin_x[i] > 0 and margin_y[j] > 0:
                # 比較「實際一起出現的機率」跟「假設獨立時會有的機率p(x)*p(y)」
                # 兩者相等,比值=1,log=0,沒有貢獻;實際比獨立時多很多,貢獻就大
                mi += pxy * math.log(pxy / (margin_x[i] * margin_y[j])) / math.log(base)
    return mi


def softmax(logits):
    """把任意實數向量(logits)轉成合法機率分布(總和=1,每個都>=0)"""
    # 先減掉最大值再exp:結果數學上完全相同,但最大的指數變成exp(0)=1,不會溢位
    # (float32下exp(89)就會變inf,Lesson 13細講這個技巧)
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    # 每項除以總和做正規化,加起來剛好是1
    return [e / total for e in exps]


def cross_entropy_loss(true_class, logits):
    """分類任務的交叉熵損失: P是one-hot(真實類別機率=1,其他=0)
    公式簡化成: H(P,Q) = -log(q(true_class)) —— 只看模型給真實類別的機率
    """
    # 先用softmax把模型輸出的logits變成機率分布Q
    probs = softmax(logits)
    # one-hot的P只有真實類別那一項不為0(等於1),其餘各項乘上0消失,
    # 所以整個加總只剩 -log(probs[true_class])
    # 模型給真實類別的機率越高,loss越接近0;機率越低,loss越大
    return -math.log(probs[true_class])


# === 🔴 ===
def perplexity(avg_cross_entropy, base="e"):
    """困惑度 = e^(交叉熵) 或 2^(交叉熵)
    語言模型困惑度50,代表平均起來像是要從50個選項裡均勻亂猜一樣困惑。
    """
    # 底數必須跟交叉熵用的log底數配對:用nat(ln)算的交叉熵配e,用bit(log2)算的配2
    if base == "e":
        return math.exp(avg_cross_entropy)
    # ** 是次方運算,2 ** x 就是2的x次方
    return 2 ** avg_cross_entropy


# === 🟡 ===
def negative_log_likelihood(labels, all_logits):
    """負對數概似(NLL)——跟cross_entropy_loss取平均後數學上完全相同"""
    # labels[k]是第k筆資料的真實類別,all_logits[k]是模型對第k筆的輸出
    # 每筆算一次cross_entropy_loss,加總後除以筆數 = 平均損失
    # 最小化平均交叉熵 = 最大化資料的概似(likelihood),兩種說法是同一件事
    return sum(
        cross_entropy_loss(label, logits)
        for label, logits in zip(labels, all_logits)
    ) / len(labels)


# === 🟡 ===
def conditional_entropy(joint_probs, base=2):
    """條件熵 H(Y|X):已知X之後,Y還剩下多少不確定性"""
    rows = len(joint_probs)
    cols = len(joint_probs[0])

    # 只需要X的邊際分布(第i列加總),用來把聯合機率換算成條件機率
    margin_x = [sum(joint_probs[i][j] for j in range(cols)) for i in range(rows)]

    h_yx = 0.0
    for i in range(rows):
        for j in range(cols):
            pxy = joint_probs[i][j]
            if pxy > 0 and margin_x[i] > 0:
                # 條件機率 P(Y=j | X=i) = P(X=i, Y=j) / P(X=i)
                p_y_given_x = pxy / margin_x[i]
                # H(Y|X) = -sum p(x,y) * log p(y|x);權重用聯合機率p(x,y)
                # -= 是「減掉」,配上log(<=1)為負,合起來累加正數
                h_yx -= pxy * math.log(p_y_given_x) / math.log(base)
    return h_yx


def joint_entropy(joint_probs, base=2):
    """聯合熵 H(X,Y):X和Y一起看的熵,H(X,Y) <= H(X)+H(Y),等號在獨立時成立"""
    total = 0.0
    # 二維list逐列、逐格走過,不需要索引i、j,直接迴圈取值
    for row in joint_probs:
        for pxy in row:
            if pxy > 0:
                # 把(X,Y)的每一種組合當成一個事件,套用一般熵公式 -sum p*log(p)
                total -= pxy * math.log(pxy) / math.log(base)
    return total


def label_smoothing_demo():
    """label smoothing示範:把one-hot硬標籤換成有一點不確定性的軟標籤,
    從資訊理論角度看就是「刻意提高target的熵」,防止模型過度自信。
    """
    print()
    print("=" * 60)
    print("LABEL SMOOTHING AND CROSS-ENTROPY")
    print("=" * 60)

    num_classes = 4
    true_class = 2
    logits = [1.0, 0.5, 3.0, 0.2]
    probs = softmax(logits)

    # 硬標籤(one-hot):真實類別是1,其他是0
    hard_target = [0.0] * num_classes
    hard_target[true_class] = 1.0

    # epsilon是平滑強度:0代表不平滑,越大代表把越多機率分給其他類別
    epsilons = [0.0, 0.05, 0.1, 0.2]
    print(f"\n  Logits:  {logits}")
    # f-string裡再套一層f-string,用join把list裡每個機率格式化成4位小數後接起來
    print(f"  Softmax: [{', '.join(f'{p:.4f}' for p in probs)}]")
    print(f"  True class: {true_class}")
    print()

    for eps in epsilons:
        # 軟標籤:每個類別先平分eps/num_classes,
        # 真實類別再額外加上(1-eps),所以總和仍是1
        # 例:eps=0.2,4類 -> 其他類別各0.05,真實類別 0.8+0.05=0.85
        soft_target = [eps / num_classes] * num_classes
        soft_target[true_class] = (1 - eps) + eps / num_classes

        # 用自然對數(base=math.e)算,跟PyTorch的交叉熵單位(nat)一致
        ce = cross_entropy(soft_target, probs, base=math.e)
        target_entropy = entropy(soft_target, base=math.e)
        label = "hard" if eps == 0.0 else f"eps={eps}"
        # {label:>8s}是靠右對齊、佔8格寬的字串
        print(f"  {label:>8s}  target={[f'{t:.3f}' for t in soft_target]}  "
              f"H(target)={target_entropy:.4f}  CE={ce:.4f}")

    print()
    print("  Higher epsilon -> higher target entropy -> acts as regularization")


def feature_selection_mi_demo():
    """用互資訊排序特徵重要性:跟目標關係越大,MI越高,雜訊特徵MI趨近0"""
    print()
    print("=" * 60)
    print("FEATURE SELECTION VIA MUTUAL INFORMATION")
    print("=" * 60)

    # 固定亂數種子,每次執行都產生同樣的資料,結果可重現
    random.seed(42)
    n = 200

    # 目標:隨機的0/1標籤
    target = [random.choice([0, 1]) for _ in range(n)]

    # 造四種特徵,跟目標的關係由強到無:
    features = {}
    # ^ 是XOR:t ^ 1 會把0變1、1變0;以10%機率翻轉目標 -> 90%跟目標一致,訊號很強
    features["strong_signal"] = [t ^ (1 if random.random() < 0.1 else 0) for t in target]
    # 35%機率翻轉 -> 只有65%一致,訊號弱
    features["weak_signal"] = [t ^ (1 if random.random() < 0.35 else 0) for t in target]
    # 完全獨立的隨機0/1,沒有任何資訊
    features["noise"] = [random.choice([0, 1]) for _ in range(n)]
    # 常數特徵:永遠是0,不可能提供資訊
    features["constant"] = [0] * n

    print(f"\n  Samples: {n}")
    print(f"  Target balance: {sum(target)}/{n - sum(target)}")
    print()

    mi_scores = []
    for name, feat in features.items():
        # 2x2的計數表:joint[特徵值][目標值] = 這種組合出現幾次
        joint = [[0, 0], [0, 0]]
        for f, t in zip(feat, target):
            joint[f][t] += 1
        # 次數除以總筆數,得到聯合機率分布,交給mutual_information
        joint_p = [[c / n for c in row] for row in joint]
        mi = mutual_information(joint_p, base=2)
        mi_scores.append((name, mi))

    # 依MI由大到小排序;key=lambda x: x[1]代表用tuple的第2個元素(MI)當排序依據
    mi_scores.sort(key=lambda x: x[1], reverse=True)
    print("  Feature MI ranking:")
    for name, mi in mi_scores:
        # 用#字元畫簡易長條圖,MI乘200再取整數當長度
        bar = "#" * int(mi * 200)
        print(f"    {name:>16s}  MI = {mi:.4f} bits  {bar}")

    print()
    print("  Strong signal has highest MI. Noise and constant have ~0.")


if __name__ == "__main__":

    # ---------- 1. 單一事件的資訊量:機率越低、越驚訝 ----------
    print("=" * 60)
    print("INFORMATION CONTENT (SURPRISE)")
    print("=" * 60)

    events = [
        ("Fair coin heads", 0.5),
        ("Rolling a 6", 1 / 6),
        ("1-in-1000 event", 0.001),
        ("Certain event", 1.0),
    ]
    for name, p in events:
        # {p:<8.4f}是靠左對齊、佔8格寬、小數4位
        print(f"  {name:20s}  p={p:<8.4f}  surprise={information_content(p):.4f} bits")

    # ---------- 2. 熵:分布越平均越難預測,熵越大 ----------
    print()
    print("=" * 60)
    print("ENTROPY")
    print("=" * 60)

    distributions = {
        "Fair coin": [0.5, 0.5],
        "Biased coin (99/1)": [0.99, 0.01],
        "Fair die (6 sides)": [1 / 6] * 6,
        "Loaded die": [0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
    }
    for name, probs in distributions.items():
        print(f"  {name:25s}  H = {entropy(probs):.4f} bits")

    # ---------- 3. 交叉熵與KL散度:模型分布跟真實分布差多遠 ----------
    print()
    print("=" * 60)
    print("CROSS-ENTROPY AND KL DIVERGENCE")
    print("=" * 60)

    # 真實分布,以及一個接近的好模型、一個很離譜的壞模型
    true_dist = [0.7, 0.2, 0.1]
    good_model = [0.6, 0.25, 0.15]
    bad_model = [0.1, 0.1, 0.8]

    h_true = entropy(true_dist)
    ce_good = cross_entropy(true_dist, good_model)
    ce_bad = cross_entropy(true_dist, bad_model)
    kl_good = kl_divergence(true_dist, good_model)
    kl_bad = kl_divergence(true_dist, bad_model)

    print(f"  True distribution:    {true_dist}")
    print(f"  Good model:           {good_model}")
    print(f"  Bad model:            {bad_model}")
    print()
    print(f"  H(true):              {h_true:.4f} bits")
    print(f"  H(true, good):        {ce_good:.4f} bits")
    print(f"  H(true, bad):         {ce_bad:.4f} bits")
    print(f"  KL(true || good):     {kl_good:.4f} bits")
    print(f"  KL(true || bad):      {kl_bad:.4f} bits")
    print()
    # 驗證恆等式 H(P,Q) = H(P) + KL(P||Q):左右兩邊數字應該一樣
    print(f"  Verify: H(P,Q) = H(P) + KL(P||Q)")
    print(f"  Good: {h_true:.4f} + {kl_good:.4f} = {h_true + kl_good:.4f}  (CE = {ce_good:.4f})")
    print(f"  Bad:  {h_true:.4f} + {kl_bad:.4f} = {h_true + kl_bad:.4f}  (CE = {ce_bad:.4f})")

    # ---------- 4. KL散度不對稱:P||Q跟Q||P算出來不一樣 ----------
    print()
    print("=" * 60)
    print("KL DIVERGENCE IS NOT SYMMETRIC")
    print("=" * 60)

    p = [0.9, 0.1]
    q = [0.5, 0.5]
    print(f"  P = {p},  Q = {q}")
    print(f"  KL(P || Q) = {kl_divergence(p, q):.4f} bits")
    print(f"  KL(Q || P) = {kl_divergence(q, p):.4f} bits")
    print(f"  They differ because KL is not a true distance metric.")

    # ---------- 5. 分類任務的交叉熵損失 ----------
    print()
    print("=" * 60)
    print("CROSS-ENTROPY LOSS FOR CLASSIFICATION")
    print("=" * 60)

    logits = [2.0, 1.0, 0.1]
    true_class = 0
    probs = softmax(logits)
    loss = cross_entropy_loss(true_class, logits)

    print(f"  Logits:       {logits}")
    # 這裡的迴圈變數p只存在於generator expression內部,不會覆蓋上面的list p
    print(f"  Softmax:      [{', '.join(f'{p:.4f}' for p in probs)}]")
    print(f"  True class:   {true_class}")
    print(f"  CE loss:      {loss:.4f} nats")
    print(f"  Perplexity:   {perplexity(loss):.2f}")

    print()
    print("  Trying different true classes with same logits:")
    # 同一組logits,真實類別換成不同的,看loss怎麼隨「模型給該類別的機率」變化
    for c in range(3):
        l = cross_entropy_loss(c, logits)
        print(f"    Class {c}: loss={l:.4f}  prob={probs[c]:.4f}")

    # ---------- 6. 交叉熵 = 負對數概似(兩種說法是同一件事) ----------
    print()
    print("=" * 60)
    print("CROSS-ENTROPY = NEGATIVE LOG-LIKELIHOOD")
    print("=" * 60)

    random.seed(42)
    n_samples = 1000
    n_classes = 3
    # 隨機造1000筆資料:真實類別是0到2的整數(randint兩端都包含)
    labels = [random.randint(0, n_classes - 1) for _ in range(n_samples)]
    # 每筆資料的logits是3個常態分布(平均0、標準差1)的隨機數;巢狀list comprehension
    all_logits = [[random.gauss(0, 1) for _ in range(n_classes)] for _ in range(n_samples)]

    ce_avg = negative_log_likelihood(labels, all_logits)
    # 換一種寫法直接算:對每筆取「模型給真實類別的機率」的log,加總,取負號,除以筆數
    nll_avg = -sum(
        math.log(softmax(lg)[lb])
        for lb, lg in zip(labels, all_logits)
    ) / n_samples

    print(f"  Samples:               {n_samples}")
    print(f"  Cross-entropy loss:    {ce_avg:.6f} nats")
    print(f"  Neg log-likelihood:    {nll_avg:.6f} nats")
    # {:.2e}是科學記號;兩種算法只差浮點誤差,大約1e-16
    print(f"  Difference:            {abs(ce_avg - nll_avg):.2e}")
    print(f"  They are identical. Minimizing CE = maximizing likelihood.")

    # ---------- 7. 互資訊:三種依賴程度的聯合分布 ----------
    print()
    print("=" * 60)
    print("MUTUAL INFORMATION")
    print("=" * 60)

    # 2x2聯合機率表,每一份總和都是1
    independent = [[0.25, 0.25], [0.25, 0.25]]  # X、Y完全獨立
    dependent = [[0.45, 0.05], [0.05, 0.45]]    # X、Y強相關(對角線機率大)
    partial = [[0.3, 0.2], [0.1, 0.4]]          # 部分相關

    print(f"  Independent:   MI = {mutual_information(independent):.4f} bits")
    print(f"  Dependent:     MI = {mutual_information(dependent):.4f} bits")
    print(f"  Partial:       MI = {mutual_information(partial):.4f} bits")

    # ---------- 8. bit與nat的換算 ----------
    print()
    print("=" * 60)
    print("BITS VS NATS")
    print("=" * 60)

    fair_coin = [0.5, 0.5]
    print(f"  Fair coin entropy:")
    print(f"    In bits (log2): {entropy(fair_coin, base=2):.4f}")
    print(f"    In nats (ln):   {entropy(fair_coin, base=math.e):.4f}")
    # log2(e)約1.4427:1 nat = 1.4427 bit;倒過來1 bit = 0.6931 nat
    print(f"    1 bit = {1 / math.log2(math.e):.4f} nats")
    print(f"    1 nat = {math.log2(math.e):.4f} bits")

    # ---------- 9. 語言模型的困惑度 ----------
    print()
    print("=" * 60)
    print("PERPLEXITY IN LANGUAGE MODELS")
    print("=" * 60)

    random.seed(123)
    vocab_size = 50
    sequence_length = 100
    # 假設一段100個token的文本,每個token的正確答案是0到49的整數
    true_tokens = [random.randint(0, vocab_size - 1) for _ in range(sequence_length)]
    # 「模型」對每個位置輸出50個隨機logits,等於一個沒訓練過的模型
    token_logits = [[random.gauss(0, 1) for _ in range(vocab_size)] for _ in range(sequence_length)]

    avg_ce = negative_log_likelihood(true_tokens, token_logits)
    ppl = perplexity(avg_ce)

    print(f"  Vocab size:        {vocab_size}")
    print(f"  Sequence length:   {sequence_length}")
    print(f"  Avg CE loss:       {avg_ce:.4f} nats")
    print(f"  Perplexity:        {ppl:.2f}")
    # 完全亂猜(每個字機率都是1/50)的困惑度剛好等於詞彙表大小50,當作比較基準
    print(f"  Random baseline:   {vocab_size:.2f} (uniform over vocab)")
    print(f"  The model is better than random if perplexity < vocab size.")

    # ---------- 10. 條件熵與聯合熵 ----------
    print()
    print("=" * 60)
    print("CONDITIONAL AND JOINT ENTROPY")
    print("=" * 60)

    joint_dep = [[0.45, 0.05], [0.05, 0.45]]
    joint_indep = [[0.25, 0.25], [0.25, 0.25]]

    print(f"\n  Dependent joint distribution: {joint_dep}")
    print(f"    Joint entropy H(X,Y):     {joint_entropy(joint_dep):.4f} bits")
    print(f"    Conditional H(Y|X):       {conditional_entropy(joint_dep):.4f} bits")
    print(f"    Mutual information I(X;Y):{mutual_information(joint_dep):.4f} bits")

    # H(X):把每一列加總得到X的邊際分布,再算熵
    hx_dep = entropy([sum(row) for row in joint_dep])
    print(f"    H(X):                     {hx_dep:.4f} bits")
    # 連鎖法則驗證:H(X,Y) = H(X) + H(Y|X)
    print(f"    Verify: H(X,Y) = H(X) + H(Y|X) = {hx_dep:.4f} + {conditional_entropy(joint_dep):.4f} = {hx_dep + conditional_entropy(joint_dep):.4f}")

    print(f"\n  Independent joint distribution: {joint_indep}")
    print(f"    Joint entropy H(X,Y):     {joint_entropy(joint_indep):.4f} bits")
    print(f"    Conditional H(Y|X):       {conditional_entropy(joint_indep):.4f} bits")
    print(f"    Mutual information I(X;Y):{mutual_information(joint_indep):.4f} bits")
    print("    When independent: H(Y|X) = H(Y) and I(X;Y) = 0")

    # ---------- 11. 兩個實作示範 ----------
    label_smoothing_demo()
    feature_selection_mi_demo()
