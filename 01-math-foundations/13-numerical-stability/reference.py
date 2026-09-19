"""
Lesson 13: Numerical Stability(數值穩定性)
浮點數是有限精度、有限範圍的近似值。訓練中的NaN、loss突然變inf、float16訓練掉準確率,
多半都來自這裡。這支程式示範溢位(overflow)、下溢(underflow)、災難性抵銷(catastrophic cancellation),
以及對應的穩定寫法:減最大值的softmax與log-sum-exp、gradient checking、gradient clipping、
float16/bfloat16與loss scaling。

檔案結構:
    🔴 減最大值的技巧:穩定版softmax與log-sum-exp(訓練時最常用到的一招)
    🟡 浮點精度、溢位/下溢、抵銷、交叉熵、梯度檢查、NaN偵測、梯度裁剪、混合精度
    🟢 穩定版sigmoid與BCE、Kahan加總、Welford變異數、layer norm、常見bug整理
數字備忘:
    float32 最大約3.4e38,exp(88.7)已接近上限,exp(89)溢位成inf
    float64(Python的float)最大約1.8e308,exp(709.78)溢位
    float16 最大65504,最小正數約6e-8;bfloat16範圍與float32相同,但只有7位尾數
"""
import math
import struct
import random


# 🔴 max-subtraction trick: stable softmax and log-sum-exp

def softmax_naive(logits):
    """直接照定義寫的softmax:exp(z_i) / sum(exp(z_j))。logits一大,exp就溢位,不能實際使用。"""
    # list comprehension:對每個logit算exp。z大於約709時,Python的math.exp直接丟OverflowError(NumPy/PyTorch則回傳inf)
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    # 每項除以總和,得到總和為1的機率分布
    return [e / total for e in exps]


def softmax_stable(logits):
    """穩定版softmax:先減去最大值再exp。數學上結果完全相同(分子分母同乘exp(-max)約掉),但不會溢位。"""
    # 找出最大的logit,當作平移量
    max_logit = max(logits)
    # 減完之後最大的一項是exp(0)=1,其餘都小於1,所以不可能溢位;
    # 總和至少是1,所以不會出現全部下溢成0再0/0的情況
    # 例:[100,101,102]減102 -> [-2,-1,0] -> exp後 [0.135, 0.368, 1.0]
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def logsumexp_naive(values):
    """直接算 log(sum(exp(v)))。v大時exp溢位,v很負時每項下溢成0、log(0)出錯。"""
    return math.log(sum(math.exp(v) for v in values))


def logsumexp_stable(values):
    """穩定版log-sum-exp:log(sum(exp(v))) = c + log(sum(exp(v - c))),取c = max(v)。
        推導:exp(v) = exp(v - c) * exp(c),把exp(c)提到sum外面,log之後變成加上c。"""
    # c取最大值,讓減完後最大項是exp(0)=1
    c = max(values)
    # sum裡至少有一項是1,所以sum>=1,log不會是-inf;最後把提出去的c加回來
    # generator expression放在sum()裡,不必先建立完整list
    return c + math.log(sum(math.exp(v - c) for v in values))


def demo_softmax_stability():
    """對照naive與stable softmax:安全範圍結果相同;logits大到100、1000、-1000時,naive失敗、stable正常。"""
    print("=" * 60)
    print("DEMO 4: Naive vs Stable Softmax")
    print("=" * 60)

    # 第一組:小logits,兩種寫法都安全,結果應該一致
    safe_logits = [2.0, 1.0, 0.1]
    print(f"\n  Safe logits: {safe_logits}")
    naive_result = softmax_naive(safe_logits)
    stable_result = softmax_stable(safe_logits)
    print(f"  Naive:  {[f'{p:.6f}' for p in naive_result]}")
    print(f"  Stable: {[f'{p:.6f}' for p in stable_result]}")
    # all(...)檢查每一對數字的差都小於1e-10;用容許誤差比較,不用==(浮點誤差)
    print(f"  Match: {all(abs(a - b) < 1e-10 for a, b in zip(naive_result, stable_result))}")

    # 第二組:中等logits。在float32下exp(100)已經是inf;
    # Python用float64,exp(100)還能算,所以這裡naive不一定失敗,用來對照
    moderate_logits = [100.0, 101.0, 102.0]
    print(f"\n  Moderate logits: {moderate_logits}")
    stable_result = softmax_stable(moderate_logits)
    print(f"  Stable: {[f'{p:.6f}' for p in stable_result]}")
    # 用try/except接住溢位:naive寫法失敗時印出訊息而不是讓程式中斷
    try:
        naive_result = softmax_naive(moderate_logits)
        print(f"  Naive:  {[f'{p:.6f}' for p in naive_result]}")
    except OverflowError:
        print("  Naive:  OVERFLOW (exp(100) too large)")

    # 第三組:exp(1000)超過float64上限(約709.78),naive一定溢位,stable只需平移就能算
    extreme_logits = [1000.0, 1001.0, 1002.0]
    print(f"\n  Extreme logits: {extreme_logits}")
    stable_result = softmax_stable(extreme_logits)
    print(f"  Stable: {[f'{p:.6f}' for p in stable_result]}")
    print("  Naive:  would be [nan, nan, nan] or OVERFLOW")

    # 第四組:很負的logits。exp(-1000)下溢成0.0,naive的總和變成0,接著0/0;
    # stable平移到[-2,-1,0],結果正常
    negative_logits = [-1000.0, -999.0, -998.0]
    print(f"\n  Very negative logits: {negative_logits}")
    stable_result = softmax_stable(negative_logits)
    print(f"  Stable: {[f'{p:.6f}' for p in stable_result]}")
    print("  Naive:  would be [0/0 = nan] (all exp() underflow to 0)")
    print()


def demo_logsumexp():
    """示範log-sum-exp在一般值、極大值、極負值、全部相等、一個特別大這幾種情況下的表現。"""
    print("=" * 60)
    print("DEMO 5: Log-Sum-Exp Trick")
    print("=" * 60)

    safe = [1.0, 2.0, 3.0]
    print(f"\n  Safe values: {safe}")
    print(f"  Naive:  {logsumexp_naive(safe):.10f}")
    print(f"  Stable: {logsumexp_stable(safe):.10f}")

    # 極大值:exp(500)在float64還沒溢位(上限709),但加總、後續運算容易出問題;float32下更早失敗
    large = [500.0, 501.0, 502.0]
    print(f"\n  Large values: {large}")
    print(f"  Stable: {logsumexp_stable(large):.10f}")
    try:
        naive = logsumexp_naive(large)
        print(f"  Naive:  {naive}")
    except OverflowError:
        print("  Naive:  OVERFLOW")

    # 極負值:naive每項exp下溢成0,log(0)出錯;stable先平移所以正常
    very_negative = [-1000.0, -999.0, -998.0]
    print(f"\n  Very negative values: {very_negative}")
    print(f"  Stable: {logsumexp_stable(very_negative):.10f}")

    # 全部相等:log(3*exp(5)) = 5 + ln(3),用來驗證結果正確
    equal = [5.0, 5.0, 5.0]
    print(f"\n  Equal values: {equal}")
    expected = 5.0 + math.log(3.0)
    print(f"  Stable:   {logsumexp_stable(equal):.10f}")
    print(f"  Expected: {expected:.10f} (= 5.0 + ln(3))")

    # 一個值遠大於其他:結果會非常接近最大值本身(其他項貢獻幾乎為0)
    one_dominant = [100.0, 1.0, 1.0]
    print(f"\n  One dominant value: {one_dominant}")
    print(f"  Stable: {logsumexp_stable(one_dominant):.10f}")
    print(f"  ~100.0 (dominated by exp(100))")
    print()


# 🟡 float precision, overflow/underflow, cancellation, cross-entropy, gradient checking, clipping, mixed precision

def demo_float_precision():
    """示範浮點數的精度限制:0.1+0.2不等於0.3、machine epsilon、以及累加很多次小數字時的誤差。"""
    print("=" * 60)
    print("DEMO 1: Floating Point Precision Limits")
    print("=" * 60)

    # 0.1與0.2在二進位是無限循環小數,存進浮點數時被截斷,所以加起來是0.30000000000000004,不等於0.3
    print(f"\n  0.1 + 0.2 = {0.1 + 0.2}")
    print(f"  0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
    print(f"  Difference from 0.3: {(0.1 + 0.2) - 0.3:.2e}")
    # 比較浮點數不要用==,要用math.isclose(容許誤差內視為相等)或abs(a-b) < epsilon
    print(f"  math.isclose(0.1 + 0.2, 0.3): {math.isclose(0.1 + 0.2, 0.3)}")

    # float32的幾個界線:最大值約3.4e38、最小正規數約1.175e-38、machine epsilon約1.19e-7
    # machine epsilon:1.0加上比它小的數,結果還是1.0(被四捨五入吸收)
    print(f"\n  Float32 max: ~{3.4028235e+38:.2e}")
    print(f"  Float32 min positive (normal): ~{1.175e-38:.2e}")
    print(f"  Float32 epsilon: ~{1.1920929e-07:.2e}")

    # Python的float是64位元,epsilon約2.2e-16,所以1e-7還能被分辨;float32則不行
    print(f"\n  1.0 + 1e-7 == 1.0?  {1.0 + 1e-7 == 1.0}")
    print(f"  1.0 + 1e-8 == 1.0?  {1.0 + 1e-8 == 1.0}")
    print(f"  (These are float64 in Python. In float32, epsilon is ~1.19e-7)")

    # 累加一百萬次1e-7:每次加法都有微小捨入誤差,誤差會累積
    # 1_000_000裡的底線只是數字分隔符,方便閱讀,等於1000000
    total_naive = 0.0
    for _ in range(1_000_000):
        total_naive += 1e-7
    # Kahan加總用一個補償變數記住每次被捨掉的低位,大幅降低累積誤差
    total_kahan = kahan_sum([1e-7] * 1_000_000)
    true_value = 1e-7 * 1_000_000

    print(f"\n  Summing 1e-7 one million times:")
    print(f"  True value:  {true_value}")
    print(f"  Naive sum:   {total_naive:.10f}  (error: {abs(total_naive - true_value):.2e})")
    print(f"  Kahan sum:   {total_kahan:.10f}  (error: {abs(total_kahan - true_value):.2e})")
    print()


def demo_overflow_underflow():
    """示範exp與log的邊界:exp太大溢位成inf、太小下溢成0,log(0)與log(負數)出錯,以及float16的上限。"""
    print("=" * 60)
    print("DEMO 3: Overflow and Underflow in exp() and log()")
    print("=" * 60)

    print("\n  exp() overflow boundary (float64 in Python):")
    # float64的exp上限約709.78,超過就OverflowError
    for x in [700, 709, 709.78, 710]:
        try:
            result = math.exp(x)
            print(f"  exp({x}) = {result:.4e}")
        except OverflowError:
            print(f"  exp({x}) = OVERFLOW")

    print("\n  exp() underflow (results become 0.0):")
    # 下溢:exp(-745)約5e-324(最小的非正規數),再小就變成0.0;下溢不會報錯,只是悄悄變0
    for x in [-700, -745, -746]:
        result = math.exp(x)
        print(f"  exp({x}) = {result}")

    print("\n  log() edge cases:")
    # log的邊界:log(0)是負無限大,math.log(0.0)在Python會丟ValueError(定義域錯誤)
    for x in [1.0, 1e-300, 1e-323, 0.0]:
        try:
            if x == 0.0:
                print(f"  log(0.0) = -inf  (mathematically)")
                result = math.log(1e-323)
                print(f"  log(1e-323) = {result:.2f}  (closest we can get)")
            else:
                result = math.log(x)
                print(f"  log({x}) = {result:.4f}")
        except ValueError:
            print(f"  log({x}) = DOMAIN ERROR")

    print("\n  Float16 overflow boundary:")
    # float16最大可表示65504,超過會變成inf(用simulate_float16模擬)
    for val in [65000.0, 65504.0, 65520.0, 70000.0]:
        f16 = simulate_float16(val)
        print(f"  float16({val}) = {f16}")
    print()


def demo_catastrophic_cancellation():
    """示範災難性抵銷:兩個幾乎相等的數相減,有效位數被抵掉,只剩捨入誤差。"""
    print("=" * 60)
    print("DEMO 2: Catastrophic Cancellation")
    print("=" * 60)

    # 資料平均值很大(一百萬)但變異很小;真正的變異數 = ((-1)^2+0^2+1^2)/3 = 2/3
    data = [1_000_000.0, 1_000_001.0, 1_000_002.0]
    true_var = 2.0 / 3.0

    # naive公式E[x^2] - E[x]^2:兩項都約1e12、彼此幾乎相等,相減時有效位數被抵掉;
    # Welford線上演算法不做這個大數相減,誤差小很多
    var_naive = variance_naive(data)
    var_welford = welford_variance(data)

    print(f"\n  Data: {data}")
    print(f"  True variance: {true_var:.10f}")
    print(f"  Naive (E[x^2] - E[x]^2): {var_naive:.10f}")
    print(f"  Welford (online):         {var_welford:.10f}")
    print(f"  Naive error:   {abs(var_naive - true_var):.2e}")
    print(f"  Welford error: {abs(var_welford - true_var):.2e}")

    # 直接相減的例子:a與b只差1e-7,相減結果的有效位數只剩最後幾位,相對誤差可高達百分之幾十(float32下更明顯)
    a = 1.0000001
    b = 1.0000000
    true_diff = 1e-7
    computed_diff = a - b
    rel_error = abs(computed_diff - true_diff) / true_diff * 100

    print(f"\n  Subtracting nearly equal numbers:")
    print(f"  a = {a}")
    print(f"  b = {b}")
    print(f"  True a - b = {true_diff}")
    print(f"  Computed:    {computed_diff}")
    print(f"  Relative error: {rel_error:.1f}%")
    print()


def log_softmax_stable(logits):
    """穩定版log-softmax:log(softmax(z_i)) = z_i - logsumexp(z)。直接在log空間算,不必先算機率再取log。"""
    # 用同樣的減最大值技巧算logsumexp
    c = max(logits)
    lse = c + math.log(sum(math.exp(z - c) for z in logits))
    # 每個logit減去logsumexp,就是它的log機率;避免了先softmax得到極小機率再取log,可能出現log(0)的問題
    return [z - lse for z in logits]


def cross_entropy_naive(true_class, logits):
    """naive版:先softmax得到機率,再取真實類別機率的-log。機率下溢成0時log(0)出錯。"""
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])


def cross_entropy_stable(true_class, logits):
    """穩定版:交叉熵 = -log_softmax[真實類別]。PyTorch的F.cross_entropy就是這個做法。"""
    log_probs = log_softmax_stable(logits)
    return -log_probs[true_class]


def demo_cross_entropy():
    """對照naive與stable交叉熵:小logits一致;大logits、非常自信但答錯的情況,stable都能得到有限的loss。"""
    print("=" * 60)
    print("DEMO 6: Stable Cross-Entropy Loss")
    print("=" * 60)

    logits = [2.0, 5.0, 1.0]
    true_class = 1

    print(f"\n  Logits: {logits}, true class: {true_class}")
    ce_naive = cross_entropy_naive(true_class, logits)
    ce_stable = cross_entropy_stable(true_class, logits)
    print(f"  Naive:  {ce_naive:.10f}")
    print(f"  Stable: {ce_stable:.10f}")
    print(f"  Match:  {abs(ce_naive - ce_stable) < 1e-10}")

    # 大logits:naive可能溢位或得到NaN,stable不受影響
    large_logits = [100.0, 105.0, 99.0]
    true_class = 1
    print(f"\n  Large logits: {large_logits}, true class: {true_class}")
    ce_stable = cross_entropy_stable(true_class, large_logits)
    print(f"  Stable: {ce_stable:.10f}")
    try:
        ce_naive = cross_entropy_naive(true_class, large_logits)
        print(f"  Naive:  {ce_naive:.10f}")
    except (OverflowError, ValueError):
        print("  Naive:  OVERFLOW or NaN")

    # 非常自信且答對:真實類別2的機率接近1,loss接近0
    confident_logits = [0.0, 0.0, 50.0]
    true_class = 2
    ce = cross_entropy_stable(true_class, confident_logits)
    print(f"\n  Very confident prediction:")
    print(f"  Logits: {confident_logits}, true class: {true_class}")
    print(f"  Loss: {ce:.10f}  (near zero, model is correct and confident)")

    # 非常自信卻答錯:真實類別是0,但模型給類別2幾乎100%,loss約50,非常大;
    # naive版此時機率會下溢成0,log(0)直接出錯
    wrong_logits = [0.0, 0.0, 50.0]
    true_class = 0
    ce = cross_entropy_stable(true_class, wrong_logits)
    print(f"\n  Very wrong prediction:")
    print(f"  Logits: {wrong_logits}, true class: {true_class}")
    print(f"  Loss: {ce:.4f}  (very large, model is confident but wrong)")
    print()


def numerical_gradient(f, x, h=1e-5):
    """用中央差分算數值梯度:df/dx_i 約等於 (f(x+h) - f(x-h)) / (2h),誤差是O(h^2),比前向差分精準。"""
    # 逐個參數求偏導:每次只微調一個參數,其他不動
    grad = []
    for i in range(len(x)):
        # x[:]複製一份list;若直接x_plus = x,兩個變數指向同一個list,改一個另一個也被改
        x_plus = x[:]
        x_minus = x[:]
        # 第i個參數加h、另一份減h
        x_plus[i] += h
        x_minus[i] -= h
        # 中央差分公式;h不能太大(近似不準)也不能太小(捨入誤差被放大),1e-5是常用折衷
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad


def check_gradient(analytical, numerical, tolerance=1e-5):
    """比較解析梯度(反向傳播算的)與數值梯度,用相對誤差判斷。回傳是否全部通過。
        經驗值:相對誤差<1e-7很好,<1e-5可接受,>1e-3多半有bug。"""
    all_ok = True
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        # 分母取兩者絕對值較大者;1e-8避免兩個梯度都接近0時除以0
        denom = max(abs(a), abs(n), 1e-8)
        # 相對誤差:差距相對於梯度本身的大小
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")
    return all_ok


def demo_gradient_checking():
    """三個測試:多項式函數、softmax交叉熵(梯度有已知的解析式 p - one_hot)、以及故意寫錯的梯度(應該被抓出來)。"""
    print("=" * 60)
    print("DEMO 8: Gradient Checking")
    print("=" * 60)

    print("\n  Test 1: f(x,y) = x^2 + 3xy + y^3")

    # 巢狀函式:只在這個demo裡使用的小函式
    def f1(params):
        x, y = params
        return x ** 2 + 3 * x * y + y ** 3

    def f1_grad(params):
        x, y = params
        # 手推的解析梯度:f = x^2+3xy+y^3,對x偏微分2x+3y,對y偏微分3x+3y^2
        return [2 * x + 3 * y, 3 * x + 3 * y ** 2]

    point = [2.0, 1.0]
    analytical = f1_grad(point)
    numerical = numerical_gradient(f1, point)
    print(f"  Point: {point}")
    check_gradient(analytical, numerical)

    print("\n  Test 2: f(x) = softmax cross-entropy")

    def f2(logits):
        return cross_entropy_stable(0, logits)

    logits = [2.0, 1.0, 0.5]
    probs = softmax_stable(logits)
    # softmax加交叉熵的梯度有漂亮的封閉形式:softmax機率 - one-hot標籤(真實類別減1)
    analytical_ce = [probs[i] - (1.0 if i == 0 else 0.0) for i in range(len(logits))]
    numerical_ce = numerical_gradient(f2, logits)
    print(f"  Logits: {logits}")
    check_gradient(analytical_ce, numerical_ce)

    print("\n  Test 3: Deliberately wrong gradient (should FAIL)")

    def f3(params):
        x, y = params
        return x ** 2 + y ** 2

    # f = x^2+y^2 在(3,4)的正確梯度是[6,8];故意給[1,1],檢查函式應該回報FAIL
    wrong_grad = [1.0, 1.0]
    numerical_f3 = numerical_gradient(f3, [3.0, 4.0])
    print(f"  Wrong analytical: {wrong_grad}")
    print(f"  Correct numerical: {[f'{g:.4f}' for g in numerical_f3]}")
    check_gradient(wrong_grad, numerical_f3)
    print()


def check_tensor(name, values):
    """檢查一組數值有沒有NaN或Inf,有就印警告。訓練除錯時常在每次forward之後檢查。"""
    # any(...):只要有一個為真就是True;math.isnan檢查是不是NaN(NaN不能用==比較,因為nan == nan 也是False)
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    n_nan = sum(1 for v in values if math.isnan(v))
    n_inf = sum(1 for v in values if math.isinf(v))
    if has_nan or has_inf:
        print(f"  WARNING {name}: {n_nan} NaN, {n_inf} Inf out of {len(values)} values")
        return False
    print(f"  OK {name}: all {len(values)} values finite")
    return True


def demo_nan_inf():
    """示範NaN與Inf怎麼出現、怎麼像病毒一樣擴散:任何運算碰到NaN結果都是NaN。"""
    print("=" * 60)
    print("DEMO 9: NaN and Inf Detection and Propagation")
    print("=" * 60)

    print("\n  How inf appears:")
    # inf的來源:除以0(NumPy/PyTorch回傳inf,純Python的1.0/0.0會丟ZeroDivisionError,這裡用float("inf")代表結果)、exp溢位、大數乘法溢位
    print(f"  1.0 / 0.0    = {float('inf')}")
    print(f"  exp(710)     = overflow -> inf")
    print(f"  1e308 * 10   = {1e308 * 10}")

    print("\n  How nan appears:")
    # nan的來源:0/0、inf-inf、inf*0、對負數開根號或取log、以及任何跟nan有關的運算
    print(f"  0.0 / 0.0        = {float('nan')}")
    print(f"  inf - inf        = {float('inf') - float('inf')}")
    print(f"  inf * 0          = {float('inf') * 0}")
    print(f"  nan + 1          = {float('nan') + 1}")
    # NaN跟任何數比較(包含自己)都是False,所以檢查NaN要用math.isnan,不能用==
    print(f"  nan == nan       = {float('nan') == float('nan')}")
    print(f"  nan < 0          = {float('nan') < 0}")
    print(f"  nan > 0          = {float('nan') > 0}")

    print("\n  NaN propagation (one nan ruins everything):")
    # 只要一個NaN,sum、mean都變成NaN;一個NaN的梯度更新會讓權重變NaN,整個訓練報廢
    values = [1.0, 2.0, float('nan'), 4.0, 5.0]
    print(f"  values = {values}")
    print(f"  sum    = {sum(values)}")
    print(f"  max    = nan (comparison with nan is always False)")
    print(f"  mean   = {sum(values) / len(values)}")

    print("\n  Tensor health checks:")
    check_tensor("weights", [0.1, -0.3, 0.5, 0.2])
    check_tensor("logits_bad", [1.0, float('inf'), -2.0])
    check_tensor("grads_bad", [0.01, float('nan'), -0.03])
    check_tensor("activations", [0.0, 0.5, 1.0, 0.3])
    print()


def clip_by_value(gradients, max_val):
    """逐元素夾在[-max_val, max_val]內。簡單,但各元素被夾的程度不同,梯度向量的方向會改變。"""
    # min(max_val, g)先把上限夾住,max(-max_val, ...)再把下限夾住
    return [max(-max_val, min(max_val, g)) for g in gradients]


def clip_by_norm(gradients, max_norm):
    """按整體長度(L2範數)裁剪:長度超過max_norm就等比例縮小,方向不變。PyTorch的clip_grad_norm_做的就是這個。"""
    # 梯度向量的長度 = sqrt(各元素平方和),即L2範數(Lesson 1)
    total_norm = math.sqrt(sum(g ** 2 for g in gradients))
    if total_norm > max_norm:
        # 縮放倍率:讓縮放後的長度剛好等於max_norm;每個元素乘同一個倍率,所以方向不變
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return list(gradients)


def demo_gradient_clipping():
    """對照clip by value與clip by norm,並模擬梯度爆炸:梯度每步變3.5倍,裁剪後長度固定不超過1。"""
    print("=" * 60)
    print("DEMO 10: Gradient Clipping")
    print("=" * 60)

    grads = [10.0, 20.0, 30.0]
    norm = math.sqrt(sum(g ** 2 for g in grads))

    print(f"\n  Gradients: {grads}")
    print(f"  Norm: {norm:.4f}")

    # by value把[10,20,30]夾成[10,15,15],各項比例變了(方向改變);by norm整體乘同一個倍率縮成約[1.34,2.67,4.01],比例不變(方向不變)
    clipped_val = clip_by_value(grads, max_val=15.0)
    clipped_norm = clip_by_norm(grads, max_norm=5.0)

    print(f"\n  Clip by value (max=15.0): {clipped_val}")
    print(f"  Clip by value changes direction: "
          f"{[g/grads[0] for g in grads]} vs {[g/clipped_val[0] for g in clipped_val]}")

    print(f"\n  Clip by norm (max=5.0): {[f'{g:.4f}' for g in clipped_norm]}")
    clipped_norm_val = math.sqrt(sum(g ** 2 for g in clipped_norm))
    print(f"  Clipped norm: {clipped_norm_val:.4f}")
    print(f"  Direction preserved: "
          f"{[round(g/grads[0], 4) for g in grads]} == "
          f"{[round(g/clipped_norm[0], 4) for g in clipped_norm]}")

    print("\n  Gradient explosion simulation:")
    grad_val = 1.0
    max_norm = 1.0
    # 模擬梯度爆炸:每一步梯度乘3.5(指數成長),沒有裁剪的話很快變成天文數字
    for step in range(8):
        grad_val *= 3.5
        clipped = clip_by_norm([grad_val], max_norm)[0]
        print(f"  Step {step}: raw_grad={grad_val:>12.2f}  clipped={clipped:>8.4f}")
    print()


def simulate_bfloat16(x):
    """模擬bfloat16:保留float32的高16位元(1位符號+8位指數+7位尾數),丟掉低16位元的尾數。
        範圍與float32相同,精度較低。這裡用截斷,實際硬體會做四捨五入。"""
    # struct.pack('f', x):把Python數字打包成float32的4個位元組
    packed = struct.pack('f', x)
    # 把4個位元組當成一個32位元整數('little'代表低位元組在前,是x86與ARM的排法)
    as_int = int.from_bytes(packed, 'little')
    # 按位元AND:0xFFFF0000保留高16位元、低16位元清成0,就是砍掉一半的尾數
    truncated = as_int & 0xFFFF0000
    # 整數轉回4個位元組
    repacked = truncated.to_bytes(4, 'little')
    # struct.unpack('f', ...)把位元組解讀回float32數值;回傳tuple,[0]取第一個
    return struct.unpack('f', repacked)[0]


def simulate_float16(x):
    """模擬float16:先打包成16位元(格式代碼'e')再解包,超過範圍時回傳inf。"""
    try:
        # 格式代碼'e'是IEEE 754半精度(float16),2個位元組;最大65504,太大的數字會丟OverflowError
        packed = struct.pack('e', x)
        return struct.unpack('e', packed)[0]
    # 超出float16範圍:正的變+inf、負的變-inf,模擬真實硬體的溢位行為
    except (OverflowError, struct.error):
        return float('inf') if x > 0 else float('-inf')


def demo_mixed_precision():
    """示範float16與bfloat16的差異,loss scaling如何避免梯度下溢,以及動態調整scale factor的規則。"""
    print("=" * 60)
    print("DEMO 11: Mixed Precision and Loss Scaling")
    print("=" * 60)

    print("\n  bfloat16 vs float16 precision:")
    # 65504是float16最大值,65536以上float16變inf、bfloat16仍然表示得出來(但精度較粗)
    test_values = [1.0, 0.1, 3.14159, 100.0, 65504.0, 65536.0, 100000.0]
    # {'value':>12s}是靠右對齊、佔12格寬的字串格式;下一行用'-'*12畫分隔線
    print(f"  {'value':>12s}  {'float16':>12s}  {'bfloat16':>12s}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}")
    for v in test_values:
        f16 = simulate_float16(v)
        bf16 = simulate_bfloat16(v)
        f16_str = f"{f16:.4f}" if not math.isinf(f16) else "inf"
        bf16_str = f"{bf16:.4f}" if not math.isinf(bf16) else "inf"
        print(f"  {v:>12.4f}  {f16_str:>12s}  {bf16_str:>12s}")

    print("\n  Loss scaling simulation:")
    # loss scaling實驗:模擬1000個很小的梯度,範圍1e-9到1e-5
    random.seed(42)
    n_grads = 1000
    tiny_grads = [random.uniform(1e-9, 1e-5) for _ in range(n_grads)]

    # float16最小正數約6e-8,比它小的梯度轉成float16會變0,模型就學不到東西(梯度下溢)
    # sum(1 for ... if ...)是「計算符合條件的個數」的慣用寫法
    zeros_without_scaling = sum(1 for g in tiny_grads if simulate_float16(g) == 0.0)

    # loss scaling:先把梯度(等於先把loss)放大1024倍,讓小梯度落在float16能表示的範圍內
    scale = 1024.0
    scaled_grads = [g * scale for g in tiny_grads]
    zeros_with_scaling = sum(1 for g in scaled_grads if simulate_float16(g) == 0.0)

    # 完整流程:放大 -> 轉float16 -> 除回原本倍率;下溢成0的比例大幅下降,更新量保持一致
    scaled_back = [simulate_float16(g * scale) / scale for g in tiny_grads]
    zeros_after_roundtrip = sum(1 for g in scaled_back if g == 0.0)

    print(f"  {n_grads} gradients in range [1e-9, 1e-5]")
    print(f"  Zeros without scaling: {zeros_without_scaling}/{n_grads} "
          f"({zeros_without_scaling/n_grads*100:.1f}%)")
    print(f"  Zeros with scaling (x{scale:.0f}): {zeros_with_scaling}/{n_grads} "
          f"({zeros_with_scaling/n_grads*100:.1f}%)")
    print(f"  Zeros after scale+convert+unscale: {zeros_after_roundtrip}/{n_grads} "
          f"({zeros_after_roundtrip/n_grads*100:.1f}%)")

    print("\n  Dynamic loss scaling simulation:")
    # 動態loss scaling:從很大的倍率開始,遇到溢位就減半,連續growth_interval步都沒溢位就加倍,自動找到合適的倍率
    scale_factor = 65536.0
    no_overflow_steps = 0
    growth_interval = 100

    print(f"  {'step':>6s}  {'scale':>12s}  {'event':s}")
    for step in range(500):
        grad = random.gauss(0, 1)
        scaled = grad * scale_factor
        if math.isinf(simulate_float16(scaled)):
            scale_factor /= 2
            no_overflow_steps = 0
            if step < 20 or step % 100 == 0:
                print(f"  {step:>6d}  {scale_factor:>12.0f}  overflow -> halved")
        else:
            no_overflow_steps += 1
            if no_overflow_steps >= growth_interval:
                scale_factor *= 2
                no_overflow_steps = 0
                if step < 100 or step % 100 == 0:
                    print(f"  {step:>6d}  {scale_factor:>12.0f}  stable -> doubled")
    print(f"  Final scale factor: {scale_factor:.0f}")
    print()


def demo_format_comparison():
    """列出各浮點格式的位元配置、範圍與適用場合,並用pi與大數比較float16跟bfloat16的差別。"""
    print("=" * 60)
    print("DEMO 14: Float Format Comparison Summary")
    print("=" * 60)

    # 三個引號的f-string可以跨多行;表格欄位:位元數、指數位數、尾數位數、有效十進位位數、最大值
    # 重點:bfloat16用float32相同的8位指數換取大範圍(適合訓練),犧牲尾數精度;float16精度較高但範圍小(適合推論)
    print(f"""
  Format     Bits  Exp  Mantissa  ~Digits  Max Value       Best For
  -------    ----  ---  --------  -------  ----------      --------
  float64    64    11   52        15-16    1.8e308         CPU training, accumulation
  float32    32    8    23        7-8      3.4e38          Default training
  float16    16    5    10        3-4      65,504          Inference
  bfloat16   16    8    7         2-3      3.4e38          GPU/TPU training
  float8     8     4    3         1-2      240             Forward pass only (H100+)
""")

    print("  Precision test (representing pi):")
    # pi的例子:float16尾數10位、bfloat16尾數7位,pi=3.14159...在這兩種格式下剛好都被表示成3.140625,誤差約0.00097;
    # 尾數位數越少,能表示的數字越稀疏,一般情況下誤差會更大
    pi = math.pi
    f16_pi = simulate_float16(pi)
    bf16_pi = simulate_bfloat16(pi)
    print(f"  float64:  {pi}")
    print(f"  float16:  {f16_pi}  (error: {abs(f16_pi - pi):.6f})")
    print(f"  bfloat16: {bf16_pi}  (error: {abs(bf16_pi - pi):.6f})")

    print("\n  Range test (large values):")
    # 範圍測試:100000超過float16上限變INF,bfloat16仍然是有限值
    for val in [100.0, 1000.0, 10000.0, 65504.0, 100000.0]:
        f16 = simulate_float16(val)
        bf16 = simulate_bfloat16(val)
        f16_ok = "ok" if not math.isinf(f16) else "INF"
        bf16_ok = "ok" if not math.isinf(bf16) else "INF"
        print(f"  {val:>10.0f}  float16={f16_ok:>4s}  bfloat16={bf16_ok:>4s}")
    print()


# 🟢 stable sigmoid/BCE, Kahan sum, Welford variance, layer norm, common bug walkthrough

def sigmoid_naive(x):
    """直接照定義:1/(1+exp(-x))。x很負時exp(-x)溢位。"""
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_stable(x):
    """分成正負兩邊算,讓exp的參數永遠小於等於0,不會溢位。
        x<0時等價改寫成 exp(x)/(1+exp(x)),分子分母同乘exp(x)。"""
    # x>=0:exp(-x)在0到1之間,安全
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        # x<0:exp(x)在0到1之間,安全;用改寫後的公式,結果與原公式相同
        z = math.exp(x)
        return z / (1.0 + z)


def binary_cross_entropy_naive(y_true, y_pred):
    """naive版二元交叉熵:先有機率y_pred,再取log。y_pred=0或1時log(0)出錯。"""
    return -(y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))


def binary_cross_entropy_stable(y_true, logit):
    """直接由logit算二元交叉熵,不經過sigmoid:max(z,0) - y*z + log(1 + exp(-|z|))的等價形式。
        PyTorch的BCEWithLogitsLoss就是這個做法,比先sigmoid再BCELoss穩定。"""
    # max_val是減最大值技巧的平移量,讓兩個exp的參數都小於等於0
    max_val = max(0.0, logit)
    # 等於 log(1 + exp(logit)) - y_true * logit,把log(exp(-m)+exp(z-m))加上m的平移還原
    return max_val + math.log(math.exp(-max_val) + math.exp(logit - max_val)) - y_true * logit


def demo_sigmoid_stability():
    """對照naive與stable sigmoid在極大、極小輸入下的結果。"""
    print("=" * 60)
    print("DEMO 7: Stable Sigmoid")
    print("=" * 60)

    test_values = [0.0, 1.0, -1.0, 10.0, -10.0, 100.0, -100.0, 500.0, -500.0, 710.0, -710.0]
    print(f"\n  {'x':>8s}  {'naive':>14s}  {'stable':>14s}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}")
    for x in test_values:
        try:
            naive = sigmoid_naive(x)
            naive_str = f"{naive:.10f}"
        except OverflowError:
            naive_str = "OVERFLOW"
        stable = sigmoid_stable(x)
        print(f"  {x:>8.1f}  {naive_str:>14s}  {stable:.10f}")
    print()


def kahan_sum(values):
    """Kahan補償加總:用compensation記住每次加法被捨掉的低位,下一次補回去,降低長序列累加的誤差。"""
    total = 0.0
    compensation = 0.0
    for v in values:
        # 先扣掉上次遺失的部分
        y = v - compensation
        # 暫時的新總和;total很大、y很小時,y的低位會在這一步被捨入丟掉
        t = total + y
        # (t - total)是實際加進去的量,減去y得到被捨掉的部分(理論上是0,實際是小誤差)
        compensation = (t - total) - y
        total = t
    return total


def welford_variance(values):
    """Welford線上演算法算(母體)變異數:一次一個數更新平均與平方差總和,不做「平方的平均 - 平均的平方」這種大數相減。"""
    n = 0
    mean = 0.0
    m2 = 0.0
    for x in values:
        n += 1
        # 新數字與舊平均的差
        delta = x - mean
        # 增量更新平均
        mean += delta / n
        # 與更新後平均的差(注意用的是新平均)
        delta2 = x - mean
        # m2累計平方差總和;delta與delta2一個用舊平均、一個用新平均,合起來是數值穩定的更新式
        m2 += delta * delta2
    if n < 2:
        return 0.0
    return m2 / n


def variance_naive(values):
    """naive公式 E[x^2] - E[x]^2。平均值很大時兩項都很大且幾乎相等,相減發生災難性抵銷。"""
    n = len(values)
    mean_x = sum(values) / n
    mean_x2 = sum(v ** 2 for v in values) / n
    return mean_x2 - mean_x ** 2


def layer_norm(values, epsilon=1e-5, gamma=1.0, beta=0.0):
    """Layer Normalization:把一組數字標準化成平均0、標準差1,再乘gamma加beta。讓每層的數值範圍保持穩定。"""
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    # epsilon(常用1e-5)避免所有值相同、變異數為0時除以0
    std = math.sqrt(var + epsilon)
    # 標準化後乘上可學習的gamma(縮放)、加上beta(平移),讓網路能還原它需要的尺度
    return [(v - mean) / std * gamma + beta for v in values]


def demo_layer_norm():
    """對照沒有與有layer norm時,數值隨層數加深的變化:沒有的會指數成長,有的維持在固定範圍。"""
    print("=" * 60)
    print("DEMO 12: Normalization as Numerical Stabilizer")
    print("=" * 60)

    print("\n  Without normalization (values grow through layers):")
    values = [1.0, 0.5, -0.3, 0.8, -0.1]
    for layer in range(10):
        # 模擬一層:乘2.5、加0.1,再套ReLU(max(0, x));每層放大2.5倍,10層後放大約9500倍
        values = [max(0, v * 2.5 + 0.1) for v in values]
        max_val = max(abs(v) for v in values)
        if layer % 2 == 0:
            print(f"  Layer {layer:>2d}: max={max_val:>12.2f}  values={[f'{v:.2f}' for v in values[:3]]}...")

    print("\n  With layer normalization (values stay bounded):")
    values = [1.0, 0.5, -0.3, 0.8, -0.1]
    for layer in range(10):
        values = [max(0, v * 2.5 + 0.1) for v in values]
        values = layer_norm(values)
        max_val = max(abs(v) for v in values)
        if layer % 2 == 0:
            print(f"  Layer {layer:>2d}: max={max_val:>6.4f}  values={[f'{v:.4f}' for v in values[:3]]}...")
    print()


def demo_common_bugs():
    """整理五個常見的數值bug:log(0)、exp溢位、大平均值下的變異數、浮點數比較、標準化的0/0。"""
    print("=" * 60)
    print("DEMO 13: Common ML Numerical Bugs")
    print("=" * 60)

    print("\n  Bug 1: log(0) from confident wrong prediction")
    # Bug 1:模型非常自信且答錯,真實類別的機率下溢成0,log(0) = -inf;穩定版交叉熵直接在log空間算,不會出事
    logits = [100.0, -100.0, -100.0]
    probs = softmax_stable(logits)
    print(f"  Softmax: {[f'{p:.2e}' for p in probs]}")
    print(f"  If true class is 1: log({probs[1]:.2e}) = ", end="")
    if probs[1] == 0.0:
        print("log(0) = -inf (CRASH)")
    else:
        print(f"{math.log(probs[1]):.2f}")
    print(f"  Stable cross-entropy handles this: {cross_entropy_stable(1, logits):.4f}")

    print("\n  Bug 2: exp() overflow in naive softmax")
    # Bug 2:exp(800)超過float64上限,naive softmax溢位
    logits = [800.0, 801.0, 802.0]
    try:
        naive = softmax_naive(logits)
        print(f"  Naive softmax: {naive}")
    except OverflowError:
        print("  Naive softmax: OverflowError (exp(800) is too large)")
    stable = softmax_stable(logits)
    print(f"  Stable softmax: {[f'{p:.6f}' for p in stable]}")

    print("\n  Bug 3: Variance underflow with large-mean data")
    # Bug 3:平均值約1億、變異數只有2,naive公式的大數相減誤差變得顯著;1e8是科學記號,等於100000000
    data = [1e8 + 1, 1e8 + 2, 1e8 + 3, 1e8 + 4, 1e8 + 5]
    var_naive = variance_naive(data)
    var_welford = welford_variance(data)
    true_var = 2.0
    print(f"  Data: [{data[0]:.0f}, ..., {data[-1]:.0f}]")
    print(f"  True variance: {true_var}")
    print(f"  Naive:   {var_naive:.6f}  (error: {abs(var_naive - true_var):.2e})")
    print(f"  Welford: {var_welford:.6f}  (error: {abs(var_welford - true_var):.2e})")

    print("\n  Bug 4: Float comparison in training loop")
    # Bug 4:累加10次0.1不會剛好等於1.0,不要用==比較浮點數
    loss = 0.0
    for _ in range(10):
        loss += 0.1
    print(f"  After 10 steps of loss += 0.1: loss = {loss}")
    print(f"  loss == 1.0? {loss == 1.0} (WRONG)")
    print(f"  math.isclose(loss, 1.0)? {math.isclose(loss, 1.0)} (CORRECT)")

    print("\n  Bug 5: NaN from 0/0 in normalization")
    # Bug 5:所有值相同時變異數為0,標準化要除以sqrt(0);解法是分母加epsilon
    values = [5.0, 5.0, 5.0, 5.0]
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    print(f"  Constant input: {values}")
    print(f"  Variance: {var}")
    print(f"  1/sqrt(var) = 1/sqrt(0) = ", end="")
    try:
        result = 1.0 / math.sqrt(var)
        print(f"{result}")
    except ZeroDivisionError:
        print("ZeroDivisionError")
    safe = 1.0 / math.sqrt(var + 1e-5)
    print(f"  1/sqrt(var + 1e-5) = {safe:.2f} (safe with epsilon)")
    print()


if __name__ == "__main__":
    demo_softmax_stability()
    demo_logsumexp()
    demo_float_precision()
    demo_overflow_underflow()
    demo_catastrophic_cancellation()
    demo_cross_entropy()
    demo_gradient_checking()
    demo_nan_inf()
    demo_gradient_clipping()
    demo_mixed_precision()
    demo_format_comparison()
    demo_sigmoid_stability()
    demo_layer_norm()
    demo_common_bugs()
