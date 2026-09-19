"""
Lesson 13: Numerical Stability——PyTorch對照
用純Python手寫的穩定版softmax、log-sum-exp、交叉熵、gradient checking、gradient clipping,
在PyTorch裡都有內建函式,而且內部已經套用同樣的數值穩定技巧。
"""
import torch
import torch.nn.functional as F


def demo_stable_softmax():
    """PyTorch的softmax/log_softmax/logsumexp內部已經先減最大值,大logits不會溢位"""
    # 三個logits都很大:直接exp在float32會溢位(exp(89)以上就是inf)
    logits = torch.tensor([100.0, 101.0, 102.0])
    # PyTorch的softmax內部已經先減最大值,所以大logits不會溢位
    # dim=0代表沿第0軸(唯一的軸)做softmax;結果總和為1
    print(f"softmax:     {F.softmax(logits, dim=0)}")
    # log_softmax直接在log空間算,比先softmax再取log精準,也不會出現log(0)
    print(f"log_softmax: {F.log_softmax(logits, dim=0)}")
    # logsumexp = log(sum(exp(x))),內部用 max + log(sum(exp(x-max))) 的技巧
    # .item()把單一元素的tensor取成Python數字才能用:.4f格式化
    print(f"logsumexp:   {torch.logsumexp(logits, dim=0).item():.4f}")

    # 對照組:沒有減最大值直接算exp,float32下三個都溢位成inf
    naive = torch.exp(logits)
    print(f"naive exp(100..102) in float32: {naive}")


def demo_cross_entropy():
    """F.cross_entropy把softmax與取log合併計算,logits再大也不會變成inf或nan"""
    # shape (1, 3):1筆資料、3個類別的logits;PyTorch分類損失的輸入格式是 (批次, 類別數)
    logits = torch.tensor([[500.0, 501.0, 502.0]])
    # target是每筆資料的真實類別編號(整數),shape (1,);這裡真實類別是2
    target = torch.tensor([2])
    # cross_entropy把log_softmax與負對數概似合併成一步計算,logits再大結果也是有限值
    # 注意輸入是「未經softmax的原始logits」,不要先自己softmax再丟進去
    print(f"cross_entropy: {F.cross_entropy(logits, target).item():.6f}")


def demo_gradcheck():
    """torch.autograd.gradcheck:用中央差分數值梯度驗證autograd算出的解析梯度"""
    def f(x):
        # f(x,y) = x^2 + 3xy + y^3,跟課程手寫的梯度檢查用同一個函數
        # unsqueeze(0)把純量變成shape (1,)的tensor,gradcheck要求輸出是tensor
        return (x[0] ** 2 + 3 * x[0] * x[1] + x[1] ** 3).unsqueeze(0)

    # gradcheck用有限差分比對梯度,需要float64的精度
    # float32精度只有約7位小數,有限差分會被捨入誤差干擾,所以用float64
    # requires_grad=True:告訴autograd要追蹤這個tensor的梯度
    x = torch.tensor([2.0, 1.0], dtype=torch.float64, requires_grad=True)
    # 參數(x,)是tuple,代表傳給f的所有輸入;比對通過回傳True,不通過會丟出例外
    print(f"gradcheck passes: {torch.autograd.gradcheck(f, (x,))}")


def demo_clip_grad_norm():
    """clip_grad_norm_:整個梯度向量的長度超過門檻時,等比例縮小,方向不變"""
    # nn.Parameter是「可訓練的權重」,可以有.grad屬性
    w = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
    # 手動指定梯度 [10,20,30],範數 = sqrt(100+400+900) = sqrt(1400) 約 37.42
    w.grad = torch.tensor([10.0, 20.0, 30.0])
    # 函式名尾端的底線_是PyTorch慣例:代表「就地修改」(in-place),直接改w.grad
    # 回傳的是縮小之前的總範數;超過max_norm=5就整體乘上 5/37.42
    total_norm = torch.nn.utils.clip_grad_norm_([w], max_norm=5.0)
    print(f"norm before clip: {total_norm.item():.2f}")
    # 縮小後的範數剛好等於5.00,方向與原本一致
    print(f"norm after clip:  {w.grad.norm().item():.2f}")


def demo_half_precision():
    """float16與bfloat16的範圍、精度差異"""
    big = torch.tensor(70000.0)
    small = torch.tensor(1e-8)
    # torch.finfo查詢浮點格式的極限值;float16最大只有65504
    print(f"float16 max: {torch.finfo(torch.float16).max}")
    # 70000超過float16上限 -> inf(溢位);bfloat16的指數位數跟float32一樣,範圍夠大,
    # 只是精度低,70000被四捨五入成70144
    print(f"70000 -> float16: {big.half().item()}, bfloat16: {big.bfloat16().item()}")
    # 1e-8太小,float16下溢成0.0;bfloat16仍能表示(約1.00e-08),
    # 這就是float16訓練需要loss scaling、bfloat16通常不需要的原因
    print(f"1e-8  -> float16: {small.half().item()}, bfloat16: {small.bfloat16().item()}")
    # machine epsilon:1.0加上多小的數還能被分辨出來,數字越大代表精度越差
    print(f"machine eps float32: {torch.finfo(torch.float32).eps}")
    print(f"machine eps float16: {torch.finfo(torch.float16).eps}")
    print(f"machine eps bfloat16: {torch.finfo(torch.bfloat16).eps}")


if __name__ == "__main__":
    demo_stable_softmax()
    demo_cross_entropy()
    demo_gradcheck()
    demo_clip_grad_norm()
    demo_half_precision()
