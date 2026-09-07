# 這堂課手打驗證過的核心邏輯,完整可執行版本在 numpy_version.py

output = np.maximum(0, weights @ inputs + bias)


# ---------- matmul:矩陣乘法,對照C++三層迴圈矩陣乘法理解 ----------

class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def matmul(self, other):
        # 矩陣乘法(真正的matrix multiply,不是逐項相乘)
        # 形狀規則:(m,n) @ (n,p) = (m,p),中間的n要對上
        # 對照C++三層迴圈:i是外層列、j是中層行、k是內層做內積加總
        # for(i) for(j) { sum=0; for(k) sum += A[i][k]*B[k][j]; result[i][j]=sum; }
        if self.cols != other.rows:
            raise ValueError(
                f"Cannot multiply shapes {self.shape} and {other.shape}: "
                f"inner dimensions {self.cols} != {other.rows}"
            )
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])


if __name__ == "__main__":
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])
    result = a.matmul(b)
    print(f"a @ b = {result.data}")  # 應該是 [[19, 22], [43, 50]]
