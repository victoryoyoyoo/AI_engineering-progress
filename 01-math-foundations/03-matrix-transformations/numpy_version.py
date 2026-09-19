import numpy as np

# NumPy 版本:把手刻的旋轉矩陣、eigenvalue求解，換成現成的函式呼叫

# 旋轉矩陣:np.cos / np.sin 對應 reference.py 裡的 math.cos / math.sin,差別是能直接處理陣列
theta = np.pi / 4  # 45 度,弧度制
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

point = np.array([1.0, 0.0])
# R @ point:矩陣乘向量;預期 (0.7071, 0.7071)
print(f"Rotate (1,0) by 45 deg: {R @ point}")

# np.diag([2, 3]):把一維陣列放到對角線,其餘補 0,得到 [[2,0],[0,3]],也就是縮放矩陣
S = np.diag([2.0, 3.0])
# S @ R:複合變換,從右邊先算 = 先旋轉(R)再縮放(S)
composed = S @ R
print(f"Scale(2,3) after Rotate(45): {composed @ point}")

# eigenvalue/eigenvector:一行取代手刻的 eigenvalues_2x2 + eigenvector_2x2
A = np.array([[2, 1], [1, 2]], dtype=float)
# np.linalg.eig 回傳兩個東西:eigenvalues(一維陣列)、eigenvectors(矩陣,每一「欄」是一個特徵向量)
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors (columns):\n{eigenvectors}")

for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]  # 每一行(column)是一個eigenvector，跟平常「一列一筆資料」的直覺相反
    # [:, i] 是切片:第一個位置的 : 代表「所有列」,第二個位置 i 是指定第 i 欄
    lam = eigenvalues[i]
    # 驗證定義 A v = λ v:兩邊印出來的數字應該相同(可能差正負號或浮點數尾巴)
    print(f"  A @ v{i} = {A @ v}, lambda * v{i} = {lam * v}")

# np.linalg.det:行列式;旋轉的 det=1(面積不變),縮放 (2,3) 的 det=6(面積放大 6 倍)
print(f"\ndet(R) = {np.linalg.det(R):.4f}")
print(f"det(S) = {np.linalg.det(S):.1f}")

# Eigendecomposition: A = V @ D @ V^-1，驗證能不能重建出原本的矩陣
# 意義:把矩陣拆成「換到 eigenvector 座標系(V^-1)→ 沿各軸伸縮(D)→ 換回來(V)」
B = np.array([[3, 1], [0, 2]], dtype=float)
vals, vecs = np.linalg.eig(B)
D = np.diag(vals)  # eigenvalue 放到對角線,得到對角矩陣 D
V = vecs           # eigenvector 組成的矩陣,每一欄一個
# np.linalg.inv:反矩陣;V @ D @ V^-1 應該還原出 B(浮點數誤差內)
reconstructed = V @ D @ np.linalg.inv(V)
print(f"\nEigendecomposition A = V @ D @ V^-1:")
print(f"Original:\n{B}")
print(f"Reconstructed:\n{reconstructed}")

# === 🟢 ===
# 3D旋轉(這堂課本體只教了2D，這裡順便看一下3D長怎樣)
# 3D 旋轉要指定「繞哪個軸」,被繞的軸那一格是 1、其餘位置放 2D 旋轉的 cos/sin
def rotation_3d_z(theta):
    # 繞 z 軸旋轉:z 座標不變(最後一列、最後一欄是 [0,0,1]),x-y 平面照 2D 旋轉
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rotation_3d_x(theta):
    # 繞 x 軸旋轉:x 座標不變(第一列、第一欄是 [1,0,0]),y-z 平面照 2D 旋轉
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

point_3d = np.array([1.0, 0.0, 0.0])
rotated_z = rotation_3d_z(np.pi / 2) @ point_3d  # (1,0,0) 繞 z 轉 90 度 → (0,1,0)
rotated_x = rotation_3d_x(np.pi / 2) @ point_3d  # (1,0,0) 剛好在 x 軸上,繞 x 轉不會動 → (1,0,0)

print(f"\n3D point: {point_3d}")
# np.round(x, 4):把 6e-17 這種浮點數誤差四捨五入成 0,輸出比較乾淨
print(f"Rotate 90 around z: {np.round(rotated_z, 4)}")
print(f"Rotate 90 around x: {np.round(rotated_x, 4)}")
