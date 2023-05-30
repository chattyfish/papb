import numpy as np

def p(X):
    """
    计算 P(X)
    """
    return np.mean(X)

def p_A_given_B(A, B):
    """
    计算 P(A|B)
    """
    return np.mean(A[B])

# 计算相关系数
def calc_corr(x,y):
    corr_matrix = np.corrcoef(x, y)
    corr = corr_matrix[0, 1]

    return corr

# sigmoid 函数, 帮助判断 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

threshold = 0.6224593312018546

print("预热...")
# 模拟随机事件 A 和 B
# 假设 A 和 B 都是二元事件，即 True 或 False
# 这里仅作示例，您可以根据实际情况进行修改
num_samples = 10000
A = np.random.choice([True, False], num_samples)
B = np.random.choice([True, False], num_samples)

print("P(A|B) =", p_A_given_B(A, B))

print("P(B|A) =", p_A_given_B(B, A))

print("P(A) =", p(A))

print("P(B) =", p(B))

print("验证 p(A|B) = p(B|A)p(A) / p(B)")
print("left =", p_A_given_B(A, B))
print("right =", p_A_given_B(B, A) * p(A) / p(B))
print("手动比对，如果 left 和 right 相差不大，则说明预热成功")

print("======== 开始 ========")
s = 100
correct = 0
for i in range(s):
    n = 100

    print("计算",n,"次")    
    print("验证 p(A|B) ∝ p(B|A)p(A)")

    x1 = []
    x2 = []
    for i in range(n):
        A = np.random.choice([True, False], num_samples)
        B = np.random.choice([True, False], num_samples)
        x1.append(p_A_given_B(A, B))
        x2.append(p_A_given_B(B, A) * p(A))

    left = np.array(x1)
    right = np.array(x2)
    corr1 = calc_corr(left,right)
    print("相关系数(-1 ~ 1):", corr1)
    print("激活：", sigmoid(corr1))
    if sigmoid(corr1) > threshold:
        print("验证通过")


    print("计算",n,"次")    
    print("验证 p(A|B) ∝ p(B|A) / p(B)")

    x1 = []
    x2 = []
    for i in range(n):
        A = np.random.choice([True, False], num_samples)
        B = np.random.choice([True, False], num_samples)
        x1.append(p_A_given_B(A, B))
        x2.append(p_A_given_B(B, A) / p(B))

    left = np.array(x1)
    right = np.array(x2)
    corr2 = calc_corr(left,right)
    print("相关系数(-1 ~ 1):", corr2)
    print("激活：", sigmoid(corr2))
    if sigmoid(corr2) > threshold:
        print("验证通过")

    if sigmoid(corr1) > threshold and sigmoid(corr2) > threshold:
        correct += 1
print("======== 结束 ========")
print("判断标准: sigmoid(相关系数) > ", threshold)
print("正确率:", correct / s)
print("如果正确率接近1, 则说明: p(A|B) ∝ p(B|A)p(A) 和 p(A|B) ∝ p(B|A) / p(B) 是等价的")


