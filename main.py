import numpy as np

# 生成伯努利分布的随机事件
def bernoulli(size, p1=0.3, p2=0.6):

    A = np.random.choice([True, False], size, p=[p1, 1-p1])
    B = np.random.choice([True, False], size, p=[p2, 1-p2])
    return A, B


# 非等概率的二元事件
def random_choice(num_samples, p1=0.5, p2=0.5):

    A = np.random.choice([True, False], num_samples, p=[p1, 1 - p1])
    B = np.random.choice([True, False], num_samples, p=[p2, 1 - p2])

    return A, B

# 等概率的二元事件
def random_choice_equal(num_samples, p1=0.5, p2=0.5):
    
    return random_choice(num_samples, p1, p2)

# 生成指数分布的随机事件
def exponential(size, lambda1=0.3, lambda2=0.6, threshold=0.5):
    A = np.random.exponential(lambda1, size) > threshold
    B = np.random.exponential(lambda2, size) > threshold
    return A, B

# 生成随机事件的函数
generate_event = random_choice_equal
#generate_event = random_choice
#generate_event = bernoulli
#generate_event = exponential

def p(X):
    """
    计算 P(X)
    """
    return np.mean(X)

def pp(A, B):
    """
    计算 P(A|B)
    """
    return np.mean(A[B])

# 计算相关系数
def calc_corr(x,y):
    corr_matrix = np.corrcoef(x, y)
    corr = corr_matrix[0, 1]

    return corr

threshold = 0.5

print("预热...")
# 模拟随机事件 A 和 B
# 假设 A 和 B 都是二元等概率事件，即 True 或 False
num_samples = 10000
A, B = generate_event(num_samples)

print("P(A|B) =", pp(A, B))

print("P(B|A) =", pp(B, A))

print("P(A) =", p(A))

print("P(B) =", p(B))

print("验证 p(A|B) = p(B|A)p(A) / p(B)")
print("left =", pp(A, B))
print("right =", pp(B, A) * p(A) / p(B))
print("肉眼观察，确保没有问题。如果 left 和 right 相差不大，则说明各个函数的实现是正确")
print("预热结束")
print("======== 开始验证 p(A|B) ∝ p(B|A)p(A) 和 p(A|B) ∝ p(B|A)/p(B) ========")
s = 100
n = 1000
correct = 0
for i in range(s):

    x1 = []
    x2 = []
    x3 = []
    for i in range(n):
        A, B = generate_event(num_samples)
        x1.append(pp(A, B))
        x2.append(pp(B, A) * p(A))
        x3.append(pp(B, A) / p(B))

    c1 = np.array(x1)
    c2 = np.array(x2)
    c3 = np.array(x3)
    corr1 = calc_corr(c1,c2)
    corr2 = calc_corr(c1,c3)
    if corr1 > threshold and corr2 > threshold:
        correct += 1
print("======== 结束 ========")

print("判断标准: 相关系数 > ", threshold)
print("样本数量:", num_samples)
print("总样本数量:", num_samples * n * s)
print("序列长度:", n)
print("计算次数",s,"次")
print("正确率:", correct / s)
print("如果正确率接近1, 则可以相信: p(A|B) ∝ p(B|A)p(A) 和 p(A|B) ∝ p(B|A)/p(B) 是成立的")


