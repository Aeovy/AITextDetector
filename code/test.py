import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# 设置字体以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def fx(x, point, k):
    y = np.where(x < point, 0.5 * x / point, 0.5 + 0.5 * (1 / (1 + np.exp(-k * (x - point)))))
    return y
def custom_mapping_function(x, point=0.9, k=100):
    if(x<point):
        y=0.5 * x / point
    else:
        y =  0.5 + 0.5 * (1 / (1 + np.exp(-k * (x - point))))
    return y
# x = np.linspace(0, 1, 1000)
# y = fx(x, 0.9, 100)
# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(['f(x)'])
# plt.grid(True)
# plt.show()

print(custom_mapping_function(0.8))