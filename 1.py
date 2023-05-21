import torch
print(torch.__version__)# 学校：湖北工业大学
# 开发人员：Barbaric Growth
# 开发时间：2022/11/7 16:18
import numpy as np
indices = np.array([[2,3], [4,5]])
values = np.array([10, 20])
shape = (6,)
result = np.zeros(shape)
np.scatter_add(result, indices, values)
print("a.shapre",indices.shape())
print(result)