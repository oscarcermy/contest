import numpy as np

# 角度转换为弧度
theta_x = np.radians(71.867)
theta_y = np.radians(0)
theta_z = np.radians(90)

# 构造旋转矩阵
R_x = np.array([
    [1, 0, 0],
    [0, np.cos(theta_x), -np.sin(theta_x)],
    [0, np.sin(theta_x), np.cos(theta_x)]
])

R_y = np.array([
    [np.cos(theta_y), 0, np.sin(theta_y)],
    [0, 1, 0],
    [-np.sin(theta_y), 0, np.cos(theta_y)]
])

R_z = np.array([
    [np.cos(theta_z), -np.sin(theta_z), 0],
    [np.sin(theta_z), np.cos(theta_z), 0],
    [0, 0, 1]
])

R = np.dot(R_z, np.dot(R_y, R_x))

# 平移向量
t = np.array([7.3589, -6.9258, 4.9583])

print("旋转矩阵 R:")
print(R)
print("平移向量 t:")
print(t)