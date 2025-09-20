import numpy as np

# 旋转矩阵
R = np.array([
     [
        -0.0,
        0.0,
        -1.0
      ],
      [
        -0.2588190451025208,
        0.9659258262890684,
        0.0
      ],
      [
        0.9659258262890684,
        0.2588190451025208,
        -0.0
      ]
])

# 计算欧拉角
theta_x = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
theta_y = np.arctan2(R[2, 1], R[2, 2])
theta_z = np.arctan2(-R[0, 0], R[1, 0])

# 将弧度转换为度
theta_x_deg = np.degrees(theta_x)
theta_y_deg = np.degrees(theta_y)
theta_z_deg = np.degrees(theta_z)

print("欧拉角 (度):")
print(f"X: {theta_x_deg}, Y: {theta_y_deg}, Z: {theta_z_deg}")