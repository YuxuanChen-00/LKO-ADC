# generate_reference_circle.py

import numpy as np
from typing import Union


def generate_reference_circle(center: np.ndarray,
                              radius_or_start: Union[float, np.ndarray],
                              steps: int) -> np.ndarray:
    """
    在三维空间中生成一个圆形参考轨迹。

    Args:
        center (np.ndarray): 一个包含3个元素的NumPy数组 [xc, yc, zc]，指定圆心的坐标。
        radius_or_start (Union[float, np.ndarray]):
            - 一个标量（float），代表圆的半径。
            - 一个包含3个元素的NumPy数组 [xs, ys, zs]，代表圆轨迹的起始点。
        steps (int): 为轨迹生成的点的数量。

    Returns:
        np.ndarray: 一个 3xN 的矩阵 (N = steps)，代表圆上点的 [x; y; z] 坐标。
    """
    xc, yc, zc = center

    # 检查第二个参数是半径还是起始点
    if np.isscalar(radius_or_start):
        # 情况1：第二个参数是标量（半径）
        radius = float(radius_or_start)

        # 生成角度。圆将从标准的0弧度位置（x轴正方向）开始。
        theta = np.linspace(0, 2 * np.pi, steps)

    else:
        # 情况2：第二个参数是向量（起始点）
        start_point = radius_or_start
        xs, ys, _ = start_point
        # 注意：我们假设起始点的z坐标与圆心定义的平面一致。
        # 我们使用xs和ys在XY平面上定义圆。

        # 将圆心到起始点的欧几里得距离计算为半径
        radius = np.sqrt((xs - xc) ** 2 + (ys - yc) ** 2)

        # 处理半径为零的边缘情况（起始点即圆心）
        if radius == 0:
            x = xc * np.ones(steps)
            y = yc * np.ones(steps)
            z = zc * np.ones(steps)
            return np.vstack((x, y, z))

        # 计算给定起始点的初始角度
        theta_start = np.arctan2(ys - yc, xs - xc)

        # 从计算出的初始角度开始，生成一个完整的圆
        theta = np.linspace(theta_start, theta_start + 2 * np.pi, steps)

    # 为轨迹生成 (x, y, z) 坐标
    x = xc + radius * np.cos(theta)
    y = yc + radius * np.sin(theta)
    z = zc * np.ones_like(theta)  # z坐标是恒定的（圆所在的平面）

    # 将坐标合并到输出矩阵中
    trajectory = np.vstack((x, y, z))

    return trajectory


if __name__ == '__main__':
    # --- 示例 ---
    center_point = np.array([1, 2, 5])

    # 示例1: 使用半径
    traj_from_radius = generate_reference_circle(center_point, 10, 100)
    print("从半径生成的轨迹形状:", traj_from_radius.shape)

    # 示例2: 使用起始点
    start_point = np.array([11, 2, 5])  # (1,2,5) + (10,0,0)
    traj_from_start = generate_reference_circle(center_point, start_point, 100)
    print("从起始点生成的轨迹形状:", traj_from_start.shape)

    # 简单的可视化
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(traj_from_radius[0, :], traj_from_radius[1, :], traj_from_radius[2, :], label='From Radius')
        ax.scatter(start_point[0], start_point[1], start_point[2], color='red', s=50, label='Start Point')
        ax.set_title("Generated Circle Trajectory")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        ax.axis('equal')
        plt.show()
    except ImportError:
        print("Matplotlib未安装，无法进行可视化。")