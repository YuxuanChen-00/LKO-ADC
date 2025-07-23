# generate_reference_square.py

import numpy as np
import warnings
from scipy.interpolate import interp1d


def generate_reference_square(center_pos: np.ndarray,
                              dist_center_to_vertex: float,
                              total_steps: int) -> np.ndarray:
    """
    生成以z轴为法向量的正方形轨迹。

    Args:
        center_pos (np.ndarray): [xc, yc, zc] 正方形中心点位置 (基坐标系)。
        dist_center_to_vertex (float): 中心点到顶点的长度。
        total_steps (int): 轨迹上的总点数 (建议 >= 4)。

    Returns:
        np.ndarray: 3xN矩阵，每列包括位置[x; y; z]。
    """
    # --- 参数检查 ---
    if total_steps < 4:
        warnings.warn("总点数(total_steps)小于4，轨迹可能无法表示完整的正方形。")

    # --- 坐标与尺寸计算 ---
    xc, yc, zc = center_pos

    # 计算正方形的半边长、边长和总周长
    half_side_length = dist_center_to_vertex / np.sqrt(2)
    L = half_side_length
    side_length = 2 * L
    perimeter = 4 * side_length

    # --- 定义轨迹路径的关键点 ---
    # 定义四个顶点 (以右下角为起点，逆时针顺序)
    v1 = np.array([xc + L, yc - L])
    v2 = np.array([xc + L, yc + L])
    v3 = np.array([xc - L, yc + L])
    v4 = np.array([xc - L, yc - L])

    # 创建一个闭环的路径，包含5个点 (起点、三个顶点、回到起点)
    path_waypoints_x = np.array([v1[0], v2[0], v3[0], v4[0], v1[0]])
    path_waypoints_y = np.array([v1[1], v2[1], v3[1], v4[1], v1[1]])

    # 定义路径上每个关键点对应的累计距离
    path_distances = np.array([0, side_length, 2 * side_length, 3 * side_length, perimeter])

    # --- 均匀插值生成轨迹点 ---
    # 在总周长上生成 'total_steps' 个均匀分布的采样距离
    # 采用 (0:N-1)/N * P 的方式，可以使生成的点均匀分布，且终点不会与起点重合
    sample_distances = np.arange(total_steps) * (perimeter / total_steps)

    # 使用线性插值来计算每个采样距离上的x, y坐标
    interp_func_x = interp1d(path_distances, path_waypoints_x)
    interp_func_y = interp1d(path_distances, path_waypoints_y)

    x = interp_func_x(sample_distances)
    y = interp_func_y(sample_distances)

    # z坐标保持不变
    z = zc * np.ones_like(x)

    # --- 合并轨迹数据 ---
    trajectory = np.vstack((x, y, z))

    return trajectory


if __name__ == '__main__':
    # --- 示例 ---
    center = np.array([5, 5, 10])
    distance_to_vertex = 5
    steps = 100
    square_traj = generate_reference_square(center, distance_to_vertex, steps)

    print("生成的正方形轨迹形状:", square_traj.shape)

    # --- 可视化 ---
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(square_traj[0, :], square_traj[1, :], '-o', markersize=2)
        ax.scatter(center[0], center[1], color='red', label='Center')
        ax.set_title("Generated Square Trajectory (XY Plane)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.axis('equal')
        ax.legend()
        plt.show()
    except ImportError:
        print("Matplotlib未安装，无法进行可视化。")