"""
动作原语定义和分类模块
======================

支持7维状态: [x, y, z, w, qx, qy, qz] + gripper
- 位置: xyz (米)
- 旋转: 四元数 wxyz
- Gripper: 开合度 (0-1)

坐标系定义 (右手坐标系):
- X轴: 正方向 = 右 (right), 负方向 = 左 (left)
- Y轴: 正方向 = 前 (forward), 负方向 = 后 (backward)
- Z轴: 正方向 = 上 (up), 负方向 = 下 (down)

旋转定义 (欧拉角):
- Roll: 绕X轴旋转
- Pitch: 绕Y轴旋转 (tilt up/down)
- Yaw: 绕Z轴旋转 (rotate clockwise/counterclockwise)
"""

import numpy as np
from typing import Tuple, List, Optional


def quat_to_euler(w: float, x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    将四元数(wxyz)转换为欧拉角(roll, pitch, yaw)
    
    Args:
        w, x, y, z: 四元数分量 (wxyz格式)
    
    Returns:
        (roll, pitch, yaw): 欧拉角 (弧度)
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def describe_move(move_vec: np.ndarray) -> str:
    """
    将运动向量转换为自然语言描述
    
    Args:
        move_vec: 7维运动向量 [dx, dy, dz, droll, dpitch, dyaw, dgripper]
                  每个分量取值 -1, 0, 1
    
    Returns:
        str: 动作描述
    """
    names = [
        {-1: "backward", 0: None, 1: "forward"},   # Y axis (前后)
        {-1: "left", 0: None, 1: "right"},          # X axis (左右)  
        {-1: "down", 0: None, 1: "up"},             # Z axis (上下)
        {},  # roll - 暂不使用
        {-1: "tilt up", 0: None, 1: "tilt down"},  # pitch
        {-1: "rotate clockwise", 0: None, 1: "rotate counterclockwise"},  # yaw
        {-1: "close gripper", 0: None, 1: "open gripper"},  # gripper
    ]
    
    # 注意: move_vec的顺序是 [x, y, z, roll, pitch, yaw, gripper]
    # 但坐标系定义是 X=right, Y=forward, Z=up
    # 所以需要映射: move_vec[0]=dx→right/left, move_vec[1]=dy→forward/backward, move_vec[2]=dz→up/down
    
    # 平移描述 (xyz)
    xyz_parts = []
    # dy (forward/backward)
    if move_vec[1] != 0:
        xyz_parts.append(names[0][int(move_vec[1])])
    # dx (left/right)
    if move_vec[0] != 0:
        xyz_parts.append(names[1][int(move_vec[0])])
    # dz (up/down)
    if move_vec[2] != 0:
        xyz_parts.append(names[2][int(move_vec[2])])
    
    if xyz_parts:
        description = "move " + " ".join(xyz_parts)
    else:
        description = ""
    
    # 旋转描述 - pitch (tilt)
    # 合并 roll 和 pitch（参考原始代码）
    if move_vec[3] == 0:
        move_vec[3] = move_vec[4]  # 如果roll为0，使用pitch
    
    if move_vec[4] != 0:  # pitch
        if description:
            description += ", "
        description += names[4][int(move_vec[4])]
    
    # 旋转描述 - yaw (rotate)
    if move_vec[5] != 0:
        if description:
            description += ", "
        description += names[5][int(move_vec[5])]
    
    # Gripper描述
    if move_vec[6] != 0:
        if description:
            description += ", "
        description += names[6][int(move_vec[6])]
    
    if not description:
        description = "stop"
    
    return description


def classify_movement(
    move: np.ndarray,
    pos_threshold: float = 0.02,
    rot_threshold: float = 0.05,
    gripper_threshold: float = 0.05,
    use_ratio_filter: bool = True,
    ratio_threshold: float = 0.3
) -> Tuple[str, np.ndarray]:
    """
    分类运动轨迹为动作原语
    
    Args:
        move: 状态序列 (window_size, 8) - [x, y, z, w, qx, qy, qz, gripper]
        pos_threshold: 位置变化阈值（米）
        rot_threshold: 旋转变化阈值（弧度）
        gripper_threshold: gripper变化阈值
        use_ratio_filter: 是否使用比例过滤（降噪）
        ratio_threshold: 比例阈值
    
    Returns:
        (description, move_vec): 动作描述和运动向量
    """
    # 计算起止点差异
    start_state = move[0]
    end_state = move[-1]
    
    # 位置差异 (xyz)
    dx = end_state[0] - start_state[0]
    dy = end_state[1] - start_state[1]
    dz = end_state[2] - start_state[2]
    
    # 旋转差异 (四元数 -> 欧拉角)
    start_euler = quat_to_euler(start_state[3], start_state[4], start_state[5], start_state[6])
    end_euler = quat_to_euler(end_state[3], end_state[4], end_state[5], end_state[6])
    
    droll = end_euler[0] - start_euler[0]
    dpitch = end_euler[1] - start_euler[1]
    dyaw = end_euler[2] - start_euler[2]
    
    # 处理角度跨越 ±π 的情况
    droll = np.arctan2(np.sin(droll), np.cos(droll))
    dpitch = np.arctan2(np.sin(dpitch), np.cos(dpitch))
    dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))
    
    # Gripper差异
    dgripper = end_state[7] - start_state[7]
    
    # 构建差异向量
    diff = np.array([dx, dy, dz, droll, dpitch, dyaw, dgripper])
    
    # 应用比例过滤（参考原始代码的归一化逻辑）
    if use_ratio_filter:
        # 位置归一化
        pos_total = np.sum(np.abs(diff[:3]))
        if pos_total > 3 * pos_threshold:
            diff[:3] *= 3 * pos_threshold / pos_total
        
        # 旋转缩放（旋转变化通常较小，需要放大权重）
        # 原始代码: diff[3:6] /= 10，这里我们使用不同的阈值来处理
    
    # 设置阈值向量
    thresholds = np.array([
        pos_threshold,      # dx
        pos_threshold,      # dy
        pos_threshold,      # dz
        rot_threshold,      # droll
        rot_threshold,      # dpitch
        rot_threshold,      # dyaw
        gripper_threshold   # dgripper
    ])
    
    # 量化为 -1, 0, 1
    move_vec = np.zeros(7, dtype=int)
    move_vec[diff > thresholds] = 1
    move_vec[diff < -thresholds] = -1
    
    # 生成描述
    description = describe_move(move_vec)
    
    return description, move_vec


def extract_primitives_from_trajectory(
    positions: np.ndarray,
    orientations: np.ndarray,
    grippers: np.ndarray,
    window_size: int = 4,
    pos_threshold: float = 0.02,
    rot_threshold: float = 0.05,
    gripper_threshold: float = 0.05
) -> List[dict]:
    """
    从轨迹中提取动作原语序列
    
    Args:
        positions: 位置序列 (N, 3) - [x, y, z]
        orientations: 四元数序列 (N, 4) - [w, x, y, z]
        grippers: gripper序列 (N,)
        window_size: 滑动窗口大小
        pos_threshold: 位置阈值
        rot_threshold: 旋转阈值
        gripper_threshold: gripper阈值
    
    Returns:
        List[dict]: 每帧的动作原语
    """
    num_steps = len(positions)
    
    # 组合状态: [x, y, z, w, qx, qy, qz, gripper]
    states = np.concatenate([
        positions,
        orientations,
        grippers.reshape(-1, 1)
    ], axis=1)
    
    primitives = []
    
    for i in range(num_steps):
        # 获取窗口
        end_idx = min(i + window_size, num_steps)
        window = states[i:end_idx]
        
        if len(window) < 2:
            # 不够长，复制上一个
            if primitives:
                primitives.append(primitives[-1].copy())
            else:
                primitives.append({
                    "primitive": "stop",
                    "move_vec": [0, 0, 0, 0, 0, 0, 0],
                    "position": positions[i].tolist(),
                    "orientation": orientations[i].tolist(),
                    "gripper": float(grippers[i])
                })
            continue
        
        description, move_vec = classify_movement(
            window,
            pos_threshold=pos_threshold,
            rot_threshold=rot_threshold,
            gripper_threshold=gripper_threshold
        )
        
        primitives.append({
            "primitive": description,
            "move_vec": move_vec.tolist(),
            "position": positions[i].tolist(),
            "orientation": orientations[i].tolist(),
            "gripper": float(grippers[i])
        })
    
    return primitives


# 测试代码
if __name__ == "__main__":
    # 测试四元数转欧拉角
    print("Testing quat_to_euler...")
    roll, pitch, yaw = quat_to_euler(1, 0, 0, 0)  # 单位四元数
    print(f"  Identity quaternion: roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f}")
    
    # 测试动作描述
    print("\nTesting describe_move...")
    test_vecs = [
        [0, 0, 0, 0, 0, 0, 0],   # stop
        [0, 1, 1, 0, 0, 0, 0],   # move forward up
        [1, 0, 0, 0, 0, 0, -1],  # move right, close gripper
        [0, 0, 0, 0, 1, 0, 0],   # tilt up
        [0, 0, 0, 0, 0, 1, 0],   # rotate clockwise
    ]
    for vec in test_vecs:
        desc = describe_move(np.array(vec))
        print(f"  {vec} -> {desc}")
    
    print("\nTests completed!")
