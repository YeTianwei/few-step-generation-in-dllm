"""
Clear CoT Label Module
======================
用于标注双臂机器人的动作原语和思维链(CoT)

主要模块:
- primitives: 动作原语定义和分类
- annotate_primitives: 从HDF5提取并标注动作原语
- generate_cot: 利用动作原语生成完整CoT
- visualize_cot: CoT可视化
- main: 主流程入口
"""

from .primitives import (
    describe_move,
    classify_movement,
    quat_to_euler,
)

__all__ = [
    "describe_move",
    "classify_movement", 
    "quat_to_euler",
]
