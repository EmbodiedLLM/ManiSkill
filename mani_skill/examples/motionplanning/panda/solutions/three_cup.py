import argparse
import gymnasium as gym
import numpy as np
import sapien
import time
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import StackCubeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.wrappers.record import RecordEpisode

def solve(env: StackCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    FINGER_LENGTH = 0.025
    env = env.unwrapped
    
    # 获取三个杯子的位置
    cup1 = env.papercup
    cup2 = env.papercup2
    cup3 = env.papercup3
    
    # 获取小球的位置
    ball = env.ball
    
    # 等待一段时间，让用户看到初始状态
    if vis:
        for _ in range(30):  # 等待1秒
            env.render_human()
            time.sleep(1/30)
    
    # # 确定哪个杯子中有小球
    # target_cup = None
    # if env.ball_cup_idx == 0:
    #     target_cup = cup1
    # elif env.ball_cup_idx == 1:
    #     target_cup = cup2
    # else:
    #     target_cup = cup3
    
    # 获取目标杯子的OBB
    obb = get_actor_obb(cup2)
    
    # 计算抓取姿态
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    
    # 搜索有效的抓取姿态
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose * delta_pose
        res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print("Fail to find a valid grasp pose")
        return -1
    
    # -------------------------------------------------------------------------- #
    # 接近目标
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    
    # -------------------------------------------------------------------------- #
    # 抓取
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()
    
    # -------------------------------------------------------------------------- #
    # 提起
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)
    
    # # 移动到目标位置
    # goal_pose = sapien.Pose(p=[0.3, 0.3, 0.2], q=grasp_pose.q)
    # planner.move_to_pose_with_screw(goal_pose)
    
    # # 释放
    # res = planner.open_gripper()
    planner.close()
    return res
