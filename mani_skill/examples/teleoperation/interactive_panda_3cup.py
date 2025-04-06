import argparse
from ast import parse
from typing import Annotated
import gymnasium as gym
import numpy as np
import sapien.core as sapien
from mani_skill.envs.sapien_env import BaseEnv
import time
import random

from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.motionplanner_stick import \
    PandaStickMotionPlanningSolver
import sapien.utils.viewer
import h5py
import json
import mani_skill.trajectory.utils as trajectory_utils
from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers.record import RecordEpisode
import tyro
from dataclasses import dataclass
from skvideo.io import vwrite
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PickCube-v1"
    obs_mode: str = "none"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "panda"
    """The robot to use. Robot setups supported for teleop in this script are panda and panda_stick"""
    record_dir: str = "demos"
    """directory to record the demonstration data and optionally videos"""
    save_video: bool = False
    """whether to save the videos of the demonstrations after collecting them all"""
    viewer_shader: str = "rt-fast"
    """the shader to use for the viewer. 'default' is fast but lower-quality shader, 'rt' and 'rt-fast' are the ray tracing shaders"""
    video_saving_shader: str = "rt-fast"
    """the shader to use for the videos of the demonstrations. 'minimal' is the fast shader, 'rt' and 'rt-fast' are the ray tracing shaders"""

scene_camera_frames = []

def parse_args() -> Args:
    return tyro.cli(Args)

def main(args: Args):
    output_dir = f"{args.record_dir}/{args.env_id}/teleop/"
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="none",
        enable_shadow=True,
        viewer_camera_configs=dict(shader_pack=args.viewer_shader)
    )
    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name="trajectory",
        save_video=False,
        info_on_video=False,
        source_type="teleoperation",
        source_desc="teleoperation via the click+drag system"
    )
    num_trajs = 0
    seed = 0
    env.reset(seed=seed)
    while True:
        print(f"Collecting trajectory {num_trajs+1}, seed={seed}")
        code = solve(env, debug=False, vis=True)
        if code == "quit":
            num_trajs += 1
            break
        elif code == "continue":
            seed += 1
            num_trajs += 1
            env.reset(seed=seed)
            continue
        elif code == "restart":
            env.reset(seed=seed, options=dict(save_trajectory=False))
    h5_file_path = env._h5_file.filename
    json_file_path = env._json_path
    env.close()
    del env
    print(f"Trajectories saved to {h5_file_path}")
    if args.save_video:
        print(f"Saving videos to {output_dir}")

        trajectory_data = h5py.File(h5_file_path)
        with open(json_file_path, "r") as f:
            json_data = json.load(f)
        env = gym.make(
            args.env_id,
            obs_mode=args.obs_mode,
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            reward_mode="none",
            human_render_camera_configs=dict(shader_pack=args.video_saving_shader),
        )
        env = RecordEpisode(
            env,
            output_dir=output_dir,
            trajectory_name="trajectory",
            save_video=True,
            info_on_video=False,
            save_trajectory=False,
            video_fps=30
        )
        for episode in json_data["episodes"]:
            traj_id = f"traj_{episode['episode_id']}"
            data = trajectory_data[traj_id]
            env.reset(**episode["reset_kwargs"])
            env_states_list = trajectory_utils.dict_to_list_of_dicts(data["env_states"])

            env.base_env.set_state_dict(env_states_list[0])
            for action in np.array(data["actions"]):
                env.step(action)

        trajectory_data.close()
        env.close()
        del env



def solve(env: BaseEnv, debug=False, vis=False):
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    robot_has_gripper = False
    if env.unwrapped.robot_uids == "panda_stick":
        planner = PandaStickMotionPlanningSolver(
            env,
            debug=debug,
            vis=vis,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            joint_acc_limits=0.5,
            joint_vel_limits=0.5,
        )
    elif env.unwrapped.robot_uids == "panda" or env.unwrapped.robot_uids == "panda_wristcam":
        robot_has_gripper = True
        planner = PandaArmMotionPlanningSolver(
            env,
            debug=debug,
            vis=vis,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            joint_acc_limits=0.5,
            joint_vel_limits=0.5,
        )
    viewer = env.render_human()

    last_checkpoint_state = None
    gripper_open = True
    def select_panda_hand():
        viewer.select_entity(sapien_utils.get_obj_by_name(env.agent.robot.links, "panda_hand")._objs[0].entity)
    
    def pick_cup_with_ball():
        print("Finding and picking up the cup containing the ball...")

        
        # 获取三个杯子和球的引用
        papercup1 = env.unwrapped.papercup
        papercup2 = env.unwrapped.papercup2
        papercup3 = env.unwrapped.papercup3
        ball = env.unwrapped.ball
        # 确定哪个杯子中有球
        ball_pos = ball.pose.p.cpu().numpy().astype(np.float32).squeeze()
        cup_positions = {
            1: papercup1.pose.p.cpu().numpy().astype(np.float32).squeeze(),
            2: papercup2.pose.p.cpu().numpy().astype(np.float32).squeeze(),
            3: papercup3.pose.p.cpu().numpy().astype(np.float32).squeeze()
        }
        
        # 计算球到每个杯子的距离
        distances = {}
        for idx, pos in cup_positions.items():
            # 计算水平距离(仅xy平面)
            dist = np.sqrt((ball_pos[0] - pos[0])**2 + (ball_pos[1] - pos[1])**2)
            distances[idx] = dist
        
        # 找到离球最近的杯子
        ball_in_cup_idx = min(distances, key=distances.get)
        print(f"Ball is in cup {ball_in_cup_idx}")
        
        # 根据索引获取目标杯子
        cups = {
            1: papercup1,
            2: papercup2,
            3: papercup3
        }
        target_cup = cups[ball_in_cup_idx]
        
        # 为抓取准备参数
        FINGER_LENGTH = 0.025
        
        # 获取目标杯子的OBB用于计算抓取
        from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb
        obb = get_actor_obb(target_cup)
        
        # 计算抓取姿态
        approaching = np.array([0, 0, -1])  # 从上方接近
        target_closing = env.unwrapped.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=FINGER_LENGTH,
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        grasp_pose = env.unwrapped.agent.build_grasp_pose(approaching, closing, center)
        
        # 搜索有效的抓取姿态
        from transforms3d.euler import euler2quat
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
            return
        
        # 首先移动到杯子上方以避免碰撞
        print("Moving above the cup...")
        pre_grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
        result = planner.move_to_pose_with_screw(pre_grasp_pose, dry_run=True)
        if result != -1:
            _, reward, _, _, info = planner.follow_path(result)
            print(f"Reward: {reward}, Info: {info}")
        else:
            print("Failed to plan path to position above cup")
            return
        
        # 接近杯子
        print("Approaching the cup...")
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        result = planner.move_to_pose_with_screw(reach_pose, dry_run=True)
        if result != -1:
            _, reward, _, _, info = planner.follow_path(result)
            print(f"Reward: {reward}, Info: {info}")
        else:
            print("Failed to plan path to approach cup")
            return
        
        # 移动到抓取位置
        print("Moving to grasp position...")
        result = planner.move_to_pose_with_screw(grasp_pose, dry_run=True)
        if result != -1:
            _, reward, _, _, info = planner.follow_path(result)
            print(f"Reward: {reward}, Info: {info}")
        else:
            print("Failed to plan path to grasp position")
            return
        
        # 抓取
        print("Closing gripper...")
        _, reward, _, _, info = planner.close_gripper()
        print(f"Reward: {reward}, Info: {info}")
        
        # 提起
        print("Lifting cup...")
        lift_pose = sapien.Pose([0, 0, 0.2]) * grasp_pose
        result = planner.move_to_pose_with_screw(lift_pose, dry_run=True)
        if result != -1:
            _, reward, _, _, info = planner.follow_path(result)
            print(f"Reward: {reward}, Info: {info}")
        else:
            print("Failed to plan lifting path")
        
        print("Pick up operation complete")


    def get_cup_positions():
        """Determine which cup is left, middle, and right based on x coordinates"""
        papercup1 = env.unwrapped.papercup
        papercup2 = env.unwrapped.papercup2
        papercup3 = env.unwrapped.papercup3
        
        # Get x coordinates of cups
        cup_positions = {
            1: papercup1.pose.p.cpu().numpy().astype(np.float32).squeeze()[0],
            2: papercup2.pose.p.cpu().numpy().astype(np.float32).squeeze()[0],
            3: papercup3.pose.p.cpu().numpy().astype(np.float32).squeeze()[0]
        }
        
        # Sort cups by x coordinate
        sorted_cups = sorted(cup_positions.items(), key=lambda x: x[1])
        
        # Map cup indices to positions
        position_map = {
            sorted_cups[0][0]: "right",
            sorted_cups[1][0]: "middle",
            sorted_cups[2][0]: "left"
        }
        
        return position_map

    def auto_play():
        global scene_camera_frames  # 使用全局变量存储帧
        scene_camera_frames = []  # 清空之前可能存在的帧
        
        env.reset()
        print("Starting auto play sequence...")
        
        # 获取主要人类渲染相机
        human_render_cameras = env.unwrapped._human_render_cameras
        if not human_render_cameras:
            print("Warning: No human render cameras found.")
            return
        
        # 通常使用第一个相机，或者特定名称的相机如果知道的话
        camera_name = list(human_render_cameras.keys())[0]  # 获取第一个相机名称
        camera = human_render_cameras[camera_name]
        print(f"Using camera: {camera_name}")
        
        # 创建保存视频的目录
        os.makedirs("videos", exist_ok=True)
        
        # 捕获初始帧
        def capture_frame():
            camera.capture()  # 拍照
            images = camera.get_obs(rgb=True)  # 只获取RGB数据
            frame = images['rgb'].squeeze().cpu().numpy()
            scene_camera_frames.append(frame)
        
        # 1. 等待1秒显示初始状态
        print("Waiting for 1 second to show initial state...")
        env.render_human()
        
        # 不使用sleep，而是连续捕获多帧来创建视频中的1秒过渡
        # 假设最终视频帧率为30fps，捕获30帧表示1秒
        for _ in range(30):
            capture_frame()
        
        # Initialize movement record
        movement_record = {
            "initial_setup": {
                "description": f"Ball under the {get_cup_positions()[env.unwrapped.current_ball_cup_idx + 1]} cup",
                "start_frame": 0,
                "end_frame": 45
            },
            "movements": [],
            "final_state": {
                "description": "",
                "frame": 0
            }
        }

        # 让杯子落下并覆盖球
        print("Stepping environment to let cups settle...")
        planner.open_gripper()
        capture_frame()
        
        # 额外0.5秒，同样用多帧捕获替代sleep
        # 15帧表示0.5秒
        for _ in range(15):
            capture_frame()
        
        current_frame = len(scene_camera_frames)
        
        # 3. 执行3次随机杯子交换动画
        print("Performing 3 shuffle animations...")
        for i in range(3):
            # 随机选择两个不同的杯子索引(1, 2, 或 3)
            available_cups = [1, 2, 3]
            cup_idx1 = random.choice(available_cups)
            available_cups.remove(cup_idx1)
            cup_idx2 = random.choice(available_cups)
            
            # Get cup positions before swap
            cup_positions = get_cup_positions()
            cup1_pos = cup_positions[cup_idx1]
            cup2_pos = cup_positions[cup_idx2]
            
            print(f"Shuffle {i+1}/3: Swapping cup {cup_idx1} ({cup1_pos}) and cup {cup_idx2} ({cup2_pos})")
            
            # Record movement start
            movement_start_frame = current_frame
            
            # 使用全局的shuffle_with_capture函数
            shuffle_with_capture(cup_idx1, cup_idx2, capture_frame=capture_frame)
            
            # Update current frame count
            current_frame = len(scene_camera_frames)
            
            # Record movement
            movement_record["movements"].append({
                "description": f"swap: {cup1_pos} cup, {cup2_pos} cup",
                "start_frame": movement_start_frame,
                "end_frame": current_frame
            })
            
            # 在交换之间暂停
            time.sleep(0.25)
            capture_frame()
            current_frame = len(scene_camera_frames)
        
        # 4. 保存视频
        frames_array = np.array(scene_camera_frames)
        print(f"Saving video with {len(frames_array)} frames")
        
        # 确保帧格式正确，并调整为适合视频的格式
        if frames_array.max() <= 1.0:
            frames_array = (frames_array * 255).astype(np.uint8)
        
        # 保存视频
        video_path = f"videos/cup_shuffle_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        vwrite(video_path, frames_array)
        print(f"Auto play complete - video saved to {video_path}")
        
        # 捕获最后一帧的分割图
        print("Capturing segmentation map of the final frame...")
        camera.capture()
        images = camera.get_obs(rgb=True, segmentation=True)
        segmentation = images['segmentation'].squeeze().cpu().numpy()
        cup_name_map = {
            0: "papercup",
            1: "papercup2",
            2: "papercup3"
        }
        ball_cup_idx = env.unwrapped.current_ball_cup_idx
        ball_cup_name = cup_name_map[ball_cup_idx]
        # 查找包含球的杯子的分割ID
        segmentation_id_map = env.unwrapped.segmentation_id_map
        for seg_id, obj in segmentation_id_map.items():
            if hasattr(obj, 'name') and obj.name == ball_cup_name:
                ball_cup_seg_id = seg_id
                break
        
        if ball_cup_seg_id is not None:
            # 创建只包含球的杯子的分割图
            ball_cup_mask = (segmentation == ball_cup_seg_id)
            ball_cup_segmentation = np.zeros_like(segmentation)
            ball_cup_segmentation[ball_cup_mask] = ball_cup_seg_id
            
            # 获取杯子的边界框
            y_coords, x_coords = np.where(ball_cup_mask)
            if len(y_coords) > 0 and len(x_coords) > 0:
                min_x, max_x = np.min(x_coords), np.max(x_coords)
                min_y, max_y = np.min(y_coords), np.max(y_coords)
                bounding_box = {
                    "min_x": int(min_x),
                    "min_y": int(min_y),
                    "max_x": int(max_x),
                    "max_y": int(max_y)
                }
                print(f"Ball cup bounding box: {bounding_box}")
                
                # 获取最后一帧的RGB图像
                camera.capture()
                rgb_images = camera.get_obs(rgb=True)
                last_frame_rgb = rgb_images['rgb'].squeeze().cpu().numpy()
                
                # 在RGB图像上绘制边界框
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(last_frame_rgb)
                
                # 创建矩形补丁表示边界框
                rect = patches.Rectangle(
                    (bounding_box["min_x"], bounding_box["min_y"]),
                    bounding_box["max_x"] - bounding_box["min_x"],
                    bounding_box["max_y"] - bounding_box["min_y"],
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect)
                
                # 添加标签
                cup_position = get_cup_positions()[env.unwrapped.current_ball_cup_idx + 1]
                ax.text(
                    bounding_box["min_x"], 
                    bounding_box["min_y"] - 10, 
                    f"Ball under {cup_position} cup", 
                    color='red', 
                    fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.7)
                )
                
                # 保存带有边界框的图像
                bbox_image_path = video_path.replace('.mp4', '_with_bbox.png')
                plt.savefig(bbox_image_path)
                plt.close()
                print(f"Image with bounding box saved to {bbox_image_path}")
            else:
                bounding_box = None
                print("Could not find bounding box for the ball cup")

        
        # Add final state to movement record
        movement_record["final_state"] = {
            "description": f"Ball is under the {get_cup_positions()[env.unwrapped.current_ball_cup_idx + 1]} cup",
            "frame": current_frame
        }
        movement_record["ball_cup_bounding_box"] = bounding_box
        # Save movement record
        json_path = video_path.replace('.mp4', '_movements.json')
        with open(json_path, 'w') as f:
            json.dump(movement_record, f, indent=2)
        print(f"Movement record saved to {json_path}")
        print(f"Final state: {movement_record['final_state']['description']}")
    
    def shuffle_with_capture(cup_idx1=None, cup_idx2=None, capture_frame=None):
        """
        执行杯子交换动画
        
        Args:
            cup_idx1: 第一个杯子的索引 (1, 2, 或 3)，如果为None则随机选择
            cup_idx2: 第二个杯子的索引 (1, 2, 或 3)，如果为None则随机选择
            capture_frame: 可选的帧捕获函数，用于视频录制，如果为None则不录制
        """
        # 如果没有指定杯子索引，则随机选择
        if cup_idx1 is None or cup_idx2 is None:
            available_cups = [1, 2, 3]
            if cup_idx1 is None:
                cup_idx1 = random.choice(available_cups)
                available_cups.remove(cup_idx1)
            if cup_idx2 is None:
                cup_idx2 = random.choice(available_cups)
        
        print(f"Playing shuffle animation between cup {cup_idx1} and cup {cup_idx2}...")
        # 获取杯子和球的引用
        papercup1 = env.unwrapped.papercup
        papercup2 = env.unwrapped.papercup2
        papercup3 = env.unwrapped.papercup3
        ball = env.unwrapped.ball  # 获取球对象
        # Set ball to kinematic(no force applied)
        ball._bodies[0].kinematic = True

        # 动画前设置杯子不受重力影响
        papercup1._bodies[0].disable_gravity = True
        papercup2._bodies[0].disable_gravity = True
        papercup3._bodies[0].disable_gravity = True

        papercup1_original_pose = sapien.Pose(
            p=papercup1.pose.p.cpu().numpy().astype(np.float32).squeeze(),
            q=papercup1.pose.q.cpu().numpy().astype(np.float32).squeeze()
        )
        papercup2_original_pose = sapien.Pose(
            p=papercup2.pose.p.cpu().numpy().astype(np.float32).squeeze(),
            q=papercup2.pose.q.cpu().numpy().astype(np.float32).squeeze()
        )
        papercup3_original_pose = sapien.Pose(
            p=papercup3.pose.p.cpu().numpy().astype(np.float32).squeeze(),
            q=papercup3.pose.q.cpu().numpy().astype(np.float32).squeeze()
        )
        
        # 根据输入索引获取指定的杯子
        cups = {
            1: (papercup1, papercup1_original_pose),
            2: (papercup2, papercup2_original_pose),
            3: (papercup3, papercup3_original_pose)
        }
        # 验证杯子索引
        assert cup_idx1 in cups, f"Invalid cup index {cup_idx1}. Please use values 1, 2, or 3."
        assert cup_idx2 in cups, f"Invalid cup index {cup_idx2}. Please use values 1, 2, or 3."
        assert cup_idx1 != cup_idx2, "Please select two different cups."
        
        # 根据输入定义要交换的杯子
        cup1, cup1_pose = cups[cup_idx1]
        cup2, cup2_pose = cups[cup_idx2]
        
        # 确定哪个杯子包含球
        ball_pos = ball.pose.p.cpu().numpy().astype(np.float32).squeeze()
        cup_positions = {
            1: papercup1.pose.p.cpu().numpy().astype(np.float32).squeeze(),
            2: papercup2.pose.p.cpu().numpy().astype(np.float32).squeeze(),
            3: papercup3.pose.p.cpu().numpy().astype(np.float32).squeeze()
        }
        
        # 计算球到每个杯子的距离
        distances = {}
        for idx, pos in cup_positions.items():
            # 计算水平距离(仅xy平面)
            dist = np.sqrt((ball_pos[0] - pos[0])**2 + (ball_pos[1] - pos[1])**2)
            distances[idx] = dist
        
        # 找到离球最近的杯子
        ball_in_cup_idx = min(distances, key=distances.get)
        ball_offset = ball_pos - cup_positions[ball_in_cup_idx]  # 球在杯中的相对位置
        
        print(f"Ball is in cup {ball_in_cup_idx}")
        
        # 总动画帧数
        total_frames = 60
        
        # 获取起始和结束位置
        start_pos1 = cup1_pose.p
        start_pos2 = cup2_pose.p
        
        # 保持恒定的z高度(停留在桌面上)
        z_height1 = start_pos1[2]
        z_height2 = start_pos2[2]
        
        # 计算杯子之间的中点(作为半圆的中心)
        midpoint_x = (start_pos1[0] + start_pos2[0]) / 2
        midpoint_y = (start_pos1[1] + start_pos2[1]) / 2
        
        # 根据杯子之间的距离计算半径
        radius = np.sqrt((start_pos1[0] - start_pos2[0])**2 + (start_pos1[1] - start_pos2[1])**2) / 2
        
        # 计算每个杯子的初始角度
        angle1 = np.arctan2(start_pos1[1] - midpoint_y, start_pos1[0] - midpoint_x)
        angle2 = np.arctan2(start_pos2[1] - midpoint_y, start_pos2[0] - midpoint_x)
        
        # 计算半圆轨迹点
        for i in range(total_frames + 1):
            t = i / total_frames  # 从0到1的归一化时间
            
            # 使用半圆插值计算新角度(180度 = π弧度)
            new_angle1 = angle1 + t * np.pi
            new_angle2 = angle2 + t * np.pi
            
            # 使用圆周运动计算杯子的新位置
            cup1_x = midpoint_x + radius * np.cos(new_angle1)
            cup1_y = midpoint_y + radius * np.sin(new_angle1)
            
            cup2_x = midpoint_x + radius * np.cos(new_angle2)
            cup2_y = midpoint_y + radius * np.sin(new_angle2)
            
            # 设置杯子的新位置，保持原始z高度
            cup1.set_pose(sapien.Pose(
                p=np.array([cup1_x, cup1_y, z_height1], dtype=np.float32),
                q=cup1_pose.q
            ))
            
            cup2.set_pose(sapien.Pose(
                p=np.array([cup2_x, cup2_y, z_height2], dtype=np.float32),
                q=cup2_pose.q
            ))
            
            # 更新球的位置以跟随它所在的杯子
            if ball_in_cup_idx == cup_idx1:
                # 球在cup1中，更新球的位置以跟随cup1
                new_ball_pos = np.array([cup1_x, cup1_y, z_height1], dtype=np.float32) + ball_offset
                ball.set_pose(sapien.Pose(
                    p=new_ball_pos,
                    q=ball.pose.q.cpu().numpy().astype(np.float32).squeeze()
                ))
            elif ball_in_cup_idx == cup_idx2:
                # 球在cup2中，更新球的位置以跟随cup2
                new_ball_pos = np.array([cup2_x, cup2_y, z_height2], dtype=np.float32) + ball_offset
                ball.set_pose(sapien.Pose(
                    p=new_ball_pos,
                    q=ball.pose.q.cpu().numpy().astype(np.float32).squeeze()
                ))
            
            # 更新模拟状态以更新视觉状态
            qpos = planner.robot.get_qpos()
            planner.robot.set_qpos(qpos)
            env.scene.step()
            env.render_human()
            
            # 如果提供了捕获函数，则捕获当前帧
            if capture_frame is not None:
                capture_frame()
            
            time.sleep(0.01)  # 平滑动画的短暂延迟
        
        print("Shuffle animation completed")

        # 动画后恢复杯子
        papercup1._bodies[0].disable_gravity = False
        papercup2._bodies[0].disable_gravity = False
        papercup3._bodies[0].disable_gravity = False

        return ball_in_cup_idx  # 返回球所在杯子的索引
    
    select_panda_hand()
    for plugin in viewer.plugins:
        if isinstance(plugin, sapien.utils.viewer.viewer.TransformWindow):
            transform_window = plugin
    while True:

        transform_window.enabled = True
        # transform_window.update_ghost_objects
        # print(transform_window.ghost_objects, transform_window._gizmo_pose)
        # planner.grasp_pose_visual.set_pose(transform_window._gizmo_pose)

        env.render_human()
        execute_current_pose = False
        if viewer.window.key_press("h"):
            print("""Available commands:
            h: print this help menu
            g: toggle gripper to close/open (if there is a gripper)
            u: move the panda hand up
            j: move the panda hand down
            arrow_keys: move the panda hand in the direction of the arrow keys
            n: execute command via motion planning to make the robot move to the target pose indicated by the ghost panda arm
            f: play shuffle animation - you will be prompted to enter two cup numbers (1-3) to swap. The ball will follow the cup it's in
            c: stop this episode and record the trajectory and move on to a new episode
            q: quit the script and stop collecting data. Save trajectories and optionally videos.
            z: start one demo
            b: pick up the cup containing the ball using motion planning
            """)
            pass
        # elif viewer.window.key_press("k"):
        #     print("Saving checkpoint")
        #     last_checkpoint_state = env.get_state_dict()
        # elif viewer.window.key_press("l"):
        #     if last_checkpoint_state is not None:
        #         print("Loading previous checkpoint")
        #         env.set_state_dict(last_checkpoint_state)
        #     else:
        #         print("Could not find previous checkpoint")
        elif viewer.window.key_press("q"):
            return "quit"
        elif viewer.window.key_press("c"):
            return "continue"
        # elif viewer.window.key_press("r"):
        #     viewer.select_entity(None)
        #     return "restart"
        # elif viewer.window.key_press("t"):
        #     # TODO (stao): change from position transform to rotation transform
        #     pass
        elif viewer.window.key_press("n"):
            execute_current_pose = True
        elif viewer.window.key_press("z"):
            auto_play()
        elif viewer.window.key_press("f"):
            shuffle_with_capture()
        elif viewer.window.key_press("b"):
            pick_cup_with_ball()
        elif viewer.window.key_press("g") and robot_has_gripper:
            if gripper_open:
                gripper_open = False
                _, reward, _ ,_, info = planner.close_gripper()
            else:
                gripper_open = True
                _, reward, _ ,_, info = planner.open_gripper()
            print(f"Reward: {reward}, Info: {info}")
        elif viewer.window.key_press("u"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, 0, -0.01])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("j"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, 0, +0.01])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("down"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[+0.01, 0, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("up"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[-0.01, 0, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("right"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, -0.01, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("left"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, +0.01, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        if execute_current_pose:
            # z-offset of end-effector gizmo to TCP position is hardcoded for the panda robot here
            if env.unwrapped.robot_uids == "panda" or env.unwrapped.robot_uids == "panda_wristcam":
                result = planner.move_to_pose_with_screw(transform_window._gizmo_pose * sapien.Pose([0, 0, 0.1]), dry_run=True)
            elif env.unwrapped.robot_uids == "panda_stick":
                result = planner.move_to_pose_with_screw(transform_window._gizmo_pose * sapien.Pose([0, 0, 0.15]), dry_run=True)
            if result != -1 and len(result["position"]) < 150:
                _, reward, _ ,_, info = planner.follow_path(result)
                print(f"Reward: {reward}, Info: {info}")
            else:
                if result == -1: print("Plan failed")
                else: print("Generated motion plan was too long. Try a closer sub-goal")
            execute_current_pose = False

    return args
if __name__ == "__main__":
    main(parse_args())
  