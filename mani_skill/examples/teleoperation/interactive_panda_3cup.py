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
        
        # 让杯子落下并覆盖球
        print("Stepping environment to let cups settle...")
        planner.open_gripper()
        capture_frame()
        
        # 额外0.5秒，同样用多帧捕获替代sleep
        # 15帧表示0.5秒
        for _ in range(15):
            capture_frame()
        
        # 3. 执行3次随机杯子交换动画
        print("Performing 3 shuffle animations...")
        for i in range(3):
            # 随机选择两个不同的杯子索引(1, 2, 或 3)
            available_cups = [1, 2, 3]
            cup_idx1 = random.choice(available_cups)
            available_cups.remove(cup_idx1)
            cup_idx2 = random.choice(available_cups)
            
            print(f"Shuffle {i+1}/3: Swapping cup {cup_idx1} and cup {cup_idx2}")
            
            # 修改perform_shuffle_animation以捕获每个动画帧
            def shuffle_with_capture(cup_idx1, cup_idx2):
                print(f"Playing shuffle animation between cup {cup_idx1} and cup {cup_idx2}...")
                # 获取杯子和球的引用
                papercup1 = env.unwrapped.papercup
                papercup2 = env.unwrapped.papercup2
                papercup3 = env.unwrapped.papercup3
                ball = env.unwrapped.ball  # 获取球对象

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
                    
                    capture_frame()
                    
                    time.sleep(0.01)  # 平滑动画的短暂延迟
                
                print("Shuffle animation completed")
                
            # 执行带捕获的交换动画
            shuffle_with_capture(cup_idx1, cup_idx2)
            
            # 在交换之间暂停
            time.sleep(0.25)
            capture_frame()
        
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
            auto_play()
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
