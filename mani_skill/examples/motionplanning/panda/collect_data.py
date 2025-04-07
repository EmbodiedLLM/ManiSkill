import os
import time
import json
import random
import numpy as np
import argparse
from tqdm import tqdm
import os.path as osp
import gymnasium as gym
import sapien.core as sapien
from skvideo.io import vwrite
import matplotlib.pyplot as plt
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.examples.motionplanning.panda.solutions import solveThreeCup as solve
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.motionplanner_stick import \
    PandaStickMotionPlanningSolver
import matplotlib.patches as patches



class DataCollector:
    def __init__(self, env_id="ThreeCup-v1", render_mode=None):
        self.env_id = env_id
        self.render_mode = render_mode
        self.env = gym.make(
            self.env_id,
            obs_mode=None,
            control_mode="pd_joint_pos",
            render_mode=render_mode,
            sensor_configs=dict(shader_pack="default"),
            human_render_camera_configs=dict(shader_pack="default"),
            viewer_camera_configs=dict(shader_pack="default"),
            sim_backend="cpu"
        )
        self.fps = 50
        self.human_render_cameras = self.env.unwrapped._human_render_cameras
        assert self.human_render_cameras, "Warning: No human render cameras found."
        self.camera_name = list(self.human_render_cameras.keys())[0]  # 获取第一个相机名称
        self.camera = self.human_render_cameras[self.camera_name]
        print(f"Using camera: {self.camera_name}")
        new_traj_name = time.strftime("%Y%m%d_%H%M%S")
        # self.env = RecordEpisode(
        #     self.env,
        #     output_dir=osp.join("demos", self.env_id, "motionplanning"),
        #     trajectory_name=new_traj_name, save_video=False,
        #     source_type="motionplanning",
        #     source_desc="official motion planning solution from ManiSkill contributors",
        #     video_fps=self.fps,
        #     record_reward=False,
        #     save_on_reset=False
        # )
        self.env.reset()
        self.scene_camera_frames = []
        self.papercup1 = self.env.unwrapped.papercup
        self.papercup2 = self.env.unwrapped.papercup2
        self.papercup3 = self.env.unwrapped.papercup3
        self.ball = self.env.unwrapped.ball
        project_path = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
        os.makedirs(osp.join(project_path, "videos"), exist_ok=True)

        assert self.env.unwrapped.control_mode in [
            "pd_joint_pos",
            "pd_joint_pos_vel",
        ], self.env.unwrapped.control_mode
        self.robot_has_gripper = False
        debug = False
        vis = True
        if self.env.unwrapped.robot_uids == "panda_stick":
            self.planner = PandaStickMotionPlanningSolver(
                self.env,
                debug=debug,
                vis=vis,
                base_pose=self.env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                joint_acc_limits=0.5,
                joint_vel_limits=0.5,
            )
        elif self.env.unwrapped.robot_uids == "panda" or self.env.unwrapped.robot_uids == "panda_wristcam":
            self.robot_has_gripper = True
            self.planner = PandaArmMotionPlanningSolver(
                self.env,
                debug=debug,
                vis=vis,
                base_pose=self.env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                joint_acc_limits=0.5,
                joint_vel_limits=0.5,
            )
        # Initialize viewer only if not using rgb_array mode
        if self.render_mode != "rgb_array":
            self.viewer = self.env.render_human()
        else:
            self.env.render_rgb_array()


    def capture_frame(self):
        self.camera.capture()  # 拍照
        images = self.camera.get_obs(rgb=True)  # 只获取RGB数据
        frame = images['rgb'].squeeze().cpu().numpy()
        self.scene_camera_frames.append(frame)
        
        # For rgb_array mode, update the environment render
        if self.render_mode == "rgb_array":
            qpos = self.planner.robot.get_qpos()
            self.planner.robot.set_qpos(qpos)
            self.env.scene.step()
            # No need to call render_human here as we're capturing directly
    
    
    
    def get_cup_positions(self):
        """Determine which cup is left, middle, and right based on x coordinates"""
        
        # Get x coordinates of cups
        cup_positions = {
            1: self.papercup1.pose.p.cpu().numpy().astype(np.float32).squeeze()[0],
            2: self.papercup2.pose.p.cpu().numpy().astype(np.float32).squeeze()[0],
            3: self.papercup3.pose.p.cpu().numpy().astype(np.float32).squeeze()[0]
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

    
    def collect_single_task(self, shuffle_count=3):
        self.env.reset()
        self.timestamp = time.strftime('%Y%m%d_%H%M%S')
        os.makedirs(f"videos/{self.timestamp}", exist_ok=True)
        self.scene_camera_frames = []
        print("Waiting for 1 second to show initial state...")
        
        # Only create viewer if not in rgb_array mode
        if self.render_mode != "rgb_array":
            self.viewer = self.env.render_human()
        else:
            self.env.render_rgb_array()
            
        for _ in range(self.fps):
            self.capture_frame()

        # 让杯子落下并覆盖球
        print("Stepping environment to let cups settle...")
        self.planner.open_gripper()
        self.capture_frame()

        # capture 0.5 seconds
        for _ in range(int(0.5 * self.fps)):
            self.capture_frame()

        current_frame = len(self.scene_camera_frames)

        # Initialize movement record
        movement_record = {
            "initial_setup": {
                "description": f"Ball under the {self.get_cup_positions()[self.env.unwrapped.current_ball_cup_idx + 1]} cup",
                "start_frame": 0,
                "end_frame": current_frame,
                "shuffle_count": shuffle_count
            },
            "movements": [],
            "final_state": {
                "description": "",
                "frame": 0
            }
        }


        # 3. 执行3次随机杯子交换动画
        print("Performing 3 shuffle animations...")
        for i in range(shuffle_count):
            # 随机选择两个不同的杯子索引(1, 2, 或 3)
            available_cups = [1, 2, 3]
            cup_idx1 = random.choice(available_cups)
            available_cups.remove(cup_idx1)
            cup_idx2 = random.choice(available_cups)
            
            # Get cup positions before swap
            cup_positions = self.get_cup_positions()
            cup1_pos = cup_positions[cup_idx1]
            cup2_pos = cup_positions[cup_idx2]
        
            print(f"Shuffle {i+1}/{shuffle_count}: Swapping cup {cup_idx1} ({cup1_pos}) and cup {cup_idx2} ({cup2_pos})")
            
            # Record movement start
            movement_start_frame = current_frame
            
            # 使用全局的shuffle_with_capture函数
            self.shuffle_with_capture(cup_idx1, cup_idx2, capture_frame=self.capture_frame)
            
            # Update current frame count
            current_frame = len(self.scene_camera_frames)
            
            # Record movement
            movement_record["movements"].append({
                "description": f"swap: {cup1_pos} cup, {cup2_pos} cup",
                "start_frame": movement_start_frame,
                "end_frame": current_frame
            })
                    
            # 在交换之间暂停
            time.sleep(0.25)
            self.capture_frame()
            current_frame = len(self.scene_camera_frames)

        # 4. 保存视频
        frames_array = np.array(self.scene_camera_frames)
        print(f"Saving video with {len(frames_array)} frames")

        # 确保帧格式正确，并调整为适合视频的格式
        if frames_array.max() <= 1.0:
            frames_array = (frames_array * 255).astype(np.uint8)

        # 保存视频
        video_path = f"videos/{self.timestamp}/{self.timestamp}_task.mp4"
        vwrite(video_path, frames_array)
        print(f"Auto play complete - video saved to {video_path}")

        # 捕获最后一帧的分割图
        print("Capturing segmentation map of the final frame...")
        self.camera.capture()
        images = self.camera.get_obs(rgb=True, segmentation=True)
        segmentation = images['segmentation'].squeeze().cpu().numpy()
        cup_name_map = {
            0: "papercup",
            1: "papercup2",
            2: "papercup3"
        }
        ball_cup_idx = self.env.unwrapped.current_ball_cup_idx
        ball_cup_name = cup_name_map[ball_cup_idx]
        # 查找包含球的杯子的分割ID
        segmentation_id_map = self.env.unwrapped.segmentation_id_map
        ball_cup_seg_id = None
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
            
            # 获取图像尺寸
            img_height, img_width = segmentation.shape
            
            # 转换为相对坐标（0-1范围）
            relative_bounding_box = {
                "min_x": float(min_x) / img_width,
                "min_y": float(min_y) / img_height,
                "max_x": float(max_x) / img_width,
                "max_y": float(max_y) / img_height
            }
            
            # 保留绝对坐标用于绘制
            absolute_bounding_box = {
                "min_x": int(min_x),
                "min_y": int(min_y),
                "max_x": int(max_x),
                "max_y": int(max_y)
            }
            
            print(f"Ball cup relative bounding box: {relative_bounding_box}")
            print(f"Ball cup absolute bounding box: {absolute_bounding_box}")
            
            # 获取最后一帧的RGB图像
            self.camera.capture()
            rgb_images = self.camera.get_obs(rgb=True)
            last_frame_rgb = rgb_images['rgb'].squeeze().cpu().numpy()
            
            # 在RGB图像上绘制边界框
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(last_frame_rgb)
            
            # 创建矩形补丁表示边界框（使用绝对坐标）
            rect = patches.Rectangle(
                (absolute_bounding_box["min_x"], absolute_bounding_box["min_y"]),
                absolute_bounding_box["max_x"] - absolute_bounding_box["min_x"],
                absolute_bounding_box["max_y"] - absolute_bounding_box["min_y"],
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            
            # 添加标签
            cup_position = self.get_cup_positions()[self.env.unwrapped.current_ball_cup_idx + 1]
            ax.text(
                absolute_bounding_box["min_x"], 
                absolute_bounding_box["min_y"] - 10, 
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
            
            # 使用相对坐标作为边界框
            bounding_box = relative_bounding_box
        else:
            bounding_box = None
            print("Could not find bounding box for the ball cup")
        
        
        # Add final state to movement record
        movement_record["final_state"] = {
            "description": f"Ball is under the {self.get_cup_positions()[self.env.unwrapped.current_ball_cup_idx + 1]} cup",
            "frame": current_frame
        }
        movement_record["ball_cup_relative_bounding_box"] = bounding_box
        
        # Save movement record
        json_path = video_path.replace('.mp4', '_movements.json')
        with open(json_path, 'w') as f:
            json.dump(movement_record, f, indent=2)
        print(f"Movement record saved to {json_path}")
        print(f"Final state: {movement_record['final_state']['description']}")
        
        # 5. 添加抓取操作并记录视频
        print("\n--- Beginning pick up operation ---")
        
        # 创建新的帧列表用于抓取动作视频
        action_frames = []
        
        def capture_action_frame():
            self.camera.capture()
            images = self.camera.get_obs(rgb=True)
            frame = images['rgb'].squeeze().cpu().numpy()
            action_frames.append(frame)
        
        # 暂停1秒显示初始状态
        for _ in range(30):
            capture_action_frame()
        
        # 执行抓取操作并捕获帧和轨迹
        ball_cup_idx, trajectory_data = self.pick_cup_with_ball(capture_frame=capture_action_frame, record_trajectory=True)
        
        # 保存抓取动作视频
        if action_frames:
            action_frames_array = np.array(action_frames)
            if action_frames_array.max() <= 1.0:
                action_frames_array = (action_frames_array * 255).astype(np.uint8)
            
            action_video_path = f"videos/{self.timestamp}/{self.timestamp}_action.mp4"
            vwrite(action_video_path, action_frames_array)
            print(f"Pick up action video saved to {action_video_path}")
            
            # 创建描述文件
            action_json_path = action_video_path.replace('.mp4', '_metadata.json')
            action_info = {
                "description": f"Picking up the {self.get_cup_positions()[ball_cup_idx]} cup containing the ball",
                "timestamp": self.timestamp,
                "frames": len(action_frames)
            }
            with open(action_json_path, 'w') as f:
                json.dump(action_info, f, indent=2)
            print(f"Action info saved to {action_json_path}")
            
            # 保存轨迹数据
            if trajectory_data:
                # 保存为JSONL格式 (JSON Lines)，每行一个JSON对象
                jsonl_trajectory_file = action_video_path.replace('.mp4', '_trajectory.jsonl')
                with open(jsonl_trajectory_file, 'w') as f:
                    # 将每个时间点的数据转换为单独的JSON行
                    for i in range(len(trajectory_data["timestamps"])):
                        # 确保数据可以被JSON序列化
                        def convert_numpy(obj):
                            if isinstance(obj, np.ndarray):
                                return obj.tolist()
                            elif isinstance(obj, np.number):
                                return obj.item()  # 将numpy标量转换为Python标量
                            elif isinstance(obj, list):
                                return [convert_numpy(item) for item in obj]
                            else:
                                return obj
                        
                        # 准备数据
                        frame_data = {
                            "frame_idx": i,  # 使用帧索引代替时间戳
                            "joint_positions": convert_numpy(trajectory_data["joint_positions"][i]),
                            "eef_positions": convert_numpy(trajectory_data["eef_positions"][i]),
                            "eef_orientations": convert_numpy(trajectory_data["eef_orientations"][i]),
                            "description": trajectory_data["descriptions"][i]
                        }
                        f.write(json.dumps(frame_data) + '\n')
                print(f"Trajectory data (JSONL format) saved to {jsonl_trajectory_file}")

    # Add a custom follow_path method to capture frames consistently at the desired FPS
    def follow_path_with_capture(self, planner, result, capture_frame=None, record_trajectory_point=None, description=""):
        """
        Follow the planned path while capturing frames at the desired FPS rate.
        
        Args:
            planner: The motion planner instance
            result: The result from the motion planning call
            capture_frame: Function to capture a frame
            record_trajectory_point: Function to record trajectory points
            description: Description of this motion for trajectory recording
        """
        if record_trajectory_point is not None:
            record_trajectory_point(f"starting_{description}")
            
        n_step = result["position"].shape[0]
        
        # Calculate how many steps to skip to maintain the proper fps
        # The motion planner runs at its own internal rate, but we want to capture at our fps rate
        # We estimate the total time of the motion based on the number of steps and control timestep
        estimated_duration = n_step * self.env.unwrapped.control_timestep
        total_frames_needed = int(estimated_duration * self.fps)
        
        # We need at least 1 frame per step, and we need to distribute frames evenly
        frames_per_step = max(1, total_frames_needed // n_step)
        
        for i in range(n_step):
            qpos = result["position"][i]
            
            # Get the gripper state from the planner
            has_gripper = hasattr(planner, 'gripper_state')
            gripper_state = planner.gripper_state if has_gripper else None
            
            # Create the appropriate action based on control mode
            if planner.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][i]
                if has_gripper:
                    action = np.hstack([qpos, qvel, [gripper_state]])
                else:
                    action = np.hstack([qpos, qvel])
            else:
                if has_gripper:
                    action = np.hstack([qpos, [gripper_state]])
                else:
                    action = qpos
            
            # Add batch dimension if needed for the environment
            if self.env.unwrapped.action_space.shape[0] != len(action):
                action = action.reshape(1, -1)
                
            # Take the step in the environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Capture frames at the desired rate 
            for _ in range(frames_per_step):
                if capture_frame is not None:
                    capture_frame()
                if record_trajectory_point is not None:
                    record_trajectory_point(f"{description}_in_progress")
            
            # Update the planner's internal state
            planner.elapsed_steps += 1
            
            # Render if visualization is enabled
            if planner.vis:
                planner.base_env.render_human()
        
        if record_trajectory_point is not None:
            record_trajectory_point(f"finished_{description}")
            
        return obs, reward, terminated, truncated, info
        
    def pick_cup_with_ball(self, capture_frame=None, record_trajectory=False):
        print("Finding and picking up the cup containing the ball...")
        action_frames = []  # Store frames if we're capturing
        
        # 初始化轨迹记录数据结构
        if record_trajectory:
            trajectory_data = {
                "joint_positions": [],  # 存储机械臂关节位置
                "eef_positions": [],    # 存储末端执行器位置
                "eef_orientations": [], # 存储末端执行器方向
                "timestamps": [],       # 时间戳
                "descriptions": []      # 每个轨迹点的描述
            }
        
        # 定义记录轨迹的函数
        def record_trajectory_point(description=""):
            if record_trajectory:
                # 获取关节位置
                qpos = self.planner.robot.get_qpos().cpu().numpy()
                # 获取末端执行器位置和方向
                eef_pose = self.env.unwrapped.agent.tcp.pose
                eef_pos = eef_pose.p.cpu().numpy()
                eef_ori = eef_pose.q.cpu().numpy()
                
                # 记录数据
                trajectory_data["joint_positions"].append(qpos.tolist())
                trajectory_data["eef_positions"].append(eef_pos.tolist())
                trajectory_data["eef_orientations"].append(eef_ori.tolist())
                trajectory_data["timestamps"].append(time.time())
                trajectory_data["descriptions"].append(description)

        # 获取三个杯子和球的引用
        papercup1 = self.papercup1
        papercup2 = self.papercup2
        papercup3 = self.papercup3
        ball = self.ball
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
        
        # Capture frame if requested
        if capture_frame is not None:
            capture_frame()
        # 记录初始轨迹点
        record_trajectory_point("initial_pose")
        
        # 为抓取准备参数
        FINGER_LENGTH = 0.025
        
        # 获取目标杯子的OBB用于计算抓取
        from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb
        obb = get_actor_obb(target_cup)
        
        # 计算抓取姿态
        approaching = np.array([0, 0, -1])  # 从上方接近
        target_closing = self.env.unwrapped.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=FINGER_LENGTH,
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        grasp_pose = self.env.unwrapped.agent.build_grasp_pose(approaching, closing, center)
        
        # 搜索有效的抓取姿态
        from transforms3d.euler import euler2quat
        angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
        angles = np.repeat(angles, 2)
        angles[1::2] *= -1
        for angle in angles:
            delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
            grasp_pose2 = grasp_pose * delta_pose
            res = self.planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
            if res == -1:
                continue
            grasp_pose = grasp_pose2
            break
        else:
            print("Fail to find a valid grasp pose")
            return None, None if record_trajectory else None
        
        # 首先移动到杯子上方以避免碰撞
        print("Moving above the cup...")
        pre_grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
        result = self.planner.move_to_pose_with_screw(pre_grasp_pose, dry_run=True)
        if result != -1:
            # 使用我们自定义的follow_path方法来确保以合适的帧率捕获
            _, reward, _, _, info = self.follow_path_with_capture(
                self.planner, 
                result, 
                capture_frame=capture_frame,
                record_trajectory_point=record_trajectory_point,
                description="moving_above_cup"
            )
            print(f"Reward: {reward}, Info: {info}")
        else:
            print("Failed to plan path to position above cup")
            return None, None if record_trajectory else None
        
        # 接近杯子
        print("Approaching the cup...")
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        result = self.planner.move_to_pose_with_screw(reach_pose, dry_run=True)
        if result != -1:
            # 使用我们自定义的follow_path方法来确保以合适的帧率捕获
            _, reward, _, _, info = self.follow_path_with_capture(
                self.planner, 
                result, 
                capture_frame=capture_frame,
                record_trajectory_point=record_trajectory_point,
                description="approaching_cup"
            )
            print(f"Reward: {reward}, Info: {info}")
        else:
            print("Failed to plan path to approach cup")
            return None, None if record_trajectory else None
        
        # 移动到抓取位置
        print("Moving to grasp position...")
        result = self.planner.move_to_pose_with_screw(grasp_pose, dry_run=True)
        if result != -1:
            # 使用我们自定义的follow_path方法来确保以合适的帧率捕获
            _, reward, _, _, info = self.follow_path_with_capture(
                self.planner, 
                result, 
                capture_frame=capture_frame,
                record_trajectory_point=record_trajectory_point,
                description="moving_to_grasp"
            )
            print(f"Reward: {reward}, Info: {info}")
        else:
            print("Failed to plan path to grasp position")
            return None, None if record_trajectory else None
        
        # 抓取
        print("Closing gripper...")
        record_trajectory_point("before_closing_gripper")
        _, reward, _, _, info = self.planner.close_gripper()
        print(f"Reward: {reward}, Info: {info}")
        record_trajectory_point("after_closing_gripper")
        
        # Capture multiple frames during gripper closing for smoother video
        if capture_frame is not None:
            for i in range(int(0.5*self.fps)):  # Capture several frames during gripper closing
                capture_frame()
                record_trajectory_point("closing_gripper")
        
        # 提起
        print("Lifting cup...")
        lift_pose = sapien.Pose([0, 0, 0.2]) * grasp_pose
        result = self.planner.move_to_pose_with_screw(lift_pose, dry_run=True)
        if result != -1:
            # 使用我们自定义的follow_path方法来确保以合适的帧率捕获
            _, reward, _, _, info = self.follow_path_with_capture(
                self.planner, 
                result, 
                capture_frame=capture_frame,
                record_trajectory_point=record_trajectory_point,
                description="lifting_cup"
            )
            print(f"Reward: {reward}, Info: {info}")
        else:
            print("Failed to plan lifting path")
        
        # Hold the pose for a moment at the end for video
        if capture_frame is not None:
            for i in range(self.fps):
                capture_frame()
                record_trajectory_point("holding_final_pose")
        
        print("Pick up operation complete")
        
        # Return the index of the cup containing the ball and trajectory data if recording
        return (ball_in_cup_idx, trajectory_data) if record_trajectory else ball_in_cup_idx
    
    def shuffle_with_capture(self, cup_idx1=None, cup_idx2=None, capture_frame=None):
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
        papercup1 = self.papercup1
        papercup2 = self.papercup2
        papercup3 = self.papercup3
        ball = self.ball
        # Set ball to kinematic(no force applied)
        ball._bodies[0].kinematic = True

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
        total_frames = 2 * self.fps
        
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

            # # # 动画前设置杯子不受重力影响
            # papercup1._bodies[0].disable_gravity = True
            # papercup2._bodies[0].disable_gravity = True
            # papercup3._bodies[0].disable_gravity = True

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
            # 使用固定的四元数 [1,0,0,0] 确保杯子始终竖直向下
            cup1.set_pose(sapien.Pose(
                p=np.array([cup1_x, cup1_y, z_height1], dtype=np.float32),
                q=np.array([1, 0, 0, 0], dtype=np.float32)
            ))
            
            cup2.set_pose(sapien.Pose(
                p=np.array([cup2_x, cup2_y, z_height2], dtype=np.float32),
                q=np.array([1, 0, 0, 0], dtype=np.float32)
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
            qpos = self.planner.robot.get_qpos()
            self.planner.robot.set_qpos(qpos)
            self.env.scene.step()
            
            # Only render if not using rgb_array mode
            if self.render_mode != "rgb_array":
                self.env.render_human()
            
            # 如果提供了捕获函数，则捕获当前帧
            if capture_frame is not None:
                capture_frame()

            # # 动画后恢复杯子
            # papercup1._bodies[0].disable_gravity = False
            # papercup2._bodies[0].disable_gravity = False
            # papercup3._bodies[0].disable_gravity = False

            
            time.sleep(0.01)  # 平滑动画的短暂延迟
        
        print("Shuffle animation completed")

        return ball_in_cup_idx  # 返回球所在杯子的索引
    

if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description='Collect data for ThreeCup task')
    parser.add_argument('--data_iters', type=int, default=2, help='Number of data iterations to collect')
    parser.add_argument('--shuffle_count', type=int, default=7, help='Number of shuffle animations per iteration (ignored if random_shuffle is True)')
    parser.add_argument('--random_shuffle', action='store_true', help='Use random shuffle count between 3 and 10')
    args = parser.parse_args()
    
    # Use arguments from command line
    data_collection = DataCollector(render_mode="human_render")
    for i in range(args.data_iters):
        print(f"\n--- Starting data collection iteration {i+1}/{args.data_iters} ---\n")
        
        # Determine shuffle count - either fixed or random
        if args.random_shuffle:
            actual_shuffle_count = random.randint(3, 10)
            print(f"Using random shuffle count: {actual_shuffle_count}")
        else:
            actual_shuffle_count = args.shuffle_count
            
        data_collection.collect_single_task(shuffle_count=actual_shuffle_count)
