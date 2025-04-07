from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
from pathlib import Path
import os.path as osp
import sapien.physx as physx

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("ThreeCup-v1", max_episode_steps=50)
class ThreeCupEnv(BaseEnv):
    """
    **Task Description:**
    The goal is to pick up the cup that contains a ball after the cups have been randomly moved around

    **Randomizations:**
    - all cups have their xy positions on top of the table scene randomized. The positions are sampled such that the cups do not collide with each other
    - the ball is placed inside one of the cups randomly

    **Success Conditions:**
    - the robot successfully grasps the cup containing the ball
    - the robot lifts the cup to a target position
    """

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, 
        ball_cup_idx=None, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.ball_cup_idx = ball_cup_idx  # 可以指定球放在哪个杯子下，如果为None则随机
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # 创建三个杯子
        papercup_builder = self.scene.create_actor_builder()
        table_assets_dir = Path(osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__))))) / "utils" / "scene_builder" / "table" / "assets"
        papercup_file = str(table_assets_dir / "papercup.glb")
        
        # 创建杯子的物理材质
        cup_physical_material = physx.PhysxMaterial(
            static_friction=0.5,    # 静摩擦系数
            dynamic_friction=0.4,  
            restitution=0.1         # 弹性恢复系数（碰撞后回弹程度）
        )

        papercup_builder.add_nonconvex_collision_from_file(
            filename=papercup_file,
            pose=sapien.Pose(),
            scale=(1, 1, 1),
            material=cup_physical_material,
            patch_radius=0,
            min_patch_radius=0,
            is_trigger=False,
        )
        # 添加视觉模型
        papercup_builder.add_visual_from_file(papercup_file)
        
        cup_mass = 0.1
        papercup_builder.set_mass_and_inertia(
            mass=cup_mass,
            cmass_local_pose=sapien.Pose(),
            inertia=np.array([0.001, 0.001, 0.001])
        )
        # 创建三个杯子物体
        self.papercup = papercup_builder.build(name="papercup")
        self.papercup2 = papercup_builder.build(name="papercup2")
        self.papercup3 = papercup_builder.build(name="papercup3")
        
        # 创建小球
        ball_builder = self.scene.create_actor_builder()
        # 创建小球的物理材质 - 较高摩擦力可以防止滑落
        ball_physical_material = physx.PhysxMaterial(
            static_friction=0.8,    # 高静摩擦系数
            dynamic_friction=0.7,   # 高动摩擦系数
            restitution=0.2         # 稍高的弹性
        )
        
        # 设置小球密度 - 小球应该有一定质量以保持稳定
        ball_density = 100  # 标准物体密度
        
        ball_builder.add_sphere_visual(radius=0.02, material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 1]))  
        ball_builder.add_sphere_collision(
            radius=0.01,
            material=ball_physical_material,
            density=ball_density
        )
        self.ball = ball_builder.build(name="ball")
        
        # 定义三个杯子的固定位置
        self.cup_positions = [
            np.array([-0.1, 0.1, 0.05]),  # 左上
            np.array([0.0, 0.0, 0.05]),   # 中间
            np.array([0.1, -0.1, 0.05])   # 右下
        ]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # 将三个杯子放在固定位置上方
            for i, cup in enumerate([self.papercup, self.papercup2, self.papercup3]):
                # 初始位置在固定位置上方
                initial_pos = torch.tensor(self.cup_positions[i], device=self.device)
                initial_pos[2] = 0.4  # 放在上方
                
                # 设置杯子的位置和朝向
                cup.set_pose(Pose.create_from_pq(
                    p=initial_pos, 
                    q=torch.tensor([1, 0, 0, 0], device=self.device)
                ))
            
            # 确定小球放在哪个杯子下方
            if self.ball_cup_idx is None:
                # 随机选择一个杯子放置小球
                cup_idx = np.random.randint(0, 3)
            else:
                # 使用指定的杯子索引
                cup_idx = self.ball_cup_idx
                
            # 保存当前回合中球所在的杯子索引
            self.current_ball_cup_idx = cup_idx
            
            # 获取杯子位置
            cups = [self.papercup, self.papercup2, self.papercup3]
            selected_cup_pos = self.cup_positions[cup_idx]
            
            # 将小球放在选中的杯子位置
            ball_pos = torch.tensor(selected_cup_pos, device=self.device)
            ball_pos[2] = 0.02  # 小球放在桌面上，杯子会罩住它
            self.ball.set_pose(Pose.create_from_pq(p=ball_pos, q=torch.tensor([1, 0, 0, 0], device=self.device)))
            
            # 让杯子落下罩住小球
            for i, cup in enumerate([self.papercup, self.papercup2, self.papercup3]):
                # 目标位置是固定位置
                target_pos = torch.tensor(self.cup_positions[i], device=self.device)
                
                # 设置杯子的位置和朝向
                cup.set_pose(Pose.create_from_pq(
                    p=target_pos, 
                    q=torch.tensor([1, 0, 0, 0], device=self.device)
                ))

    def evaluate(self):
        # 检查是否成功抓取了包含小球的杯子
        ball_pos = self.ball.pose.p
        cup1_pos = self.papercup.pose.p
        cup2_pos = self.papercup2.pose.p
        cup3_pos = self.papercup3.pose.p
        
        # 计算小球到各个杯子的距离
        dist1 = torch.linalg.norm(ball_pos - cup1_pos, axis=1)
        dist2 = torch.linalg.norm(ball_pos - cup2_pos, axis=1)
        dist3 = torch.linalg.norm(ball_pos - cup3_pos, axis=1)
        
        # 找出距离最小的杯子
        min_dist = torch.minimum(torch.minimum(dist1, dist2), dist3)
        target_cup = None
        if torch.all(min_dist == dist1):
            target_cup = self.papercup
        elif torch.all(min_dist == dist2):
            target_cup = self.papercup2
        else:
            target_cup = self.papercup3
        
        # 检查是否成功抓取了目标杯子
        is_target_cup_grasped = self.agent.is_grasping(target_cup)
        
        # 检查杯子是否被提起
        cup_height = target_cup.pose.p[:, 2]
        is_cup_lifted = cup_height > 0.1
        
        success = is_target_cup_grasped & is_cup_lifted
        
        return {
            "is_target_cup_grasped": is_target_cup_grasped,
            "is_cup_lifted": is_cup_lifted,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                papercup_pose=self.papercup.pose.raw_pose,
                papercup2_pose=self.papercup2.pose.raw_pose,
                papercup3_pose=self.papercup3.pose.raw_pose,
                ball_pose=self.ball.pose.raw_pose,
                tcp_to_papercup_pos=self.papercup.pose.p - self.agent.tcp.pose.p,
                tcp_to_papercup2_pos=self.papercup2.pose.p - self.agent.tcp.pose.p,
                tcp_to_papercup3_pos=self.papercup3.pose.p - self.agent.tcp.pose.p,
                tcp_to_ball_pos=self.ball.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        
        # 确定哪个杯子中有小球
        ball_pos = self.ball.pose.p
        cup1_pos = self.papercup.pose.p
        cup2_pos = self.papercup2.pose.p
        cup3_pos = self.papercup3.pose.p
        
        # 计算小球到各个杯子的距离
        dist1 = torch.linalg.norm(ball_pos - cup1_pos, axis=1)
        dist2 = torch.linalg.norm(ball_pos - cup2_pos, axis=1)
        dist3 = torch.linalg.norm(ball_pos - cup3_pos, axis=1)
        
        # 找出距离最小的杯子
        min_dist = torch.minimum(torch.minimum(dist1, dist2), dist3)
        target_cup = None
        if torch.all(min_dist == dist1):
            target_cup = self.papercup
        elif torch.all(min_dist == dist2):
            target_cup = self.papercup2
        else:
            target_cup = self.papercup3
        
        # 计算机械臂到目标杯子的距离
        target_cup_to_tcp_dist = torch.linalg.norm(tcp_pose - target_cup.pose.p, axis=1)
        reward = 2 * (1 - torch.tanh(5 * target_cup_to_tcp_dist))
        
        # 抓取奖励
        is_target_cup_grasped = self.agent.is_grasping(target_cup)
        reward[is_target_cup_grasped] = 4
        
        # 提起奖励
        cup_height = target_cup.pose.p[:, 2]
        is_cup_lifted = cup_height > 0.1
        reward[is_cup_lifted] = 6
        
        # 成功奖励
        success = is_target_cup_grasped & is_cup_lifted
        reward[success] = 8
        
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8
