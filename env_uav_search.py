"""
ICCT 2023 论文复现 - 多无人机搜索环境
论文: Multi-UAV Searching Trajectory Optimization Algorithm based on Deep Reinforcement Learning
"""

import numpy as np
import random


class UAVSearchEnv:
    """多无人机协同搜索环境（基于论文描述）"""

    # 方向映射：0-7 对应 8 个方向
    DIRECTION_VECTORS = [
        (0, 0),  # 0: 停留（论文未明确，但需要）
        (0, 1),  # 1: 上
        (0, -1),  # 2: 下
        (1, 0),  # 3: 右
        (-1, 0),  # 4: 左
        (1, 1),  # 5: 右上
        (1, -1),  # 6: 右下
        (-1, 1),  # 7: 左上
        (-1, -1),  # 8: 左下
    ]
    # 论文图2：受转弯半径限制，只有 3 个方向可选（当前方向 + 左右相邻）
    _ALLOWED_DIR_OFFSETS = [-1, 0, 1]  # 相对于当前方向的偏移

    def __init__(
        self,
        grid_size=(10, 10),  # 论文实际为 10km x 10km，离散化为 Lx=Ly 格
        num_uavs=4,
        num_users=10,
        max_steps=200,  # 论文 episode length = 200
        d_max=7.5,  # 最大通信距离（km），论文 7500m
        d_min=0.15,  # 最小碰撞距离（km），论文 150m
        uav_altitude=0.1,  # km，论文 100m
        user_speed=1,  # 用户每步可移动的格数
        v_max=1,  # 无人机速度（格/步）
    ):
        self.grid_size = grid_size
        self.Lx, self.Ly = grid_size
        self.num_uavs = num_uavs
        self.num_users = num_users
        self.max_steps = max_steps
        self.d_max = d_max
        self.d_min = d_min
        self.uav_altitude = uav_altitude
        self.user_speed = user_speed
        self.v_max = v_max

        # 奖励权重（论文 rn = α|qn| + β pn + c_dis + L_fly）
        self.alpha = 1.0  # 搜索到目标的奖励权重（|qn| 是经过格数，论文有点模糊）
        self.beta = 5.0  # 找到用户的奖励权重
        self.w_energy = 0.1  # 能量惩罚权重
        self.w_connectivity = 2.0  # 连通性惩罚权重
        self.w_collision = 10.0  # 碰撞惩罚权重
        self.w_boundary = 5.0  # 飞出边界惩罚权重
        self.w_coverage = 0.5  # 覆盖奖励权重

        self.reset()

    def reset(self):
        """重置环境"""
        self.current_step = 0

        # 初始化用户位置（随机分布）
        self.user_positions = []
        for _ in range(self.num_users):
            x = random.randint(0, self.Lx - 1)
            y = random.randint(0, self.Ly - 1)
            self.user_positions.append([x, y])

        # 用户是否已被找到
        self.user_found = [False] * self.num_users

        # 用户运动方向（随机初始化）
        self.user_directions = [random.randint(1, 8) for _ in range(self.num_users)]

        # 无人机初始化（随机位置，分散分布）
        self.uav_positions = []
        self.uav_directions = []  # 当前飞行方向 1-8
        self.uav_visited = []  # 经过的格数 qn
        self.uav_found_users = []  # 每个无人机找到的用户集合

        for i in range(self.num_uavs):
            x = random.randint(0, self.Lx - 1)
            y = random.randint(0, self.Ly - 1)
            self.uav_positions.append([x, y])
            self.uav_directions.append(random.randint(1, 8))
            self.uav_visited.append(0)
            self.uav_found_users.append(set())

        # 覆盖地图（每个格子的搜索次数）
        self.coverage_map = np.zeros((self.Lx, self.Ly), dtype=np.int32)

        return self._get_observations()

    def step(self, actions):
        """执行一步

        actions: list of dict, 每个元素 {'v': speed, 'd': direction}
          其中 direction 是相对于当前方向的偏移（-1, 0, 1），对应论文图2
          或者直接用 direction 0-8
        """
        self.current_step += 1
        rewards = [0.0] * self.num_uavs
        infos = []

        # 1. 移动用户（随机游走）
        self._move_users()

        # 2. 移动无人机
        for i in range(self.num_uavs):
            action = actions[i]
            # 速度（论文 vn）
            speed = action.get('v', 1)
            # 方向偏移（论文 fn）
            dir_offset = action.get('d', 0)

            # 计算新方向：只能从当前方向转向相邻方向（论文图2，3个选择）
            # 论文图2: 当前方向 → 只能到达3个相邻位置
            old_dir = self.uav_directions[i]
            new_dir = old_dir + dir_offset
            # 确保方向在1-8范围内
            new_dir = ((new_dir - 1) % 8) + 1
            self.uav_directions[i] = new_dir

            # 移动
            dx, dy = self.DIRECTION_VECTORS[new_dir]
            old_x, old_y = self.uav_positions[i]
            new_x = old_x + dx * min(speed, self.v_max)
            new_y = old_y + dy * min(speed, self.v_max)

            # 边界检查（论文约束 x∈[0,Lx], y∈[0,Ly]）
            in_bounds = True
            if new_x < 0 or new_x >= self.Lx or new_y < 0 or new_y >= self.Ly:
                in_bounds = False
                # 惩罚飞出边界（论文 L_fly）
                rewards[i] -= self.w_boundary
                # 不移动
                new_x, new_y = old_x, old_y

            self.uav_positions[i] = [new_x, new_y]

            if in_bounds:
                self.uav_visited[i] += 1

            # 更新覆盖地图
            gx, gy = int(new_x), int(new_y)
            if 0 <= gx < self.Lx and 0 <= gy < self.Ly:
                self.coverage_map[gx, gy] += 1

        # 3. 检查搜索到用户
        for i in range(self.num_uavs):
            ux, uy = self.uav_positions[i]
            # 检查是否有用户在同一格且未被找到
            for u_idx in range(self.num_users):
                if self.user_found[u_idx]:
                    continue
                ux_u, uy_u = self.user_positions[u_idx]
                # 同一格视为找到
                if int(ux) == int(ux_u) and int(uy) == int(uy_u):
                    self.user_found[u_idx] = True
                    self.uav_found_users[i].add(u_idx)
                    rewards[i] += self.beta  # β * pn

        # 4. 检查连通性（论文约束 dij <= d_max）
        connectivity_penalty = 0.0
        for i in range(self.num_uavs):
            # 找最近的无人机
            min_dist = float('inf')
            for j in range(self.num_uavs):
                if i == j:
                    continue
                dist = self._distance_3d(i, j)
                min_dist = min(min_dist, dist)
            if min_dist > self.d_max:
                connectivity_penalty -= self.w_connectivity

        # 5. 检查碰撞（论文 dij > d_min）
        collision_penalty = 0.0
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                dist = self._distance_2d(i, j)
                if dist < self.d_min:
                    collision_penalty -= self.w_collision

        # 6. 覆盖奖励（探索鼓励）
        coverage_rewards = []
        for i in range(self.num_uavs):
            ux, uy = self.uav_positions[i]
            gx, gy = int(ux), int(uy)
            if 0 <= gx < self.Lx and 0 <= gy < self.Ly:
                visited_count = self.coverage_map[gx, gy]
                if visited_count == 1:
                    coverage_rewards.append(self.w_coverage)
                else:
                    coverage_rewards.append(0)
            else:
                coverage_rewards.append(0)

        # 7. 能量惩罚（论文 α|qn| 部分，经过越多越耗能）
        energy_penalties = []
        for i in range(self.num_uavs):
            energy_penalties.append(-self.w_energy)  # 每步都有能量消耗

        # 汇总奖励（论文 rn = α|qn| + βpn + c_dis + L_fly）
        for i in range(self.num_uavs):
            rewards[i] += self.alpha * self.uav_visited[i] * 0.001  # 很小的正奖励鼓励移动
            rewards[i] += connectivity_penalty  # 共享连通性惩罚
            rewards[i] += collision_penalty  # 共享碰撞惩罚
            rewards[i] += energy_penalties[i]
            rewards[i] += coverage_rewards[i]

        # 检查是否结束
        done = (self.current_step >= self.max_steps) or (sum(self.user_found) >= self.num_users)

        # 计算信息
        infos = {
            'found_users': sum(self.user_found),
            'total_users': self.num_users,
            'total_visited': sum(self.uav_visited),
            'step': self.current_step,
        }

        return self._get_observations(), rewards, done, infos

    def _move_users(self):
        """移动用户（随机游走，每步可移动到相邻格）"""
        for i in range(self.num_users):
            if self.user_found[i]:
                continue  # 找到的用户不再移动？论文未明确
            # 随机改变方向
            if random.random() < 0.3:
                self.user_directions[i] = random.randint(1, 8)

            dx, dy = self.DIRECTION_VECTORS[self.user_directions[i]]
            old_x, old_y = self.user_positions[i]
            new_x = old_x + dx * self.user_speed
            new_y = old_y + dy * self.user_speed

            # 边界限制
            new_x = max(0, min(self.Lx - 1, new_x))
            new_y = max(0, min(self.Ly - 1, new_y))
            self.user_positions[i] = [int(new_x), int(new_y)]

    def _distance_2d(self, i, j):
        """2D欧氏距离"""
        xi, yi = self.uav_positions[i]
        xj, yj = self.uav_positions[j]
        return np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

    def _distance_3d(self, i, j):
        """3D欧氏距离（含高度差）"""
        d2d = self._distance_2d(i, j)
        return np.sqrt(d2d ** 2 + (2 * self.uav_altitude) ** 2)

    def _get_observations(self):
        """获取所有无人机的观测（论文 on(t) = {US1, US2, ..., USN}）

        每个无人机观测包括：
        - 所有无人机位置
        - 自己找到的用户数
        """
        observations = []
        for i in range(self.num_uavs):
            obs = []
            # 所有无人机的位置和状态
            for j in range(self.num_uavs):
                x, y = self.uav_positions[j]
                d = self.uav_directions[j]
                q = self.uav_visited[j]
                # 归一化到 [0, 1]
                obs.extend([
                    x / self.Lx,
                    y / self.Ly,
                    d / 8.0,
                    q / self.max_steps,
                    len(self.uav_found_users[j]) / self.num_users,
                ])
            observations.append(np.array(obs, dtype=np.float32))
        return observations

    def get_global_state(self):
        """获取全局状态（用于集中式 Critic，论文 CTDE）"""
        state = []
        for obs in self._get_observations():
            state.extend(obs.tolist())
        # 添加全局信息
        state.append(sum(self.user_found) / self.num_users)
        state.append(self.current_step / self.max_steps)
        return np.array(state, dtype=np.float32)

    def get_valid_actions(self, uav_idx):
        """获取某无人机在当前方向下的有效动作

        论文图2：受转弯半径限制，只能选择3个方向（当前方向+左右相邻）
        返回：[{'v': v, 'd': d}, ...]
        """
        actions = []
        # 论文 vn 和 fn，这里简化
        # 速度选项：0（悬停）、1（慢速）、2（快速）
        for v in [1]:
            for offset in [-1, 0, 1]:
                actions.append({'v': v, 'd': offset})
        return actions

    def render_ascii(self):
        """ASCII渲染（调试用）"""
        grid = [[' . ' for _ in range(self.Ly)] for _ in range(self.Lx)]

        # 标记用户
        for i, pos in enumerate(self.user_positions):
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < self.Lx and 0 <= y < self.Ly:
                if self.user_found[i]:
                    grid[x][y] = f'U{i}'
                else:
                    grid[x][y] = f'u{i}'

        # 标记无人机
        for i, pos in enumerate(self.uav_positions):
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < self.Lx and 0 <= y < self.Ly:
                grid[x][y] = f'D{i}'

        print('+' + '----' * self.Ly + '+')
        for row in grid:
            print('|' + '|'.join(f'{c:^3}' for c in row) + '|')
            print('+' + '----' * self.Ly + '+')
        print(f'Step: {self.current_step}, Found: {sum(self.user_found)}/{self.num_users}')
