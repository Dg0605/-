import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import random

# 1. 미사일 요격 환경 (Gym 스타일)
class MissileEnv:
    def __init__(self):
        self.dt = 0.1
        self.max_steps = 150
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset()

    def reset(self):
        # 적 미사일 초기 위치 및 속도 (포물선 궤적)
        self.target_pos = np.array([100.0, 100.0, 80.0])
        self.target_vel = np.array([-20.0, -15.0, 5.0])
        self.gravity = np.array([0, 0, -9.8])
        
        # 요격 미사일 초기 상태
        self.interceptor_pos = np.array([0.0, 0.0, 0.0])
        self.interceptor_vel = np.array([0.0, 0.0, 0.0])
        
        self.steps = 0
        self.traj_target = [self.target_pos.copy()]
        self.traj_interceptor = [self.interceptor_pos.copy()]
        
        return self._get_obs()

    def _get_obs(self):
        # 상대적 위치와 속도를 모델의 입력으로 사용
        rel_pos = self.target_pos - self.interceptor_pos
        rel_vel = self.target_vel - self.interceptor_vel
        return np.concatenate([rel_pos, rel_vel]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        
        # 적 미사일 물리 업데이트
        self.target_vel += self.gravity * self.dt
        self.target_pos += self.target_vel * self.dt
        
        # 행동(Action) 적용: 가속도 제어 [-1, 0, 1] 범위를 스케일링
        accel = action * 30.0 
        self.interceptor_vel += accel * self.dt
        self.interceptor_pos += self.interceptor_vel * self.dt
        
        self.traj_target.append(self.target_pos.copy())
        self.traj_interceptor.append(self.interceptor_pos.copy())
        
        # 보상 설계
        dist = np.linalg.norm(self.target_pos - self.interceptor_pos)
        reward = -dist * 0.1 # 기본적으로 거리가 멀면 감점
        
        done = False
        if dist < 3.0: # 요격 성공
            reward += 500.0
            done = True
        elif self.steps >= self.max_steps or self.target_pos[2] < 0: # 실패
            reward -= 100.0
            done = True
            
        return self._get_obs(), reward, done

# 2. 신경망 모델 (Actor-Critic 또는 단순 DQN용 Q-Net)
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim) # 각 축별 가속도 결정
        )
        
    def forward(self, x):
        return self.net(x)

# 3. 학습 및 시각화 함수
def train_and_visualize():
    env = MissileEnv()
    state_dim = 6 # (x,y,z, vx,vy,vz)
    action_dim = 3 # (ax, ay, az)
    
    model = PolicyNet(state_dim, action_dim).to(env.device)
    # 실제 학습 루프는 이 부분에 PPO나 DQN 알고리즘을 추가하면 됩니다.
    # 여기서는 구조를 보여드리기 위해 '추론' 시뮬레이션 위주로 진행합니다.

    # 시뮬레이션 실행 (학습된 정책 가정 - 비례 제어 로직 예시)
    state = env.reset()
    done = False
    while not done:
        # 강화학습 모델이 예측해야 할 값 대신 간단한 벡터 추적 로직 사용
        rel_pos = state[:3]
        action = rel_pos / (np.linalg.norm(rel_pos) + 1e-6) # 방향 벡터
        state, reward, done = env.step(action)

    # 시각화
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    t_traj = np.array(env.traj_target)
    i_traj = np.array(env.traj_interceptor)
    
    ax.plot(t_traj[:,0], t_traj[:,1], t_traj[:,2], 'r-', label='Enemy (Target)', linewidth=2)
    ax.plot(i_traj[:,0], i_traj[:,1], i_traj[:,2], 'b--', label='AI Interceptor', alpha=0.8)
    ax.scatter(t_traj[-1,0], t_traj[-1,1], t_traj[-1,2], color='orange', s=200, marker='*', label='Impact')
    
    ax.set_title(f"Interception RL Simulation (Steps: {env.steps})")
    ax.legend()
    plt.show()

train_and_visualize()
