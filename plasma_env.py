import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class TokamakEnv(gym.Env):
    
    def __init__(self):
        super().__init__()
        
        # Constraints
        self.dt = 0.001
        self.max_z = 0.3
        self.alpha = 0.1
        self.L = 0.2
        self.R = 0.78
        self.max_voltage = 300.0
        self.max_steps = 2000
        self.gamma = 25.0
        self.lambda_v = 0.1
        self.prev_V = None
        
        self.norm_z = 0.3
        self.norm_dz = 10.0
        self.norm_I = 100.0
        
        #definin action space [V]
        self.action_space = spaces.Box(low=-1.0 ,high=1.0 ,shape=(1,),dtype=np.float32)
        
        #defn observation space [Z, dZ, I]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,),dtype=np.float32)
        
    def reset(self, seed = None, options = None):
        super().reset(seed=seed)

        self.state = np.random.uniform(low=-0.01, high=0.01, size=(3,)).astype(np.float32)
        #this will amount to the delay to providing required voltage to the reactor 
        self.action_queue = [0]*8       # latency = dt*size = 0.001*8 = 8ms
        z, dz, _ = self.state
        self.prev_V = z**2 + self.lambda_v * dz**2
        self.steps = 0
        
        return self._get_norm_state(), {}

    def _get_norm_state(self):
        z, dz, I = self.state
        norm = np.array([z / self.norm_z, dz / self.norm_dz, I / self.norm_I], dtype=np.float32)
        return np.clip(norm, -1.0, 1.0)
    
    def step(self, action):
        z, dz, In = self.state
        
        self.action_queue.append(float(action[0]))
        delayed_action = self.action_queue.pop(0)
        V = delayed_action * self.max_voltage

        #self.gamma = min(self.gamma + 0.001, 35.0)
        
        dI = (V - self.R * In)/(self.L)
        In1 = In + (dI*self.dt)
        
        force = (self.gamma * z) - (self.alpha * In1)
        
        dz1 = dz + (force * self.dt)
        z1 = z + (dz1*self.dt)
        
        z1 += np.random.normal(0, 1e-4)
        dz1 += np.random.normal(0, 1e-3)
        
        self.steps += 1
        terminated = bool(abs(z1)>self.max_z)
        truncated = bool(self.steps >= self.max_steps)
        
        # Compute energy (Lyapunov function)
        V = z1**2 + self.lambda_v * dz1**2
        reward = self.prev_V - V
        self.prev_V = V
        reward -= 0.001 * (action[0] ** 2)
        reward += 0.5

        # Termination penalty
        if terminated:
            reward -= 50.0
        
        self.state = np.array([z1, dz1, In1], dtype=np.float32)
        return self._get_norm_state(), float(reward), terminated, truncated, {}