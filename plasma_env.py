import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from queue import Queue

class TokamakEnv(gym.Env):
    
    def __init__(self):
        super().__init__()
        
        # Constraints
        self.dt = 0.001
        self.max_z = 1
        self.alpha = 30
        self.L = 1.2
        self.R = 0.78
        self.max_voltage = 150.0
        self.max_steps = 2000
        
        #definin action space [V]
        self.action_space = spaces.Box(low=-1.0 ,high=1.0 ,shape=(1,),dtype=np.float32)
        
        #defn observation space [Z, dZ, I]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,),dtype=np.float32)
        
    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        self.state = np.random.uniform(low=-0.01, high=0.01, size=(3,)).astype(np.float32)
        #this will amount to the delay to providing required voltage to the reactor 
        aq = Queue(maxsize=8)
        for _ in range(8):
            aq.put(0.0)
        self.action_queue = aq
        self.gamma = 49.0
        self.steps = 0
        
        return self.state, {}

    
    def step(self, action):
        z, dz, In = self.state
        
        delayed_action = self.action_queue.get()
        self.action_queue.put(float(action[0]))
        V = delayed_action * self.max_voltage
        
        fluc = 0.07
        self.gamma += fluc
        
        dI = (V - self.R * In)/(self.L)
        In1 = In + (dI*self.dt)
        
        force = (self.gamma * z) - (self.alpha * In1)
        
        dz1 = dz + (force * self.dt)
        z1 = z + (dz1*self.dt)
        
        self.steps += 1
        terminated = bool(abs(z1)>self.max_z)
        truncated = bool(self.steps >= self.max_steps)
        
        if terminated:
            reward = -1000
        else:
            reward = 1.0 - (10.0 * (z1**2)) - (0.1 * (dz1**2)) - (0.001 * (action[0]**2))
        self.state = np.array([z1, dz1, In1], dtype=np.float32)
        
        return self.state, float(reward), terminated, truncated, {}
        

if __name__ == "__main__":
    env = TokamakEnv()
    obs, _ = env.reset();
    print("---- No Agent Test ----")
    for i in range(1000):
        action = env.action_space.sample() 
        obs, reward, term, trunc, _ = env.step(action)
        if term:
            print(f"CRASHED at Step {i}! (Z = {obs[0]:.4f})")
            break