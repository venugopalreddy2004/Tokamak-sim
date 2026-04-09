import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plasma_env import TokamakEnv
from mamba_ssm import Mamba
from torch.distributions import Normal, TransformedDistribution, TanhTransform
from collections import deque
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StateBuffer:
    def __init__(self, size, state_dim):
        self.size = size
        self.state_dim = state_dim
        self.buffer = deque(maxlen=size)

    def reset(self, state):
        self.buffer.clear()
        for _ in range(self.size):
            self.buffer.append(state)

    def append(self, state):
        self.buffer.append(state)

    def get(self):
        return np.array(self.buffer, dtype=np.float32)  # (T, state_dim)

class MambaAgent(nn.Module):
    def __init__(self, state_dim=3, d_model=64, action_dim=1):
        super().__init__()
        self.embedding = nn.Linear(state_dim, d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2
        )
        
        self.critic = nn.Linear(d_model, 1)
        self.actor_mean = nn.Linear(d_model, action_dim)
        self.actor_std = nn.Linear(d_model, action_dim)
        
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
    
    def forward(self, x):
        embed = F.silu(self.embedding(x)) 
        mamba_out = self.mamba(embed)
        mamba_out = F.layer_norm(mamba_out, mamba_out.shape[-1:])
        mamba_out = F.silu(mamba_out)
        mamba_out = mamba_out[:, -1]
        
        value = self.critic(mamba_out).squeeze(-1)
        
        action_mean = self.actor_mean(mamba_out) 
        log_std = torch.clamp(self.actor_std(mamba_out), -1, 1)
        action_sd = torch.exp(log_std).expand_as(action_mean)
        dist = Normal(action_mean, action_sd)
        
        return dist, value

def collect_trajectories(env, agent, state_buffer, steps_per_rollout, device=DEVICE):
    '''
        loop for N steps
            get state seq 
            run policy
            sample action
            step env
            store everything
        
        return rollout; it has states, actions,
        log_probs, rewards, values, dones
    '''
    states, actions, log_probs, rewards, values, dones=[], [], [], [], [], []
    episode_lengths=[]
    episode_rewards=[]
    
    state,_=env.reset()
    state_buffer.reset(state)
    
    episode_step, episode_reward=0, 0
    agent.eval()
    best_len = 0
    
    for step in range(steps_per_rollout):
        ''' 1. get state seq '''
        state_seq = torch.tensor(state_buffer.get(), dtype=torch.float32, device=device)    # [T, 3]
        state_seq = state_seq.unsqueeze(0)   #[1, T, 3]
        
        
        '''2. run policy '''
        with torch.no_grad():
            dist, val = agent(state_seq)
        
        
        '''3. sample action '''
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        action_np = torch.clamp(action, -1, 1).cpu().numpy()
        
        
        '''4. step env'''
        next_state, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated
        
        '''5. store everything'''
        states.append(state_seq.squeeze(0))     
        actions.append(action.squeeze(0))       
        log_probs.append(log_prob.squeeze(0))   
        values.append(val.squeeze(0).detach())  

        rewards.append(reward/100.0)
        dones.append(done)
        
        '''6. update episode and buffer'''
        episode_step+=1
        episode_reward+=reward
        
        state_buffer.append(next_state)
        state=next_state
        
        if done:
            episode_lengths.append(episode_step)
            episode_rewards.append(episode_reward)
            
            state, _ = env.reset()
            state_buffer.reset(state)
            
            episode_step=0
            episode_reward=0

    rollout = {
        "states": torch.stack(states),
        "actions": torch.stack(actions),
        "log_probs": torch.stack(log_probs),
        "rewards": torch.tensor(rewards, dtype=torch.float32, device=device),
        "values": torch.stack(values),
        "dones": torch.tensor(dones, dtype=torch.float32, device=device)
    }
    
    if len(episode_lengths) > 0:
        print(f"Avg Ep Len: {np.mean(episode_lengths):.2f}, Avg Reward: {np.mean(episode_rewards):.2f}")
    else:
        print("No episodes finished in this rollout")
    return rollout

def compute_gae(rollout, gamma=0.99, lam=0.95):
    rewards = rollout["rewards"]
    values = rollout["values"]
    dones = rollout["dones"]
    
    adv = torch.zeros_like(rewards)
    gae=0
    next_val=values[-1]
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_val * (1-dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv[t] = gae
        next_val = values[t]
    
    returns = adv + values
    returns = torch.clamp(returns, -100, 100)
    return adv, returns

def ppo_update(agent, optimizer, rollout, adv, returns,
               clip_eps=0.1, K_epochs=2, batch_size=64):
    
    states = rollout["states"]
    actions = rollout["actions"]
    old_log_probs = rollout["log_probs"].detach()
    old_values = rollout["values"].detach()
    
    dataset_size = states.size(0)
    
    agent.train()
    
    for _ in range(K_epochs):
        indices = torch.randperm(dataset_size, device=states.device)
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            
            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_adv = adv[batch_idx]
            batch_returns = returns[batch_idx]
            
            dist, values = agent(batch_states)
            
            new_log_probs = dist.log_prob(batch_actions).sum(-1)
            
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            approx_kl = (batch_old_log_probs - new_log_probs).mean()
            if approx_kl > 0.02:
                break
            
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_adv
            
            actor_loss = -torch.min(surr1, surr2).mean()
            value_pred_clipped = old_values[batch_idx] + \
                (values - old_values[batch_idx]).clamp(-0.2, 0.2)

            value_loss1 = (values - batch_returns).pow(2)
            value_loss2 = (value_pred_clipped - batch_returns).pow(2)

            critic_loss = torch.max(value_loss1, value_loss2).mean()
            entropy = dist.entropy().sum(-1).mean()
            
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()


def train():
    env = TokamakEnv()
    agent = MambaAgent().to(DEVICE)
    nb_iter = 120
    steps_per_rollout = 4096

    
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
    
    state_buffer = StateBuffer(size=32, state_dim=3)
    
    for i in range(nb_iter):
        rollout = collect_trajectories(
            env, agent, state_buffer, steps_per_rollout
        )
        adv, returns = compute_gae(rollout)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = torch.clamp(adv, -10, 10)
        
        ppo_update(agent, optimizer, rollout, adv, returns)
        print(f"Iteration {i} completed")
    torch.save(agent.state_dict(), "tokamak_mamba.pth")
        
    

if __name__=="__main__":
    train()
    