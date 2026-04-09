import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plasma_env import TokamakEnv
from mamba_ppo import MambaAgent, StateBuffer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_saved_model(model_path="best_model.pth", steps=2000):
    env = TokamakEnv()
    agent = MambaAgent().to(DEVICE)
    agent.load_state_dict(torch.load(model_path, map_location=DEVICE))
    agent.eval()

    state_buffer = StateBuffer(size=32, state_dim=3)

    state, _ = env.reset()
    state_buffer.reset(state)

    zs, dzs, actions = [], [], []

    # ---- 2D PLOTS ----
    plt.ion()
    fig1, axs = plt.subplots(3, 1, figsize=(8, 8))

    line_z, = axs[0].plot([], [])
    line_dz, = axs[1].plot([], [])
    line_action, = axs[2].plot([], [])

    axs[0].set_title("z")
    axs[1].set_title("dz")
    axs[2].set_title("action")

    R, r = 1.0, 0.3
    theta = np.linspace(0, 2*np.pi, 100)

    for step in range(steps):
        state_seq = torch.tensor(state_buffer.get(), dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            dist, _ = agent(state_seq)

        action = dist.mean
        action_np = action.squeeze(0).cpu().numpy()

        next_state, _, done, _, _ = env.step(action_np)

        z, dz, _ = env.state

        zs.append(z)
        dzs.append(dz)
        actions.append(action_np[0])

        state_buffer.append(next_state)
        state = next_state

        # ---- UPDATE 2D ----
        x = np.arange(len(zs))

        line_z.set_data(x, zs)
        line_dz.set_data(x, dzs)
        line_action.set_data(x, actions)

        for ax in axs:
            ax.relim()
            ax.autoscale_view()

        fig1.canvas.draw()
        fig1.canvas.flush_events()

        if done:
            break

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    visualize_saved_model("tokamak_mamba.pth")