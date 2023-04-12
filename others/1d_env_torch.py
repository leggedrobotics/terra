import torch
import torch.nn as nn
import gymnasium as gym

torch.autograd.set_detect_anomaly(True)


class Line(gym.Env):
    def __init__(self, batch_size: int, device: str, reward_l: int = 5) -> None:
        super().__init__()
        self.reward_l = torch.ones(batch_size, 1, device=device) * reward_l
        self.device = device
        self.state = torch.randint(size=(batch_size, 1), device=device, low=-2, high=2).to(dtype=torch.float)
        # self.dones_before = torch.zeros(batch_size, 1, device=device, dtype=torch.bool)
        self.batch_size = batch_size
    
    def step(self, u):
        assert torch.equal(
            torch.where(torch.logical_or(u.clone() == 1, u.clone() == -1), u.abs(), 0),
            torch.ones_like(u)
        ), f"{u=}"

        self.state = self.state + u

        dones = self.state >= self.reward_l

        rewards = self.state - self.reward_l

        return self.state, rewards, dones, {}
    
    def reset(self):
        self.state = torch.randint(size=(self.batch_size, 1), device=self.device, low=-2, high=2).to(dtype=torch.float)
        return self.state
    
    def render(self, mode="human"):
        print(f"{self.state=}")
    
    def close(self):
        pass


class Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, input):
        return self.net(input)


if __name__ == "__main__":
    batch_size = 10
    epochs = 3
    max_rollout = 50
    device = "cpu"

    env = Line(batch_size, device)
    policy = Policy()

    optimizer = torch.optim.SGD(policy.parameters(), 0.001)

    for i in range(epochs):
        cum_rewards = 0
        state = env.reset()
        for r_i in range(max_rollout):
            probs = policy(state)

            # probs = Categorical(logits=logits)

            # u = probs.sample().to(dtype=torch.float)
            # u -= 0.5
            # u *= 2

            # u = u.unsqueeze(-1)

            # ---

            # delta_prob = probs[..., [0]] - probs[..., [1]]
            # print(f"{delta_prob.requires_grad=}")

            # u = torch.where(
            #     delta_prob > 0,
            #     -1,
            #     1
            # )

            # ---

            probs = probs - probs.min(-1, keepdim=True)[0].abs()

            probs = probs / torch.where(probs == 0, 1, probs)

            u = probs @ torch.tensor([-1, 1], dtype=torch.float)

            u = u.unsqueeze(-1)

            state, rewards, dones, _ = env.step(u)
            cum_rewards += rewards.sum()

            if dones.sum().item() == batch_size:
                break

        env.render()

        optimizer.zero_grad()
        cum_rewards.backward()
        optimizer.step()
