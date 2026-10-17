"""CartPole(棒立て)を DQN で学習し、展示用 HTML が読むデータ web/models.js を書き出す。

使い方:
    python train.py                 # 既定 600 エピソード
    python train.py --episodes 1000 --seed 1
    python train.py --robust --episodes 1000   # いじわる付きで鍛える → web/models_robust.js
    python train.py --robust --minutes 240     # 時間で止める(4時間)
"""
import argparse
import copy
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HERE = Path(__file__).resolve().parent

# 展示で「学習の途中」として見せるため、この練習回数の時点のネットを保存する
SNAP_EPISODES = [0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1500,
                 2000, 3000, 4000, 5000, 7500, 10000, 15000, 20000, 30000]


class QNet(nn.Module):
    def __init__(self, n_obs, n_act, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_act),
        )

    def forward(self, x):
        return self.net(x)


class RobustCartPole(gym.Wrapper):
    """棒の長さ・重力を毎回変え、ときどき棒をつつく「いじわる」付きの練習環境。"""

    def __init__(self, env, rng, poke_prob=0.02):
        super().__init__(env)
        self.rng = rng
        self.poke_prob = poke_prob   # 1ステップごとにつつく確率(0.02 ≒ 1秒に1回)

    def reset(self, **kwargs):
        u = self.env.unwrapped
        u.length = float(self.rng.uniform(0.25, 1.0))   # 展示の「短い」〜「長い」
        u.polemass_length = u.masspole * u.length
        u.gravity = float(np.exp(self.rng.uniform(np.log(1.6), np.log(24.8))))   # 月〜木星
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.rng.random() < self.poke_prob:
            u = self.env.unwrapped
            u.state = np.array(u.state, dtype=np.float64)
            u.state[3] += self.rng.choice([-1, 1]) * self.rng.uniform(0.3, 1.0)
        return self.env.step(action)


def make_env(robust, rng=None):
    env = gym.make("CartPole-v1")
    return RobustCartPole(env, rng if rng is not None else np.random.default_rng()) if robust else env


class Replay:
    """経験をためておくリングバッファ。"""

    def __init__(self, cap, n_obs):
        self.s = np.zeros((cap, n_obs), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.s2 = np.zeros((cap, n_obs), np.float32)
        self.d = np.zeros(cap, np.float32)
        self.cap, self.i, self.n = cap, 0, 0

    def add(self, s, a, r, s2, d):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, b, device):
        idx = np.random.randint(0, self.n, b)
        t = lambda x: torch.as_tensor(x[idx], device=device)
        return t(self.s), t(self.a), t(self.r), t(self.s2), t(self.d)


def export_layers(model):
    layers = []
    for m in model.net:
        if isinstance(m, nn.Linear):
            layers.append({
                "W": np.round(m.weight.detach().cpu().numpy().astype(float), 4).tolist(),
                "b": np.round(m.bias.detach().cpu().numpy().astype(float), 4).tolist(),
            })
    return layers


@torch.no_grad()
def evaluate(model, device, episodes, robust=False, seed=10_000):
    """ランダム行動なしで何ステップ立てられるかの平均(最大 500)。robust なら練習と同じいじわる付き。"""
    env = make_env(robust, np.random.default_rng(seed))
    total = 0.0
    for k in range(episodes):
        obs, _ = env.reset(seed=seed + k)
        done = False
        while not done:
            a = model(torch.as_tensor(obs, device=device).unsqueeze(0)).argmax(1).item()
            obs, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
    env.close()
    return total / episodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--minutes", type=float, default=None, help="指定すると回数ではなく時間(分)で止める")
    p.add_argument("--save-every", type=float, default=10, help="途中結果を書き出す間隔(分)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--buffer", type=int, default=50_000)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--eps-decay", type=float, default=1000)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--robust", action="store_true", help="棒の長さ・重力の変化とつつきありで練習する")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if args.out is None:
        args.out = HERE / "web" / ("models_robust.js" if args.robust else "models.js")
    var_name = "CARTPOLE_DATA_ROBUST" if args.robust else "CARTPOLE_DATA"
    n_eval = 5 if args.robust else 3   # いじわる付きは成績がばらつくので多めにテスト

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)

    env = make_env(args.robust, np.random.default_rng(args.seed))
    n_obs, n_act = env.observation_space.shape[0], env.action_space.n
    policy = QNet(n_obs, n_act, args.hidden).to(device)
    target = copy.deepcopy(policy)
    opt = optim.AdamW(policy.parameters(), lr=args.lr, amsgrad=True)
    buf = Replay(args.buffer, n_obs)

    returns, evals, snapshots = [], [], []
    best_score, best_state, best_episode = -1.0, None, 0
    steps = 0
    t0 = time.time()

    def snap(ep, model, label=None):
        return {"episode": ep, "label": label,
                "eval": evaluate(model, device, 5, args.robust), "layers": export_layers(model)}

    def write_data(ep_done):
        """ここまでの結果を書き出す(途中保存と最終保存で共用)。最高成績のネットの評価値を返す。"""
        best = QNet(n_obs, n_act, args.hidden).to(device)
        best.load_state_dict(best_state)
        best_snap = snap(best_episode, best, "best")
        last = [] if ep_done in SNAP_EPISODES else [snap(ep_done, policy)]
        data = {
            "meta": {
                "episodes": ep_done,
                "hidden": args.hidden,
                "seed": args.seed,
                "robust": args.robust,
                "device": str(device),
                "seconds": round(time.time() - t0, 1),
                "created": datetime.now().isoformat(timespec="seconds"),
            },
            "returns": returns,
            "evals": evals,
            "snapshots": snapshots + last + [best_snap],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_name(args.out.name + ".tmp")
        tmp.write_text(f"window.{var_name} = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
        tmp.replace(args.out)   # 書き込み途中で止まっても前回のファイルが壊れないように
        ckpt_dir = HERE / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        torch.save(best_state, ckpt_dir / ("best_robust.pt" if args.robust else "best.pt"))
        return best_snap["eval"]

    def keep_going(ep):
        return time.time() - t0 < args.minutes * 60 if args.minutes else ep < args.episodes

    if args.minutes:
        print(f"{args.minutes:.0f} 分間学習します。{args.save_every:.0f} 分ごとに途中保存し、"
              "Ctrl+C で中断してもそこまでの結果を保存します。", flush=True)
    last_save = time.time()
    ep = 0
    try:
        while keep_going(ep):
            if ep in SNAP_EPISODES:
                snapshots.append(snap(ep, policy))
            obs, _ = env.reset(seed=args.seed * 1_000_000 + ep)
            ret = 0.0
            while True:
                eps = 0.05 + 0.85 * math.exp(-steps / args.eps_decay)
                if random.random() < eps:
                    a = env.action_space.sample()
                else:
                    with torch.no_grad():
                        a = policy(torch.as_tensor(obs, device=device).unsqueeze(0)).argmax(1).item()
                obs2, r, term, trunc, _ = env.step(a)
                buf.add(obs, a, r, obs2, float(term))
                obs = obs2
                ret += r
                steps += 1

                if buf.n >= args.batch:
                    s, a_, r_, s2, d = buf.sample(args.batch, device)
                    q = policy(s).gather(1, a_.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        y = r_ + args.gamma * target(s2).max(1).values * (1 - d)
                    loss = nn.functional.smooth_l1_loss(q, y)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(policy.parameters(), 100)
                    opt.step()
                    with torch.no_grad():
                        for tp, pp in zip(target.parameters(), policy.parameters()):
                            tp.lerp_(pp, args.tau)

                if term or trunc:
                    break
            returns.append(ret)
            ep += 1

            if ep % args.eval_every == 0:
                score = evaluate(policy, device, n_eval, args.robust)
                evals.append({"episode": ep, "score": score})
                if score >= best_score:
                    best_score, best_state, best_episode = score, copy.deepcopy(policy.state_dict()), ep
                avg = sum(returns[-10:]) / len(returns[-10:])
                print(f"ep {ep:6d}  return {ret:5.0f}  avg10 {avg:6.1f}  eval {score:5.1f}  "
                      f"best {best_score:5.1f}  eps {eps:.2f}  {(time.time() - t0) / 60:6.1f}分", flush=True)

            if best_state is not None and time.time() - last_save > args.save_every * 60:
                write_data(ep)
                last_save = time.time()
                print(f"  -- 途中保存しました({ep} 回目)", flush=True)
    except KeyboardInterrupt:
        print("\n中断しました。ここまでの結果を保存します…", flush=True)

    if best_state is None:
        print("テストが一度も行われる前に終わったので保存しませんでした。")
        return
    score = write_data(ep)
    print(f"best: episode {best_episode}, eval(5) {score:.1f}")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)  {(time.time() - t0) / 60:.1f} 分")


if __name__ == "__main__":
    main()
