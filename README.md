# 07 Learning Distance-to-Goal Functions  
## for Goal-Conditioned RL in Sparse Environments
Artem Voronov · Vladislav Ulendeev · Sofia Gamershmidt · Petr Kuderov  
*Updated: 10 July 2025*

---

## Problem Statement
In **goal-conditioned reinforcement learning (RL)** an agent must move from its current state $s$ to a desired goal $g$.  
With a **sparse reward**—zero everywhere and +1 only at the goal—the agent seldom receives positive feedback, so learning stalls.  
Classic fixes such as *Hindsight Experience Replay* help but can still fail in mazes with dead ends.

A simple remedy is to provide a dense hint: the **distance-to-goal** $D(s,g)$.  
If the agent knows (or is rewarded for decreasing) this distance it can discover good actions sooner.

> **Research question:**  
> Can a learned distance function speed up PPO on the `PointMaze_UMaze-v3` task?

We compare seven variants: plain PPO (baseline) and six that use a distance model either for **R**eward shaping, as an **O**bserved feature, or **B**oth—each trained with **Sup**ervised labels or **TD** bootstrapping.

<p align="center">
  <img src="https://github.com/user-attachments/assets/30a7d649-4af1-4d44-8779-91b292b4ba5b" width="28%" alt="Maze environment">
  <img src="https://github.com/user-attachments/assets/275f2cb3-7d9e-42d5-8d30-4932c7bd406f" width="33%" alt="Supervised distance field">
  <img src="https://github.com/user-attachments/assets/e37d153c-438e-4dbd-be8f-756fa0f7a2b8" width="33%" alt="TD distance field">
</p>
<p align="center"><em>Figure 1 – Task and learned distance fields (lighter ≈ farther).</em></p>

---

## Method Summary
We jointly train the PPO policy and the distance module in *stages*:

1. **Cold-start roll-out**    
   Run the current policy for *N* sparse-reward episodes; log transitions.

2. **Build dataset**    
   For every trajectory enumerate all state–goal pairs $(s,g)$, compute their path distance $d^\star$, and store $(s,g,d^\star)$.

3. **Train distance model $f_\theta$**  
   * **Sup:**  minimise $\lVert f_\theta(s,g)-d^\star\rVert^{2}$  
   * **TD:**   enforce $f_\theta(s,g)\approx 1+f_\theta(s',g)$ for each transition $s\!\to\!s'$ (and 0 at the goal)

4. **Retrain PPO for *M* updates** using *one* of three ways to inject distance:  

   | Mode | Mechanism |
   |------|-----------|
   | **R** | Reward shaping:  $r_t \gets r_t^{\text{sparse}} + \gamma\,[\Phi(s_{t+1})-\Phi(s_t)]$, where $\Phi(s)=-f_\theta(s,g)$ |
   | **O** | Observation feature: append $d_t=f_\theta(s_t,g)$ to the state |
   | **B** | Use **R** and **O** simultaneously |

Policy exploration enlarges the dataset; a better distance model in turn speeds up the next stage.

**Variants tested:** Baseline, Sup-R, Sup-O, Sup-B, TD-R, TD-O, TD-B.

---

## Experimental Setup
* 300 k environment steps per run, 10 random seeds  
* Metric: mean episodic return (success rate)  
* PPO hyper-parameters and network size (2 × 64) shared across variants  
* Distance nets trained on 100 k pairs (Sup) or 5 epochs of roll-outs (TD)
---

## Results
* **Reward shaping** (Sup-R, TD-R).  
* **Observation only**.  
* **Both signals** (Sup-B).

| Variant | Avg. Return | Seeds Solved |
|---------|------------:|-------------:|
| PPO Baseline | 37.46 ± 3.4 | 52 / 100 |
| Sup-R | 168.43 ± 21.89 | 52 / 100 |
| Sup-O | 112.45 ± 12.81 | 31 / 100 |
| Sup-B | 38.67 ± 15.1 | 50 / 100 |
| TD-R | — | — |
| TD-O | — | — |
| TD-B | — | — |

---

## Future Work
1. **Harder tasks & generalisation** – deeper mazes, unseen goal layouts.  
2. **Distance-guided exploration** – follow steepest predicted descent or short local plans.  
3. **Online distance refinement** – update $f_\theta$ continuously as PPO explores new states.

---

## Conclusion
A learned distance-to-goal function—especially when used for potential-based shaping—turns a nearly unsolvable sparse-reward maze into a reliably solved task, cutting both data requirements and seed variance.
