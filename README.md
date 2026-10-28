# Implementation-of-Iterative-Policy-Evaluation-for-a-Finite-MDP
## Aim

To implement iterative policy evaluation using Gymnasium and estimate the state-value function $V^\pi(s)$ for a fixed random policy.

---
## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy
```

---

## Environment Used

The experiment uses the **FrozenLake-v1** environment from Gymnasium.

FrozenLake is a grid-based reinforcement learning environment where the agent starts from a start state and tries to reach the goal state without falling into holes.

For the default 4 x 4 FrozenLake map:

| Component | Description |
|---|---|
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching goal, 0 otherwise |
| Terminal states | Goal and holes |

---

## Problem Statement

Evaluate a fixed random policy in the FrozenLake-v1 environment.

The agent follows a random policy, where each of the four actions is selected with equal probability:

$$
\pi(a|s) = \frac{1}{4}
$$

This probability refers to the policy's action-selection probability. The environment transition probabilities are obtained from Gymnasium using `env.P[state][action]`. If `is_slippery=True`, the agent may not move in the intended direction due to stochastic transitions.

The objective is to estimate the state-value function:

$$
V^\pi(s)
$$

---

## Theory

The state-value function under policy $pi$, denoted by $V^\pi(s)$, represents the expected return starting from state $s$ and following policy $pi$.

The Bellman expectation equation is:

```math
V^\pi(s) =
\sum_a \pi(a|s)
\sum_{s'} P(s'|s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
```

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $\pi(a \mid s)$ | Probability of selecting action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $\gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $\pi$ |

---
## Algorithm

1. Create the FrozenLake-v1 environment using Gymnasium.
2. Access the transition model of the environment.
3. Initialize \(V(s)=0\) for all states.
4. Define a random policy where each action has equal probability.
5. For each state:
   - For each action:
     - Read transition probability, next state, reward, and terminal status.
     - Apply the Bellman expectation equation.
6. Repeat until the value function converges.
7. Display the final value function as a 4 x 4 grid.

---

## Program

```python
import gymnasium as gym
import numpy as np

# Create FrozenLake environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

# Access the unwrapped environment to use the transition model
env = env.unwrapped

# Number of states and actions
n_states = env.observation_space.n
n_actions = env.action_space.n

# Parameters
gamma = 0.99
theta = 1e-8

# Random policy: each action has equal probability
policy = np.ones((n_states, n_actions)) / n_actions

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    """
    Evaluates a given policy using iterative policy evaluation and returns V and the iteration count.
    """
    V = np.zeros(n_states)
    iterations = 0

    while True:
        delta = 0
        iterations += 1

        for state in range(n_states):
            old_value = V[state]
            new_value = 0

            for action in range(n_actions):
                action_probability = policy[state][action]

                for probability, next_state, reward, terminated in env.P[state][action]:
                    new_value += action_probability * probability * (
                        reward + gamma * V[next_state] * (not terminated)
                    )

            V[state] = new_value
            delta = max(delta, abs(old_value - V[state]))

        if delta < theta:
            break

    return V, iterations

# Run policy evaluation
V, iterations = policy_evaluation(env, policy, gamma, theta)

print("Name: Sriram R")
print("Register Number: 212223230215")
print(f"Number of iterations to converge: {iterations}")
print("\nState-Value Function:")
print(V)

print("Name: Sriram R")
print("Register Number: 212223230215")
print("\nState-Value Function as 4x4 Grid:")
print(np.round(V.reshape(4, 4), 4))

env.close()
```

---

## Output
Gamma value is :0.99

<img width="744" height="359" alt="image" src="https://github.com/user-attachments/assets/f6d74fe6-2b05-4667-b06e-05cc5da70cd5" />

Gamma value is :0.9

<img width="630" height="358" alt="image" src="https://github.com/user-attachments/assets/55b28794-1ed1-4bcf-8639-e40559bbc019" />


## Result

Iterative policy evaluation was implemented successfully using the Gymnasium FrozenLake environment. The state-value function for the fixed random policy was estimated using the Bellman expectation equation.

---

## Inference

```text
When gamma decreases from 0.99 to 0.9, the state values decrease because future rewards are given less importance.

The number of iterations decreases from 54 to 38, showing faster convergence with a lower gamma value.

A higher gamma value considers long-term rewards more, while a lower gamma value focuses more on immediate rewards.


```




---


