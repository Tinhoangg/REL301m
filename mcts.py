"""Monte Carlo Tree Search with PUCT for Go."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Node:
    to_play: int
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    priors: dict[int, float] = field(default_factory=dict)
    children: dict[int, "Node"] = field(default_factory=dict)
    expanded: bool = False

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class NetworkEvaluator:
    """Convert a policy-value network into MCTS priors and values."""

    def __init__(self, model=None, device=None):
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def evaluate(self, env):
        legal_actions = env.legal_actions()
        action_size = env.pass_action + 1
        if not legal_actions:
            return np.zeros(action_size, dtype=np.float32), 0.0

        if self.model is None:
            priors = np.zeros(action_size, dtype=np.float32)
            priors[legal_actions] = 1.0 / len(legal_actions)
            return priors, 0.0

        state = torch.from_numpy(env.encode()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.model(state)
            probs = F.softmax(logits[0], dim=0).detach().cpu().numpy().astype(np.float32)

        priors = np.zeros(action_size, dtype=np.float32)
        usable = min(action_size, probs.shape[0])
        priors[:usable] = probs[:usable]
        mask = np.zeros(action_size, dtype=np.float32)
        mask[legal_actions] = 1.0
        priors *= mask
        total = float(priors.sum())
        if total <= 0.0:
            priors[legal_actions] = 1.0 / len(legal_actions)
        else:
            priors /= total
        return priors, float(value.item())


class MCTS:
    def __init__(self, evaluator=None, num_simulations: int = 100, c_puct: float = 1.5):
        self.evaluator = evaluator or NetworkEvaluator()
        self.num_simulations = int(num_simulations)
        self.c_puct = float(c_puct)

    def search(self, env, add_noise: bool = False, dirichlet_alpha: float = 0.3, exploration_fraction: float = 0.25):
        """Run MCTS from ``env`` and return a visit-count policy."""
        root = Node(to_play=env.current_player)
        self._expand(root, env)
        if add_noise:
            self._add_root_noise(root, dirichlet_alpha, exploration_fraction)

        for _ in range(self.num_simulations):
            sim_env = env.clone()
            node = root
            path = [node]

            while node.expanded and node.priors and not sim_env.done:
                action = self._select_action(node)
                sim_env.step(action)
                if action not in node.children:
                    node.children[action] = Node(to_play=sim_env.current_player, prior=node.priors[action])
                node = node.children[action]
                path.append(node)
                if not node.expanded:
                    break

            if sim_env.done:
                winner = sim_env.winner()
                leaf_player = node.to_play
                value = 0.0 if winner == 0 else (1.0 if winner == leaf_player else -1.0)
            else:
                value = self._expand(node, sim_env)
                leaf_player = sim_env.current_player

            self._backpropagate(path, value, leaf_player)

        return self._visit_policy(root, env.pass_action + 1)

    def select_action(self, env, temperature: float = 1.0, add_noise: bool = False) -> tuple[int, np.ndarray]:
        """Return ``(action, policy)`` from MCTS visit counts."""
        policy = self.search(env, add_noise=add_noise)
        legal_actions = env.legal_actions()
        if not legal_actions:
            return env.pass_action, policy

        legal_probs = policy[legal_actions]
        if temperature <= 1e-6:
            action = legal_actions[int(np.argmax(legal_probs))]
            return int(action), policy

        adjusted = np.power(legal_probs, 1.0 / temperature)
        total = float(adjusted.sum())
        if total <= 0.0:
            adjusted = np.ones_like(adjusted) / len(adjusted)
        else:
            adjusted /= total
        action = int(np.random.choice(legal_actions, p=adjusted))
        return action, policy

    def _expand(self, node: Node, env) -> float:
        priors, value = self.evaluator.evaluate(env)
        node.priors = {action: float(priors[action]) for action in env.legal_actions()}
        node.expanded = True
        return value

    def _select_action(self, node: Node) -> int:
        best_score = -float("inf")
        best_action = None
        parent_sqrt = np.sqrt(max(1, node.visit_count))

        for action, prior in node.priors.items():
            child = node.children.get(action)
            child_visits = 0 if child is None else child.visit_count
            if child is None:
                q_value = 0.0
            elif child.to_play == node.to_play:
                q_value = child.value()
            else:
                q_value = -child.value()
            exploration = self.c_puct * prior * parent_sqrt / (1 + child_visits)
            score = q_value + exploration
            if score > best_score:
                best_score = score
                best_action = action

        return int(best_action)

    def _backpropagate(self, path: list[Node], value: float, leaf_player: int):
        for node in reversed(path):
            node_value = value if node.to_play == leaf_player else -value
            node.value_sum += node_value
            node.visit_count += 1

    def _visit_policy(self, root: Node, action_size: int) -> np.ndarray:
        policy = np.zeros(action_size, dtype=np.float32)
        for action, child in root.children.items():
            policy[action] = child.visit_count
        total = float(policy.sum())
        if total <= 0.0:
            legal = list(root.priors)
            if legal:
                policy[legal] = 1.0 / len(legal)
        else:
            policy /= total
        return policy

    def _add_root_noise(self, root: Node, alpha: float, fraction: float):
        actions = list(root.priors)
        if not actions:
            return
        noise = np.random.dirichlet([alpha] * len(actions))
        for action, sample in zip(actions, noise):
            root.priors[action] = (1 - fraction) * root.priors[action] + fraction * float(sample)
