"""
Turn-based evaluate: İki agent'ı sırayla oynatır, trajectory toplar.
StateVisualizer ile uyumlu dict döner.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv


def turn_based_evaluate(agent0, agent1, layout_name, horizon=400, num_games=1, display=False):
    """
    Turn-based oyun çalıştırır.

    Her timestep:
      1. agent0 hareket eder (agent1 STAY)
      2. agent1 hareket eder (agent0 STAY)

    Args:
        agent0: Player 0 (index=0) agent
        agent1: Player 1 (index=1) agent
        layout_name: Layout ismi
        horizon: Maksimum timestep
        num_games: Oyun sayısı
        display: True ise her adımı yazdır

    Returns:
        StateVisualizer ile uyumlu trajectory dict
    """
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)

    all_ep_states = []
    all_ep_actions = []
    all_ep_rewards = []
    all_ep_dones = []
    all_ep_infos = []
    all_ep_returns = []
    all_ep_lengths = []
    all_mdp_params = []
    all_env_params = []

    for game_idx in range(num_games):
        env.reset()

        # Agent'ları hazırla
        agent0.reset()
        agent1.reset()
        agent0.set_agent_index(0)
        agent1.set_agent_index(1)

        # set_mdp varsa çağır (BeliefAgentV2 için gerekli)
        for agent in (agent0, agent1):
            if hasattr(agent, "set_mdp"):
                import inspect
                sig = inspect.signature(agent.set_mdp)
                if "initial_state" in sig.parameters:
                    agent.set_mdp(mdp, initial_state=env.state)
                else:
                    agent.set_mdp(mdp)

        states = []
        actions = []
        rewards = []
        dones = []
        infos = []
        total_reward = 0

        for t in range(horizon):
            state = env.state

            # Adım 1: Agent 0 hareket eder
            a0, _ = agent0.action(state)
            joint_action_0 = (a0, Action.STAY)
            s_after_0, r0, done0, info0 = env.step(joint_action_0)
            total_reward += r0

            # Kaydet (agent0'ın adımı)
            states.append(state)
            actions.append(joint_action_0)
            rewards.append(r0)
            dones.append(done0)
            infos.append(info0)

            if display:
                print(f"  Game {game_idx+1} T={t} A0={a0} R={r0}")

            if done0:
                break

            # Adım 2: Agent 1 hareket eder
            state = env.state
            a1, _ = agent1.action(state)
            joint_action_1 = (Action.STAY, a1)
            s_after_1, r1, done1, info1 = env.step(joint_action_1)
            total_reward += r1

            # Kaydet (agent1'in adımı)
            states.append(state)
            actions.append(joint_action_1)
            rewards.append(r1)
            dones.append(done1)
            infos.append(info1)

            if display:
                print(f"  Game {game_idx+1} T={t} A1={a1} R={r1}")

            if done1:
                break

        all_ep_states.append(states)
        all_ep_actions.append(actions)
        all_ep_rewards.append(rewards)
        all_ep_dones.append(dones)
        all_ep_infos.append(infos)
        all_ep_returns.append(total_reward)
        all_ep_lengths.append(len(states))
        all_mdp_params.append(mdp.mdp_params)
        all_env_params.append(env.env_params)

        if display:
            print(f"Game {game_idx+1}: Score={total_reward}, Steps={len(states)}")

    return {
        "ep_states": all_ep_states,
        "ep_actions": all_ep_actions,
        "ep_rewards": all_ep_rewards,
        "ep_dones": all_ep_dones,
        "ep_infos": all_ep_infos,
        "ep_returns": all_ep_returns,
        "ep_lengths": all_ep_lengths,
        "mdp_params": all_mdp_params,
        "env_params": all_env_params,
        "metadatas": [{}] * num_games,
    }
