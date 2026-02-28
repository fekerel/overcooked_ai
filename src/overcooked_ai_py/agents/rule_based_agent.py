import numpy as np
from collections import defaultdict

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.agents.agent import Agent


class RuleBasedAgent(Agent):
    """
    Basit refleks ajanı.

    Karar kuralları (sabit öncelik sırası):
      1. Elinde çorba varsa → servis noktasına git
      2. Elinde tabak varsa → hazır/pişen tencereye git
      3. Elinde soğan varsa → tencereye koy
      4. Elin boşsa ve çorba hazırsa → tabak al
      5. Elin boşsa ve tencere doluysa → pişir
      6. Elin boşsa → soğan al

    A* yol planlamayı MotionPlanner üzerinden kullanır,
    ama hedef seçiminde hiçbir zeka yok — ilk geçerli hedefi alır.
    """

    def __init__(self, mlam, auto_unstuck=True):
        self.mlam = mlam
        self.mdp = self.mlam.mdp
        self.auto_unstuck = auto_unstuck
        self.reset()

    def reset(self):
        super().reset()
        self.prev_state = None

    def action(self, state):
        player = state.players[self.agent_index]
        am = self.mlam

        pot_states = self.mdp.get_pot_states(state)
        counter_objects = self.mdp.get_counter_objects_dict(
            state, list(self.mdp.terrain_pos_dict["X"])
        )

        # KURAL ZİNCİRİ 
        motion_goals = None

        if player.has_object():
            obj = player.get_object()

            if obj.name == "soup":
                # KURAL 1: Çorbayı servis et
                motion_goals = am.deliver_soup_actions()

            elif obj.name == "dish":
                # KURAL 2: Tabakla çorbayı al
                motion_goals = am.pickup_soup_with_dish_actions(
                    pot_states, only_nearly_ready=True
                )

            elif obj.name == "onion":
                # KURAL 3: Soğanı tencereye koy
                motion_goals = am.put_onion_in_pot_actions(pot_states)

            elif obj.name == "tomato":
                motion_goals = am.put_tomato_in_pot_actions(pot_states)

        else:
            ready_soups = pot_states["ready"]
            cooking_soups = pot_states["cooking"]
            full_not_cooking = self.mdp.get_full_but_not_cooking_pots(pot_states)

            if len(ready_soups) > 0 or len(cooking_soups) > 0:
                # KURAL 4: Çorba hazır/pişiyor → tabak al
                motion_goals = am.pickup_dish_actions(counter_objects)

            elif len(full_not_cooking) > 0:
                # KURAL 5: Tencere dolu ama pişmiyor → pişir
                cook_dict = defaultdict(list)
                cook_dict["{}_items".format(3)] = full_not_cooking
                motion_goals = am.start_cooking_actions(cook_dict)

            else:
                # KURAL 6: Soğan al
                motion_goals = am.pickup_onion_actions(counter_objects)

        # Geçersiz hedefleri filtrele
        if motion_goals:
            motion_goals = [
                mg for mg in motion_goals
                if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                    player.pos_and_or, mg
                )
            ]

        # Hiçbir hedef bulunamadıysa en yakın özelliğe git
        if not motion_goals or len(motion_goals) == 0:
            motion_goals = am.go_to_closest_feature_actions(player)
            motion_goals = [
                mg for mg in motion_goals
                if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                    player.pos_and_or, mg
                )
            ]

        if not motion_goals or len(motion_goals) == 0:
            chosen_action = Action.STAY
        else:
            # İlk geçerli hedefi al
            goal = motion_goals[0]
            try:
                action_plan, _, _ = self.mlam.motion_planner.get_plan(
                    player.pos_and_or, goal
                )
                chosen_action = action_plan[0]
            except Exception:
                chosen_action = Action.STAY

        # Takılma kontrolü
        if self.auto_unstuck:
            chosen_action = self._unstuck_check(state, chosen_action)

        self.prev_state = state
        action_probs = Agent.a_probs_from_action(chosen_action)
        return chosen_action, {"action_probs": action_probs}

    def _unstuck_check(self, state, chosen_action):
        """Takılırsa rastgele hareket üret."""
        if self.prev_state is None:
            return chosen_action

        if state.players_pos_and_or == self.prev_state.players_pos_and_or:
            import itertools
            if self.agent_index == 0:
                joint_actions = list(
                    itertools.product(Action.ALL_ACTIONS, [Action.STAY])
                )
            else:
                joint_actions = list(
                    itertools.product([Action.STAY], Action.ALL_ACTIONS)
                )

            unblocking = []
            for j_a in joint_actions:
                new_state, _ = self.mdp.get_state_transition(state, j_a)
                if new_state.player_positions != self.prev_state.player_positions:
                    unblocking.append(j_a)

            if len(unblocking) > 0:
                chosen = unblocking[np.random.choice(len(unblocking))]
                chosen_action = chosen[self.agent_index]

        return chosen_action