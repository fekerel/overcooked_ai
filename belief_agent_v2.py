"""
BeliefAgent V2 — Bayesian Belief-Update Tabanlı İnsan Niyeti Tahmini

Overcooked ortamında insanın niyetini Bayes kuralıyla tahmin eden AI agent.
"""

import numpy as np
import random
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import Recipe
from overcooked_ai_py.planning.planners import (
    MediumLevelActionManager,
    NO_COUNTERS_PARAMS,
)

try:
    from belief_display import BeliefDisplay
    DISPLAY_AVAILABLE = True
except ImportError:
    DISPLAY_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
#  NİYET KÜMESİ (X_n)
# ═══════════════════════════════════════════════════════════════
# Gizli değişken X_n: İnsanın n. turdaki niyeti.
# Bu 8 niyet, Overcooked iş akışının tüm aşamalarını kapsar:
#   ingredient al → pota koy → dish al → soup al → teslim et
INTENTS = [
    "GET_ONION",          # 0: Soğan dispenser'ından soğan al
    "GET_TOMATO",         # 1: Domates dispenser'ından domates al
    "GET_DISH",           # 2: Tabak dispenser'ından tabak al
    "PUT_ONION_IN_POT",   # 3: Soğanı kazana koy
    "PUT_TOMATO_IN_POT",  # 4: Domatesi kazana koy
    "PICKUP_SOUP",        # 5: Hazır çorbayı kazandan al
    "DELIVER_SOUP",       # 6: Çorbayı servis noktasına teslim et
    "WAIT_FOR_SOUP",      # 7: Çorbanın pişmesini bekle
]
INTENT_TO_IDX = {name: i for i, name in enumerate(INTENTS)}
NUM_INTENTS = len(INTENTS)  # 8


# ═══════════════════════════════════════════════════════════════
#  FEATURE KÜMESİ (E_n)
# ═══════════════════════════════════════════════════════════════
# İnsanın hamlesinden çıkarılan 18 binary (0/1) özellik.
# 4 kategori:
#   Yön (5): hangi hedefe yaklaştı
#   Eylem (6): ne aldı / koydu / teslim etti
#   Durum (5): ne tutuyor
#   Hareket (1): yerinde mi kaldı
FEATURES = [
    # Yön özellikleri
    "toward_onion_dispenser",     # 0
    "toward_tomato_dispenser",    # 1
    "toward_dish_dispenser",      # 2
    "toward_pot",                 # 3
    "toward_serving",             # 4
    # Eylem özellikleri
    "picked_onion",               # 5
    "picked_tomato",              # 6
    "picked_dish",                # 7
    "picked_soup",                # 8
    "placed_onion_in_pot",        # 9
    "placed_tomato_in_pot",       # 10
    "delivered_soup",             # 11
    # Durum özellikleri
    "holding_dish",               # 12
    "holding_onion",              # 13
    "holding_tomato",             # 14
    "holding_soup",               # 15
    "holding_nothing",            # 16
    # Hareket özelliği
    "no_movement",                # 17
]
FEAT_TO_IDX = {name: i for i, name in enumerate(FEATURES)}
NUM_FEATURES = len(FEATURES)  # 18


# ═══════════════════════════════════════════════════════════════
#  AĞIRLIK MATRİSİ (W)
# ═══════════════════════════════════════════════════════════════
# P(E_n | X_n) ∝ exp(w_x · E_n)
#
# 8 satır (niyet) × 18 sütun (feature)
# Her hücre: "bu feature aktifse bu niyetin ne kadar olası olduğu"
#
# Ağırlık değerleri:
#   +5 = kesin olay (picked_onion → GET_ONION)
#   +3 = güçlü ipucu (toward_X → GET_X)
#   +2 = bağlam ipucu (holding_onion → PUT_ONION_IN_POT)
#   -1 = zıt ipucu (toward_dish → GET_ONION olası değil)
#    0 = ilgisiz
#
#  Sütunlar: t_on t_to t_di t_po t_sv p_on p_to p_di p_so pl_o pl_t d_so h_di h_on h_to h_so h_no no_m
WEIGHT_MATRIX = np.array([
    # GET_ONION:          soğan dispenser'ına yaklaşma(+3) ve soğan alma(+5) güçlü sinyal
    [ 3.0, -1.0, -1.0, -0.5, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    # GET_TOMATO:         domates dispenser'ına yaklaşma(+3) ve domates alma(+5)
    [-1.0,  3.0, -1.0, -0.5, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    # GET_DISH:           tabak dispenser'ına yaklaşma(+3) ve tabak alma(+5)
    [-1.0, -1.0,  3.0, -0.5, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    # PUT_ONION_IN_POT:   kazana yaklaşma(+3) + soğan koyma(+5) + elde soğan(+2)
    [-0.5,  0.0,  0.0,  3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0,-1.0, 0.0],
    # PUT_TOMATO_IN_POT:  kazana yaklaşma(+3) + domates koyma(+5) + elde domates(+2)
    [ 0.0, -0.5,  0.0,  3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 2.0, 0.0,-1.0, 0.0],
    # PICKUP_SOUP:        kazana yaklaşma(+3) + çorba alma(+5) + elde tabak(+2)
    [ 0.0,  0.0,  0.0,  3.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,-1.0, 0.0],
    # DELIVER_SOUP:       servise yaklaşma(+3) + teslim(+5) + elde çorba(+2)
    [ 0.0,  0.0,  0.0,  0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 2.0,-1.0, 0.0],
    # WAIT_FOR_SOUP:      kazana yaklaşma(+1) + hareket etmeme(+3) + elde tabak(+2)
    [ 0.0,  0.0,  0.0,  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,-2.0, 3.0],
])


# ═══════════════════════════════════════════════════════════════
#  GEÇİŞ TABLOLARI — P(X_{n+1} | X_n, Z_n)
# ═══════════════════════════════════════════════════════════════
# Her event (Z_n) için 8×8 matris.
# T[i][j] = mevcut intent i'den sonraki intent j'ye geçiş olasılığı.
# Satırlar toplanınca 1 olmalı.
#
# Intent sırası: GET_ON GET_TO GET_DI PUT_ON PUT_TO PICK_S DELIV  WAIT
#                  0      1      2      3      4      5     6      7

def _make_stay_row(idx, stay=0.8):
    """Yardımcı: idx intent'inde %stay kal, kalan uniform dağıl."""
    row = np.full(NUM_INTENTS, (1.0 - stay) / (NUM_INTENTS - 1))
    row[idx] = stay
    return row

# "none" — hiçbir özel olay olmadı, çoğunlukla aynı niyette kal
_T_none = np.array([_make_stay_row(i, 0.85) for i in range(NUM_INTENTS)])

# "placed_onion" — soğanı pota koydu
# PUT_ONION_IN_POT'tan çıkış: tekrar soğan al (%40), tabak al (%30), bekle (%20)
_T_placed_onion = _T_none.copy()
_T_placed_onion[INTENT_TO_IDX["PUT_ONION_IN_POT"]] = [
    0.40, 0.00, 0.30, 0.05, 0.00, 0.00, 0.00, 0.25
]

# "placed_tomato" — domatesi pota koydu
_T_placed_tomato = _T_none.copy()
_T_placed_tomato[INTENT_TO_IDX["PUT_TOMATO_IN_POT"]] = [
    0.00, 0.40, 0.30, 0.00, 0.05, 0.00, 0.00, 0.25
]

# "picked_dish" — tabak aldı → büyük olasılıkla soup almaya gidecek
_T_picked_dish = _T_none.copy()
_T_picked_dish[INTENT_TO_IDX["GET_DISH"]] = [
    0.00, 0.00, 0.05, 0.00, 0.00, 0.70, 0.00, 0.25
]

# "picked_soup" — çorbayı aldı → teslim etmeye gidecek
_T_picked_soup = _T_none.copy()
_T_picked_soup[INTENT_TO_IDX["PICKUP_SOUP"]] = [
    0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.90, 0.05
]

# "delivered_soup" — çorbayı teslim etti → yeni döngü başlıyor
_T_delivered = _T_none.copy()
_T_delivered[INTENT_TO_IDX["DELIVER_SOUP"]] = [
    0.35, 0.25, 0.20, 0.00, 0.00, 0.00, 0.00, 0.20
]

# Event → Geçiş matrisi eşleştirmesi
TRANSITION_TABLES = {
    "none":           _T_none,
    "placed_onion":   _T_placed_onion,
    "placed_tomato":  _T_placed_tomato,
    "picked_dish":    _T_picked_dish,
    "picked_soup":    _T_picked_soup,
    "delivered_soup": _T_delivered,
}

# Event tespiti: observation feature → event adı
EVENT_FEATURES = [
    ("placed_onion_in_pot", "placed_onion"),
    ("placed_tomato_in_pot", "placed_tomato"),
    ("delivered_soup",       "delivered_soup"),
    ("picked_dish",          "picked_dish"),
    ("picked_soup",          "picked_soup"),
]


class BeliefAgentV2(Agent):
    """
    Bayesian belief-update tabanlı AI agent.
    
    Her turda:
    1. İnsanın hamlesini gözlemle (observation)
    2. Bayes kuralıyla belief güncelle 
    3. Geçiş modeliyle bir sonraki turu tahmin et
    4. Belief'e dayalı karar ver
    """

    def __init__(self):
        super().__init__()
        self.mdp = None
        self._mp = None           # MotionPlanner (MLAM'ın iç pathfinder'ı)
        self._mlam = None         # MediumLevelActionManager
        self._intent_mask = np.ones(NUM_INTENTS)  # layout'a göre 0/1 maske
        self._display = None
        self._last_action = None  # tıkanma tespiti için
        if DISPLAY_AVAILABLE:
            self._display = BeliefDisplay(INTENTS)

    def reset(self):
        super().reset()
        self.prev_state = None
        self.belief = np.ones(NUM_INTENTS) / NUM_INTENTS  # uniform başlangıç
        self._posterior = self.belief.copy()  # karar için kullanılan posterior
        self._last_action = None

    def set_mdp(self, mdp, initial_state=None):
        """
        Layout yüklendikten sonra çağrılır.
        - MLAM + MotionPlanner oluştur (pathfinding için)
        - Layout'ta eksik dispenser varsa ilgili intent'leri maskele
        - initial_state verilmişse prev_state olarak kaydet
        """
        self.mdp = mdp

        # Counter'ları da dahil eden parametreler
        counter_locs = mdp.get_counter_locations()
        mlam_params = {
            "start_orientations": False,
            "wait_allowed": False,
            "counter_goals": counter_locs,
            "counter_drop": counter_locs,
            "counter_pickup": counter_locs,
            "same_motion_goals": True,
        }
        self._mlam = MediumLevelActionManager.from_pickle_or_compute(
            mdp, mlam_params, force_compute=False,
        )
        self._mp = self._mlam.joint_motion_planner.motion_planner

        # ── Intent maskeleme ──
        # Eğer layout'ta domates dispenser'ı yoksa domates intent'leri kapalı
        # Eğer layout'ta soğan dispenser'ı yoksa soğan intent'leri kapalı
        mask = np.ones(NUM_INTENTS)
        if not mdp.get_tomato_dispenser_locations():
            mask[INTENT_TO_IDX["GET_TOMATO"]] = 0
            mask[INTENT_TO_IDX["PUT_TOMATO_IN_POT"]] = 0
        if not mdp.get_onion_dispenser_locations():
            mask[INTENT_TO_IDX["GET_ONION"]] = 0
            mask[INTENT_TO_IDX["PUT_ONION_IN_POT"]] = 0
        self._intent_mask = mask

        # Belief'i maskeye göre yeniden normalize et
        self.belief *= self._intent_mask
        s = self.belief.sum()
        if s > 0:
            self.belief /= s
        self._posterior = self.belief.copy()

        # Display'e başlangıç belief'ini gönder
        if self._display:
            self._display.update(self.belief, "B'(X_1) başlangıç", "posterior")
            self._display.update(self.belief, "B'(X_1) başlangıç", "predicted")

        # Başlangıç state'i verilmişse kaydet
        if initial_state is not None:
            self.prev_state = initial_state

        # ── Dispenser erişim haritası ──
        # Her oyuncunun hangi dispenser türlerine erişebildiğini hesapla
        # Layout sabit olduğu için bir kere hesaplamak yeterli
        self._player_can_reach = {0: set(), 1: set()}
        dispenser_map = {
            "onion": mdp.get_onion_dispenser_locations(),
            "tomato": mdp.get_tomato_dispenser_locations(),
        }
        if initial_state is not None:
            for pidx in (0, 1):
                start = initial_state.players[pidx].pos_and_or
                for ing, locs in dispenser_map.items():
                    if not locs:
                        continue
                    cost = self._mp.min_cost_to_feature(start, locs)
                    if cost < float("inf"):
                        self._player_can_reach[pidx].add(ing)

    def action(self, state):
        """
        Her turda çağrılır.
        1. prev_state varsa → observation çıkar, belief update, transition
        2. Karar ver (_decide)
        3. prev_state güncelle
        """
        if self.prev_state is not None:
            obs = self._extract_observation(self.prev_state, state)
            # Posterior hesapla — self._posterior güncellenir, self.belief değişmez
            self._belief_update(obs)
            # Display: posterior panel
            if self._display:
                self._display.update(self._posterior, "B(X_n) posterior", "posterior")
            # Transition model — self.belief = B'(X_{n+1}) (sonraki turun prior'u)
            self._predict_next_belief(obs)
            # Display: predicted panel
            if self._display:
                self._display.update(self.belief, "B'(X_{n+1}) predicted", "predicted")

        result = self._decide(state)

        # Tıkanma tespiti
        result = self._unstuck(state, result)

        self._last_action = result[0]
        self.prev_state = state
        return result

    def _unstuck(self, state, result):
        """
        Tıkanma tespiti ve çözümü.
        Önceki turda hareket aksiyonu verdik ama pozisyon değişmediyse
        → komşu yürünebilir karelerden rastgele birine git.
        """
        ai = state.players[self.agent_index]
        if (self._last_action in Direction.ALL_DIRECTIONS
                and self.prev_state is not None
                and self.prev_state.players[self.agent_index].pos_and_or == ai.pos_and_or):
            walkable = set(self.mdp.get_valid_player_positions())
            valid = []
            for d in Direction.ALL_DIRECTIONS:
                new_pos = Action.move_in_direction(ai.position, d)
                if new_pos in walkable:
                    valid.append(d)
            if valid:
                return (random.choice(valid), {})
        return result

    def _decide(self, state):
        """
        Karar koordinatörü:
        - Katman 1 (fiziksel kurallar) → sonuç varsa döndür
        - Katman 2 (belief-tabanlı) → eli boşken ne almalıyım?
        """
        result = self._physical_rules(state)
        if result is not None:
            return result
        return self._belief_decision(state)

    # ─────────────────────────────────
    #  Katman 2: Belief-Tabanlı Karar
    # ─────────────────────────────────

    def _belief_decision(self, state):
        """
        Eli boş AI: posterior'a ve pot durumuna bakarak ne alacağını seç.

        1. Tüm potlar dolu (cooking/ready) → tabak al
        2. Eksik pot var, ingredient < 2  → pot'a uyumlu ingredient al
        3. Eksik pot var, ingredient == 2 → insanın intent'ine bak:
           - İnsan ingredient taşıyorsa → tabak al
           - Değilse → ingredient al
        Fallback: STAY
        """
        MAX = Recipe.MAX_NUM_INGREDIENTS

        # Pot analizi: en dolu eksik pot'u bul
        best_pot = None
        best_count = -1
        best_ingredient = None
        all_full = True

        for pot_loc in self.mdp.get_pot_locations():
            if not state.has_object(pot_loc):
                # Boş pot
                all_full = False
                if best_count < 0:
                    best_pot = pot_loc
                    best_count = 0
                    best_ingredient = None
                continue

            obj = state.get_object(pot_loc)
            if hasattr(obj, "ingredients"):
                count = len(obj.ingredients)
                if count < MAX:
                    all_full = False
                    if count > best_count:
                        best_pot = pot_loc
                        best_count = count
                        best_ingredient = obj.ingredients[0] if obj.ingredients else None

        # Adım 1: Tüm potlar dolu → tabak al
        if all_full:
            result = self._go(state, self.mdp.get_dish_dispenser_locations())
            return result if result is not None else (Action.STAY, {})

        # Adım 2-3: Eksik pot var
        if best_pot is not None:
            onion_disps = self.mdp.get_onion_dispenser_locations()
            tomato_disps = self.mdp.get_tomato_dispenser_locations()
            top_intent = INTENTS[np.argmax(self._posterior)]

            # Hangi ingredient? Pot'ta varsa aynısını, yoksa intent'e göre
            target_ing = best_ingredient
            if target_ing is None:
                # Boş pot → insanın intent'ine bakarak ingredient belirle
                if top_intent in ("GET_ONION", "PUT_ONION_IN_POT") and onion_disps:
                    target_ing = "onion"
                elif top_intent in ("GET_TOMATO", "PUT_TOMATO_IN_POT") and tomato_disps:
                    target_ing = "tomato"
                else:
                    # Belirsiz → layout'un ilk available ingredient'ı
                    target_ing = "onion" if onion_disps else "tomato" if tomato_disps else None

            if target_ing is None:
                return Action.STAY, {}

            # İnsan bu ingredient'a erişemiyor mu?
            # Erişemiyorsa AI kesinlikle bu ingredient'ı almalı (intent'e bakmadan)
            human_idx = 1 - self.agent_index
            human_can = self._player_can_reach.get(human_idx, set())
            if target_ing not in human_can:
                # İnsan bu ingredient'ı taşıyamaz → AI alsın
                if target_ing == "onion" and onion_disps:
                    result = self._go(state, onion_disps)
                    if result is not None:
                        return result
                elif target_ing == "tomato" and tomato_disps:
                    result = self._go(state, tomato_disps)
                    if result is not None:
                        return result

            # Adım 3: Pot 2 ingredient'liyse insanın intent'ine bak
            if best_count == 2:
                ingredient_intents = {"GET_ONION", "GET_TOMATO",
                                      "PUT_ONION_IN_POT", "PUT_TOMATO_IN_POT"}
                if top_intent in ingredient_intents:
                    # İnsan ingredient taşıyacak → AI tabak alsın
                    result = self._go(state, self.mdp.get_dish_dispenser_locations())
                    if result is not None:
                        return result

            # Ingredient al
            if target_ing == "onion" and onion_disps:
                result = self._go(state, onion_disps)
                if result is not None:
                    return result
            elif target_ing == "tomato" and tomato_disps:
                result = self._go(state, tomato_disps)
                if result is not None:
                    return result

        return Action.STAY, {}

    # ─────────────────────────────────
    #  Observation Extraction
    # ─────────────────────────────────

    def _extract_observation(self, prev_state, curr_state):
        """
        S_{n-1} ve S_n karşılaştırmasıyla E_n (18-boyutlu binary) vektör oluştur.
        
        İnsan oyuncunun:
        - Pozisyon değişimi → toward_X features
        - Tutulan nesne değişimi → picked_X, placed_X features
        - Mevcut durumu → holding_X features
        - Hareketsizlik → no_movement feature
        """
        hi = 1 - self.agent_index  # insan player index
        prev_p = prev_state.players[hi]
        curr_p = curr_state.players[hi]

        prev_pos = prev_p.position
        curr_pos = curr_p.position
        prev_held = prev_p.held_object.name if prev_p.held_object else None
        curr_held = curr_p.held_object.name if curr_p.held_object else None

        obs = np.zeros(NUM_FEATURES)

        # ── Nesne alma (pickup): önceki turda eli boştu, şimdi bir şey tutuyor ──
        if prev_held is None:
            pickup_map = {
                "onion": "picked_onion",
                "tomato": "picked_tomato",
                "dish": "picked_dish",
                "soup": "picked_soup",
            }
            if curr_held in pickup_map:
                obs[FEAT_TO_IDX[pickup_map[curr_held]]] = 1

        # ── Pot'a koyma: elinden ingredient düştü VE pot'taki sayı arttı ──
        if prev_held == "onion" and curr_held is None:
            if self._count_in_pots(curr_state, "onion") > self._count_in_pots(prev_state, "onion"):
                obs[FEAT_TO_IDX["placed_onion_in_pot"]] = 1

        if prev_held == "tomato" and curr_held is None:
            if self._count_in_pots(curr_state, "tomato") > self._count_in_pots(prev_state, "tomato"):
                obs[FEAT_TO_IDX["placed_tomato_in_pot"]] = 1

        # ── Soup teslim: elinde soup vardı, şimdi yok ──
        if prev_held == "soup" and curr_held is None:
            obs[FEAT_TO_IDX["delivered_soup"]] = 1

        # ── Şu an ne tutuyor? ──
        holding_map = {
            "dish": "holding_dish",
            "onion": "holding_onion",
            "tomato": "holding_tomato",
            "soup": "holding_soup",
        }
        if curr_held in holding_map:
            obs[FEAT_TO_IDX[holding_map[curr_held]]] = 1
        else:
            obs[FEAT_TO_IDX["holding_nothing"]] = 1

        # ── Hareket yönü: nereye yaklaştı? (MLAM pathfinding mesafesi ile) ──
        if prev_pos != curr_pos:
            # İnsanın önceki ve şimdiki pozisyon+yönelim bilgisi
            prev_por = prev_p.pos_and_or  # ((x,y), direction)
            curr_por = curr_p.pos_and_or
            direction_targets = {
                "toward_onion_dispenser": self.mdp.get_onion_dispenser_locations(),
                "toward_tomato_dispenser": self.mdp.get_tomato_dispenser_locations(),
                "toward_dish_dispenser": self.mdp.get_dish_dispenser_locations(),
                "toward_pot": self.mdp.get_pot_locations(),
                "toward_serving": self.mdp.get_serving_locations(),
            }
            for feat_name, positions in direction_targets.items():
                if positions and self._closer_mlam(prev_por, curr_por, positions):
                    obs[FEAT_TO_IDX[feat_name]] = 1
        else:
            obs[FEAT_TO_IDX["no_movement"]] = 1

        return obs

    def _closer_mlam(self, prev_por, curr_por, target_positions):
        """
        MLAM pathfinding mesafesi ile: insan hedefe yaklaştı mı?

        min_cost_to_feature(pos_and_or, feature_pos_list) →
          o feature'a ulaşmak için gereken minimum adım sayısı (duvarları hesaba katar).

        prev_por/curr_por: ((x,y), (dx,dy)) formatında pos_and_or
        target_positions: terrain lokasyonlarının listesi [(x,y), ...]
        """
        try:
            prev_cost = self._mp.min_cost_to_feature(prev_por, target_positions)
            curr_cost = self._mp.min_cost_to_feature(curr_por, target_positions)
            return curr_cost < prev_cost
        except Exception:
            # Erişilemeyen pozisyon varsa Manhattan fallback
            return self._closer(prev_por[0], curr_por[0], target_positions)

    @staticmethod
    def _closer(old_pos, new_pos, targets):
        """Manhattan fallback — MLAM başarısız olursa kullanılır."""
        def min_manhattan(p):
            return min(abs(p[0] - t[0]) + abs(p[1] - t[1]) for t in targets)
        return min_manhattan(new_pos) < min_manhattan(old_pos)

    def _count_in_pots(self, state, ingredient):
        """Tüm pot'lardaki belirli ingredient toplam sayısı."""
        total = 0
        for pot_loc in self.mdp.get_pot_locations():
            if not state.has_object(pot_loc):
                continue
            obj = state.get_object(pot_loc)
            if hasattr(obj, "ingredients"):
                total += sum(1 for ing in obj.ingredients if ing == ingredient)
        return total

    # ─────────────────────────────────
    #  Belief Update (Bayes Kuralı)
    # ─────────────────────────────────

    def _belief_update(self, obs):
        """
        Bayes güncellemesi:  P(X_n | E_n) ∝ P(E_n | X_n) × P(X_n)

        1. scores = W @ obs          → her intent için uyum skoru
        2. likelihood = exp(scores)  → olabilirlik
        3. posterior = likelihood × prior (mevcut belief)
        4. maskeleme + normalize
        """
        scores = WEIGHT_MATRIX @ obs              # (8,18) @ (18,) → (8,)
        likelihood = np.exp(scores)                # (8,)
        posterior = likelihood * self.belief        # element-wise × prior (self.belief = B'(X_n))
        posterior *= self._intent_mask             # layout maskeleme
        s = posterior.sum()
        if s > 0:
            self._posterior = posterior / s
        else:
            self._posterior = self.belief.copy()
        # self.belief değişmedi — hala B'(X_n) (önceki turun predicted'ı)

    # ─────────────────────────────────
    #  Transition Model (Geçiş Modeli)
    # ─────────────────────────────────

    @staticmethod
    def _extract_event(obs):
        """
        Observation vektöründen hangi tamamlanmış event olduğunu çıkar.
        İlk eşleşen event döner (öncelik sırası: place > deliver > pick).
        Hiçbiri yoksa "none".
        """
        for feat_name, event_name in EVENT_FEATURES:
            if obs[FEAT_TO_IDX[feat_name]] == 1:
                return event_name
        return "none"

    def _predict_next_belief(self, obs):
        """
        Geçiş modeli: B'(X_{n+1}) = Σ_x P(X_{n+1}|X_n=x, Z_n) × B(X_n=x)

        1. Event tespit et (Z_n)
        2. İlgili geçiş matrisini seç (T)
        3. predicted = T^T @ belief  (matris-vektör çarpımı)
        4. Maskele + normalize
        5. self.belief = predicted  (sonraki turun prior'u olarak sakla)
        """
        event = self._extract_event(obs)
        T = TRANSITION_TABLES[event]               # (8,8)
        predicted = T.T @ self._posterior           # T × B(X_n) = B'(X_{n+1})
        predicted *= self._intent_mask              # layout maskeleme
        s = predicted.sum()
        if s > 0:
            predicted /= s
        self.belief = predicted                     # sonraki turun prior'u B'(X_{n+1})

    # ─────────────────────────────────
    #  Pathfinding Helper (_go)
    # ─────────────────────────────────

    def _go(self, state, target_positions):
        """
        MLAM ile hedef pozisyonlardan en yakınına git.
        Döner: (aksiyon, {}) veya None erişilemezse.

        target_positions: terrain lokasyonları [(x,y), ...]
        MotionPlanner bu lokasyonlara ulaşmak için gerekli ilk adımı verir.
        """
        ai = state.players[self.agent_index]
        start = ai.pos_and_or

        # Hedef pozisyonlar için motion_goals bul
        # motion_goals_for_pos: terrain_pos → [(pos, orient), ...] (yaklaşma noktaları)
        goals = []
        for tpos in target_positions:
            mg = self._mp.motion_goals_for_pos.get(tpos, [])
            goals.extend(mg)

        if not goals:
            return None

        # En yakın goal'a olan plan
        min_cost = float("inf")
        best_action = None
        for g in goals:
            if not self._mp.is_valid_motion_start_goal_pair(start, g):
                continue
            try:
                action_plan, _, cost = self._mp.get_plan(start, g)
                if cost < min_cost:
                    min_cost = cost
                    best_action = action_plan[0] if action_plan else Action.INTERACT
            except Exception as e:
                print(f"[_go] get_plan failed: start={start}, goal={g}, error={e}")
                continue

        if best_action is None:
            return None
        return best_action, {}

    # ─────────────────────────────────
    #  Pot Uyumluluk Helpers
    # ─────────────────────────────────

    def _compatible_pots(self, state, ingredient):
        """
        Elindeki ingredient ile uyumlu pot'ları döndür:
        - Boş pot'lar (hiç ingredient yok)
        - Aynı ingredient'i içeren ve henüz dolu olmayan pot'lar
        """
        MAX = Recipe.MAX_NUM_INGREDIENTS  # genelde 3
        result = []
        for pot_loc in self.mdp.get_pot_locations():
            if not state.has_object(pot_loc):
                result.append(pot_loc)  # boş pot
                continue
            obj = state.get_object(pot_loc)
            if not hasattr(obj, "ingredients"):
                result.append(pot_loc)  # boş pot (obje var ama ingredient yok)
            elif len(obj.ingredients) < MAX:
                # Pot'ta ne var?
                if all(ing == ingredient for ing in obj.ingredients):
                    result.append(pot_loc)  # aynı ingredient, henüz dolu değil
        return result

    def _ready_pots(self, state):
        """Hazır soup içeren pot lokasyonları."""
        result = []
        for pot_loc in self.mdp.get_pot_locations():
            if state.has_object(pot_loc):
                obj = state.get_object(pot_loc)
                if obj.name == "soup" and obj.is_ready:
                    result.append(pot_loc)
        return result

    def _cooking_pots(self, state):
        """Pişmekte olan (henüz ready olmayan ama dolu) pot lokasyonları."""
        MAX = Recipe.MAX_NUM_INGREDIENTS
        result = []
        for pot_loc in self.mdp.get_pot_locations():
            if state.has_object(pot_loc):
                obj = state.get_object(pot_loc)
                if obj.name == "soup" and not obj.is_ready and len(obj.ingredients) >= MAX:
                    result.append(pot_loc)
        return result

    def _nearest_counter(self, state):
        """
        Boş counter lokasyonları (nesne olmayan, terrain 'X').
        AI'ın nesne bırakabileceği yerler.
        """
        counters = []
        for pos in self.mdp.get_counter_locations():
            if not state.has_object(pos):
                counters.append(pos)
        return counters

    def _midpoint_counter(self, state):
        """
        İki oyuncunun orta noktasına en yakın boş counter'ları döner.
        Manhattan distance ile sıralı, en yakından en uzağa.
        """
        counters = self._nearest_counter(state)
        if not counters:
            return []
        p0 = state.players[0].position
        p1 = state.players[1].position
        mid = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
        counters.sort(key=lambda c: abs(c[0] - mid[0]) + abs(c[1] - mid[1]))
        return counters

    # ─────────────────────────────────
    #  Katman 1: Fiziksel Güvenlik Kuralları
    # ─────────────────────────────────

    def _physical_rules(self, state):
        """
        Mevcut state'e bakarak zorunlu kararlar. Belief'e bakmaz.
        Döner: (aksiyon, {}) veya None (karar veremediyse).

        Kural sırası:
        1. Elde soup → serving'e git
        2. Elde dish + ready pot → pot'a git (soup al)
        3. Elde dish + pişen pot → pot'a git (bekle/interact dene)
        4. Elde dish + hiçbiri yok → counter'a bırak
        5. Elde ingredient + uyumlu pot → pot'a git (koy)
        6. Elde ingredient + uyumlu pot yok → counter'a bırak
        _go None dönebilir (erişilemez) → sonraki kurala/fallback'e geç
        """
        ai = state.players[self.agent_index]
        held = ai.held_object

        if held is None:
            return None  # eli boş, Katman 2'ye düş

        obj_name = held.name

        # Kural 1: Elde soup → serving'e git
        if obj_name == "soup":
            result = self._go(state, self.mdp.get_serving_locations())
            if result is not None:
                return result
            # Serving'e erişilemiyorsa counter'a bırak
            counters = self._midpoint_counter(state)
            if counters:
                result = self._go(state, counters)
                if result is not None:
                    return result
            return Action.STAY, {}

        # Kural 2-4: Elde dish
        if obj_name == "dish":
            ready = self._ready_pots(state)
            if ready:
                result = self._go(state, ready)
                if result is not None:
                    return result                       # Kural 2
            cooking = self._cooking_pots(state)
            if cooking:
                result = self._go(state, cooking)
                if result is not None:
                    return result                       # Kural 3
            counters = self._midpoint_counter(state)
            if counters:
                result = self._go(state, counters)
                if result is not None:
                    return result                       # Kural 4
            return Action.STAY, {}

        # Kural 5-6: Elde ingredient (onion/tomato)
        if obj_name in ("onion", "tomato"):
            compat = self._compatible_pots(state, obj_name)
            if compat:
                result = self._go(state, compat)
                if result is not None:
                    return result                       # Kural 5
            # Pot'a erişilemiyorsa veya uyumlu pot yoksa → counter'a bırak
            counters = self._midpoint_counter(state)
            if counters:
                result = self._go(state, counters)
                if result is not None:
                    return result                       # Kural 6
            return Action.STAY, {}

        return None  # bilinmeyen nesne, Katman 2'ye düş
