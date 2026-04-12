# Oyuncu 1 (mavi sef): Oyuncu
# Oyuncu 2 (yesil sef): Yapay zeka ajanı
#
# Kontroller:
#   W / Yukari ok   = Yukari git
#   A / Sol ok      = Sola git
#   S / Asagi ok    = Asagi git
#   D / Sag ok      = Saga git
#   SPACE           = Etkilesim
#   Q               = Bekle
#   ESC             = Oyundan cik
#
# Sira tabanlı: Once oyuncu hareket eder, sonra ajan hareket eder.


import sys
import os
import pygame
from pygame.locals import *

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

# BeliefAgent import: workspace root'tan
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from belief_agent_v2 import BeliefAgentV2 as BeliefAgent
from order_display import render_orders


# --- AYARLAR ---
DEFAULT_LAYOUT = "forced_coordination_tomato"       # Varsayılan harita ismi
LAYOUT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LAYOUT
if LAYOUT.endswith(".layout"):
    LAYOUT = LAYOUT[:-len(".layout")]
HORIZON = 400                 # Maksimum adim sayisi
WINDOW_SCALE = 3              # Pencere buyukluk (1=kucuk, 3=buyuk)
FPS = 30                      # Pencere FPS

# Tuslar -> aksiyonlar
KEY_ACTION_MAP = {
    K_UP:    Direction.NORTH,
    K_w:     Direction.NORTH,
    K_DOWN:  Direction.SOUTH,
    K_s:     Direction.SOUTH,
    K_LEFT:  Direction.WEST,
    K_a:     Direction.WEST,
    K_RIGHT: Direction.EAST,
    K_d:     Direction.EAST,
    K_SPACE: Action.INTERACT,
    K_q:     Action.STAY,
}


def render_to_window(vis, state, grid, window, score, timestep, turn_info, orders=None):
    hud_data = {
        "timestep": timestep,
        "score": score,
        "turn": turn_info,
    }

    game_surface = vis.render_state(state, grid, hud_data=hud_data)
    win_w, win_h = window.get_size()

    if orders:
        # Sipariş panelini oyunun altına ekle
        order_panel = render_orders(orders, game_surface.get_width())
        panel_h = order_panel.get_height()
        combined = pygame.Surface((game_surface.get_width(), game_surface.get_height() + panel_h))
        combined.blit(game_surface, (0, 0))
        combined.blit(order_panel, (0, game_surface.get_height()))
        scaled = pygame.transform.scale(combined, (win_w, win_h))
    else:
        scaled = pygame.transform.scale(game_surface, (win_w, win_h))

    window.blit(scaled, (0, 0))
    pygame.display.flip()


def get_player_action(events):
    """Pygame event'lerinden oyuncu aksiyonunu al. None donerse bekle."""
    for event in events:
        if event.type == QUIT:
            return "QUIT"
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                return "QUIT"
            if event.key in KEY_ACTION_MAP:
                return KEY_ACTION_MAP[event.key]
    return None  # Henuz tus basilmadi


def main():
    print(f"Harita yukleniyor: {LAYOUT}")
    # Layout'a göre geçerli siparişleri belirle
    layout_orders = []
    # Geçici MDP oluştur (ingredient varlığını kontrol etmek için)
    _tmp_mdp = OvercookedGridworld.from_layout_name(LAYOUT)
    if _tmp_mdp.get_onion_dispenser_locations():
        layout_orders.append({"ingredients": ("onion", "onion", "onion")})
    if _tmp_mdp.get_tomato_dispenser_locations():
        layout_orders.append({"ingredients": ("tomato", "tomato", "tomato")})
    if not layout_orders:
        # Hiçbiri yoksa fallback (olmaması lazım ama güvenlik)
        layout_orders.append({"ingredients": ("onion", "onion", "onion")})

    mdp = OvercookedGridworld.from_layout_name(
        LAYOUT,
        start_all_orders=layout_orders,
        recipe_values=[20] * len(layout_orders),
        recipe_times=[20] * len(layout_orders),
    )
    env = OvercookedEnv.from_mdp(mdp, horizon=HORIZON, info_level=0)

    print("AI agent hazirlaniyor (yol hesaplaniyor)...")
    ai_agent = BeliefAgent()
    ai_agent.reset()
    ai_agent.set_agent_index(1)
    ai_agent.set_mdp(mdp, initial_state=env.state)
    print("Hazir.")

    # Gorsellestirici
    vis = StateVisualizer()
    grid = mdp.terrain_mtx

    test_surface = vis.render_state(env.state, grid)
    base_w, base_h = test_surface.get_size()

    # Sipariş paneli yüksekliğini de hesaba kat
    orders = env.state.all_orders
    if orders:
        test_order_panel = render_orders(orders, base_w)
        base_h += test_order_panel.get_height()

    pygame.init()
    # Ekrana sığacak şekilde otomatik ölçekle
    screen_info = pygame.display.Info()
    max_w = int(screen_info.current_w * 0.80)
    max_h = int(screen_info.current_h * 0.80)
    auto_scale = min(max_w / base_w, max_h / base_h, WINDOW_SCALE)
    win_w = int(base_w * auto_scale)
    win_h = int(base_h * auto_scale)

    window = pygame.display.set_mode((win_w, win_h), HWSURFACE | DOUBLEBUF | RESIZABLE)
    pygame.display.set_caption(f"Overcooked Turn-Based - {LAYOUT}")
    clock = pygame.time.Clock()

    score = 0
    turn = "INSAN"

    # Baslangic ekranini goster
    render_to_window(vis, env.state, grid, window, score, env.state.timestep, "SENIN SIRAN (WASD + SPACE)", orders)

    print("\n=== OYUN BASLADI ===")
    print("WASD = hareket, SPACE = etkilesim, Q = bekle, ESC = cik")
    print()

    running = True
    while running and not env.is_done():
        # ADIM 1: INSAN OYUNCUNUN SIRASI

        turn = "SENIN SIRAN (WASD + SPACE)"
        render_to_window(vis, env.state, grid, window, score, env.state.timestep, turn, orders)

        # İnsan aksiyonunu bekle
        human_action = None
        while human_action is None:
            clock.tick(FPS)
            events = pygame.event.get()
            human_action = get_player_action(events)
            if human_action == "QUIT":
                running = False
                break

            for event in events:
                if event.type == VIDEORESIZE:
                    window = pygame.display.set_mode(
                        event.dict["size"], HWSURFACE | DOUBLEBUF | RESIZABLE
                    )
                    render_to_window(vis, env.state, grid, window, score, env.state.timestep, turn, orders)

        if not running:
            break

        # Insan hareket eder, ajan bekler
        state, reward, done, info = env.step((human_action, Action.STAY))
        score += reward

        if reward > 0:
            print(f"  +{reward} ODUL! (insan hareketi sonrasi)")

        # Gorseli guncelle
        render_to_window(vis, state, grid, window, score, state.timestep, "AI DUSUNUYOR...", orders)

        if env.is_done():
            break

        pygame.time.wait(200)

        # ADIM 2: AI'NIN SIRASI
        ai_action, _ = ai_agent.action(state)
        state, reward, done, info = env.step((Action.STAY, ai_action))
        score += reward

        if reward > 0:
            print(f"  +{reward} ODUL! (AI hareketi sonrasi)")

        # Gorseli guncelle
        render_to_window(vis, state, grid, window, score, state.timestep, "SENIN SIRAN (WASD + SPACE)", orders)

    print(f"\n=== OYUN BITTI ===")
    print(f"Toplam adim: {env.state.timestep}")
    print(f"Toplam skor: {score}")

    if running:
        font = pygame.font.Font(None, 48)
        text = font.render(f"OYUN BITTI! Skor: {score}", True, (255, 255, 255))
        overlay = pygame.Surface(window.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        window.blit(overlay, (0, 0))
        text_rect = text.get_rect(center=(window.get_width()//2, window.get_height()//2))
        window.blit(text, text_rect)
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    waiting = False

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()