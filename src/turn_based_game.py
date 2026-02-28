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
import pygame
from pygame.locals import *

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.agents.rule_based_agent import RuleBasedAgent
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer


# --- AYARLAR ---
LAYOUT = "five_by_five"       # Harita ismi
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


def render_to_window(vis, state, grid, window, score, timestep, turn_info):
    hud_data = {
        "timestep": timestep,
        "score": score,
        "turn": turn_info,
    }

    surface = vis.render_state(state, grid, hud_data=hud_data)

    scaled = pygame.transform.scale(surface, window.get_size())
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
    mdp = OvercookedGridworld.from_layout_name(LAYOUT)
    env = OvercookedEnv.from_mdp(mdp, horizon=HORIZON, info_level=0)

    print("AI agent hazirlaniyor (yol hesaplaniyor)...")
    mlam = MediumLevelActionManager.from_pickle_or_compute(
        mdp, NO_COUNTERS_PARAMS, force_compute=True
    )
    ai_agent = RuleBasedAgent(mlam)
    ai_agent.set_agent_index(1)
    ai_agent.set_mdp(mdp)
    print("Hazir.")

    # Gorsellestirici
    vis = StateVisualizer()
    grid = mdp.terrain_mtx

    test_surface = vis.render_state(env.state, grid)
    base_w, base_h = test_surface.get_size()
    win_w = base_w * WINDOW_SCALE
    win_h = base_h * WINDOW_SCALE

    pygame.init()
    window = pygame.display.set_mode((win_w, win_h), HWSURFACE | DOUBLEBUF | RESIZABLE)
    pygame.display.set_caption(f"Overcooked Turn-Based - {LAYOUT}")
    clock = pygame.time.Clock()

    score = 0
    turn = "INSAN" 

    # Baslangic ekranini goster
    render_to_window(vis, env.state, grid, window, score, env.state.timestep, "SENIN SIRAN (WASD + SPACE)")

    print("\n=== OYUN BASLADI ===")
    print("WASD = hareket, SPACE = etkilesim, Q = bekle, ESC = cik")
    print()

    running = True
    while running and not env.is_done():
        # ADIM 1: INSAN OYUNCUNUN SIRASI

        turn = "SENIN SIRAN (WASD + SPACE)"
        render_to_window(vis, env.state, grid, window, score, env.state.timestep, turn)

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
                    render_to_window(vis, env.state, grid, window, score, env.state.timestep, turn)

        if not running:
            break

        # Insan hareket eder, ajan bekler
        state, reward, done, info = env.step((human_action, Action.STAY))
        score += reward

        if reward > 0:
            print(f"  +{reward} ODUL! (insan hareketi sonrasi)")

        # Gorseli guncelle
        render_to_window(vis, state, grid, window, score, state.timestep, "AI DUSUNUYOR...")

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
        render_to_window(vis, state, grid, window, score, state.timestep, "SENIN SIRAN (WASD + SPACE)")

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