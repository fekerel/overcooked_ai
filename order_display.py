"""
Sipariş (order) bilgisini pygame surface olarak render eder.
turn_based_game'den çağrılır.
"""
import pygame

# Malzeme renkleri
INGREDIENT_COLORS = {
    "onion": (200, 180, 50),    # sarımsı
    "tomato": (220, 50, 50),    # kırmızı
}
BG_COLOR = (40, 40, 40)
TEXT_COLOR = (220, 220, 220)
BORDER_COLOR = (100, 100, 100)


def render_orders(orders, width, font_size=18):
    """
    Sipariş listesini gösteren bir pygame.Surface döndürür.

    Args:
        orders: list of Recipe nesneleri. Her birinin .ingredients attribute'u var.
                Veya list of tuple/list of ingredient strings.
        width: Surface genişliği (oyun penceresiyle eşleşmeli).
        font_size: Yazı boyutu.

    Returns:
        pygame.Surface
    """
    if not pygame.font.get_init():
        pygame.font.init()

    font = pygame.font.Font(None, font_size)
    circle_r = 8       # malzeme dairesi yarıçapı
    padding = 8
    order_h = max(circle_r * 2 + 4, font_size + 4)
    panel_h = padding + len(orders) * (order_h + padding) + padding + font_size

    surface = pygame.Surface((width, panel_h))
    surface.fill(BG_COLOR)

    # Başlık
    title = font.render("Siparisler", True, TEXT_COLOR)
    surface.blit(title, (padding, padding))

    y = padding + font_size + padding

    for i, order in enumerate(orders):
        # Recipe nesnesi mi, yoksa tuple/list mi?
        if hasattr(order, "ingredients"):
            ingredients = order.ingredients
        elif isinstance(order, dict) and "ingredients" in order:
            ingredients = order["ingredients"]
        else:
            ingredients = list(order)

        # Sipariş numarası
        label = font.render(f"#{i+1}:", True, TEXT_COLOR)
        surface.blit(label, (padding, y + 2))

        # Malzeme daireleri
        x = padding + label.get_width() + 8
        for ing in ingredients:
            color = INGREDIENT_COLORS.get(ing, (150, 150, 150))
            pygame.draw.circle(surface, color, (x + circle_r, y + order_h // 2), circle_r)
            pygame.draw.circle(surface, BORDER_COLOR, (x + circle_r, y + order_h // 2), circle_r, 1)
            x += circle_r * 2 + 6

        # Malzeme isimleri (küçük yazı)
        desc = " + ".join(ingredients)
        desc_surf = font.render(f"({desc})", True, (160, 160, 160))
        surface.blit(desc_surf, (x + 4, y + 2))

        y += order_h + padding

    # Alt çizgi
    pygame.draw.line(surface, BORDER_COLOR, (0, 0), (width, 0), 1)

    return surface
