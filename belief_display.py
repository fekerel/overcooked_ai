"""
Ayrı bir Tkinter penceresinde belief dağılımını bar chart olarak gösterir.
İki panel: B(X_n) posterior ve B'(X_{n+1}) predicted yan yana.
"""

import threading
import tkinter as tk


class BeliefDisplay:
    """Thread-safe Tkinter belief görselleştirici — iki panel."""

    def __init__(self, intent_names, panel_width=380, bar_height=26):
        self._intents = intent_names
        self._n = len(intent_names)
        self._pw = panel_width
        self._bar_h = bar_height
        self._pad = 4
        self._label_w = 160
        self._bar_w = panel_width - self._label_w - 55
        self._panel_h = self._n * (self._bar_h + self._pad) + self._pad
        self._gap = 20  # paneller arası boşluk

        self._root = None
        self._ready = threading.Event()

        # Her panel için widget referansları
        self._panels = {}  # "posterior" ve "predicted"

        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        self._ready.wait()

    def _run(self):
        self._root = tk.Tk()
        self._root.title("Belief Distribution")
        self._root.resizable(False, False)

        top = tk.Frame(self._root)
        top.pack(padx=5, pady=5)

        for idx, key in enumerate(("posterior", "predicted")):
            frame = tk.Frame(top)
            frame.grid(row=0, column=idx, padx=(0 if idx == 0 else self._gap, 0))

            title_var = tk.StringVar(value="Waiting...")
            lbl = tk.Label(frame, textvariable=title_var,
                           font=("Consolas", 10, "bold"), anchor="w")
            lbl.pack(fill="x")

            canvas = tk.Canvas(frame, width=self._pw,
                               height=self._panel_h, bg="white")
            canvas.pack()

            bars = []
            labels = []
            val_texts = []
            for i, name in enumerate(self._intents):
                y = self._pad + i * (self._bar_h + self._pad)
                l = canvas.create_text(
                    self._label_w - 5, y + self._bar_h // 2,
                    text=name, anchor="e", font=("Consolas", 9))
                labels.append(l)
                b = canvas.create_rectangle(
                    self._label_w, y, self._label_w + 1, y + self._bar_h,
                    fill="#4A90D9", outline="#2C5F8A")
                bars.append(b)
                v = canvas.create_text(
                    self._label_w + self._bar_w + 5, y + self._bar_h // 2,
                    text="0.000", anchor="w", font=("Consolas", 9))
                val_texts.append(v)

            self._panels[key] = {
                "title_var": title_var,
                "canvas": canvas,
                "bars": bars,
                "val_texts": val_texts,
            }

        self._ready.set()
        self._root.mainloop()

    def update(self, belief, label="", panel="posterior"):
        """Belief vektörünü güncelle. panel='posterior' veya 'predicted'."""
        if self._root is None:
            return
        self._root.after(0, self._do_update, belief.copy(), label, panel)

    def _do_update(self, belief, label, panel):
        p = self._panels.get(panel)
        if p is None:
            return

        p["title_var"].set(label if label else panel)
        canvas = p["canvas"]
        max_val = max(belief.max(), 0.001)

        for i in range(self._n):
            y = self._pad + i * (self._bar_h + self._pad)
            val = belief[i]
            bar_len = max(1, int((val / max_val) * self._bar_w))

            canvas.coords(p["bars"][i],
                          self._label_w, y,
                          self._label_w + bar_len, y + self._bar_h)

            intensity = int(200 * (1 - val)) + 40
            if panel == "posterior":
                color = f"#{40:02x}{intensity:02x}{220:02x}"  # mavi
            else:
                color = f"#{40:02x}{180:02x}{intensity:02x}"  # yeşil
            canvas.itemconfig(p["bars"][i], fill=color)
            canvas.itemconfig(p["val_texts"][i], text=f"{val:.3f}")
