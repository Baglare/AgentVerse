"""Read-only policy playback through each skill's environment."""

import numpy as np

from .storage import AgentStore
from .skills import skill_spec


class ReplaySession:
    def __init__(self, seed, q=None, skill="navigation"):
        if seed < 0:
            raise ValueError("Seed must be nonnegative")
        if q is not None:
            AgentStore.validate_q(q, skill)
        self.seed = seed
        self.q = None if q is None else q.copy()
        self.skill = skill
        self.env = skill_spec(skill)[0]()
        self.reset()

    def reset(self):
        # Match the first episode of M1 evaluation, including random actions.
        starts = np.random.default_rng(np.random.SeedSequence([self.seed, 1]))
        self.rng = np.random.default_rng(np.random.SeedSequence([self.seed, 2]))
        start = int(starts.choice(self.env.starts))
        self.state, _ = self.env.reset(seed=self.seed, options={"start": start})
        self.total_reward = 0.0
        self.last_action = None
        self.collision = False
        self.terminated = False
        self.truncated = False

    def step(self):
        if self.env.done:
            return False
        self.last_action = (int(self.rng.integers(self.env.action_space.n))
                            if self.q is None else int(self.q[self.state].argmax()))
        self.state, reward, self.terminated, self.truncated, info = self.env.step(self.last_action)
        self.total_reward += reward
        self.collision = info["collision"]
        return True


class ReplayViewer:
    def __init__(self, session, label, interval_ms=300):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.widgets import Button

        if interval_ms <= 0:
            raise ValueError("Interval must be positive")
        self.session = session
        self.label = label
        self.playing = False
        self.figure, self.axes = plt.subplots(figsize=(7, 7))
        self.figure.subplots_adjust(bottom=0.22, top=0.82)
        world_map = skill_spec(session.skill)[1]["map"]
        floor = np.array([[cell != "#" for cell in row] for row in world_map], dtype=int)
        self.axes.imshow(floor, cmap=ListedColormap(["#343a40", "#f1f3f5"]), vmin=0, vmax=1)
        self.axes.set_xticks(np.arange(-0.5, len(world_map[0]), 1), minor=True)
        self.axes.set_yticks(np.arange(-0.5, len(world_map), 1), minor=True)
        self.axes.grid(which="minor", color="white", linewidth=2)
        self.axes.tick_params(which="both", bottom=False, left=False,
                              labelbottom=False, labelleft=False)
        r, c = session.env.cells[session.env.goal if session.skill == "navigation" else session.env.base]
        self.axes.scatter([c], [r], marker="*", s=650, color="#2f9e44", label="Hedef" if session.skill == "navigation" else "Us", zorder=3)
        self.resource_markers = []
        if session.skill == "resource_collection":
            for i, position in enumerate(session.env.resources):
                r, c = session.env.cells[position]
                self.resource_markers.append(self.axes.scatter(
                    [c], [r], marker="D", s=180, color="#f08c00", label=f"Kaynak {i + 1}", zorder=3))
        self.marker = self.axes.scatter([], [], s=230, color="#1971c2", label=label, zorder=4)
        self.axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
        self.next_button = Button(self.figure.add_axes((0.10, 0.05, 0.23, 0.06)), "Adim")
        self.play_button = Button(self.figure.add_axes((0.38, 0.05, 0.23, 0.06)), "Baslat")
        self.reset_button = Button(self.figure.add_axes((0.66, 0.05, 0.23, 0.06)), "Sifirla")
        self.next_button.on_clicked(self.advance)
        self.play_button.on_clicked(self.toggle)
        self.reset_button.on_clicked(self.reset)
        self.timer = self.figure.canvas.new_timer(interval=interval_ms)
        self.timer.add_callback(self.tick)
        self.figure.canvas.mpl_connect("close_event", self.close)
        self.draw()

    def draw(self):
        session = self.session
        position = session.state if session.skill == "navigation" else session.env.position
        r, c = session.env.cells[position]
        inventory = ""
        if session.skill == "resource_collection":
            inventory = f" | Kaynaklar: {session.env.collected.bit_count()}/2"
            for i, marker in enumerate(self.resource_markers):
                marker.set_visible(not bool(session.env.collected & (1 << i)))
        self.marker.set_offsets([[c, r]])
        status = ("Basarili" if session.terminated else "Adim siniri" if session.truncated
                  else "Calisiyor" if self.playing else "Duraklatildi")
        action = "-" if session.last_action is None else ("Yukari", "Sag", "Asagi", "Sol")[session.last_action]
        self.axes.set_title(f"{self.label} | seed={session.seed}\n"
                            f"Adim: {session.env.steps}/100 | Odul: {session.total_reward:.2f}\n"
                            f"{status}{inventory} | Hareket: {action} | Duvar: {session.collision}")
        self.figure.canvas.draw_idle()

    def pause(self):
        self.playing = False
        self.timer.stop()
        self.play_button.label.set_text("Baslat")

    def advance(self, event=None):
        self.pause()
        self.session.step()
        self.draw()

    def toggle(self, event=None):
        if self.playing:
            self.pause()
        elif not self.session.env.done:
            self.playing = True
            self.play_button.label.set_text("Duraklat")
            self.timer.start()
        self.draw()

    def tick(self):
        if self.playing:
            self.session.step()
            if self.session.env.done:
                self.pause()
            self.draw()

    def reset(self, event=None):
        self.pause()
        self.session.reset()
        self.draw()

    def close(self, event=None):
        self.pause()
        self.session.env.close()

    def show(self):
        import matplotlib.pyplot as plt
        try:
            plt.show(block=True)
        finally:
            self.close()
            plt.close(self.figure)


def replay(root, name, mode, seed, interval_ms, skill="navigation", version=None):
    import matplotlib
    from matplotlib.backends.registry import backend_registry

    _, framework = backend_registry.resolve_backend(matplotlib.get_backend())
    if framework is None:
        raise ValueError("Replay requires an interactive Matplotlib backend (e.g. MPLBACKEND=TkAgg)")
    if mode == "random" and version is not None:
        raise ValueError("Random replay does not use a saved version")
    q = None
    label = "Rastgele ajan"
    if mode == "trained":
        store = AgentStore(root, name)
        q, metadata, _ = store.load(skill, version)
        version = metadata["version"]
        label = f"{store.identity['name']} ({version[:8]})"
    session = ReplaySession(seed, q, skill)
    viewer = ReplayViewer(session, label, interval_ms)
    viewer.show()
    return {"skill_id": skill, "success": session.terminated, "mode": mode, "seed": seed, "version": version,
            "steps": session.env.steps, "reward": session.total_reward}
