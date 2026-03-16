import argparse
import ctypes
import sys

import numpy as np
import pygame
import torch
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_BUFFER_BIT,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_LINEAR,
    GL_LINES,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINTS,
    GL_PROGRAM_POINT_SIZE,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_STATIC_DRAW,
    GL_STREAM_DRAW,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLE_STRIP,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_SHADER,
    glActiveTexture,
    glBindBuffer,
    glBindTexture,
    glBlendFunc,
    glBufferData,
    glBufferSubData,
    glClear,
    glClearColor,
    glDisableVertexAttribArray,
    glDrawArrays,
    glEnable,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenTextures,
    glGetAttribLocation,
    glGetUniformLocation,
    glTexImage2D,
    glTexParameteri,
    glTexSubImage2D,
    glUniform1i,
    glUniform2f,
    glUniform4f,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
)
from OpenGL.GL.shaders import compileProgram, compileShader


TENSION = 0.3
DRAW_LINES = False
_REF_GRID = 30       # reference grid size for scaling damping with resolution
_DAMPING_MAX = 0.15  # effective damping at reverb=0 (waves die in ~2 cycles)
_DAMPING_POW = 3     # cubic curve — high reverb end is sensitive, low end is gentle
_REVERB_DEFAULT = 0.5  # reverb=0.5 ≈ old DAMPING=0.02 feel
DAMPING = _DAMPING_MAX * (1.0 - _REVERB_DEFAULT) ** _DAMPING_POW  # ≈ 0.019

_ANCHOR_LAYOUT_CENTERED = "centered"
_ANCHOR_LAYOUT_GOLDEN = "golden"
_SLIDER_REVERB = "Reverb"
_SLIDER_GRID_SIZE = "Grid Size"
_SLIDER_WAVE_RADIUS = "Wave Radius"

wave_radius = 100
grid_size = 60
anchors_enabled = True
anchor_layout = _ANCHOR_LAYOUT_GOLDEN
IS_RUNNING = False

WAVE_AREA_HEIGHT = 1000
CONTROL_AREA_HEIGHT = 100
WIDTH = 1000
HEIGHT = WAVE_AREA_HEIGHT + CONTROL_AREA_HEIGHT

BG_COLOR = (18, 18, 24)
UI_BG_COLOR = (36, 36, 44)
GRID_COLOR = (180, 180, 190, 0.3)

sliders = [
    {
        "name": "Tension",
        "min": 0.05,
        "max": 0.9,
        "value": TENSION,
        "x": 50,
        "y": WAVE_AREA_HEIGHT + 30,
        "width": 200,
        "height": 10,
        "text": f"{TENSION:.3f}",
        "editing": False,
        "textbox_rect": (260, WAVE_AREA_HEIGHT + 20, 60, 30),
    },
    {
        "name": _SLIDER_REVERB,
        "min": 0.0,
        "max": 1.0,
        "value": _REVERB_DEFAULT,
        "x": 350,
        "y": WAVE_AREA_HEIGHT + 30,
        "width": 200,
        "height": 10,
        "text": f"{_REVERB_DEFAULT:.3f}",
        "editing": False,
        "textbox_rect": (560, WAVE_AREA_HEIGHT + 20, 60, 30),
    },
    {
        "name": _SLIDER_GRID_SIZE,
        "min": 10,
        "max": 128,
        "value": float(grid_size),
        "x": 50,
        "y": WAVE_AREA_HEIGHT + 80,
        "width": 200,
        "height": 10,
        "text": str(grid_size),
        "editing": False,
        "textbox_rect": (260, WAVE_AREA_HEIGHT + 70, 60, 30),
    },
    {
        "name": _SLIDER_WAVE_RADIUS,
        "min": 10,
        "max": 300,
        "value": float(wave_radius),
        "x": 350,
        "y": WAVE_AREA_HEIGHT + 80,
        "width": 200,
        "height": 10,
        "text": str(wave_radius),
        "editing": False,
        "textbox_rect": (560, WAVE_AREA_HEIGHT + 70, 60, 30),
    },
]

buttons = [
    {
        "name": "Start",
        "x": 625,
        "y": WAVE_AREA_HEIGHT + 20,
        "width": 80,
        "height": 40,
        "state": IS_RUNNING,
    },
    {
        "name": "Layout",
        "x": 715,
        "y": WAVE_AREA_HEIGHT + 20,
        "width": 80,
        "height": 40,
        "state": True,
    },
    {
        "name": "Anchors",
        "x": 625,
        "y": WAVE_AREA_HEIGHT + 70,
        "width": 80,
        "height": 40,
        "state": anchors_enabled,
    },
    {
        "name": "Sensors",
        "x": 715,
        "y": WAVE_AREA_HEIGHT + 70,
        "width": 80,
        "height": 40,
        "state": True,   # True = sensors connected
    },
]

_fonts = {}
_label_surfaces = {}

# ── Global physics state ────────────────────────────────────────────────────────
_res = None             # WaveReservoir | None
_node_pos = None        # (N, 2) float32 pixel positions for GL
_line_vertices = None
_node_x_t = None        # GPU tensor of pixel x coords (N,)
_node_y_t = None        # GPU tensor of pixel y coords (N,)
_node_type_f = None     # (N,) float32: 0=free, 0.5=input, 1=anchor
_state_upload = None    # (N, 3) temp for GL upload

# ── Brain mode globals ──────────────────────────────────────────────────────────
_brain = None           # WimBrain | None
_brain_state = {}       # {"fwd": 0, "turn": 0, "eat": 0, "reward": 0, "step": 0}
_client = None          # game client socket
_sensors_on = True      # False = obs injection severed; mouse poke only


def point_in_rect(px, py, rx, ry, rw, rh):
    return (rx <= px <= rx + rw) and (ry <= py <= ry + rh)


def _layout_controls():
    sliders[0]["x"] = 50
    sliders[0]["y"] = WAVE_AREA_HEIGHT + 30
    sliders[0]["textbox_rect"] = (260, WAVE_AREA_HEIGHT + 20, 60, 30)

    sliders[1]["x"] = 350
    sliders[1]["y"] = WAVE_AREA_HEIGHT + 30
    sliders[1]["textbox_rect"] = (560, WAVE_AREA_HEIGHT + 20, 60, 30)

    sliders[2]["x"] = 50
    sliders[2]["y"] = WAVE_AREA_HEIGHT + 80
    sliders[2]["textbox_rect"] = (260, WAVE_AREA_HEIGHT + 70, 60, 30)

    sliders[3]["x"] = 350
    sliders[3]["y"] = WAVE_AREA_HEIGHT + 80
    sliders[3]["textbox_rect"] = (560, WAVE_AREA_HEIGHT + 70, 60, 30)

    right_x = max(625, WIDTH - 175)
    buttons[0]["x"] = right_x
    buttons[0]["y"] = WAVE_AREA_HEIGHT + 20
    buttons[1]["x"] = right_x + 90
    buttons[1]["y"] = WAVE_AREA_HEIGHT + 20
    buttons[2]["x"] = right_x
    buttons[2]["y"] = WAVE_AREA_HEIGHT + 70
    buttons[3]["x"] = right_x + 90
    buttons[3]["y"] = WAVE_AREA_HEIGHT + 70


def _configure_window(width, wave_height, controls_height):
    global WIDTH, WAVE_AREA_HEIGHT, CONTROL_AREA_HEIGHT, HEIGHT

    WIDTH = max(800, int(width))
    WAVE_AREA_HEIGHT = max(400, int(wave_height))
    CONTROL_AREA_HEIGHT = max(120, int(controls_height))
    HEIGHT = WAVE_AREA_HEIGHT + CONTROL_AREA_HEIGHT
    _layout_controls()


def _init_fonts():
    global _fonts, _label_surfaces
    _fonts = {
        "sm": pygame.font.SysFont(None, 20),
        "md": pygame.font.SysFont(None, 24),
    }
    _label_surfaces = {
        s["name"]: _fonts["sm"].render(f"{s['name']}:", True, (200, 200, 210))
        for s in sliders
    }



def _build_line_vertices(size, node_pos):
    idx_grid = np.arange(size * size, dtype=np.int32).reshape(size, size)
    offsets = ((1, 0), (0, 1), (1, 1), (1, -1))
    segments = []
    for dx, dy in offsets:
        x0 = slice(0, size - dx if dx >= 0 else size)
        x1 = slice(dx if dx >= 0 else 0, size)
        if dy >= 0:
            y0 = slice(0, size - dy)
            y1 = slice(dy, size)
        else:
            y0 = slice(-dy, size)
            y1 = slice(0, size + dy)
        src = idx_grid[x0, y0].ravel()
        dst = idx_grid[x1, y1].ravel()
        if src.size == 0:
            continue
        pair = np.empty((src.size * 2, 2), dtype=np.float32)
        pair[0::2] = node_pos[src]
        pair[1::2] = node_pos[dst]
        segments.append(pair)
    if not segments:
        return np.zeros((0, 2), dtype=np.float32)
    return np.ascontiguousarray(np.vstack(segments), dtype=np.float32)


def regenerate_grid():
    global _res, _node_pos, _line_vertices
    global _node_x_t, _node_y_t, _node_type_f, _state_upload

    from wim_brain import WaveReservoir

    size = int(grid_size)
    n = size * size

    x_idx, y_idx = np.meshgrid(
        np.arange(size, dtype=np.int32),
        np.arange(size, dtype=np.int32),
        indexing="ij",
    )

    x_spacing = WIDTH / size
    y_spacing = WAVE_AREA_HEIGHT / size
    node_x = (x_idx.astype(np.float32) * x_spacing + x_spacing * 0.5).ravel()
    node_y = (y_idx.astype(np.float32) * y_spacing + y_spacing * 0.5).ravel()
    _node_pos = np.ascontiguousarray(np.column_stack((node_x, node_y)), dtype=np.float32)

    _res = WaveReservoir(
        grid_size=size,
        input_dim=263,
        tension=TENSION,
        damping=DAMPING * (_REF_GRID / size),
        noise_scale=0.0,
        input_scale=1.0,
        anchors=anchors_enabled,
        anchor_layout=anchor_layout,
        seed=42,
    )

    _node_x_t = torch.from_numpy(node_x).to(_res._dev)
    _node_y_t = torch.from_numpy(node_y).to(_res._dev)

    # node_type_f: 0.0=free, 1.0=anchor (input nodes marked 0.5 only in brain mode)
    _node_type_f = np.zeros(n, dtype=np.float32)
    anchor_np = _res._is_anchor.cpu().numpy()
    _node_type_f[anchor_np] = 1.0

    _state_upload = np.zeros((n, 3), dtype=np.float32)
    _line_vertices = _build_line_vertices(size, _node_pos)


def create_wave_at(px, py, is_drag=False):
    if _res is None:
        return
    radius = float(wave_radius)
    scale = np.float32(0.3 if is_drag else 1.0)
    dx = _node_x_t - px
    dy = _node_y_t - py
    dist2 = dx * dx + dy * dy
    mask = _res._free_mask & (dist2 < radius * radius)
    if not mask.any():
        return
    dist = dist2[mask].sqrt()
    impact = (1.0 - dist / radius) * scale * 2.0
    _res._disp[mask] = impact
    _res._vel[mask] = 0.0
    _res._energy[mask] = 1.0


def handle_slider_event(slider, mouse_x):
    mx = max(slider["x"], min(mouse_x, slider["x"] + slider["width"]))
    ratio = (mx - slider["x"]) / slider["width"]
    new_val = slider["min"] + ratio * (slider["max"] - slider["min"])
    if slider["name"] in (_SLIDER_GRID_SIZE, _SLIDER_WAVE_RADIUS):
        slider["text"] = str(int(round(new_val)))
        slider["value"] = float(round(new_val))
    else:
        slider["text"] = f"{new_val:.3f}"
        slider["value"] = new_val


def handle_text_input(slider, event):
    if event.key == pygame.K_RETURN:
        try:
            if slider["name"] == _SLIDER_GRID_SIZE:
                val = max(1, int(slider["text"]))
                slider["value"] = float(val)
                slider["text"] = str(val)
            elif slider["name"] == _SLIDER_WAVE_RADIUS:
                val = int(slider["text"])
                val = max(int(slider["min"]), min(int(slider["max"]), val))
                slider["value"] = float(val)
                slider["text"] = str(val)
            else:
                val = float(slider["text"])
                val = max(slider["min"], min(slider["max"], val))
                slider["value"] = val
                slider["text"] = f"{val:.3f}"
        except ValueError:
            slider["text"] = (
                str(int(round(slider["value"])))
                if slider["name"] in (_SLIDER_GRID_SIZE, _SLIDER_WAVE_RADIUS)
                else f"{slider['value']:.3f}"
            )
        slider["editing"] = False
        return True
    if event.key == pygame.K_ESCAPE:
        slider["editing"] = False
        return True
    if event.key == pygame.K_BACKSPACE:
        slider["text"] = slider["text"][:-1]
        return True
    if event.unicode:
        slider["text"] += event.unicode
        return True
    return False


def apply_slider_values(renderer):
    global TENSION, DAMPING, wave_radius, grid_size

    old_size = grid_size
    for slider in sliders:
        name = slider["name"]
        if name == "Tension":
            TENSION = float(slider["value"])
        elif name == _SLIDER_REVERB:
            DAMPING = _DAMPING_MAX * (1.0 - float(slider["value"])) ** _DAMPING_POW
        elif name == _SLIDER_WAVE_RADIUS:
            wave_radius = int(round(slider["value"]))
        elif name == _SLIDER_GRID_SIZE:
            grid_size = int(round(slider["value"]))

    if _res is not None:
        _res._tension = TENSION
        _res._damping = DAMPING * (_REF_GRID / grid_size)

    if grid_size != old_size:
        regenerate_grid()
        renderer.set_grid(_node_pos, _line_vertices)


def update_simulation():
    if _res is not None:
        steps = max(1, round(grid_size / _REF_GRID))
        for _ in range(steps):
            _res.step_wave()


def draw_slider(surface, slider):
    line_y = slider["y"] + slider["height"] // 2
    pygame.draw.line(
        surface,
        (80, 80, 90),
        (slider["x"], line_y),
        (slider["x"] + slider["width"], line_y),
        3,
    )
    slider_value = min(max(slider["value"], slider["min"]), slider["max"])
    ratio = (slider_value - slider["min"]) / (slider["max"] - slider["min"])
    handle_x = slider["x"] + ratio * slider["width"]
    pygame.draw.circle(surface, (220, 80, 70), (int(handle_x), int(line_y)), 8)
    surface.blit(_label_surfaces[slider["name"]], (slider["x"], slider["y"] - 20))
    draw_textbox(surface, slider)


def draw_textbox(surface, slider):
    rect = slider["textbox_rect"]
    bg = (70, 70, 50) if slider["editing"] else (50, 50, 58)
    pygame.draw.rect(surface, bg, rect)
    ts = _fonts["sm"].render(slider["text"], True, (200, 200, 210))
    surface.blit(ts, ts.get_rect(center=(rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)))
    pygame.draw.rect(surface, (80, 80, 90), rect, 2)


def draw_button(surface, button):
    rect = (button["x"], button["y"], button["width"], button["height"])
    if button["name"] == "Start":
        bg = (50, 200, 50) if IS_RUNNING else (200, 50, 50)
        label = "Stop" if IS_RUNNING else "Start"
    elif button["name"] == "Anchors":
        bg = (50, 200, 50) if anchors_enabled else (200, 50, 50)
        label = "Anchors On" if anchors_enabled else "Anchors Off"
    elif button["name"] == "Sensors":
        bg = (50, 200, 50) if _sensors_on else (200, 130, 50)
        label = "Sensors" if _sensors_on else "Severed"
    else:
        bg = (70, 120, 220) if anchor_layout == _ANCHOR_LAYOUT_CENTERED else (210, 150, 55)
        label = "Center" if anchor_layout == _ANCHOR_LAYOUT_CENTERED else "Golden"
    pygame.draw.rect(surface, bg, rect, border_radius=5)
    ts = _fonts["md"].render(label, True, (255, 255, 255))
    surface.blit(ts, ts.get_rect(center=(button["x"] + button["width"] // 2, button["y"] + button["height"] // 2)))


def redraw_ui_surface(surface, brain_state=None):
    surface.fill((0, 0, 0, 0))
    pygame.draw.rect(surface, (*UI_BG_COLOR, 255), (0, WAVE_AREA_HEIGHT, WIDTH, CONTROL_AREA_HEIGHT))
    for slider in sliders:
        draw_slider(surface, slider)
    buttons[0]["state"] = IS_RUNNING
    buttons[1]["state"] = True
    buttons[2]["state"] = anchors_enabled
    buttons[3]["state"] = _sensors_on
    for button in buttons:
        draw_button(surface, button)

    if brain_state:
        _draw_brain_overlay(surface, brain_state)


def _draw_brain_overlay(surface, brain_state):
    """Draw fwd/turn/eat/reward/step on a single line at the bottom of the control area."""
    font = _fonts["sm"]
    fwd    = float(brain_state.get("fwd", 0.0))
    turn   = float(brain_state.get("turn", 0.0))
    eat    = float(brain_state.get("eat", 0.0))
    reward = float(brain_state.get("reward", 0.0))
    step   = int(brain_state.get("step", 0))

    y  = WAVE_AREA_HEIGHT + CONTROL_AREA_HEIGHT - 20
    bar_w = 60
    bar_h = 8
    bar_y = y + 5
    x = 10

    def _inline_bar(label, val):
        nonlocal x
        lbl = font.render(f"{label}:", True, (180, 180, 190))
        surface.blit(lbl, (x, y))
        x += lbl.get_width() + 3
        pygame.draw.rect(surface, (60, 60, 70), (x, bar_y, bar_w, bar_h))
        cx = x + bar_w // 2
        fill = int(val * bar_w / 2)
        if fill >= 0:
            pygame.draw.rect(surface, (60, 180, 60), (cx, bar_y, max(fill, 1), bar_h))
        else:
            pygame.draw.rect(surface, (220, 90, 60), (cx + fill, bar_y, -fill, bar_h))
        x += bar_w + 3
        val_s = font.render(f"{val:+.2f}", True, (200, 200, 210))
        surface.blit(val_s, (x, y))
        x += val_s.get_width() + 14

    _inline_bar("fwd",  fwd)
    _inline_bar("turn", turn)

    eat_color = (60, 210, 60) if eat > 0.5 else (120, 120, 130)
    eat_s = font.render("eat: ON" if eat > 0.5 else "eat: off", True, eat_color)
    surface.blit(eat_s, (x, y))
    x += eat_s.get_width() + 14

    rew_s = font.render(f"reward: {reward:+.3f}", True, (120, 140, 220))
    surface.blit(rew_s, (x, y))
    x += rew_s.get_width() + 14

    step_s = font.render(f"step: {step}", True, (140, 140, 150))
    surface.blit(step_s, (x, y))


class GLRenderer:
    def __init__(self):
        self._node_program = self._build_program(_NODE_VERTEX_SHADER, _NODE_FRAGMENT_SHADER)
        self._line_program = self._build_program(_LINE_VERTEX_SHADER, _LINE_FRAGMENT_SHADER)
        self._ui_program = self._build_program(_UI_VERTEX_SHADER, _UI_FRAGMENT_SHADER)

        self._node_pos_loc = glGetAttribLocation(self._node_program, "a_pos")
        self._node_state_loc = glGetAttribLocation(self._node_program, "a_state")
        self._node_view_loc = glGetUniformLocation(self._node_program, "u_view")

        self._line_pos_loc = glGetAttribLocation(self._line_program, "a_pos")
        self._line_view_loc = glGetUniformLocation(self._line_program, "u_view")
        self._line_color_loc = glGetUniformLocation(self._line_program, "u_color")

        self._ui_vert_loc = glGetAttribLocation(self._ui_program, "a_vert")
        self._ui_screen_loc = glGetUniformLocation(self._ui_program, "u_screen")
        self._ui_tex_loc = glGetUniformLocation(self._ui_program, "u_tex")

        self._pos_vbo = glGenBuffers(1)
        self._state_vbo = glGenBuffers(1)
        self._line_vbo = glGenBuffers(1)
        self._ui_vbo = glGenBuffers(1)
        self._ui_tex = glGenTextures(1)

        self._node_count = 0
        self._line_count = 0
        self._ui_bytes = None

        glViewport(0, 0, WIDTH, HEIGHT)
        glClearColor(*(c / 255.0 for c in BG_COLOR), 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_PROGRAM_POINT_SIZE)

        ui_quad = np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [WIDTH, 0.0, 1.0, 1.0],
                [0.0, HEIGHT, 0.0, 0.0],
                [WIDTH, HEIGHT, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        glBindBuffer(GL_ARRAY_BUFFER, self._ui_vbo)
        glBufferData(GL_ARRAY_BUFFER, ui_quad.nbytes, ui_quad, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        blank = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
        glBindTexture(GL_TEXTURE_2D, self._ui_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            WIDTH,
            HEIGHT,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            blank,
        )
        glBindTexture(GL_TEXTURE_2D, 0)

    @staticmethod
    def _build_program(vertex_src, fragment_src):
        return compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER),
        )

    def set_grid(self, node_pos, line_vertices):
        self._node_count = int(node_pos.shape[0])
        self._line_count = int(line_vertices.shape[0])

        glBindBuffer(GL_ARRAY_BUFFER, self._pos_vbo)
        glBufferData(GL_ARRAY_BUFFER, node_pos.nbytes, np.ascontiguousarray(node_pos, dtype=np.float32), GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self._state_vbo)
        glBufferData(GL_ARRAY_BUFFER, self._node_count * 3 * 4, None, GL_STREAM_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self._line_vbo)
        glBufferData(
            GL_ARRAY_BUFFER,
            line_vertices.nbytes,
            np.ascontiguousarray(line_vertices, dtype=np.float32),
            GL_STATIC_DRAW,
        )
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def update_ui_texture(self, surface):
        ui_bytes = pygame.image.tostring(surface, "RGBA", True)
        if ui_bytes == self._ui_bytes:
            return
        self._ui_bytes = ui_bytes
        glBindTexture(GL_TEXTURE_2D, self._ui_tex)
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            WIDTH,
            HEIGHT,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            ui_bytes,
        )
        glBindTexture(GL_TEXTURE_2D, 0)

    def render(self, state_upload, draw_lines):
        glClear(GL_COLOR_BUFFER_BIT)
        # Wave area only — GL y=0 is bottom, so offset by CONTROL_AREA_HEIGHT
        glViewport(0, CONTROL_AREA_HEIGHT, WIDTH, WAVE_AREA_HEIGHT)
        self._render_nodes(state_upload)
        if draw_lines and self._line_count:
            self._render_lines()
        # Full window for UI overlay
        glViewport(0, 0, WIDTH, HEIGHT)
        self._render_ui()

    def _render_nodes(self, state_upload):
        glBindBuffer(GL_ARRAY_BUFFER, self._state_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, state_upload.nbytes, state_upload)

        glUseProgram(self._node_program)
        glUniform2f(self._node_view_loc, float(WIDTH), float(WAVE_AREA_HEIGHT))

        glBindBuffer(GL_ARRAY_BUFFER, self._pos_vbo)
        glEnableVertexAttribArray(self._node_pos_loc)
        glVertexAttribPointer(self._node_pos_loc, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        glBindBuffer(GL_ARRAY_BUFFER, self._state_vbo)
        glEnableVertexAttribArray(self._node_state_loc)
        glVertexAttribPointer(self._node_state_loc, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        glDrawArrays(GL_POINTS, 0, self._node_count)

        glDisableVertexAttribArray(self._node_pos_loc)
        glDisableVertexAttribArray(self._node_state_loc)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glUseProgram(0)

    def _render_lines(self):
        glUseProgram(self._line_program)
        glUniform2f(self._line_view_loc, float(WIDTH), float(WAVE_AREA_HEIGHT))
        glUniform4f(self._line_color_loc, *GRID_COLOR)

        glBindBuffer(GL_ARRAY_BUFFER, self._line_vbo)
        glEnableVertexAttribArray(self._line_pos_loc)
        glVertexAttribPointer(self._line_pos_loc, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glDrawArrays(GL_LINES, 0, self._line_count)

        glDisableVertexAttribArray(self._line_pos_loc)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glUseProgram(0)

    def _render_ui(self):
        glUseProgram(self._ui_program)
        glUniform2f(self._ui_screen_loc, float(WIDTH), float(HEIGHT))
        glUniform1i(self._ui_tex_loc, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._ui_tex)

        glBindBuffer(GL_ARRAY_BUFFER, self._ui_vbo)
        glEnableVertexAttribArray(self._ui_vert_loc)
        glVertexAttribPointer(self._ui_vert_loc, 4, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        glDisableVertexAttribArray(self._ui_vert_loc)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)


_NODE_VERTEX_SHADER = """
#version 120
attribute vec2 a_pos;
attribute vec3 a_state;
uniform vec2 u_view;
varying vec4 v_color;

vec3 hls_to_rgb(float hue_deg, float light, float sat) {
    float h = hue_deg / 60.0;
    float c = (1.0 - abs(2.0 * light - 1.0)) * sat;
    float x = c * (1.0 - abs(mod(h, 2.0) - 1.0));
    vec3 rgb;
    if (h < 1.0) {
        rgb = vec3(c, x, 0.0);
    } else if (h < 2.0) {
        rgb = vec3(x, c, 0.0);
    } else if (h < 3.0) {
        rgb = vec3(0.0, c, x);
    } else if (h < 4.0) {
        rgb = vec3(0.0, x, c);
    } else if (h < 5.0) {
        rgb = vec3(x, 0.0, c);
    } else {
        rgb = vec3(c, 0.0, x);
    }
    float m = light - 0.5 * c;
    return clamp(rgb + vec3(m), 0.0, 1.0);
}

void main() {
    float disp = a_state.x;
    float energy = clamp(a_state.y, 0.0, 1.0);
    float ntype = a_state.z;

    if (ntype > 0.92) {
        // Anchor node — golden on dark bg
        float l = 0.45 + energy * 0.25;
        v_color = vec4(hls_to_rgb(42.0, l, 0.7), 0.95);
    } else if (ntype > 0.82) {
        // Barrier node (corpus callosum) — dim neutral
        v_color = vec4(0.25, 0.25, 0.30, 0.7);
    } else if (ntype > 0.25) {
        // Input node (brain mode) — cyan
        float l = 0.4 + energy * 0.3;
        v_color = vec4(hls_to_rgb(185.0, l, 0.75), 0.9);
    } else {
        // Free node: white at rest, blue peak, red trough
        float pos = clamp( disp, 0.0, 1.0);
        float neg = clamp(-disp, 0.0, 1.0);
        float strength = clamp(pos + neg, 0.0, 1.0);
        // White base, blend toward blue (peak) or red (trough)
        float r = 1.0 - pos * 0.95;          // red drops on positive disp
        float g = 1.0 - strength * 0.85;     // green drops on any displacement
        float b = 1.0 - neg * 0.95;          // blue drops on negative disp
        // Dim slightly when no energy (idle nodes fade to soft grey)
        float e = 0.8 + clamp(energy, 0.0, 1.0) * 0.2;
        v_color = vec4(r * e, g * e, b * e, 0.95);
    }

    float base_r = 4.0;
    if (ntype > 0.92) base_r = 6.0;       // anchor
    else if (ntype > 0.82) base_r = 3.0;  // barrier (small, subtle)
    else if (ntype > 0.25) base_r = 5.0;  // input
    gl_PointSize = base_r * 2.0 + abs(disp) * 1.5;

    float ndc_x = (a_pos.x / u_view.x) * 2.0 - 1.0;
    float ndc_y = 1.0 - (a_pos.y / u_view.y) * 2.0;
    gl_Position = vec4(ndc_x, ndc_y, 0.0, 1.0);
}
"""

_NODE_FRAGMENT_SHADER = """
#version 120
varying vec4 v_color;

void main() {
    gl_FragColor = v_color;
}
"""

_LINE_VERTEX_SHADER = """
#version 120
attribute vec2 a_pos;
uniform vec2 u_view;

void main() {
    float ndc_x = (a_pos.x / u_view.x) * 2.0 - 1.0;
    float ndc_y = 1.0 - (a_pos.y / u_view.y) * 2.0;
    gl_Position = vec4(ndc_x, ndc_y, 0.0, 1.0);
}
"""

_LINE_FRAGMENT_SHADER = """
#version 120
uniform vec4 u_color;

void main() {
    gl_FragColor = u_color;
}
"""

_UI_VERTEX_SHADER = """
#version 120
attribute vec4 a_vert;
uniform vec2 u_screen;
varying vec2 v_tex;

void main() {
    float ndc_x = (a_vert.x / u_screen.x) * 2.0 - 1.0;
    float ndc_y = 1.0 - (a_vert.y / u_screen.y) * 2.0;
    gl_Position = vec4(ndc_x, ndc_y, 0.0, 1.0);
    v_tex = a_vert.zw;
}
"""

_UI_FRAGMENT_SHADER = """
#version 120
uniform sampler2D u_tex;
varying vec2 v_tex;

void main() {
    gl_FragColor = texture2D(u_tex, v_tex);
}
"""


def _render_frame(renderer):
    """Upload reservoir state to GPU and render."""
    disp_np   = _res._disp.cpu().numpy()
    energy_np = _res._energy.cpu().numpy()
    _state_upload[:, 0] = disp_np
    _state_upload[:, 1] = energy_np
    _state_upload[:, 2] = _node_type_f
    renderer.render(_state_upload, DRAW_LINES)


def _set_caption(clock):
    fps = clock.get_fps()
    mode = " [BRAIN]" if _brain is not None else ""
    if fps <= 0:
        pygame.display.set_caption(f"Wave Reservoir GPU{mode}")
    else:
        pygame.display.set_caption(f"Wave Reservoir GPU{mode}  {fps:5.1f} FPS")


def _parse_args():
    parser = argparse.ArgumentParser(description="GPU wave reservoir visualizer.")
    parser.add_argument("--width", type=int, default=WIDTH,
                        help="Window width in pixels (min 800)")
    parser.add_argument("--wave-height", type=int, default=WAVE_AREA_HEIGHT,
                        help="Wave area height in pixels (min 400)")
    parser.add_argument("--controls-height", type=int, default=CONTROL_AREA_HEIGHT,
                        help="Control bar height in pixels (min 120)")
    parser.add_argument("--brain", action="store_true",
                        help="Enable brain mode (load WimBrain, connect to game)")
    parser.add_argument("--config", metavar="PATH", default=None,
                        help="Brain config JSON (used with --brain)")
    parser.add_argument("--headless", type=int, default=0, metavar="N",
                        help="Brain mode: run N steps then exit (0 = connect to game)")
    return parser.parse_args()


def _setup_brain(args):
    """Load WimBrain, point _res at the brain's reservoir, mark input nodes."""
    global _brain, _res, _node_type_f, _node_pos, _line_vertices
    global _node_x_t, _node_y_t, _state_upload

    import json
    from wim_brain import WimBrain, WAVE_DEFAULT_CONFIG

    if args.config:
        with open(args.config) as _f:
            _raw = json.load(_f)
        cfg = {**WAVE_DEFAULT_CONFIG, **{k: v for k, v in _raw.items() if not k.startswith("_")}}
    else:
        cfg = dict(WAVE_DEFAULT_CONFIG)
    dev_str = cfg.get("device", "cuda")
    _brain = WimBrain(config=cfg, device=dev_str)
    _res = _brain._reservoir

    # Rebuild node positions to match the brain's grid size
    size = _res.grid_size
    n = size * size

    x_idx, y_idx = np.meshgrid(
        np.arange(size, dtype=np.int32),
        np.arange(size, dtype=np.int32),
        indexing="ij",
    )
    x_spacing = WIDTH / size
    y_spacing = WAVE_AREA_HEIGHT / size
    node_x = (x_idx.astype(np.float32) * x_spacing + x_spacing * 0.5).ravel()
    node_y = (y_idx.astype(np.float32) * y_spacing + y_spacing * 0.5).ravel()
    _node_pos = np.ascontiguousarray(np.column_stack((node_x, node_y)), dtype=np.float32)

    dev = _res._dev
    _node_x_t = torch.from_numpy(node_x).to(dev)
    _node_y_t = torch.from_numpy(node_y).to(dev)

    _node_type_f = np.zeros(n, dtype=np.float32)
    anchor_np = _res._is_anchor.cpu().numpy()
    _node_type_f[anchor_np] = 1.0
    # Mark barrier nodes (corpus callosum) distinctly
    barrier_np = _res._is_barrier.cpu().numpy()
    _node_type_f[barrier_np] = 0.9
    # Mark input nodes as 0.5 (cyan in brain mode)
    input_np = _res._input_nodes.cpu().numpy()
    _node_type_f[input_np] = 0.5

    _state_upload = np.zeros((n, 3), dtype=np.float32)
    _line_vertices = _build_line_vertices(size, _node_pos)

    print(f"[brain] WimBrain loaded  grid={size}x{size}  device={dev}  "
          f"input_nodes={len(input_np)}")


def _brain_tick():
    """One game step: recv obs, forward, learn, send action."""
    global _brain_state
    if _client is None:
        return
    obs, reward, done, _ = _client.recv_obs()
    if _sensors_on:
        fwd, turn, eat = _brain.forward(obs)
        _brain.learn(reward)
    else:
        # Sensors severed: step wave physics only, compute action from current state
        _res.step_wave()
        with torch.no_grad():
            h = torch.tanh(_res._disp)
            raw = _brain.W_out @ h + _brain.b_out
            fwd  = float(torch.tanh(raw[0]))
            turn = float(torch.tanh(raw[1]))
            eat  = 1.0 if float(raw[2]) > _brain._eat_threshold else 0.0
        reward = 0.0
    _client.send_action((fwd, turn, eat))
    if done:
        _brain.reset_state()
    _brain_state = {
        "fwd":    fwd,
        "turn":   turn,
        "eat":    eat,
        "reward": reward,
        "step":   _brain_state.get("step", 0) + 1,
    }


def main():
    global IS_RUNNING, anchors_enabled, anchor_layout, DRAW_LINES
    global grid_size, _client, _sensors_on

    args = _parse_args()
    _configure_window(args.width, args.wave_height, args.controls_height)

    pygame.init()
    _init_fonts()

    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 2)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

    try:
        pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF, vsync=1)
    except TypeError:
        pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)

    clock = pygame.time.Clock()
    ui_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    renderer = GLRenderer()

    if args.brain:
        _setup_brain(args)
        # Connect to game client
        try:
            import sys as _sys
            import os as _os
            _here = _os.path.dirname(_os.path.abspath(__file__))
            if _here not in _sys.path:
                _sys.path.insert(0, _here)
            from connector import AgentConnector
            _client = AgentConnector()
            _client.connect(name=_brain._observer.name)
            print("[brain] Connected to game.")
        except Exception as e:
            print(f"[brain] Could not connect to game: {e}")
    else:
        regenerate_grid()

    renderer.set_grid(_node_pos, _line_vertices)

    if not args.brain:
        create_wave_at(WIDTH // 2, WAVE_AREA_HEIGHT // 2, False)

    redraw_ui_surface(ui_surface, _brain_state if _brain else None)
    renderer.update_ui_texture(ui_surface)

    running = True
    mouse_down = False
    active_slider = None
    active_textbox = None
    ui_dirty = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                mouse_down = True
                active_textbox = None
                ui_dirty = True
                for slider in sliders:
                    slider["editing"] = False

                if my < WAVE_AREA_HEIGHT:
                    create_wave_at(mx, my, False)
                else:
                    for slider in sliders:
                        tbx, tby, tbw, tbh = slider["textbox_rect"]
                        slider["editing"] = False
                        if point_in_rect(mx, my, tbx, tby, tbw, tbh):
                            slider["editing"] = True
                            active_textbox = slider
                            break
                        if point_in_rect(mx, my, slider["x"], slider["y"], slider["width"], slider["height"] + 16):
                            active_slider = slider
                            handle_slider_event(slider, mx)
                            apply_slider_values(renderer)
                            break
                    else:
                        for button in buttons:
                            if point_in_rect(mx, my, button["x"], button["y"], button["width"], button["height"]):
                                if button["name"] == "Start":
                                    IS_RUNNING = not IS_RUNNING
                                elif button["name"] == "Layout":
                                    anchor_layout = (
                                        _ANCHOR_LAYOUT_GOLDEN
                                        if anchor_layout == _ANCHOR_LAYOUT_CENTERED
                                        else _ANCHOR_LAYOUT_CENTERED
                                    )
                                    if anchors_enabled and _brain is None:
                                        regenerate_grid()
                                        renderer.set_grid(_node_pos, _line_vertices)
                                elif button["name"] == "Anchors":
                                    anchors_enabled = not anchors_enabled
                                    if _brain is None:
                                        regenerate_grid()
                                        renderer.set_grid(_node_pos, _line_vertices)
                                elif button["name"] == "Sensors":
                                    _sensors_on = not _sensors_on
                                    buttons[3]["state"] = _sensors_on
                                break

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                mouse_down = False
                active_slider = None

            elif event.type == pygame.MOUSEMOTION and mouse_down:
                mx, my = event.pos
                if active_slider is None and my < WAVE_AREA_HEIGHT:
                    create_wave_at(mx, my, True)
                elif active_slider is not None:
                    handle_slider_event(active_slider, mx)
                    apply_slider_values(renderer)
                    ui_dirty = True

            elif event.type == pygame.KEYDOWN:
                if active_textbox is not None and active_textbox["editing"]:
                    if handle_text_input(active_textbox, event):
                        apply_slider_values(renderer)
                        ui_dirty = True
                elif event.key == pygame.K_l:
                    DRAW_LINES = not DRAW_LINES
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if _brain is not None:
            # Brain mode: one game step per frame, paced by game
            _brain_tick()
            ui_dirty = True
        elif IS_RUNNING:
            update_simulation()

        if ui_dirty:
            redraw_ui_surface(ui_surface, _brain_state if _brain else None)
            renderer.update_ui_texture(ui_surface)
            ui_dirty = False
        elif buttons[0]["state"] != IS_RUNNING or buttons[2]["state"] != anchors_enabled:
            redraw_ui_surface(ui_surface, _brain_state if _brain else None)
            renderer.update_ui_texture(ui_surface)

        _render_frame(renderer)
        pygame.display.flip()
        _set_caption(clock)

        if _brain is None:
            clock.tick(240)  # standalone: max fps

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
