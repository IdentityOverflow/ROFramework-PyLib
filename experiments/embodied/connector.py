"""
Real-time connector between the embodied game and an external AI process.

Architecture
------------
Two independent ZeroMQ sockets carry data in opposite directions:

  OBS socket  PUSH (game) → PULL (client)   game → AI
  ACT socket  PULL (game) ← PUSH (client)   AI → game

The game publishes a fresh observation every step (~60 fps) without waiting.
The client pushes actions whenever it is ready; the game polls non-blocking
and uses the last received action if nothing new has arrived.

Both sides are fully decoupled in time — the client may run at any rate.

Quick start
-----------
Game side  (inside game.py or your own render loop):

    conn = GameConnector()
    conn.start()                    # binds ports, ready to accept client

    # inside the step loop:
    ai_action = conn.recv_action()  # non-blocking; returns last action
    world.step(ai_action=ai_action, ...)
    conn.send_obs(world)            # publishes obs + reward + done flag

    conn.close()                    # on exit

Client side  (external process / AI agent):

    client = AgentConnector()
    client.connect()                # connects to the game

    obs, reward, done = client.recv_obs()   # blocks until first frame
    while True:
        action = my_agent.act(obs)          # (fwd, turn, eat) in [-1,1]
        client.send_action(action)
        obs, reward, done = client.recv_obs()

OBS packet layout  (struct, little-endian):
    int32   step_count
    uint8   done  (1 = world just reset)
    float32 reward  (= current valence, ∈ [-1, 1])
    float32 obs[OBS_SIZE]   (all values ∈ [0, 1])

ACT packet layout:
    float32 fwd    ∈ [-1, 1]
    float32 turn   ∈ [-1, 1]
    float32 eat    ∈  {0, 1}
"""

from __future__ import annotations

import struct
import numpy as np
from typing import Tuple

try:
    import zmq
except ImportError:
    raise ImportError("pyzmq is required:  pip install pyzmq")

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from env import World, OBS_SIZE

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_HOST     = "127.0.0.1"
DEFAULT_OBS_PORT = 5557   # game PUSHes obs here
DEFAULT_ACT_PORT = 5558   # game PULLs actions here

# ── Wire format ───────────────────────────────────────────────────────────────
_OBS_HEADER_FMT  = "<iB"                      # step(int32), done(uint8)
_OBS_HEADER_SIZE = struct.calcsize(_OBS_HEADER_FMT)   # 5 bytes
_ACT_FMT         = "<fff"                     # fwd, turn, eat  (float32 × 3)
_ACT_SIZE        = struct.calcsize(_ACT_FMT)  # 12 bytes

_DEFAULT_ACTION = (0.0, 0.0, 0.0)


def _pack_obs(world: World) -> bytes:
    """Serialise current AI state into a wire packet."""
    obs    = world.get_ai_observation()          # float32 (OBS_SIZE,)
    reward = float(world.ai.meters.valence)
    done   = 0                                   # reset is detected by step_count drop
    header = struct.pack(_OBS_HEADER_FMT, world.step_count, done)
    return header + struct.pack("<f", reward) + obs.tobytes()


def _unpack_obs(data: bytes) -> Tuple[np.ndarray, float, bool, int]:
    """Deserialise a wire packet into (obs, reward, done, step)."""
    step, done_flag = struct.unpack_from(_OBS_HEADER_FMT, data, 0)
    reward = struct.unpack_from("<f", data, _OBS_HEADER_SIZE)[0]
    obs_start = _OBS_HEADER_SIZE + 4
    obs = np.frombuffer(data[obs_start:], dtype=np.float32).copy()
    return obs, float(reward), bool(done_flag), int(step)


def _pack_action(fwd: float, turn: float, eat: float) -> bytes:
    return struct.pack(_ACT_FMT, fwd, turn, eat)


def _unpack_action(data: bytes) -> Tuple[float, float, float]:
    return struct.unpack(_ACT_FMT, data)


# ── Game-side connector ────────────────────────────────────────────────────────

class GameConnector:
    """
    Runs inside the game process.  Call start() once, then on every step:
        ai_action = conn.recv_action()
        world.step(ai_action=ai_action, ...)
        conn.send_obs(world)
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        obs_port: int = DEFAULT_OBS_PORT,
        act_port: int = DEFAULT_ACT_PORT,
    ) -> None:
        self._host     = host
        self._obs_port = obs_port
        self._act_port = act_port
        self._ctx: zmq.Context | None     = None
        self._obs_sock: zmq.Socket | None = None
        self._act_sock: zmq.Socket | None = None
        self._last_action: Tuple[float, float, float] = _DEFAULT_ACTION
        self.connected = False

    def start(self) -> None:
        """Bind both sockets.  Call once before the game loop."""
        self._ctx = zmq.Context()

        self._obs_sock = self._ctx.socket(zmq.PUSH)
        self._obs_sock.setsockopt(zmq.SNDHWM, 2)    # drop old frames, never block
        self._obs_sock.bind(f"tcp://{self._host}:{self._obs_port}")

        self._act_sock = self._ctx.socket(zmq.PULL)
        self._act_sock.setsockopt(zmq.RCVHWM, 2)
        self._act_sock.bind(f"tcp://{self._host}:{self._act_port}")

        self.connected = True

    def send_obs(self, world: World) -> None:
        """Push current AI observation to the client (non-blocking, drops if full)."""
        if not self.connected:
            return
        try:
            self._obs_sock.send(_pack_obs(world), zmq.NOBLOCK)
        except zmq.Again:
            pass   # client not consuming fast enough; drop frame

    def recv_action(self) -> Tuple[float, float, float]:
        """
        Poll for a new action from the client.  Non-blocking: returns the
        last received action if nothing new has arrived this frame.
        """
        if not self.connected:
            return _DEFAULT_ACTION
        try:
            data = self._act_sock.recv(zmq.NOBLOCK)
            self._last_action = _unpack_action(data)
        except zmq.Again:
            pass   # no new action; reuse last
        return self._last_action

    def close(self) -> None:
        if self._obs_sock:
            self._obs_sock.close()
        if self._act_sock:
            self._act_sock.close()
        if self._ctx:
            self._ctx.term()
        self.connected = False

    def __enter__(self) -> "GameConnector":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── Client-side connector ─────────────────────────────────────────────────────

class AgentConnector:
    """
    Runs in the external AI process.  Example:

        client = AgentConnector()
        client.connect()
        obs, reward, done = client.recv_obs()
        while True:
            action = agent.act(obs)
            client.send_action(action)
            obs, reward, done = client.recv_obs()
        client.close()
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        obs_port: int = DEFAULT_OBS_PORT,
        act_port: int = DEFAULT_ACT_PORT,
    ) -> None:
        self._host     = host
        self._obs_port = obs_port
        self._act_port = act_port
        self._ctx: zmq.Context | None     = None
        self._obs_sock: zmq.Socket | None = None
        self._act_sock: zmq.Socket | None = None

    def connect(self) -> None:
        """Connect to the running game.  Call once before the agent loop."""
        self._ctx = zmq.Context()

        self._obs_sock = self._ctx.socket(zmq.PULL)
        self._obs_sock.setsockopt(zmq.RCVHWM, 2)
        self._obs_sock.connect(f"tcp://{self._host}:{self._obs_port}")

        self._act_sock = self._ctx.socket(zmq.PUSH)
        self._act_sock.setsockopt(zmq.SNDHWM, 2)
        self._act_sock.connect(f"tcp://{self._host}:{self._act_port}")

    def recv_obs(
        self, timeout_ms: int = 5000
    ) -> Tuple[np.ndarray, float, bool, int]:
        """
        Receive the next observation frame from the game.

        Returns (obs, reward, done, step):
          obs     float32 array of shape (OBS_SIZE,), all values ∈ [0, 1]
          reward  float ∈ [-1, 1] — current valence of the AI entity
          done    bool — True when the world just reset
          step    int  — game step counter since last reset
        Raises zmq.Again if no frame arrives within timeout_ms.
        """
        if not self._obs_sock.poll(timeout_ms):
            raise zmq.Again("No observation received within timeout")
        data = self._obs_sock.recv()
        return _unpack_obs(data)

    def send_action(
        self,
        action: Tuple[float, float, float] | np.ndarray,
    ) -> None:
        """
        Send an action to the game (non-blocking).

        action: (fwd, turn, eat)
          fwd   ∈ [-1, 1]   forward/reverse speed
          turn  ∈ [-1, 1]   right/left turn rate
          eat   ∈  {0, 1}   1 = eat / grasp food
        """
        if isinstance(action, np.ndarray):
            fwd, turn, eat = float(action[0]), float(action[1]), float(action[2])
        else:
            fwd, turn, eat = action
        try:
            self._act_sock.send(_pack_action(fwd, turn, eat), zmq.NOBLOCK)
        except zmq.Again:
            pass

    def close(self) -> None:
        if self._obs_sock:
            self._obs_sock.close()
        if self._act_sock:
            self._act_sock.close()
        if self._ctx:
            self._ctx.term()

    def __enter__(self) -> "AgentConnector":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def obs_size(self) -> int:
        return OBS_SIZE
