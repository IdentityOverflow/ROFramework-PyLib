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
from typing import List, Tuple

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
REGISTER_PORT    = 5556   # game REP socket — brains REQ here to get a slot
MONITOR_PORT     = 5555   # game PUB socket — monitors SUB here (read-only, no body)
_REG_RESP_FMT    = "<BHH"  # slot_id (uint8), obs_port (uint16), act_port (uint16)

# ── Wire format ───────────────────────────────────────────────────────────────
_OBS_HEADER_FMT  = "<iB"                      # step(int32), done(uint8)
_OBS_HEADER_SIZE = struct.calcsize(_OBS_HEADER_FMT)   # 5 bytes
# Monitor broadcast: n_agents(1) | step(4) | per-agent: reward(4) | obs(OBS_SIZE*4)
_MON_HEADER_FMT  = "<Bi"   # n_agents (uint8), step (int32)
_ACT_FMT         = "<fff"                     # fwd, turn, eat  (float32 × 3)
_ACT_SIZE        = struct.calcsize(_ACT_FMT)  # 12 bytes

_DEFAULT_ACTION = (0.0, 0.0, 0.0)


def _pack_obs(world: World, entity) -> bytes:
    """Serialise entity state into a wire packet."""
    obs    = world.get_observation(entity)
    reward = float(entity.meters.valence)
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
            self._obs_sock.send(_pack_obs(world, world.ai), zmq.NOBLOCK)
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


# ── Multi-agent game connector ────────────────────────────────────────────────

class MultiGameConnector:
    """
    Multi-agent variant of GameConnector.

    Brains register on REGISTER_PORT (5556) via REQ/REP; the game responds
    with a slot ID and dedicated port pair.  Slot N gets:
        obs_port = DEFAULT_OBS_PORT + 2 * N   (PUSH → PULL)
        act_port = DEFAULT_ACT_PORT + 2 * N   (PULL ← PUSH)

    Typical use in the game loop:
        conn = MultiGameConnector()
        conn.start()
        while running:
            new_slots = conn.poll_registrations()
            for slot in new_slots:
                agent_slot = world.add_agent()
            actions = conn.recv_actions(len(world.agents))
            world.step(ai_actions=actions, ...)
            conn.send_obs_all(world)
        conn.close()
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        register_port: int = REGISTER_PORT,
        monitor_port: int = MONITOR_PORT,
    ) -> None:
        self._host       = host
        self._reg_port   = register_port
        self._mon_port   = monitor_port
        self._ctx: zmq.Context | None     = None
        self._reg_sock: zmq.Socket | None = None
        self._mon_sock: zmq.Socket | None = None
        self._slots: dict = {}       # slot_id → {obs_sock, act_sock, last_action}
        self._next_slot: int = 0
        self._free_slots: List[int] = []   # recycled slot IDs from disconnected brains

    def start(self) -> None:
        """Bind the registration and monitor sockets.  Call once before the game loop."""
        self._ctx = zmq.Context()
        self._reg_sock = self._ctx.socket(zmq.REP)
        self._reg_sock.bind(f"tcp://{self._host}:{self._reg_port}")
        self._mon_sock = self._ctx.socket(zmq.PUB)
        self._mon_sock.setsockopt(zmq.SNDHWM, 2)
        self._mon_sock.bind(f"tcp://{self._host}:{self._mon_port}")
        print(f"[connector] registration port {self._reg_port}  "
              f"monitor port {self._mon_port}  "
              f"(obs base: {DEFAULT_OBS_PORT}, act base: {DEFAULT_ACT_PORT})")

    def poll_registrations(self, current_step: int = 0) -> List[int]:
        """
        Non-blocking check for new brain registrations.
        Creates a socket pair per new brain and returns list of new slot IDs.
        """
        if self._reg_sock is None:
            return []
        new_slots: List[int] = []
        while True:
            try:
                self._reg_sock.recv(zmq.NOBLOCK)   # message content ignored
            except zmq.Again:
                break
            # Reuse the lowest freed slot ID so world.agents index stays aligned
            if self._free_slots:
                self._free_slots.sort()
                slot_id = self._free_slots.pop(0)
            else:
                slot_id = self._next_slot
                self._next_slot += 1
            obs_port = DEFAULT_OBS_PORT + 2 * slot_id
            act_port = DEFAULT_ACT_PORT + 2 * slot_id
            resp = struct.pack(_REG_RESP_FMT, slot_id, obs_port, act_port)
            self._reg_sock.send(resp)

            obs_sock = self._ctx.socket(zmq.PUSH)
            obs_sock.setsockopt(zmq.SNDHWM, 2)
            obs_sock.bind(f"tcp://{self._host}:{obs_port}")

            act_sock = self._ctx.socket(zmq.PULL)
            act_sock.setsockopt(zmq.RCVHWM, 2)
            act_sock.bind(f"tcp://{self._host}:{act_port}")

            self._slots[slot_id] = {
                "obs_sock":      obs_sock,
                "act_sock":      act_sock,
                "last_action":   _DEFAULT_ACTION,
                "last_recv_step": current_step,   # grace period from registration
            }
            new_slots.append(slot_id)
            print(f"[connector] brain registered: slot {slot_id} "
                  f"(obs:{obs_port}  act:{act_port})")
        return new_slots

    def recv_actions(self, n_agents: int, current_step: int = 0) -> List[Tuple[float, float, float]]:
        """
        Poll actions for all registered slots.
        Returns a list of length n_agents indexed by slot ID.
        """
        actions: List[Tuple[float, float, float]] = [_DEFAULT_ACTION] * n_agents
        for slot_id, slot in self._slots.items():
            if slot_id >= n_agents:
                continue
            try:
                data = slot["act_sock"].recv(zmq.NOBLOCK)
                slot["last_action"]    = _unpack_action(data)
                slot["last_recv_step"] = current_step
            except zmq.Again:
                pass
            actions[slot_id] = slot["last_action"]
        return actions

    def disconnected_slots(self, current_step: int, timeout: int = 180) -> List[int]:
        """
        Return slot IDs that haven't sent any action within ``timeout`` steps.
        Use to detect crashed or disconnected brains.
        """
        return [
            slot_id for slot_id, slot in self._slots.items()
            if current_step - slot.get("last_recv_step", current_step) > timeout
        ]

    def remove_slot(self, slot_id: int) -> None:
        """Close and remove sockets for a disconnected brain; recycle the slot ID."""
        slot = self._slots.pop(slot_id, None)
        if slot:
            slot["obs_sock"].close()
            slot["act_sock"].close()
            self._free_slots.append(slot_id)
            print(f"[connector] slot {slot_id} removed (disconnected)")

    def send_obs(self, world: "World", slot_id: int) -> None:
        """Send observation for one slot."""
        slot = self._slots.get(slot_id)
        if slot is None:
            return
        if slot_id >= len(world.agents) or world.agents[slot_id] is None:
            return
        entity = world.agents[slot_id]
        try:
            slot["obs_sock"].send(_pack_obs(world, entity), zmq.NOBLOCK)
        except zmq.Again:
            pass

    def send_obs_all(self, world: "World") -> None:
        """Send per-slot observations to each brain and broadcast to monitors."""
        for slot_id in self._slots:
            self.send_obs(world, slot_id)
        self._broadcast_monitor(world)

    def _broadcast_monitor(self, world: "World") -> None:
        """Publish all agents' obs on the monitor PUB socket."""
        if self._mon_sock is None or not world.agents:
            return
        active = [a for a in world.agents if a is not None]
        n = len(active)
        data = struct.pack(_MON_HEADER_FMT, n, world.step_count)
        for agent in active:
            reward = float(agent.meters.valence)
            obs    = world.get_observation(agent)
            data  += struct.pack("<f", reward) + obs.tobytes()
        try:
            self._mon_sock.send(data, zmq.NOBLOCK)
        except zmq.Again:
            pass

    @property
    def n_slots(self) -> int:
        """Number of brains that have registered."""
        return len(self._slots)

    def close(self) -> None:
        if self._reg_sock:
            self._reg_sock.close()
        if self._mon_sock:
            self._mon_sock.close()
        for slot in self._slots.values():
            slot["obs_sock"].close()
            slot["act_sock"].close()
        if self._ctx:
            self._ctx.term()
        self._slots.clear()

    def __enter__(self) -> "MultiGameConnector":
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
        self._slot_id: int = 0

    def connect(
        self,
        host: str | None = None,
        register_port: int = REGISTER_PORT,
    ) -> None:
        """
        Connect to the running game.

        Attempts registration on ``register_port`` first (500 ms timeout).
        On success, uses the game-assigned slot and port pair.
        On timeout (old game without multi-agent support), falls back to the
        default ports supplied at construction time.
        """
        host = host or self._host
        self._ctx = zmq.Context()

        obs_port = self._obs_port
        act_port = self._act_port

        reg_sock = self._ctx.socket(zmq.REQ)
        reg_sock.setsockopt(zmq.RCVTIMEO, 500)
        reg_sock.connect(f"tcp://{host}:{register_port}")
        try:
            reg_sock.send(b"R")
            data = reg_sock.recv()
            self._slot_id, obs_port, act_port = struct.unpack(_REG_RESP_FMT, data)
            print(f"[brain] registered as slot {self._slot_id} "
                  f"(obs:{obs_port}  act:{act_port})")
        except zmq.Again:
            self._slot_id = 0
            print(f"[brain] registration timeout — connecting on default ports "
                  f"(obs:{obs_port}  act:{act_port})")
        finally:
            reg_sock.close()

        self._obs_sock = self._ctx.socket(zmq.PULL)
        self._obs_sock.setsockopt(zmq.RCVHWM, 2)
        self._obs_sock.connect(f"tcp://{host}:{obs_port}")

        self._act_sock = self._ctx.socket(zmq.PUSH)
        self._act_sock.setsockopt(zmq.SNDHWM, 2)
        self._act_sock.connect(f"tcp://{host}:{act_port}")

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

    @property
    def slot_id(self) -> int:
        """Slot index assigned by the game during registration."""
        return self._slot_id


# ── Monitor connector (read-only, no body spawned) ────────────────────────────

class MonitorConnector:
    """
    Subscribes to the game's monitor broadcast (PUB socket on MONITOR_PORT).
    Does NOT register as a brain — no agent body is spawned.

    Usage:
        mon = MonitorConnector()
        mon.connect()
        while True:
            step, agents = mon.recv()   # agents: [{reward, obs}, ...]
            for i, a in enumerate(agents):
                print(i, a["reward"], a["obs"][260:263])
        mon.close()
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = MONITOR_PORT,
    ) -> None:
        self._host = host
        self._port = port
        self._ctx:  zmq.Context | None = None
        self._sock: zmq.Socket  | None = None

    def connect(self) -> None:
        self._ctx  = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVHWM, 2)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sock.connect(f"tcp://{self._host}:{self._port}")

    def recv(self, timeout_ms: int = 5000) -> Tuple[int, List[dict]]:
        """
        Receive the next broadcast frame.

        Returns (step, agents) where agents is a list of dicts
        [{reward: float, obs: np.ndarray(OBS_SIZE,)}], one per agent.
        Raises zmq.Again on timeout.
        """
        if not self._sock.poll(timeout_ms):
            raise zmq.Again("No monitor frame within timeout")
        data = self._sock.recv()
        n_agents, step = struct.unpack_from(_MON_HEADER_FMT, data, 0)
        off    = struct.calcsize(_MON_HEADER_FMT)
        agents = []
        for _ in range(n_agents):
            reward = struct.unpack_from("<f", data, off)[0]
            off   += 4
            obs    = np.frombuffer(data[off: off + OBS_SIZE * 4], dtype=np.float32).copy()
            off   += OBS_SIZE * 4
            agents.append({"reward": float(reward), "obs": obs})
        return int(step), agents

    def close(self) -> None:
        if self._sock:
            self._sock.close()
        if self._ctx:
            self._ctx.term()

    def __enter__(self) -> "MonitorConnector":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()
