"""Vehicular edge computing environment with digital twin, trust and attacks (TG-Diff prototype).

Event-stepped simulation (DT = 50 ms). Each generated task is one decision: the policy receives a
fixed-size observation over 7 candidate slots and returns a slot index.

Slots: 0 local | 1-2 RSUs in coverage (closest first) | 3 cloud (via closest RSU) | 4-6 three closest helpers (V2V)

Threats (configurable):
  * black-hole RSU (RSU-2 drops p_rsu of received tasks)
  * malicious helper vehicles (a fraction mal_frac of helpers drop p_veh of tasks)
  * false state reports: malicious nodes advertise an empty queue
Defenses available to policies:
  * behaviour trust: beta reputation (s+1)/(s+f+2)
  * digital-twin consistency: the RSU-side twin predicts each node's backlog from the tasks it
    dispatched (with synchronisation noise); deviation between report and twin lowers trust
"""
import math
import numpy as np

ROAD, LANES, DT = 1500.0, 4, 0.05
RSU_X, RSU_R = (250.0, 750.0, 1250.0), 300.0
F_RSU, F_VEH, F_HELP, F_CLOUD = 12e9, 1e9, 4e9, 60e9
BACKHAUL, V2V_R, P_TX, KAPPA, BEACON = 0.08, 160.0, 0.5, 1e-27, 0.25
NSLOT, NFEAT = 7, 11
OBS_DIM = NSLOT * NFEAT + 3


def rate(d, bw):
    snr = 1000.0 * (max(d, 10.0) / 50.0) ** -3
    return bw * math.log2(1.0 + snr)


class Node:
    __slots__ = ("id", "kind", "x", "veh", "busy", "mal", "s", "f", "twin")

    def __init__(self, id, kind, x=0.0, veh=None, mal=False):
        self.id, self.kind, self.x, self.veh, self.mal = id, kind, x, veh, mal
        self.busy, self.s, self.f, self.twin = 0.0, 0.0, 0.0, 0.0

    def pos(self):
        return self.veh["x"] if self.kind == "veh" else self.x


class VECEnv:
    def __init__(self, nveh=40, rate_=0.6, mal_frac=0.0, black_hole=False, false_reports=False,
                 p_rsu=0.6, p_veh=0.8, twin_noise=0.02, gamma_dt=5.0, seed=0):
        self.cfg = dict(nveh=nveh, rate=rate_, mal_frac=mal_frac, black_hole=black_hole,
                        false_reports=false_reports, p_rsu=p_rsu, p_veh=p_veh)
        self.twin_noise, self.gamma_dt = twin_noise, gamma_dt
        self.obs_use_dt = True
        self.rng = np.random.default_rng(seed)        # traffic
        self.rp = np.random.default_rng(seed + 991)   # attack outcomes
        self.t = 0.0
        self.nodes, self.veh = [], []
        for i, x in enumerate(RSU_X):
            self.nodes.append(Node(f"RSU-{i+1}", "rsu", x, mal=black_hole and i == 1))
        self.cloud = Node("Cloud", "cloud", ROAD / 2)
        self.nodes.append(self.cloud)
        for i in range(nveh):
            lane = i % LANES
            d = 1 if lane < 2 else -1
            v = {"x": self.rng.uniform(0, ROAD), "lane": lane, "sp": d * self.rng.uniform(18, 32), "busy": 0.0, "node": None}
            helper, mal_draw = self.rng.random() < 0.3, self.rng.random()
            if helper:
                v["node"] = Node(f"H-{i+1}", "veh", veh=v, mal=mal_draw < mal_frac)
                self.nodes.append(v["node"])
            self.veh.append(v)
        self.snap, self.snap_t = {n.id: 0.0 for n in self.nodes}, -1.0
        self.pending = []
        self.stats = dict(done=0, ok=0, lat=0.0, en=0.0, lost=0, dec=0, over=0.0)

    # ---------- trust ----------
    def beta_trust(self, n):
        return (n.s + 1.0) / (n.s + n.f + 2.0)

    def reported_backlog(self, n):
        if n.mal and self.cfg["false_reports"]:
            return 0.0
        return max(0.0, self.snap[n.id] - self.t)

    def twin_backlog(self, n):
        return max(0.0, n.twin - self.t)

    def dt_dev(self, n):
        return abs(self.reported_backlog(n) - self.twin_backlog(n))

    def trust(self, n, use_dt=True):
        tb = self.beta_trust(n)
        return tb * math.exp(-self.gamma_dt * self.dt_dev(n)) if use_dt else tb

    # ---------- candidates and observation ----------
    def candidates(self, v, task):
        slots = [None] * NSLOT
        D, C = task["D"], task["C"]
        slots[0] = dict(node=None, kind="local", tx=0.0, wait=max(0.0, v["busy"] - self.t), f=F_VEH,
                        en=KAPPA * F_VEH ** 2 * C)
        rsus = sorted([(abs(n.x - v["x"]), n) for n in self.nodes if n.kind == "rsu" and math.hypot(n.x - v["x"], 30) <= RSU_R], key=lambda z: z[0])[:2]
        for k, (_, n) in enumerate(rsus):
            tx = D / rate(math.hypot(n.x - v["x"], 30), 20e6)
            slots[1 + k] = dict(node=n, kind="rsu", tx=tx, f=F_RSU, en=P_TX * tx)
        if rsus:
            n0 = rsus[0][1]
            tx = D / rate(math.hypot(n0.x - v["x"], 30), 20e6) + BACKHAUL
            slots[3] = dict(node=self.cloud, kind="cloud", via=n0, tx=tx, f=F_CLOUD, en=P_TX * (tx - BACKHAUL))
        hs = sorted([(abs(n.veh["x"] - v["x"]), n) for n in self.nodes if n.kind == "veh" and n.veh is not v and abs(n.veh["x"] - v["x"]) <= V2V_R], key=lambda z: z[0])[:3]
        for k, (d, n) in enumerate(hs):
            tx = D / rate(d, 10e6)
            slots[4 + k] = dict(node=n, kind="veh", tx=tx, f=F_HELP, en=P_TX * tx)
        return slots

    def est_delay(self, s, task, truth=False):
        if s["kind"] == "local":
            return s["wait"] + task["C"] / F_VEH
        n = s["node"]
        if s["kind"] == "cloud":
            return s["tx"] + task["C"] / F_CLOUD + BACKHAUL
        back = max(0.0, n.busy - self.t) if truth else self.reported_backlog(n)
        return s["tx"] + back + task["C"] / s["f"] + 0.005

    def observe(self, v, task, slots):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        T = task["T"]
        for i, s in enumerate(slots):
            if s is None:
                continue
            n = s["node"]
            ty = {"local": 0, "rsu": 1, "cloud": 2, "veh": 3}[s["kind"]]
            rep = 0.0 if n is None or n.kind == "cloud" else self.reported_backlog(n)
            tr = 1.0 if n is None or n.kind == "cloud" else self.trust(n, self.obs_use_dt)
            dev = 0.0 if n is None or n.kind == "cloud" or not self.obs_use_dt else self.dt_dev(n)
            f = [1.0, ty == 0, ty == 1, ty == 2, ty == 3, min(self.est_delay(s, task) / T, 5.0),
                 min(s["tx"] / T, 5.0), min(s["en"], 3.0), min(rep / T, 5.0), tr, min(dev / T, 5.0)]
            obs[i * NFEAT:(i + 1) * NFEAT] = f
        obs[-3:] = [task["D"] / 4e6, task["C"] / 1.5e9, task["T"]]
        return obs

    # ---------- dynamics ----------
    def step(self, decide, record=None):
        """Advance DT seconds. decide(env, v, task, slots, obs) -> (slot index, overhead s)."""
        self.t += DT
        if self.snap_t < 0 or self.t - self.snap_t >= BEACON:
            self.snap = {n.id: n.busy for n in self.nodes}
            self.snap_t = self.t
        for v in self.veh:
            v["x"] = (v["x"] + v["sp"] * DT) % ROAD
        for v in self.veh:
            if self.rng.random() >= self.cfg["rate"] * DT:
                continue
            task = {"D": self.rng.uniform(1, 4) * 1e6, "C": self.rng.uniform(0.3, 1.5) * 1e9, "T": self.rng.uniform(0.4, 1.0), "born": self.t}
            slots = self.candidates(v, task)
            obs = self.observe(v, task, slots)
            a, over = decide(self, v, task, slots, obs)
            if slots[a] is None:
                a = 0
            if record is not None:
                record(self, v, task, slots, obs, a)
            self.stats["dec"] += 1
            self.stats["over"] += over
            self.dispatch(v, task, slots[a], over)
        for p in list(self.pending):
            due = p["born"] + p["T"] if p["dropped"] else p["finish"]
            if self.t < due:
                continue
            self.pending.remove(p)
            st = self.stats
            st["done"] += 1
            st["en"] += p["en"]
            if p["dropped"]:
                st["lost"] += 1
                if p["node"] is not None:
                    p["node"].f += 1
                continue
            lat = p["finish"] - p["born"]
            st["lat"] += lat
            if lat <= p["T"]:
                st["ok"] += 1
                if p["node"] is not None:
                    p["node"].s += 1
            elif p["node"] is not None:
                p["node"].s += 0.5

    def dispatch(self, v, task, s, over):
        start, n, dropped = self.t + over, s["node"], False
        if n is None:
            v["busy"] = max(v["busy"], start) + task["C"] / F_VEH
            finish = v["busy"]
        elif n.kind == "cloud":
            finish = start + s["tx"] + task["C"] / F_CLOUD + BACKHAUL
        else:
            arrive = start + s["tx"]
            n.busy = max(n.busy, arrive) + task["C"] / s["f"]
            n.twin = max(n.twin, arrive) + task["C"] / s["f"] + self.rp.normal(0, self.twin_noise)
            finish = n.busy + 0.004
            if n.mal and self.rp.random() < (self.cfg["p_rsu"] if n.kind == "rsu" else self.cfg["p_veh"]):
                dropped = True
        self.pending.append(dict(born=task["born"], T=task["T"], finish=finish, en=s["en"], dropped=dropped, node=n))

    def run(self, decide, seconds=60.0, record=None):
        while self.t < seconds:
            self.step(decide, record)
        return self.summary()

    def summary(self):
        s = self.stats
        fin = s["done"] - s["lost"]
        return dict(succ=s["ok"] / max(1, s["done"]), lat=s["lat"] / max(1, fin), en=s["en"] / max(1, s["done"]),
                    lost=s["lost"], lost_rate=s["lost"] / max(1, s["done"]), over=s["over"] / max(1, s["dec"]), n=s["done"])


def cost(env, s, task, truth=False):
    return env.est_delay(s, task, truth) / task["T"] + 0.6 * s["en"]
