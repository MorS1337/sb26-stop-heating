"""Reduced-order thermal model of one building's finite-difference step.

Exact step (finite_difference_solver.py:968-1025), in the notation of the
extracted operators:

    A y = C/dt T + q + T_amb (d + g_amb) + M_c T_m               (1)
    q_a = (I + G Q^T A^-1 P)^-1 G (T_s - Q^T y)                  (2)
    T'  = y + A^-1 P q_a                                         (3)
    T_m' = r T_m + g T'                                          (4)

with A symmetric positive definite, C = diag(capacitance_rate) and g_amb, d,
M_c, (r, g), P, Q from the solver.

Projection in the energy inner product
--------------------------------------
A^-1 C is not symmetric, and a plain Galerkin projection of it onto a truncated
basis produces spurious eigenvalues outside the unit disc -- measured: at
m = 200 the state grew by e^1.2 per step and left the physical range inside 12
steps. But C^1/2 A^-1 C^1/2 *is* symmetric with spectrum in (0, 1], and its
Galerkin projection inherits that spectrum (Rayleigh quotients of a symmetric
matrix are bounded by its eigenvalues), so it is unconditionally stable. With

    u = C^1/2 T,   S = C^1/2 A^-1 C^1/2,   u_k = S u_{k-1} + C^1/2 A^-1 f_k

projecting u onto an orthonormal basis W gives the reduced step below, whose
operator is W^T S W: symmetric, and inside the unit disc by construction. The
norm is also the physically right one -- a deviation is weighted by the heat
capacity it sits in, which is what a temperature error costs in stored energy.

    b    = Ac s + Aq q_zone + dv T_amb + Mr u_m                 [m]
    q_a  = (I + G S_a)^-1 G (T_s - Qq b)                        [k]
    s'   = b + Pr q_a                                           [m]
    u_m' = r u_m + g (Zm_u s')                                  [ng]
    z    = Zo s'                                                [n_zones]

Precomputed once per (basis, factorization), all as W^T C^1/2 A^-1 (.):

    Ac = W^T C^1/2 A^-1 C^1/2 W    Aq = W^T C^1/2 A^-1 Zs   (Zs = zone spread)
    dv = W^T C^1/2 A^-1 (d+g_amb)  P r = W^T C^1/2 A^-1 P
    Mr = W^T C^1/2 A^-1 (M_c . S_m)  (S_m = 0/1 scatter onto the mass groups)
    S_a = Q^T A^-1 P (exact, k x k)  Qq = Q^T C^-1/2 W
    Zo = Zavg C^-1/2 W               Zm_u = Zm C^-1/2 W

``S_a`` is formed exactly rather than through the basis: the air-side loop is
stiff (g/(C/dt) is ~10 on a room CV), so a projected S makes it spuriously
ill-conditioned and the state diverges.

Per-step cost is O(m^2 + m n_zones + m k + k^3) dense flops, with no sparse
solve: the real step is two SuperLU triangular solves over 81k-149k unknowns.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

OPS = Path(__file__).resolve().parent / "ops"
BASIS = Path(__file__).resolve().parent / "basis"


# --------------------------------------------------------------------------- #
# operators
# --------------------------------------------------------------------------- #
class Ops:
    """Extracted timestep operators plus the zone and mass-group bookkeeping."""

    def __init__(self, building: str, root: Path = OPS):
        d = root / building
        self.building = building
        self.meta = json.loads((d / "meta.json").read_text())
        self.A = sp.load_npz(d / "A.npz").tocsc()
        self.P = sp.csc_matrix(sp.load_npz(d / "applied.npz"))
        self.Q = sp.csr_matrix(sp.load_npz(d / "sensed.npz"))  # (k, n)
        z = sp.load_npz(d / "zones.npz").tocsr()
        v = np.load(d / "vectors.npz")
        self.solved = v["solved"]
        self.dt = float(v["dt"])
        self.c = v["capacitance_rate"]
        self.sqrtc = np.sqrt(self.c)
        self.d = v["dirichlet_conductance"]
        self.g_amb = v["ambient_conductance"]
        self.mc = v["mass_coupling"]
        self.retain = v["mass_retained"]
        self.gain = v["mass_gained"]
        self.T0 = v["T0"]
        self.T0_m = v["T0_m"]
        self.zones = list(self.meta["zones"])
        self.n = int(self.solved.size)
        self.nz = len(self.zones)
        self.nfull = int(np.prod(self.meta["grid"]))
        self.shape = tuple(self.meta["grid"])
        self.k = self.P.shape[1]
        # zone averaging (rows sum to 1) and spreading (columns sum to 1)
        self.Zavg = sp.csr_matrix(z)
        self.zcount = np.diff(self.Zavg.indptr).astype(np.float64)
        # Zs is the map from a zone's total power to the per-CV power the step
        # applies: apply_thermal_power_zone (building.py:1123) writes
        # power / n_cv to every CV of the zone, so its entries are 1/n_cv.
        # Zind is the plain indicator (0/1), which is what turns a set of zone
        # means back into a field or into a VAV's sensed temperature.
        self.Zs = sp.csc_matrix(self.Zavg.T)
        self.Zind = sp.csc_matrix(self.Zs @ sp.diags(self.zcount))
        # mass groups: one per zone, plus the mass CVs that fall in no zone
        inside = np.zeros(self.n, dtype=bool)
        inside[self.Zavg.indices] = True
        mass = self.mc > 0.0
        self.rest_idx = np.flatnonzero(mass & ~inside)
        groups = [
            self.Zavg.indices[self.Zavg.indptr[i] : self.Zavg.indptr[i + 1]]
            for i in range(self.nz)
        ]
        if self.rest_idx.size:
            groups.append(self.rest_idx)
        self.mcount = np.array([gr.size for gr in groups], dtype=np.float64)
        ind = np.concatenate(groups)
        ptr = np.concatenate([[0], np.cumsum(self.mcount.astype(np.int64))])
        self.Zm = sp.csr_matrix(
            (np.repeat(1.0 / self.mcount, self.mcount.astype(np.int64)), ind, ptr),
            shape=(len(groups), self.n),
        )
        self.ng = len(groups)
        self.factor = spla.splu(self.A, permc_spec="MMD_AT_PLUS_A")

    def solve(self, X) -> np.ndarray:
        x = np.asarray(X, dtype=np.float64)
        return self.factor.solve(x)

    def usolve(self, F) -> np.ndarray:
        """C^1/2 A^-1 F: one temperature-space forcing to its energy-norm image."""
        F = np.asarray(F, dtype=np.float64)
        return self.solve(F) * self.sqrtc[:, None]

    def zone_means(self, x) -> np.ndarray:
        return np.asarray(self.Zavg @ x).ravel()


def describe_ops(building: str) -> dict:
    """Evidence for the two structural claims the ROM relies on."""
    o = Ops(building)
    m = o.mc > 0
    return dict(
        building=building,
        n_solved=o.n,
        n_zones=o.nz,
        n_air_terms=o.k,
        A_nnz=int(o.A.nnz),
        mass_cvs=int(m.sum()),
        mass_coupling_unique=np.unique(np.round(o.mc[m], 9)).tolist(),
        retain_unique=np.unique(np.round(o.retain[m], 9)).tolist(),
        gain_unique=np.unique(np.round(o.gain[m], 9)).tolist(),
        dirichlet_nonzero=int((o.d != 0).sum()),
        ambient_nonzero=int((o.g_amb != 0).sum()),
        capacitance_rate_unique=np.unique(np.round(o.c, 6)).tolist(),
        mass_cvs_outside_zones=int(o.rest_idx.size),
        mass_groups=o.ng,
    )


# --------------------------------------------------------------------------- #
# basis construction (all in the energy norm, u = C^1/2 T)
# --------------------------------------------------------------------------- #
def orth(X: np.ndarray, base: np.ndarray | None = None, tol: float = 1e-10) -> np.ndarray:
    """Modified Gram-Schmidt with one reorthogonalisation pass."""
    X = np.asarray(X, dtype=np.float64)
    Q = np.empty((X.shape[0], 0)) if base is None else np.array(base, dtype=np.float64)
    for j in range(X.shape[1]):
        v = X[:, j].copy()
        scale = np.linalg.norm(v)
        if scale == 0.0:
            continue
        for _ in range(2):
            if Q.shape[1]:
                v -= Q @ (Q.T @ v)
        nrm = np.linalg.norm(v)
        if nrm > tol * scale:
            Q = np.hstack([Q, (v / nrm).reshape(-1, 1)])
    return Q


def _seed_temperature(o: Ops) -> np.ndarray:
    """What the step can push into the field, in temperature space."""
    return np.hstack(
        [
            o.T0[o.solved].reshape(-1, 1),
            o.P.toarray(),
            (o.d + o.g_amb).reshape(-1, 1),
            np.ones((o.n, 1)),
        ]
    )


def _seed_u(o: Ops, direct: bool = True) -> np.ndarray:
    """Krylov seeds in the energy norm: the images of the temperature sources.

    ``C^1/2 A^-1 F`` is where one step sends each source; ``C^1/2 F`` is the
    source as a state. The second family costs n_zones + k columns, which at
    m = 400 is a third of the budget, so it is switchable: with ``direct=False``
    the same basis size buys four Krylov iterations instead of two.
    """
    F = _seed_temperature(o)
    if not direct:
        return o.usolve(F)
    borne = o.sqrtc[:, None] * np.hstack([np.asarray(o.Zs.todense()), o.P.toarray()])
    return np.hstack([o.usolve(F), borne])


def krylov_basis(o: Ops, m: int, direct: bool = True) -> np.ndarray:
    """Block Krylov of S = C^1/2 A^-1 C^1/2, columns ordered by iteration."""
    W = orth(_seed_u(o, direct))
    if W.shape[1] > m:
        return W[:, :m]
    while W.shape[1] < m:
        block = W[:, -min(W.shape[1], 96) :]
        Y = o.solve(o.sqrtc[:, None] * block) * o.sqrtc[:, None]
        # orth returns base first, so drop it before stacking.
        new = orth(Y, base=W)[:, W.shape[1] :]
        if new.shape[1] == 0:
            break
        W = np.hstack([W, new])
    return W[:, :m]


def pod_basis(
    o: Ops,
    snaps: np.ndarray | None,
    m: int,
    base: np.ndarray | None = None,
    stride: int = 1,
    norm: str = "energy",
) -> np.ndarray:
    """Leading left singular vectors of the logged fields.

    ``norm="energy"`` measures reconstruction error in u = C^1/2 T, i.e. by the
    heat capacity a deviation sits in. ``norm="temperature"`` measures it in K,
    which spends basis budget on the low-capacity wall CVs instead -- they are
    down-weighted 28x in the energy norm (sqrt(383) against sqrt(0.48)) although
    the step's forcing C/dt T reads them all the same way.
    """
    if snaps is None or len(snaps) == 0 or m <= 0:
        return np.zeros((o.n, 0))
    X = np.asarray(snaps[::stride], dtype=np.float64).T[o.solved]
    if norm == "energy":
        X = o.sqrtc[:, None] * X
    if base is not None and base.shape[1]:
        X = X - base @ (base.T @ X)
    w, U = np.linalg.eigh(X.T @ X)
    order = np.argsort(w)[::-1]
    w, U = w[order], U[:, order]
    keep = w > max(w[0], 1e-300) * 1e-16
    w, U = w[keep][:m], U[:, keep][:, :m]
    if len(w) == 0:
        return np.zeros((o.n, 0))
    modes = X @ (U / np.sqrt(w))
    if norm != "energy":
        modes = o.sqrtc[:, None] * modes  # into u-space, keeping the T-norm subspace
    Q = orth(modes, base=base)
    return Q[:, 0 if base is None else base.shape[1] :]


def build_basis(
    o: Ops,
    m: int = 200,
    kind: str = "hybrid",
    snaps: np.ndarray | None = None,
    n_krylov: int | None = None,
    cache: bool = False,
    direct: bool = True,
    norm: str = "energy",
) -> np.ndarray:
    """Returns an (n x m') energy-orthonormal basis, m' <= m."""
    tag = f"{o.building}_m{m}_{kind}"
    f = BASIS / f"{tag}.npy"
    if cache and f.exists():
        return np.load(f).astype(np.float64)
    if norm != "energy" and kind != "pod":
        raise ValueError(
            "the temperature-norm basis is only defined for kind='pod': the "
            "Krylov seeds and the residual deflation live in the energy norm, "
            "and mixing the two spaces produces a basis that is not nested."
        )
    if kind == "krylov":
        W = krylov_basis(o, m, direct)
    elif kind == "pod":
        W = pod_basis(o, snaps, m, norm=norm)
    elif kind == "hybrid":
        nk = n_krylov if n_krylov is not None else max(m // 2, 16)
        Wk = krylov_basis(o, nk, direct)
        Wp = pod_basis(o, snaps, m - Wk.shape[1], base=Wk, norm=norm)
        W = np.hstack([Wk, Wp])
    else:
        raise ValueError(f"unknown basis kind {kind!r}")
    if cache:
        BASIS.mkdir(parents=True, exist_ok=True)
        np.save(f, W.astype(np.float32))
    return W


# --------------------------------------------------------------------------- #
# reduced model
# --------------------------------------------------------------------------- #
class ROM:
    """Projected timestep. ``step`` is the whole per-step cost of the surrogate."""

    def __init__(
        self,
        building: str,
        m: int = 200,
        kind: str = "hybrid",
        snaps: np.ndarray | None = None,
        n_krylov: int | None = None,
        ops: Ops | None = None,
        direct: bool = True,
        norm: str = "energy",
    ):
        o = ops or Ops(building)
        self.o = o
        self.kind = kind
        self.m_requested = m
        t0 = time.time()
        W = build_basis(o, m, kind, snaps, n_krylov, direct=direct, norm=norm)
        self.basis_seconds = time.time() - t0
        self.W = np.ascontiguousarray(W, dtype=np.float64)
        self._reduce()
        self.reset()

    # -- precomputation ---------------------------------------------------- #
    def _reduce(self) -> None:
        o, W = self.o, self.W
        t0 = time.time()
        self.Ac = self._U(o.sqrtc[:, None] * W)  # W^T C^1/2 A^-1 C^1/2 W
        self.Aq = self._U(o.Zs.toarray())
        self.dv = self._U((o.d + o.g_amb).reshape(-1, 1)).ravel()
        self.Mr = self._U(o.mc[:, None] * self._scatter(o))
        AinvP = o.solve(o.P.toarray())
        self.Pr = W.T @ (o.sqrtc[:, None] * AinvP)
        self.Sa = np.asarray(o.Q @ AinvP)  # exact k x k
        inv = (W / o.sqrtc[:, None])
        self.Qq = np.asarray(o.Q @ inv)
        self.Zo = np.asarray(o.Zavg @ inv)
        self.Zm_u = np.asarray(o.Zm @ inv)
        self.reduce_seconds = time.time() - t0

    def _U(self, F: np.ndarray) -> np.ndarray:
        """W^T C^1/2 A^-1 F, solved in column blocks to bound memory."""
        F = np.ascontiguousarray(F, dtype=np.float64)
        W = self.W
        out = np.empty((W.shape[1], F.shape[1]))
        step = max(1, int(2e7 // max(1, F.shape[0])))
        sc = self.o.sqrtc[:, None]
        for s in range(0, F.shape[1], step):
            Y = self.o.solve(F[:, s : s + step]) * sc
            out[:, s : s + step] = W.T @ Y
        return out

    @staticmethod
    def _scatter(o: Ops) -> np.ndarray:
        """0/1 map from mass-group means back to CVs."""
        return np.asarray((o.Zm.T @ sp.diags(o.mcount)).todense())

    # -- state ------------------------------------------------------------- #
    def reset(self, T0: np.ndarray | None = None, T0_m: np.ndarray | None = None) -> None:
        o = self.o
        x = o.T0[o.solved] if T0 is None else np.asarray(T0, float)[o.solved]
        self.s = self.W.T @ (o.sqrtc * x)
        tm = o.T0_m if T0_m is None else np.asarray(T0_m, float)
        self.u_m = np.asarray(o.Zm @ tm).ravel()
        self._ambient = float(x.mean())

    @property
    def mc(self) -> float:
        return float(self.o.mc[self.o.mc > 0][0])

    @property
    def r(self) -> float:
        return float(self.o.retain[self.o.mc > 0][0])

    @property
    def g(self) -> float:
        return float(self.o.gain[self.o.mc > 0][0])

    @property
    def rho(self) -> float:
        """Largest eigenvalue magnitude of the reduced state operator."""
        return float(np.abs(np.linalg.eigvals(self.Ac)).max())

    def step(self, q_zone, G, T_supply, T_amb):
        """Advances one timestep; arguments are the real step's own inputs."""
        b = (
            self.Ac @ self.s
            + self.Aq @ np.asarray(q_zone, dtype=np.float64)
            + self.dv * float(T_amb)
            + self.Mr @ self.u_m
        )
        G = np.asarray(G, dtype=np.float64)
        rhs = G * (np.asarray(T_supply, dtype=np.float64) - self.Qq @ b)
        if G.any():
            q_a = np.linalg.solve(np.eye(G.size) + G[:, None] * self.Sa, rhs)
        else:
            q_a = rhs
        self.s = b + self.Pr @ q_a
        self.u_m = self.r * self.u_m + self.g * (self.Zm_u @ self.s)
        return q_a

    def zone_means(self) -> np.ndarray:
        return self.Zo @ self.s

    def temperature(self) -> np.ndarray:
        """Solved-CV temperatures, C^-1/2 W s (O(n m); not on the hot path)."""
        return (self.W @ self.s) / self.o.sqrtc

    def field(self) -> np.ndarray:
        """Full grid, exterior CVs at the last ambient temperature."""
        out = np.full(self.o.nfull, self._ambient)
        out[self.o.solved] = self.temperature()
        return out.reshape(self.o.shape)

    def set_ambient(self, T_amb: float) -> None:
        self._ambient = float(T_amb)


def main(argv=None) -> int:
    import argparse  # noqa: PLC0415

    p = argparse.ArgumentParser(description="describe extracted operators")
    p.add_argument("buildings", nargs="+")
    a = p.parse_args(argv)
    for b in a.buildings:
        print(json.dumps(describe_ops(b), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
