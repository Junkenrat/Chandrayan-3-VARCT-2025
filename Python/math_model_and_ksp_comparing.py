from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


# =========================
# ПАРАМЕТРЫ
# =========================
@dataclass
class Params:
    # Планета (Kerbin)
    planet_radius: float = 600_000.0  # m
    G: float = 6.67e-11
    M: float = 5.292e22

    # Времена
    sep_time: float = 78.0
    end_time: float = 120.0
    dt: float = 0.05

    # Массы
    mass_start: float = 656_000.0         # кг
    booster_fuel_mass: float = 442_000.0  # кг (равномерно до sep_time)
    dry_mass_to_drop: float = 84_000.0    # кг (сбрасываем в момент sep)

    # Тяга
    thrust_before_sep: float = 1.18e7     # Н
    thrust_after_sep: float = 0.0         # Н (после separation)

    # θ=90° — вверх, θ=0° — горизонтально
    pitch_heights_m: Tuple[float, ...] = (0.5e3, 2e3, 5e3, 10e3, 30e3, 45e3, 60e3)
    pitch_degrees: Tuple[float, ...] = (90.0, 80.0, 70.0, 60.0, 40.0, 20.0, 0.0)

    air_density0: float = 1.225   # кг/м^3
    scale_height: float = 5600.0  # м
    drag_coef: float = 0.30       # Cx
    ref_area: float = float(np.pi * (2.5 ** 2))  # S = pi r^2, r=2.5 м

    # Численная защита
    speed_eps: float = 1e-6

    @property
    def mu(self) -> float:
        """mu = G*M."""
        return self.G * self.M

    @property
    def fuel_flow(self) -> float:
        """Положительный расход топлива (кг/с) до sep_time."""
        return self.booster_fuel_mass / self.sep_time

    @property
    def mass_after_sep(self) -> float:
        """Масса сразу после сброса (скачок)."""
        m_before_sep = self.mass_start - self.booster_fuel_mass
        return m_before_sep - self.dry_mass_to_drop


# =========================
# ФУНКЦИИ СРЕДЫ/АЭРО/ТАНГАЖ
# =========================
def g_at_altitude(h_m: float, p: Params) -> float:
    """g(h) = mu / (R+h)^2."""
    h = max(0.0, float(h_m))
    return p.mu / (p.planet_radius + h) ** 2


def pitch_deg_from_altitude(h_m: float, p: Params) -> float:
    """θ(h) по табличным точкам с линейной интерполяцией."""
    return float(np.interp(
        float(h_m),
        np.array(p.pitch_heights_m, dtype=float),
        np.array(p.pitch_degrees, dtype=float),
        left=p.pitch_degrees[0],
        right=p.pitch_degrees[-1],
    ))


def air_density(h_m: float, p: Params) -> float:
    """rho(h) = rho0 * exp(-h/H)."""
    h = max(0.0, float(h_m))
    return p.air_density0 * np.exp(-h / p.scale_height)


def drag_magnitude(h_m: float, speed: float, p: Params) -> float:
    """D = 0.5*rho(h)*v^2*Cx*S."""
    v = max(0.0, float(speed))
    rho = air_density(h_m, p)
    return 0.5 * rho * (v ** 2) * p.drag_coef * p.ref_area


# =========================
# RK4
# =========================
def rk4_step(deriv, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    k1 = deriv(t, y)
    k2 = deriv(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = deriv(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = deriv(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# =========================
# МОДЕЛИРОВАНИЕ ЧЕРЕЗ ВЕКТОРЫ
# =========================
def simulate_vector(p: Params) -> Dict[str, np.ndarray]:
    """
    Интегрируем векторно:
      до sep: y = [x, h, vx, vy, m]
      после sep: y = [x, h, vx, vy] (m = const)
    Силы:
      F_T = T * e_theta
      F_g = (0, -m*g(h))
      F_d = -D * v_hat
    """

    # ---------- Фаза 0 (до separation): [x, h, vx, vy, m]
    def deriv_stage0(t: float, y: np.ndarray) -> np.ndarray:
        x, h, vx, vy, m = map(float, y)
        m = max(m, 1.0)

        v_vec = np.array([vx, vy], dtype=float)
        speed = float(np.hypot(vx, vy))
        speed_safe = max(speed, p.speed_eps)
        v_hat = v_vec / speed_safe  # единичный вдоль скорости (при v≈0 станет ~0/eps)

        theta = np.deg2rad(pitch_deg_from_altitude(h, p))
        e_theta = np.array([np.cos(theta), np.sin(theta)], dtype=float)

        g = g_at_altitude(h, p)
        D = drag_magnitude(h, speed, p)

        F_T = p.thrust_before_sep * e_theta
        F_g = np.array([0.0, -m * g], dtype=float)
        F_d = -D * v_hat

        a = (F_T + F_g + F_d) / m

        dx = vx
        dh = vy
        dvx = a[0]
        dvy = a[1]
        dm = -p.fuel_flow

        return np.array([dx, dh, dvx, dvy, dm], dtype=float)

    t0 = np.arange(0.0, p.sep_time + p.dt, p.dt)
    stage0 = np.zeros((len(t0), 5), dtype=float)

    # Старт: x=0, h=0, v~0 вверх (чтобы задать направление)
    y = np.array([0.0, 0.0, 0.0, 1e-3, p.mass_start], dtype=float)
    stage0[0] = y
    for i in range(len(t0) - 1):
        y = rk4_step(deriv_stage0, t0[i], y, p.dt)
        # защита от отрицательной высоты и массы
        y[1] = max(y[1], 0.0)
        y[4] = max(y[4], 1.0)
        stage0[i + 1] = y

    x_sep, h_sep, vx_sep, vy_sep, m_sep_before_drop = stage0[-1]

    # ---------- Событие separation: мгновенный сброс сухой массы
    m_sep = max(m_sep_before_drop - p.dry_mass_to_drop, 1.0)

    # ---------- Фаза 1 (после separation): [x, h, vx, vy], масса фикс.
    def deriv_stage1(t: float, y: np.ndarray) -> np.ndarray:
        x, h, vx, vy = map(float, y)

        m = float(p.mass_after_sep)

        v_vec = np.array([vx, vy], dtype=float)
        speed = float(np.hypot(vx, vy))
        speed_safe = max(speed, p.speed_eps)
        v_hat = v_vec / speed_safe

        g = g_at_altitude(h, p)
        D = drag_magnitude(h, speed, p)

        F_T = p.thrust_after_sep * np.array([1.0, 0.0], dtype=float)  # T=0, направление не важно
        F_g = np.array([0.0, -m * g], dtype=float)
        F_d = -D * v_hat

        a = (F_T + F_g + F_d) / m

        dx = vx
        dh = vy
        dvx = a[0]
        dvy = a[1]

        return np.array([dx, dh, dvx, dvy], dtype=float)

    t1 = np.arange(p.sep_time, p.end_time + p.dt, p.dt)
    stage1 = np.zeros((len(t1), 4), dtype=float)

    y = np.array([x_sep, h_sep, vx_sep, vy_sep], dtype=float)
    stage1[0] = y
    for i in range(len(t1) - 1):
        y = rk4_step(deriv_stage1, t1[i], y, p.dt)
        y[1] = max(y[1], 0.0)
        stage1[i + 1] = y

    # ---------- Склейка
    t_all = np.concatenate([t0, t1[1:]])

    x_all = np.concatenate([stage0[:, 0], stage1[1:, 0]])
    h_all = np.concatenate([stage0[:, 1], stage1[1:, 1]])

    vx_all = np.concatenate([stage0[:, 2], stage1[1:, 2]])
    vy_all = np.concatenate([stage0[:, 3], stage1[1:, 3]])
    speed_all = np.hypot(vx_all, vy_all)

    m0_hist = stage0[:, 4]
    m1_hist = np.full_like(t1, p.mass_after_sep, dtype=float)
    mass_all = np.concatenate([m0_hist, m1_hist[1:]])

    return {
        "t": t_all,
        "x": x_all,
        "alt": h_all,
        "vx": vx_all,
        "vy": vy_all,
        "speed": speed_all,
        "mass": mass_all,
    }


# =========================
# KSP (kRPC)
# =========================
@dataclass
class KspLogParams:
    sample_dt: float = 0.2
    max_time: float = 120.0


def read_ksp_telemetry(cfg: KspLogParams) -> Dict[str, np.ndarray]:
    import krpc

    conn = krpc.connect(name="ksp_compare_plots_vector")
    vessel = conn.space_center.active_vessel
    rf = vessel.orbit.body.reference_frame

    t0 = conn.space_center.ut

    t, speed, alt, mass = [], [], [], []

    while True:
        time_s = float(conn.space_center.ut - t0)

        flight_rf = vessel.flight(rf)
        t.append(time_s)
        speed.append(float(flight_rf.speed))
        alt.append(float(vessel.flight().surface_altitude))
        mass.append(float(vessel.mass))

        if time_s >= cfg.max_time:
            break

        time.sleep(cfg.sample_dt)

    return {
        "t": np.array(t, dtype=float),
        "speed": np.array(speed, dtype=float),
        "alt": np.array(alt, dtype=float),
        "mass": np.array(mass, dtype=float),
    }


# =========================
# ГРАФИКИ
# =========================
def save_compare_plot(
    t_model: np.ndarray,
    y_model: np.ndarray,
    t_ksp: np.ndarray,
    y_ksp: np.ndarray,
    title: str,
    ylabel: str,
    filename: str,
    sep_time: float,
) -> None:
    plt.figure()
    plt.plot(t_model, y_model, "--", linewidth=2, label="Математическая модель")
    plt.plot(t_ksp, y_ksp, "-", linewidth=2, label="KSP")
    plt.axvline(sep_time, linestyle=":", linewidth=1.5, label=f"sep_time = {sep_time:g} s")
    plt.xlabel("Время от старта, s")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Сохранён файл: {filename}")


def main() -> None:
    # 1) Модель (векторная)
    params = Params()
    model = simulate_vector(params)

    # 2) KSP
    ksp = read_ksp_telemetry(KspLogParams())

    # 3) Графики сравнения
    save_compare_plot(
        model["t"], model["speed"],
        ksp["t"], ksp["speed"],
        "Скорость v(t)",
        "Скорость, m/s",
        "speed_vs_time.png",
        params.sep_time,
    )
    save_compare_plot(
        model["t"], model["alt"],
        ksp["t"], ksp["alt"],
        "Высота h(t)",
        "Высота, m",
        "height_vs_time.png",
        params.sep_time,
    )
    save_compare_plot(
        model["t"], model["mass"],
        ksp["t"], ksp["mass"],
        "Масса m(t)",
        "Масса, kg",
        "mass_vs_time.png",
        params.sep_time,
    )


if __name__ == "__main__":
    main()
