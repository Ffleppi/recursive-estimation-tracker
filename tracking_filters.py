from typing import Any
import numpy as np


NUM_STATES = 9
PSI = 3
THETA = 4
SPEED = 5
R_PSI = 6
R_THETA = 7
ACC = 8


def _wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _blend_angle(old, new, alpha):
    return np.arctan2(
        (1.0 - alpha) * np.sin(old) + alpha * np.sin(new),
        (1.0 - alpha) * np.cos(old) + alpha * np.cos(new),
    )


def _refine_from_position_history(x: np.ndarray, history: list,
                                  c: Any):
    history.append(x[:3].copy())
    if len(history) > 16:
        del history[0]
    if len(history) < 9:
        return

    steps = min(12, len(history) - 1)
    delta = history[-1] - history[-1 - steps]
    horizontal = np.hypot(delta[0], delta[1])
    distance = np.linalg.norm(delta)
    if distance < 30.0:
        return

    reliability = np.clip((distance - 30.0) / 220.0, 0.0, 1.0)
    x[PSI] = _blend_angle(x[PSI], np.arctan2(delta[1], delta[0]), 0.08 * reliability)
    x[THETA] = _blend_angle(
        x[THETA],
        np.arctan2(delta[2], max(horizontal, 1e-6)),
        0.08 * reliability,
    )
    a_speed = 0.06 * reliability
    x[SPEED] = max(
        0.0,
        (1.0 - a_speed) * x[SPEED] + a_speed * distance / (steps * c.Ts),
    )


def _initial_mean_and_cov(c: Any, tuned=False):
    xm = np.array([
        0.0,
        0.0,
        c.z0,
        0.0,
        0.5 * (c.theta0_min + c.theta0_max),
        0.5 * (c.p0_min + c.p0_max),
        0.0,
        0.0,
        0.0,
    ], dtype=float)

    psi_var = (2.0 * c.psi_bound) ** 2 / 12.0
    theta_var = (c.theta0_max - c.theta0_min) ** 2 / 12.0
    speed_var = (c.p0_max - c.p0_min) ** 2 / 12.0

    z_var = 2.0 ** 2 if tuned else 0.0
    rate_var = 0.02 ** 2 if tuned else 0.0

    # Initial horizontal position is uniformly distributed over a disk.
    Pm = np.diag([
        0.25 * c.R ** 2,
        0.25 * c.R ** 2,
        z_var,
        psi_var,
        theta_var,
        speed_var,
        rate_var,
        rate_var,
        rate_var,
    ])
    return xm, Pm


def _theta_target(c: Any, k: int):
    if k >= int(c.k_switch_frac * c.N):
        return -c.theta_dive
    return c.theta_dive


def _predict_one(x: np.ndarray, c: Any, k: int, clamp=False):
    Ts = c.Ts
    xn = x.copy()

    psi = x[PSI]
    theta = x[THETA]
    speed = max(0.0, x[SPEED]) if clamp else x[SPEED]
    r_psi = x[R_PSI]
    r_theta = x[R_THETA]
    acc = x[ACC]

    cp = np.cos(psi)
    sp = np.sin(psi)
    ct = np.cos(theta)
    st = np.sin(theta)

    xn[0] = x[0] + Ts * speed * ct * cp
    xn[1] = x[1] + Ts * speed * ct * sp
    xn[2] = x[2] + Ts * speed * st
    xn[PSI] = psi + Ts * r_psi
    xn[THETA] = theta + Ts * r_theta
    xn[SPEED] = speed + Ts * (acc - c.cd * speed)

    theta_ref = _theta_target(c, k)
    xn[R_PSI] = r_psi + c.K_yaw * (c.omega_yaw - r_psi)
    xn[R_THETA] = (
        r_theta
        + c.K_theta_p * (theta_ref - theta)
        - c.K_theta_d * r_theta
    )
    xn[ACC] = acc + c.K_p_p * (c.p_cruise - speed) - c.K_p_d * acc

    if clamp:
        xn[PSI] = _wrap_angle(xn[PSI])
        xn[THETA] = _wrap_angle(xn[THETA])
        xn[THETA] = np.clip(xn[THETA], -1.2, 1.2)
        xn[2] = max(0.0, xn[2])
        xn[SPEED] = max(0.0, xn[SPEED])
    return xn


def _transition_jacobian(x: np.ndarray, c: Any, clamp=False):
    Ts = c.Ts
    F = np.eye(NUM_STATES)

    psi = x[PSI]
    theta = x[THETA]
    speed = max(0.0, x[SPEED]) if clamp else x[SPEED]

    cp = np.cos(psi)
    sp = np.sin(psi)
    ct = np.cos(theta)
    st = np.sin(theta)

    F[0, PSI] = -Ts * speed * ct * sp
    F[0, THETA] = -Ts * speed * st * cp
    F[0, SPEED] = Ts * ct * cp

    F[1, PSI] = Ts * speed * ct * cp
    F[1, THETA] = -Ts * speed * st * sp
    F[1, SPEED] = Ts * ct * sp

    F[2, THETA] = Ts * speed * ct
    F[2, SPEED] = Ts * st

    F[PSI, R_PSI] = Ts
    F[THETA, R_THETA] = Ts
    F[SPEED, SPEED] = 1.0 - Ts * c.cd
    F[SPEED, ACC] = Ts

    F[R_PSI, :] = 0.0
    F[R_PSI, R_PSI] = 1.0 - c.K_yaw

    F[R_THETA, :] = 0.0
    F[R_THETA, THETA] = -c.K_theta_p
    F[R_THETA, R_THETA] = 1.0 - c.K_theta_d

    F[ACC, :] = 0.0
    F[ACC, SPEED] = -c.K_p_p
    F[ACC, ACC] = 1.0 - c.K_p_d
    return F


def _process_cov(c: Any, tuned=False):
    Q = np.zeros((NUM_STATES, NUM_STATES))

    if tuned:
        # Slack absorbs model mismatch for the free-form non-Gaussian estimator.
        Q[0, 0] = 0.5 ** 2
        Q[1, 1] = 0.5 ** 2
        Q[2, 2] = 0.5 ** 2
        Q[PSI, PSI] = 0.002 ** 2
        Q[THETA, THETA] = 0.0015 ** 2
        Q[SPEED, SPEED] = 0.01 ** 2

    Q[R_PSI, R_PSI] = c.sigma_r_psi ** 2
    Q[R_THETA, R_THETA] = c.sigma_r_theta ** 2
    Q[ACC, ACC] = c.sigma_a ** 2
    return Q


def _measurement_matrices(x: np.ndarray, measurement: np.ndarray,
                          c: Any, sonar_scale=1.0):
    rows = []
    z = []
    z_pred = []
    variances = []

    if not np.any(np.isnan(measurement[:2])):
        sigma_u = c.sigma_base + c.alpha * max(0.0, x[2])
        for meas_idx, state_idx in ((0, 0), (1, 1)):
            row = np.zeros(NUM_STATES)
            row[state_idx] = 1.0
            rows.append(row)
            z.append(measurement[meas_idx])
            z_pred.append(x[state_idx])
            variances.append(sigma_u ** 2)

    for meas_idx, state_idx in ((2, 0), (3, 1), (4, 2)):
        row = np.zeros(NUM_STATES)
        row[state_idx] = 1.0
        rows.append(row)
        z.append(measurement[meas_idx])
        z_pred.append(x[state_idx])
        variances.append((sonar_scale * c.sigma_S) ** 2)

    return (
        np.vstack(rows),
        np.array(z, dtype=float),
        np.array(z_pred, dtype=float),
        np.diag(variances),
    )


def _normalize_state(x: np.ndarray, clamp=False):
    if clamp:
        x[PSI] = _wrap_angle(x[PSI])
        x[THETA] = _wrap_angle(x[THETA])
        x[THETA] = np.clip(x[THETA], -1.2, 1.2)
        x[2] = max(0.0, x[2])
        x[SPEED] = max(0.0, x[SPEED])


def _estimate_step(xm: np.ndarray, Pm: np.ndarray, k: int,
                   measurement: np.ndarray, c: Any,
                   Q: np.ndarray, sonar_scale=1.0, robust_sonar=False,
                   clamp=False):
    xp = _predict_one(xm, c, k, clamp)
    F = _transition_jacobian(xm, c, clamp)
    Pp = F @ Pm @ F.T + Q
    Pp = 0.5 * (Pp + Pp.T)

    H, z, z_pred, R = _measurement_matrices(xp, measurement, c, sonar_scale)
    innovation = z - z_pred

    if robust_sonar:
        sonar_start = 2 if not np.any(np.isnan(measurement[:2])) else 0
        for i in range(sonar_start, innovation.size):
            if abs(innovation[i]) > 2.5 * c.sigma_S:
                R[i, i] = (2.5 * c.sigma_S) ** 2

    S = H @ Pp @ H.T + R
    K = np.linalg.solve(S.T, (Pp @ H.T).T).T

    xm = xp + K @ innovation
    Pm = Pp - K @ H @ Pp
    Pm = 0.5 * (Pm + Pm.T)
    _normalize_state(xm, clamp)
    return xm, Pm


def _estimate_step_ekf(xm: np.ndarray, Pm: np.ndarray, k: int,
                       measurement: np.ndarray, c: Any,
                       Q: np.ndarray):
    xp = _predict_one(xm, c, k, clamp=False)
    F = _transition_jacobian(xm, c, clamp=False)
    Pp = F @ Pm @ F.T + Q

    H, z, z_pred, R = _measurement_matrices(xp, measurement, c)
    S = H @ Pp @ H.T + R
    K = Pp @ H.T @ np.linalg.inv(S)

    xm = xp + K @ (z - z_pred)
    Pm = Pp - K @ H @ Pp
    return xm, Pm


class GaussianTrackingEKF:
    """
    Extended Kalman Filter for a nonlinear tracking problem under Gaussian noise.

    Args:
        model_constants : Any
            Constants known to the estimator (initial state bounds, process
            noise parameters, etc...)
    """

    def __init__(
            self,
            model_constants: Any,
    ):
        self.constant = model_constants
        self.xm = None
        self.Pm = None
        self.k = 0
        self.Q = _process_cov(model_constants)

    def initialize(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialize the estimator. Set up any internal state required by the
        filter and return the initial posterior mean and covariance.

        Returns:
            xm : np.ndarray, dim: (num_states,)
                The initial posterior state mean. The order of states is
                x = [x, y, z, psi, theta, p, r_psi, r_theta, a].
            Pm : np.ndarray, dim: (num_states, num_states)
                The initial posterior state covariance.
        """
        self.xm, self.Pm = _initial_mean_and_cov(self.constant)
        self.k = 0
        return self.xm.copy(), self.Pm.copy()

    def estimate(
            self,
            measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Update the estimator with a new measurement and return the posterior
        mean and covariance at the current time step.
        """
        self.xm, self.Pm = _estimate_step_ekf(
            self.xm, self.Pm, self.k, measurement, self.constant, self.Q
        )
        self.k += 1
        return self.xm.copy(), self.Pm.copy()


class RobustNonGaussianTrackingFilter:
    """
    Estimator for a nonlinear tracking problem under non-Gaussian noise.

    Args:
        model_constants : Any
            Constants known to the estimator.
    """

    def __init__(
            self,
            model_constants: Any,
    ):
        self.constant = model_constants
        self.xm = None
        self.Pm = None
        self.k = 0
        self.Q = _process_cov(model_constants, tuned=True)
        self.pos_history = []

    def initialize(self) -> np.ndarray:
        """
        Initialize the estimator. Set up any internal state required by the
        filter and return the initial state estimate.
        """
        self.xm, self.Pm = _initial_mean_and_cov(self.constant, tuned=True)
        self.k = 0
        self.pos_history = [self.xm[:3].copy()]
        return self.xm.copy()

    def estimate(
            self,
            measurement: np.ndarray,
    ) -> np.ndarray:
        """
        Update the estimator with a new measurement and return the posterior
        state estimate at the current time step.
        """
        self.xm, self.Pm = _estimate_step(
            self.xm,
            self.Pm,
            self.k,
            measurement,
            self.constant,
            self.Q,
            sonar_scale=0.15,
            robust_sonar=True,
            clamp=True,
        )
        _refine_from_position_history(self.xm, self.pos_history, self.constant)
        self.k += 1
        return self.xm.copy()
