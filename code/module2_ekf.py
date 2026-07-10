import numpy as np
import json


class UAVKalmanFilter:
    def __init__(self, dt=0.5):
        self.dt = dt
        self.x = np.zeros(3)
        self.P = np.eye(3) * 1.0
        self.Q = np.diag([1e-10, 1e-10, 0.01])
        self.R = np.diag([1e-8, 1e-8, 2.0])
        self.initialized = False
        self.step_count = 0
        self.v_lat = 0.0
        self.v_lon = 0.0

    def step(self, lat, lon, alt, speed, heading):
        self.step_count += 1

        if not self.initialized:
            self.x = np.array([lat, lon, alt])
            self.initialized = True
            return {
                "estimated_lat": lat,
                "estimated_lon": lon,
                "estimated_alt": alt,
                "innovation_residual": 0.0,
                "status": "initializing"
            }

        x_pred = np.array([
            self.x[0] + self.v_lat * self.dt,
            self.x[1] + self.v_lon * self.dt,
            self.x[2]
        ])

        F = np.eye(3)
        P_pred = F @ self.P @ F.T + self.Q

        z = np.array([lat, lon, alt])
        H = np.eye(3)
        innovation = z - x_pred
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ innovation
        self.P = (np.eye(3) - K @ H) @ P_pred

        self.v_lat = (lat - (self.x[0] - K[0, 0] * innovation[0])) / self.dt
        self.v_lon = (lon - (self.x[1] - K[1, 1] * innovation[1])) / self.dt

        mahalanobis = float(np.sqrt(innovation.T @ np.linalg.inv(S) @ innovation))

        if mahalanobis < 3.0:
            status = "normal"
        elif mahalanobis < 10.0:
            status = "warning"
        else:
            status = "suspicious"

        return {
            "estimated_lat": round(float(self.x[0]), 6),
            "estimated_lon": round(float(self.x[1]), 6),
            "estimated_alt": round(float(self.x[2]), 2),
            "innovation_residual": round(mahalanobis, 4),
            "status": status
        }


if __name__ == "__main__":
    ekf = UAVKalmanFilter(dt=0.5)

    normal_readings = [
        (28.6139, 77.2090, 400, 18.0, 90),
        (28.6139, 77.2091, 401, 18.1, 90),
        (28.6139, 77.2092, 401, 17.9, 91),
        (28.6140, 77.2093, 402, 18.0, 90),
        (28.6140, 77.2094, 401, 18.2, 89),
    ]

    spoofed_reading = (28.6200, 77.2200, 450, 5.0, 45)

    print("=== Normal flight readings ===")
    for i, (lat, lon, alt, speed, heading) in enumerate(normal_readings):
        result = ekf.step(lat, lon, alt, speed, heading)
        print(f"Reading {i+1}: residual = {result['innovation_residual']}  ->  {result['status']}")

    print("\n=== Spoofed GPS reading ===")
    result = ekf.step(*spoofed_reading)
    print(f"Spoofed : residual = {result['innovation_residual']}  ->  {result['status']}")
    print("\nFull output:")
    print(json.dumps(result, indent=2))