"""Locust load test for deployed UAIS API (optional Gate P P15).

Usage (against running API):
  locust -f deploy/loadtest/locustfile.py --host http://127.0.0.1:8000
"""

from __future__ import annotations

import os

from locust import HttpUser, between, task


class UaisUser(HttpUser):
    wait_time = between(0.05, 0.2)
    api_key = os.getenv("UAIS_LOADTEST_API_KEY", "replace-me")

    @task(3)
    def health(self) -> None:
        self.client.get("/health")

    @task(2)
    def kga_health(self) -> None:
        self.client.get("/kga/health")

    @task(1)
    def kga_decide_proxy(self) -> None:
        self.client.post(
            "/decide",
            headers={"X-API-Key": self.api_key},
            json={
                "calib_scores": [0.1, 0.2, 0.15, 0.18, 0.12],
                "test_scores": [0.5, 0.55, 0.52, 0.48, 0.51],
                "alpha": 0.1,
                "cert_mode": "proxy",
            },
        )
