from locust import HttpUser, task, between
import json
import random


class DataInsightUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        response = self.client.post("/api/sessions/new")
        self.session_id = response.json()["session_id"]

    @task(3)
    def query_agent(self):
        queries = [
            "Show me the first 10 rows",
            "What is the correlation between price and area?",
            "Describe the dataset",
            "Find outliers in the price column",
            "Show summary statistics"
        ]

        self.client.get(
            "/api/agent/chat-stream",
            params={
                "message": random.choice(queries),
                "session_id": self.session_id,
                "web_search_enabled": "false"
            },
            name="/api/agent/chat-stream"
        )

    @task(1)
    def get_sessions(self):
        self.client.get("/api/sessions")

    @task(1)
    def upload_dataset(self):
        csv_content = """price,area,bedrooms
300000,1500,3
450000,2000,4
250000,1200,2"""

        files = {"file": ("test.csv", csv_content, "text/csv")}
        self.client.post(
            "/api/upload",
            files=files,
            data={"session_id": self.session_id}
        )
