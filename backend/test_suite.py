import unittest
import json
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import COLLECTION_NAME, PDF_PATH, CHROMA_DB_PATH
from backend.retriever import get_retriever
from backend.llm_service import is_small_talk, build_system_prompt, get_llm_service

class TestAcrossAIRagSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.retriever = get_retriever()
        cls.llm_service = get_llm_service()

    def test_01_paths_and_database(self):
        """Verify knowledge document exists and ChromaDB is populated with 800+ clean chunks."""
        self.assertTrue(PDF_PATH.exists(), f"Handbook PDF missing at {PDF_PATH}")
        self.assertTrue(CHROMA_DB_PATH.exists(), f"ChromaDB missing at {CHROMA_DB_PATH}")
        count = self.retriever.collection.count()
        self.assertGreaterEqual(count, 800, f"Expected >= 800 chunks, found {count}")

    def test_02_health_endpoint(self):
        """Verify /api/health reports online status without backend telemetry."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["app_name"], "Across AI")
        self.assertEqual(data["version"], "2.0.0")
        self.assertNotIn("database", data)
        self.assertNotIn("llm", data)

    def test_03_handbook_stats_endpoint(self):
        """Verify /api/handbook/stats reports 12 chapters, 26 forms, and 30 FAQs."""
        res = self.client.get("/api/handbook/stats")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["chapters_count"], 12)
        self.assertEqual(data["forms_count"], 26)
        self.assertEqual(data["faqs_count"], 30)

    def test_04_hybrid_retriever_exact_codes(self):
        """Verify hybrid retriever boosts exact codes: HR-03-02, HR-F07, Band D, USD 50."""
        # Query 1: Notice period for Band D
        results_band = self.retriever.retrieve("What is the notice period for Band D?", top_k=3)
        self.assertTrue(any("Band" in r["content"] or "notice" in r["content"].lower() for r in results_band))

        # Query 2: Form HR-F07
        results_f07 = self.retriever.retrieve("HR-F07", top_k=2)
        self.assertTrue(any("HR-F07" in r["metadata"].get("policy", "") or "HR-F07" in r["content"] for r in results_f07))

        # Query 3: Policy HR-03-02
        results_0302 = self.retriever.retrieve("HR-03-02", top_k=2)
        self.assertTrue(any("HR-03-02" in r["content"] or "HR-03-02" in r["metadata"].get("policy", "") for r in results_0302))

        # Query 4: Gift limit USD 50
        results_gift = self.retriever.retrieve("What is the gift limit in USD 50?", top_k=2)
        self.assertTrue(any("50" in r["content"] for r in results_gift))

    def test_05_small_talk_detection(self):
        """Verify Mode 1 small talk classifier identifies greetings correctly."""
        self.assertTrue(is_small_talk("hi"))
        self.assertTrue(is_small_talk("hello there"))
        self.assertTrue(is_small_talk("good morning"))
        self.assertTrue(is_small_talk("how are you doing?"))
        self.assertTrue(is_small_talk("who are you"))
        self.assertTrue(is_small_talk("thank you"))
        
        # Policy queries should not be small talk
        self.assertFalse(is_small_talk("What is the notice period for Band D?"))
        self.assertFalse(is_small_talk("How do I apply for annual leave using HR-F01?"))
        self.assertFalse(is_small_talk("Can I carry forward unused annual leave?"))

    def test_06_system_prompt_builder(self):
        """Verify system prompt accurately places context into <reference_context> without handbook/PDF references."""
        prompt = build_system_prompt(context="MGI Policy 1.1 Test Context", small_talk=False)
        self.assertIn("You are Across AI, the official, intelligent HR Assistant", prompt)
        self.assertIn("<reference_context>", prompt)
        self.assertIn("MGI Policy 1.1 Test Context", prompt)
        self.assertNotIn("<handbook_context>", prompt)

    def test_07_chat_endpoint_mode1_greeting(self):
        """Verify Mode 1 greeting returns natural reply without citing policy text."""
        res = self.client.post("/api/chat", json={"message": "Hello!", "history": []})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["is_small_talk"])
        self.assertEqual(len(data.get("citations", [])), 0)
        self.assertGreater(len(data["reply"]), 0)

    def test_08_chat_endpoint_mode2_policy(self):
        """Verify Mode 2 policy question returns structured answer without citations or telemetry."""
        res = self.client.post("/api/chat", json={
            "message": "What is the notice period for Band D?",
            "history": []
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["is_small_talk"])
        self.assertEqual(len(data.get("citations", [])), 0)
        self.assertGreater(len(data["reply"]), 0)

    def test_09_chat_stream_endpoint(self):
        """Verify SSE streaming endpoint emits tokens and done signal without citations."""
        events = []
        with self.client.stream("POST", "/api/chat/stream", json={
            "message": "What is the gift limit?",
            "history": []
        }) as response:
            self.assertEqual(response.status_code, 200)
            for line in response.iter_lines():
                if line and line.startswith("data: "):
                    events.append(json.loads(line[6:]))

        self.assertGreater(len(events), 0)
        event_types = [e.get("type") for e in events]
        self.assertNotIn("citations", event_types)
        self.assertIn("token", event_types)
        self.assertIn("done", event_types)

    def test_10_frontend_index_served(self):
        """Verify GET / returns 200 with HTML frontend."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers.get("content-type", ""))
        self.assertIn("Across AI", res.text)
        self.assertIn("MGI HR INTELLIGENCE", res.text)

if __name__ == "__main__":
    unittest.main()
