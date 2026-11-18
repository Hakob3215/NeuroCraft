import os
import sys
import asyncio
import unittest
import torch
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

sys.path.insert(0, os.path.dirname(__file__))
from neural_daemon import NeuralStackServer

# Initialize once across all test methods
shared_server = None

class TestNeuralDaemonEndpoints(AioHTTPTestCase):
    @classmethod
    def setUpClass(cls):
        global shared_server
        if shared_server is None:
            shared_server = NeuralStackServer(slm_model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct", load_slm=True)
            shared_server.initialize_models()
        super().setUpClass()

    async def get_application(self):
        self.server = shared_server
        app = web.Application()
        app.router.add_get("/health", self.server.handle_health)
        app.router.add_post("/reflex", self.server.handle_reflex)
        app.router.add_post("/chunk_scan", self.server.handle_chunk_scan)
        app.router.add_post("/llm/chat", self.server.handle_llm_chat)
        app.router.add_post("/vla/predict", self.server.handle_vla_predict)
        app.router.add_post("/dqn_sparring/predict", self.server.handle_dqn_sparring_predict)
        app.router.add_post("/pause_slm", self.server.handle_pause_slm)
        app.router.add_post("/reload_adapter", self.server.handle_reload_adapter)
        return app

    async def test_01_health(self):
        resp = await self.client.request("GET", "/health")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("vram_allocated_mb", data)
        print("✓ Health check endpoint verified:", data["device"])

    async def test_02_reflex(self):
        payload = {
            "voxels": [0] * 3375,
            "entities": [{"name": "creeper", "rx": 2.0, "ry": 0.0, "rz": 1.0, "distance": 2.2, "isHostile": True}],
            "bot_state": {"health": 20.0, "food": 20.0, "onGround": True}
        }
        resp = await self.client.request("POST", "/reflex", json=payload)
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertIn("aim_delta", data)
        self.assertIn("actions", data)
        self.assertIn("hazard_urgency", data)
        print(f"✓ Reflex endpoint verified (latency: {data['latency_ms']}ms)")

    async def test_03_chunk_scan(self):
        payload = {"sparse_blocks": [{"dx": 0, "dy": -1, "dz": 0, "name": "stone"}]}
        resp = await self.client.request("POST", "/chunk_scan", json=payload)
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertIn("recommended_build_offset", data)
        self.assertIn("flatness_confidence", data)
        print(f"✓ Chunk scan endpoint verified (latency: {data['latency_ms']}ms)")

    async def test_04_llm_chat(self):
        payload = {
            "system_prompt": "You are a helpful Minecraft AI assistant. Respond with valid JSON.",
            "user_prompt": "Respond with {\"action\": \"idle\"}",
            "max_new_tokens": 40
        }
        resp = await self.client.request("POST", "/llm/chat", json=payload)
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertIn("content", data)
        self.assertIn("tokens_generated", data)
        self.assertIn("tokens_per_sec", data)
        print(f"✓ LLM chat endpoint verified: generated {data['tokens_generated']} tokens in {data['latency_ms']}ms ({data['tokens_per_sec']} tok/s)")

    async def test_05_vla_predict(self):
        payload = {
            "system_prompt": "You are NeuroAgent companion. Respond using native action tokens.",
            "user_prompt": "Look out, creeper nearby!",
            "proprioception": [0.0] * 24,
            "max_new_tokens": 30
        }
        resp = await self.client.request("POST", "/vla/predict", json=payload)
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertIn("motor_controls", data)
        self.assertIn("intent", data)
        mc = data["motor_controls"]
        self.assertIn("delta_yaw", mc)
        self.assertIn("action_chunk", mc)
        self.assertEqual(len(mc["action_chunk"]), 8)
        print(f"✓ VLA predict verified: 8-tick continuous motor chunk generated in {data['latency_ms']}ms")

    async def test_06_pause_and_reload(self):
        resp_pause = await self.client.request("POST", "/pause_slm")
        self.assertEqual(resp_pause.status, 200)
        data_pause = await resp_pause.json()
        self.assertEqual(data_pause["status"], "paused")
        print(f"✓ Pause SLM verified: VRAM allocated = {data_pause['vram_allocated_mb']} MiB")

        resp_reload = await self.client.request("POST", "/reload_adapter")
        self.assertEqual(resp_reload.status, 200)
        data_reload = await resp_reload.json()
        self.assertEqual(data_reload["status"], "ok")
        print(f"✓ Reload adapter verified: SLM device = {data_reload['slm_device']}, VRAM = {data_reload['vram_allocated_mb']} MiB")

if __name__ == '__main__':
    unittest.main()
