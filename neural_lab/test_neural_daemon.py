"""
Unit tests for neural_daemon.py FastTokenStoppingCriteria
Covers:
1. Simple JSON top-level termination
2. Nested JSON structures
3. JSON with escaped quotes and braces inside string values
4. Markdown wrapped JSON
5. Tool synthesis JSON with JS function code
6. Standalone discrete VLA action (shield_block)
7. Combat discrete VLA action with target (kite_combat creeper)
8. Equip action with item and slot (must not stop before slot)
9. Drop item with quantity (no short-circuiting)
10. Mine block with block parameter
11. Conversational text (no premature cutoff)
12. VLA speech with braces (stop_on_json=False)
"""

import os
import sys
import unittest
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from neural_daemon import FastTokenStoppingCriteria, ALL_MINECRAFT_SPECIAL_TOKENS

class TestFastTokenStoppingCriteria(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct', local_files_only=True)
        common_tokens = [
            "{", "}", '{"', '{\n', '}\n', '}\n\n', '"', '":', '": "', '",', ':', ',', ' ', '\n',
            "true", "false", "null", "<|im_end|>", "<|im_start|>", "<|endoftext|>"
        ]
        for s in common_tokens + ALL_MINECRAFT_SPECIAL_TOKENS:
            try:
                for tid in cls.tok.encode(s, add_special_tokens=False):
                    if tid not in FastTokenStoppingCriteria._global_token_str_cache:
                        FastTokenStoppingCriteria._global_token_str_cache[tid] = cls.tok.decode([tid], skip_special_tokens=False)
            except Exception:
                pass

    def _eval_sequence(self, text, stop_on_json=True, stop_on_action=True):
        token_ids = self.tok.encode(text, add_special_tokens=False)
        crit = FastTokenStoppingCriteria(self.tok, prompt_len=0, stop_on_json=stop_on_json, stop_on_action=stop_on_action)
        for i in range(1, len(token_ids) + 1):
            if crit(torch.tensor([token_ids[:i]])):
                return i, self.tok.decode(token_ids[:i])
        return None, text

    def test_simple_json(self):
        idx, stopped = self._eval_sequence('{"action": "mine", "block": "stone"} trailing notes')
        self.assertIsNotNone(idx)
        self.assertTrue(stopped.endswith('stone"}'))

    def test_nested_json(self):
        idx, stopped = self._eval_sequence('{"thought": "test", "plan": [{"step": 1, "action": "craft"}]} trailing chat')
        self.assertIsNotNone(idx)
        self.assertTrue(stopped.endswith('"craft"}]}'))

    def test_escaped_quotes_and_braces(self):
        idx, stopped = self._eval_sequence('{"text": "A string with \\" quotes and { braces } inside"} trailing')
        self.assertIsNotNone(idx)
        self.assertTrue(stopped.endswith('inside"}'))

    def test_markdown_json(self):
        idx, stopped = self._eval_sequence('```json\n{"action": "craft"}\n```\nExtra notes here')
        self.assertIsNotNone(idx)
        self.assertIn('{"action": "craft"}', stopped)
        self.assertNotIn('Extra notes', stopped)

    def test_tool_synthesis_js_code(self):
        js_json = '{\n  "description": "dance",\n  "functionBodyCode": "async function dance(bot) { if (true) { bot.jump(); } }",\n  "tags": ["dance"]\n} trailing'
        idx, stopped = self._eval_sequence(js_json)
        self.assertIsNotNone(idx)
        self.assertTrue(stopped.endswith('["dance"]\n}'))

    def test_standalone_action(self):
        idx, stopped = self._eval_sequence('<|act:shield_block|> trailing text')
        self.assertIsNotNone(idx)
        self.assertTrue(stopped.endswith('<|act:shield_block|>'))

    def test_combat_action_with_target(self):
        idx, stopped = self._eval_sequence('<|act:kite_combat|> <|target:creeper|> trailing text')
        self.assertIsNotNone(idx)
        self.assertTrue(stopped.endswith('<|target:creeper|>'))

    def test_equip_action_with_slot(self):
        idx, stopped = self._eval_sequence('<|act:equip|> <|item:iron_sword|> <|slot:mainhand|> trailing text')
        self.assertIsNotNone(idx)
        self.assertTrue(stopped.endswith('<|slot:mainhand|>'))

    def test_drop_item_with_qty(self):
        idx, stopped = self._eval_sequence('<|intent:hybrid|> Sure! <|act:drop_item|> <|item:torch|> <|qty:5|> trailing')
        self.assertIsNotNone(idx)
        self.assertTrue(stopped.endswith('<|qty:5|>'))

    def test_mine_block_with_block(self):
        idx, stopped = self._eval_sequence('<|act:mine_block|> <|block:iron_ore|> trailing')
        self.assertIsNotNone(idx)
        self.assertTrue(stopped.endswith('<|block:iron_ore|>'))

    def test_conversational_text_no_premature_cutoff(self):
        chat_text = (
            "Minecraft is a 2011 sandbox video game created by Markus 'Notch' Persson and released by Mojang Studios. "
            "In Minecraft, players explore a blocky, procedurally generated 3D world with virtually infinite terrain, "
            "and may discover and extract raw materials, craft tools and items, and build structures or earthworks. "
            "Depending on the game mode, players can fight computer-controlled mobs, as well as cooperate with or compete "
            "against other players in the same world."
        )
        idx, _ = self._eval_sequence(chat_text, stop_on_json=True, stop_on_action=True)
        self.assertIsNone(idx, "Conversational text should not stop prematurely")

    def test_vla_speech_with_braces(self):
        vla_speech = "I found {3} diamonds in the cavern! Let me know if you want them."
        idx, _ = self._eval_sequence(vla_speech, stop_on_json=False, stop_on_action=True)
        self.assertIsNone(idx, "VLA speech with braces should not stop when stop_on_json=False")

if __name__ == '__main__':
    unittest.main()
