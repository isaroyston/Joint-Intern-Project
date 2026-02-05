"""
Unit tests for Prompt Attack Dataset
"""

import unittest
import tempfile
import os
from src.datasets import PromptAttackDataset


class TestPromptAttackDataset(unittest.TestCase):
    """Test cases for PromptAttackDataset class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.dataset = PromptAttackDataset()
    
    def test_initialization(self):
        """Test dataset initialization"""
        self.assertIsNotNone(self.dataset)
        self.assertTrue(len(self.dataset.get_all_attacks()) > 0)
    
    def test_get_all_attacks(self):
        """Test getting all attacks"""
        attacks = self.dataset.get_all_attacks()
        self.assertIsInstance(attacks, list)
        self.assertTrue(len(attacks) >= 15)
    
    def test_get_attacks_by_category(self):
        """Test filtering attacks by category"""
        attacks = self.dataset.get_attacks_by_category("instruction_override")
        self.assertIsInstance(attacks, list)
        for attack in attacks:
            self.assertEqual(attack['category'], "instruction_override")
    
    def test_get_attacks_by_severity(self):
        """Test filtering attacks by severity"""
        high_severity = self.dataset.get_attacks_by_severity("high")
        self.assertIsInstance(high_severity, list)
        for attack in high_severity:
            self.assertEqual(attack['severity'], "high")
    
    def test_get_categories(self):
        """Test getting all categories"""
        categories = self.dataset.get_categories()
        self.assertIsInstance(categories, list)
        self.assertTrue(len(categories) > 0)
    
    def test_save_and_load(self):
        """Test saving and loading dataset"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            self.dataset.save_to_file(filepath)
            self.assertTrue(os.path.exists(filepath))
            
            new_dataset = PromptAttackDataset()
            new_dataset.load_from_file(filepath)
            
            self.assertEqual(
                len(new_dataset.get_all_attacks()),
                len(self.dataset.get_all_attacks())
            )
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_add_custom_attack(self):
        """Test adding custom attack"""
        initial_count = len(self.dataset.get_all_attacks())
        
        custom_attack = {
            "id": "custom_1",
            "name": "Custom Attack",
            "category": "test_category",
            "severity": "low",
            "prompt": "Test prompt",
            "description": "Test description"
        }
        
        self.dataset.add_custom_attack(custom_attack)
        self.assertEqual(len(self.dataset.get_all_attacks()), initial_count + 1)


if __name__ == '__main__':
    unittest.main()
