import unittest
import torch
from src.models.model import Generator, Discriminator

class TestModels(unittest.TestCase):
    def test_generator(self):
        model = Generator()
        input_tensor = torch.randn(1, 100)
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 1, 28, 28))

    def test_discriminator(self):
        model = Discriminator()
        input_tensor = torch.randn(1, 1, 28, 28)
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 1))

if __name__ == '__main__':
    unittest.main()