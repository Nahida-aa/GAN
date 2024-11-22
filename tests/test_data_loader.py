import unittest
from src.data.data_loader import get_data_loader

class TestDataLoader(unittest.TestCase):
    def test_data_loader(self):
        data_loader = get_data_loader(batch_size=64)
        self.assertIsNotNone(data_loader)
        for images, labels in data_loader:
            self.assertEqual(images.shape[0], 64)
            break

if __name__ == '__main__':
    unittest.main()