import unittest
import numpy as np
from neural_net_from_scratch import main

class TestMain(unittest.TestCase):
    def test_main_passes(self):
        # Create a sample input
        inputs = np.array([1, 2, 3])
        expected_outputs = np.array([4, 5, 6])

        # Run the main function with the sample input
        outputs = main(inputs)

        # Check if the output matches the expected output
        np.testing.assert_array_equal(outputs, expected_outputs)

    def test_main_handles_invalid_input(self):
        # Test that the main function raises an error for invalid input
        with self.assertRaises(ValueError):
            main("not a numpy array")

if __name__ == "__main__":
    unittest.main()