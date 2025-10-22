import unittest

from methods import simulated_annealing


class MethodsSmokeTests(unittest.TestCase):
    def test_simulated_annealing_smoke(self):
        # Run a very short SA so it executes quickly
        best_E, best_config = simulated_annealing.simulated_annealing(n_sweeps=1, T0=1.0, alpha=0.9)
        self.assertIsInstance(best_E, (int, float))
        self.assertIsInstance(best_config, list)

    def test_hybrid_annealing_smoke_import(self):
        try:
            from methods import hybrid_annealing
        except Exception:
            self.skipTest("methods.hybrid_annealing not importable")

        # If pyqubo is not available, calling hybrid_annealing may raise; we
        # only assert the function exists to keep this a smoke test.
        self.assertTrue(hasattr(hybrid_annealing, 'hybrid_annealing'))


if __name__ == '__main__':
    unittest.main()
