import unittest

import pandas as pd

from prepare_trout_training_labels import prepare_labels


class TrainingLabelTests(unittest.TestCase):
    def tables(self):
        master = pd.DataFrame({"scale_id": list("abcdefg"), "fish_key": ["f1"] * 4 + ["f2"] * 3,
                               "path": ["image.png"] * 7, "label": [2, None, 6, None, 1, 2, None],
                               "split": ["labeled"] * 7})
        manifest = pd.DataFrame({"scale_id": ["a", "e", "f"], "fish_key": ["f1", "f2", "f2"],
                                 "split": ["train"] * 3})
        return master, manifest

    def test_quality_never_propagates_and_conflicts_block_age(self):
        master, manifest = self.tables()
        quality = pd.DataFrame({"scale_id": ["b", "g"], "quality_gt": ["readable"] * 2})
        rows = prepare_labels(master, manifest, quality).set_index("scale_id")
        self.assertEqual(rows.loc["b", "age_label"], 2)
        self.assertEqual(rows.loc["b", "age_label_source"], "fish_propagated")
        self.assertEqual(rows.loc["c", "quality_gt"], "bad")
        self.assertTrue(pd.isna(rows.loc["c", "age_label"]))
        self.assertEqual(rows.loc["d", "quality_gt"], "unknown")
        self.assertTrue(pd.isna(rows.loc["d", "age_label"]))
        self.assertTrue(pd.isna(rows.loc["g", "age_label"]))
        self.assertFalse(rows.loc["e", "age_train_eligible"])

    def test_evaluation_direct_only(self):
        master, manifest = self.tables()
        manifest["split"] = "test"
        quality = pd.DataFrame({"scale_id": ["b"], "quality_gt": ["readable"]})
        rows = prepare_labels(master, manifest, quality).set_index("scale_id")
        self.assertTrue(rows.loc["a", "age_evaluation_eligible"])
        self.assertFalse(rows.loc["b", "age_evaluation_eligible"])
        self.assertFalse(rows.age_train_eligible.any())

    def test_leakage_and_quality_conflict_rejected(self):
        master, manifest = self.tables()
        leak = pd.concat([manifest, pd.DataFrame({"scale_id": ["b"], "fish_key": ["f1"], "split": ["test"]})])
        with self.assertRaises(ValueError):
            prepare_labels(master, leak)
        with self.assertRaises(ValueError):
            prepare_labels(master, manifest, pd.DataFrame({"scale_id": ["c"], "quality_gt": ["readable"]}))


if __name__ == "__main__":
    unittest.main()
