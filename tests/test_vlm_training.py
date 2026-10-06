import unittest

import pandas as pd

from prepare_trout_training_labels import prepare_labels
from train_trout_vlm import select_rows


class VLMTrainingTests(unittest.TestCase):
    def setUp(self):
        master = pd.DataFrame({"scale_id": list("abcdef"), "fish_key": ["f1", "f1", "f2", "f3", "f4", "f5"],
                               "path": ["image.png"] * 6, "label": [1, None, 6, 2, 6, 3]})
        manifest = pd.DataFrame({"scale_id": list("acdef"), "fish_key": ["f1", "f2", "f3", "f4", "f5"],
                                 "split": ["train", "train", "validation", "validation", "test"]})
        quality = pd.DataFrame({"scale_id": ["b"], "quality_gt": ["readable"]})
        self.rows = prepare_labels(master, manifest, quality)

    def test_age_includes_propagation_only_for_training(self):
        train, val = select_rows(self.rows, "age")
        self.assertEqual(set(train.scale_id), {"a", "b"})
        self.assertEqual(set(val.scale_id), {"d"})
        self.assertEqual(train.answer.iloc[0], '{"prediction": 1}')

    def test_quality_uses_bad_and_readable_not_test(self):
        train, val = select_rows(self.rows, "quality")
        self.assertEqual(set(train.scale_id), {"a", "b", "c"})
        self.assertEqual(set(val.scale_id), {"d", "e"})
        self.assertNotIn("f", set(train.scale_id) | set(val.scale_id))

    def test_invalid_flags_fail(self):
        self.rows["age_train_eligible"] = "False-ish"
        with self.assertRaises(ValueError):
            select_rows(self.rows, "age")


if __name__ == "__main__":
    unittest.main()
