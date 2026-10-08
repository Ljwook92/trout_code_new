import unittest
import random

import pandas as pd
from PIL import Image

from prepare_trout_training_labels import prepare_labels
from train_trout_vlm import select_rows, augment_image, transform_image


class VLMTrainingTests(unittest.TestCase):
    def test_validation_and_disabled_training_unchanged(self):
        image = Image.new("RGB", (40, 20), (210, 200, 180))
        for split in ["validation", "test"]:
            self.assertIs(augment_image(image, split, enabled=True), image)
        self.assertIs(augment_image(image, "train", enabled=False), image)

    def test_augmentation_reproducible_and_frame_preserved(self):
        image = Image.new("RGB", (40, 20), (210, 200, 180))
        image.putpixel((20, 10), (0, 0, 0))
        a = augment_image(image, "train", True, rng=random.Random(100))
        b = augment_image(image, "train", True, rng=random.Random(100))
        self.assertEqual(a.size, b.size)
        self.assertEqual(a.tobytes(), b.tobytes())
        rotated = transform_image(image, angle=15)
        self.assertGreater(rotated.width, image.width)
        self.assertGreater(rotated.height, image.height)
        self.assertEqual(rotated.getpixel((0, 0)), (210, 200, 180))

    def test_fixed_perturbations_and_invalid_parameters(self):
        image = Image.new("RGB", (40, 20), (100, 100, 100))
        self.assertEqual(transform_image(image, brightness=0.8).getpixel((5, 5)), (80, 80, 80))
        self.assertEqual(transform_image(image, angle=90).size, (20, 40))
        with self.assertRaises(ValueError):
            transform_image(image, contrast=0)
        with self.assertRaises(ValueError):
            transform_image(image, angle=float("nan"))

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
