import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from train_trout_vlm import MODEL, REVISION
from trout_feature_memory import FeatureMemory, file_sha, training_memory_rows, class_prototypes


class FeatureMemoryTests(unittest.TestCase):
    def table(self):
        return pd.DataFrame({"scale_id": ["a", "b", "c", "d"], "fish_key": ["f1", "f1", "f2", "f3"],
            "path": ["a.png", "b.png", "c.png", "d.png"], "split": ["train", "train", "train", "validation"],
            "quality_gt": ["readable", "bad", "readable", "readable"], "quality_source": ["expert_direct"] * 4,
            "age4": [2, None, 3, 1], "age_label_source": ["expert_direct", "unlabeled", "expert_direct", "expert_direct"],
            "age_train_eligible": [True, False, True, False]})

    def test_separate_quality_age_train_only(self):
        rows = training_memory_rows(self.table())
        self.assertEqual(rows.scale_id.tolist(), ["a", "b", "c"])
        self.assertEqual(rows.age_class.tolist(), [2, -1, 3])
        self.assertEqual(rows.quality_class.tolist(), [0, 1, 0])
        self.assertNotIn("d", rows.scale_id.tolist())

    def test_split_leakage_rejected(self):
        table = self.table()
        table.loc[3, "fish_key"] = "f1"
        with self.assertRaises(ValueError):
            training_memory_rows(table)

    def test_prototypes_balance_fish_not_image_count(self):
        rows = pd.DataFrame({"fish_key": ["f1"] * 9 + ["f2"], "quality_class": [0] * 10, "age_class": [2] * 10})
        vectors = np.asarray([[1., 0.]] * 9 + [[0., 1.]])
        proto = class_prototypes(rows, vectors)["age_2"]
        self.assertAlmostEqual(proto[0], proto[1])

    def test_memory_checksums_query_and_same_fish_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            table = self.table()
            for i in table.index:
                path = root / f"image_{i}.png"
                path.write_bytes(b"fixture image")
                table.loc[i, "path"] = str(path)
            labels = root / "labels.csv"
            table.to_csv(labels, index=False)
            rows = training_memory_rows(table)
            entries = rows.to_dict("records")
            for row in entries:
                row["image_sha256"] = file_sha(row["path"])
            memory = root / "memory"
            memory.mkdir()
            (memory / "entries.jsonl").write_text("\n".join(json.dumps(r) for r in entries))
            np.save(memory / "features.npy", np.asarray([[1., 0.], [.9, .1], [0., 1.]], dtype=np.float32))
            hashes = {"quality": "q", "age": "a"}
            meta = {"labels_sha256": file_sha(labels), "model": MODEL, "revision": REVISION,
                    "adapter_hashes": hashes, "pixels": 200704,
                    "artifact_hashes": {name: file_sha(memory / name) for name in ["features.npy", "entries.jsonl"]}}
            (memory / "memory_config.json").write_text(json.dumps(meta))
            index = FeatureMemory(memory, labels, hashes)
            hits = index.search([1., 0.], "quality", top_k=2)
            self.assertEqual([h["fish_key"] for h in hits], ["f1", "f2"])
            self.assertEqual(index.search([1., 0.], "age", exclude_fish="f1")[0]["class"], 3)
            self.assertEqual(len(index.search([1., 0.], "quality", exclude_fish="f1")), 1)
            (memory / "entries.jsonl").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                FeatureMemory(memory, labels, hashes)


if __name__ == "__main__":
    unittest.main()
