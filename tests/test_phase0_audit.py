"""Synthetic structural tests only; these are not the missing G1 diagnostics."""
import csv
import importlib.util
import io
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/phase0/audit_renge_assets.py"
SPEC = importlib.util.spec_from_file_location("audit_renge", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def run_fixture():
    rows = []
    for n, (gsm, (day, library)) in enumerate(audit.SAMPLES.items()):
        for run in range(2):
            row = {key: "" for key in audit.ENA_FIELDS.split(",")}
            row.update(run_accession=f"SRR{n * 2 + run}", library_name=gsm,
                       study_accession="PRJNA879051", read_count="10", base_count="910",
                       experiment_title=f"{gsm}: cDNA library from {library} of hiPSC at day{day} after transduction")
            rows.append(row)
    return rows


def encode(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=audit.ENA_FIELDS.split(","), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


class RunMetadataTests(unittest.TestCase):
    def test_resolves_both_library_types(self):
        rows = audit.parse_runs(encode(run_fixture()))
        self.assertEqual(sum(r["library_type"] == "mRNA" for r in rows), 8)
        self.assertEqual(sum(r["library_type"] == "gRNA" for r in rows), 8)

    def test_rejects_title_sample_mismatch(self):
        rows = run_fixture()
        rows[0]["experiment_title"] = rows[0]["experiment_title"].replace("mRNA", "gRNA")
        with self.assertRaises(ValueError):
            audit.parse_runs(encode(rows))

    def test_rejects_duplicate_run(self):
        rows = run_fixture()
        rows[1]["run_accession"] = rows[0]["run_accession"]
        with self.assertRaises(ValueError):
            audit.parse_runs(encode(rows))


class MatrixTests(unittest.TestCase):
    features = [["ENSG1", "GENE", "Gene Expression"], ["g1", "CTRL_1", "CRISPR Guide Capture"]]

    def test_valid_sparse_counts(self):
        matrix = coo_matrix(np.array([[1, 2], [0, 3]], dtype=np.int64))
        self.assertEqual(audit.validate_matrix(matrix, self.features, ["a", "b"]).nnz, 3)

    def test_rejects_duplicate_barcodes(self):
        with self.assertRaises(ValueError):
            audit.validate_matrix(coo_matrix(np.eye(2, dtype=int)), self.features, ["a", "a"])

    def test_rejects_negative_counts(self):
        with self.assertRaises(ValueError):
            audit.validate_matrix(coo_matrix(np.array([[1, -2], [0, 3]])), self.features, ["a", "b"])

    def test_rejects_duplicate_coordinates(self):
        matrix = coo_matrix(([1, 2], ([0, 0], [0, 0])), shape=(2, 2))
        with self.assertRaises(ValueError):
            audit.validate_matrix(matrix, self.features, ["a", "b"])

    def test_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            audit.validate_matrix(coo_matrix(np.ones((2, 3), dtype=int)), self.features, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
