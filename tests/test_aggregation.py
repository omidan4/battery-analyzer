import unittest

from src import aggregation


class AggregationTests(unittest.TestCase):
    def test_parse_timestamp_with_subseconds(self):
        ts = aggregation.parse_timestamp("2025-06-03 18:56:10.123")
        self.assertEqual(ts.year, 2025)
        self.assertEqual(ts.second, 10)
        self.assertEqual(ts.microsecond, 123000)

    def test_aggregate_rows_to_seconds(self):
        rows = [
            {"timestamp": "2025-06-03 18:56:10.100", "value": 1.0},
            {"timestamp": "2025-06-03 18:56:10.900", "value": 3.0},
            {"timestamp": "2025-06-03 18:56:11.100", "value": 5.0},
        ]

        aggregates = aggregation.aggregate_rows_to_seconds(rows, numeric_columns=["value"])

        self.assertEqual(len(aggregates), 2)
        first = aggregates[0]
        self.assertEqual(first["timestamp"], "2025-06-03 18:56:10")
        self.assertEqual(first["raw_sample_count"], 2.0)
        self.assertEqual(first["value_mean"], 2.0)
        self.assertEqual(first["value_min"], 1.0)
        self.assertEqual(first["value_max"], 3.0)
        self.assertEqual(first["value_count"], 2.0)


if __name__ == "__main__":
    unittest.main()
