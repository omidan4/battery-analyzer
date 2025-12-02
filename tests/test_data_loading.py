import tempfile
import unittest
from pathlib import Path

from src import data_loading


class DataLoadingTests(unittest.TestCase):
    def _write_csv(self, tmp_dir: Path, content: str) -> Path:
        path = tmp_dir / "sample.csv"
        path.write_text(content, encoding="utf-8")
        return path

    def test_load_sensor_csv_handles_bom_and_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_text = (
                "sep=,\nTime,INV_DC_Bus_Voltage,INV_DC_Bus_Current\n"
                "2025-06-03 18:56:10,0.0400 Volts,10 Amps\n"
            )
            path = self._write_csv(tmp_path, csv_text)
            rules = data_loading.default_column_rules()

            rows = data_loading.load_sensor_csv(path, rules)

            self.assertEqual(
                rows,
                [
                    {
                        "timestamp": "2025-06-03 18:56:10",
                        "inv_dc_bus_voltage": 0.04,
                        "inv_dc_bus_current": 10.0,
                    }
                ],
            )

    def test_verbose_column_is_renamed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_text = (
                '"Time","sensorReading {messageName=""M165_Motor_Position_Info"", rawCAN=""165"", signalName=""INV_Motor_Speed""}"\n'
                '"2025-06-03 18:56:10","200"\n'
            )
            path = self._write_csv(tmp_path, csv_text)
            rules = data_loading.default_column_rules()

            rows = data_loading.load_sensor_csv(path, rules)

            self.assertEqual(rows[0]["inv_motor_speed"], 200.0)
            self.assertIn("timestamp", rows[0])

    def test_required_columns_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_text = "Time,INV_DC_Bus_Voltage\n,0.5\n"
            path = self._write_csv(tmp_path, csv_text)
            rules = data_loading.default_column_rules()

            with self.assertRaises(ValueError):
                data_loading.load_sensor_csv(path, rules)


if __name__ == "__main__":
    unittest.main()
