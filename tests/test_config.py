import importlib
import os
import shutil
import unittest
from pathlib import Path


class ConfigTests(unittest.TestCase):
    def test_get_settings_loads_values_from_dotenv_file(self):
        tmp_dir = Path("tests/.tmp_config_env")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        try:
            env_file = tmp_dir / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "ANTHROPIC_API_KEY=test-key",
                        "ANTHROPIC_BASE_URL=https://anthropic.example.com",
                        "ANTHROPIC_MODEL=claude-test-model",
                        "TITLE_AGENT_MODEL=claude-title-model",
                        "DATABASE_URL=postgresql://user:pass@localhost:5432/astral",
                        "MEMORY_WINDOW_SIZE=8",
                        "MEMORY_SUMMARY_TRIGGER=12",
                    ]
                ),
                encoding="utf-8",
            )

            old_cwd = os.getcwd()
            old_environ = os.environ.copy()

            try:
                os.environ.clear()
                os.chdir(tmp_dir)

                config_module = importlib.import_module("app.core.config")
                importlib.reload(config_module)
                config_module.get_settings.cache_clear()

                settings = config_module.get_settings()
            finally:
                os.chdir(old_cwd)
                os.environ.clear()
                os.environ.update(old_environ)
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

        self.assertEqual(settings.anthropic_api_key, "test-key")
        self.assertEqual(settings.anthropic_base_url, "https://anthropic.example.com")
        self.assertEqual(settings.anthropic_model, "claude-test-model")
        self.assertEqual(settings.title_agent_model, "claude-title-model")
        self.assertEqual(settings.database_url, "postgresql://user:pass@localhost:5432/astral")
        self.assertEqual(settings.memory_window_size, 8)
        self.assertEqual(settings.memory_summary_trigger, 12)

    def test_validate_settings_rejects_missing_anthropic_api_key(self):
        config_module = importlib.import_module("app.core.config")
        importlib.reload(config_module)

        settings = config_module.Settings(
            anthropic_api_key="",
            anthropic_base_url=None,
            anthropic_model="claude-test-model",
            title_agent_model=None,
            database_url="postgresql://user:pass@localhost:5432/astral",
            memory_window_size=8,
            memory_summary_trigger=12,
        )

        with self.assertRaises(config_module.ConfigurationError) as context:
            config_module.validate_settings(settings)

        self.assertIn("ANTHROPIC_API_KEY", str(context.exception))

    def test_validate_settings_rejects_invalid_database_url(self):
        config_module = importlib.import_module("app.core.config")
        importlib.reload(config_module)

        settings = config_module.Settings(
            anthropic_api_key="test-key",
            anthropic_base_url=None,
            anthropic_model="claude-test-model",
            title_agent_model=None,
            database_url="sqlite:///tmp.db",
            memory_window_size=8,
            memory_summary_trigger=12,
        )

        with self.assertRaises(config_module.ConfigurationError) as context:
            config_module.validate_settings(settings)

        self.assertIn("DATABASE_URL", str(context.exception))

    def test_validate_settings_rejects_invalid_memory_limits(self):
        config_module = importlib.import_module("app.core.config")
        importlib.reload(config_module)

        settings = config_module.Settings(
            anthropic_api_key="test-key",
            anthropic_base_url=None,
            anthropic_model="claude-test-model",
            title_agent_model=None,
            database_url="postgresql://user:pass@localhost:5432/astral",
            memory_window_size=12,
            memory_summary_trigger=12,
        )

        with self.assertRaises(config_module.ConfigurationError) as context:
            config_module.validate_settings(settings)

        self.assertIn("MEMORY_SUMMARY_TRIGGER", str(context.exception))

    def test_validate_settings_rejects_missing_anthropic_model(self):
        config_module = importlib.import_module("app.core.config")
        importlib.reload(config_module)

        settings = config_module.Settings(
            anthropic_api_key="test-key",
            anthropic_base_url=None,
            anthropic_model="",
            title_agent_model="title-model",
            database_url="postgresql://user:pass@localhost:5432/astral",
            memory_window_size=8,
            memory_summary_trigger=12,
        )

        with self.assertRaises(config_module.ConfigurationError) as context:
            config_module.validate_settings(settings)

        self.assertIn("ANTHROPIC_MODEL", str(context.exception))

    def test_validate_settings_rejects_invalid_anthropic_base_url(self):
        config_module = importlib.import_module("app.core.config")
        importlib.reload(config_module)

        settings = config_module.Settings(
            anthropic_api_key="test-key",
            anthropic_base_url="invalid-url",
            anthropic_model="claude-test-model",
            title_agent_model="title-model",
            database_url="postgresql://user:pass@localhost:5432/astral",
            memory_window_size=8,
            memory_summary_trigger=12,
        )

        with self.assertRaises(config_module.ConfigurationError) as context:
            config_module.validate_settings(settings)

        self.assertIn("ANTHROPIC_BASE_URL", str(context.exception))


if __name__ == "__main__":
    unittest.main()
