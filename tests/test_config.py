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
                        "LLM_PROVIDER=openai",
                        "LLM_API_KEY=test-key",
                        "LLM_BASE_URL=https://openai.example.com",
                        "LLM_MODEL=gpt-4o-mini",
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

        self.assertEqual(settings.chat_endpoint.provider, "openai")
        self.assertEqual(settings.chat_endpoint.api_key, "test-key")
        self.assertEqual(settings.chat_endpoint.base_url, "https://openai.example.com")
        self.assertEqual(settings.chat_endpoint.model, "gpt-4o-mini")

    def test_validate_settings_rejects_missing_chat_api_key(self):
        config_module = importlib.import_module("app.core.config")
        importlib.reload(config_module)

        settings = config_module.Settings(
            chat_endpoint=config_module.ModelEndpointSettings(
                provider="anthropic",
                api_key="",
                base_url=None,
                model="claude-test-model",
            ),
            database_url="postgresql://user:pass@localhost:5432/astral",
            memory_window_size=8,
            memory_summary_trigger=12,
        )

        with self.assertRaises(config_module.ConfigurationError) as context:
            config_module.validate_settings(settings)

        self.assertIn("LLM_API_KEY", str(context.exception))

    def test_validate_settings_rejects_invalid_database_url(self):
        config_module = importlib.import_module("app.core.config")
        importlib.reload(config_module)

        settings = config_module.Settings(
            chat_endpoint=config_module.ModelEndpointSettings(
                provider="anthropic",
                api_key="test-key",
                base_url=None,
                model="claude-test-model",
            ),
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
            chat_endpoint=config_module.ModelEndpointSettings(
                provider="anthropic",
                api_key="test-key",
                base_url=None,
                model="claude-test-model",
            ),
            database_url="postgresql://user:pass@localhost:5432/astral",
            memory_window_size=12,
            memory_summary_trigger=12,
        )

        with self.assertRaises(config_module.ConfigurationError) as context:
            config_module.validate_settings(settings)

        self.assertIn("MEMORY_SUMMARY_TRIGGER", str(context.exception))

    def test_validate_settings_rejects_invalid_provider(self):
        config_module = importlib.import_module("app.core.config")
        importlib.reload(config_module)

        settings = config_module.Settings(
            chat_endpoint=config_module.ModelEndpointSettings(
                provider="invalid-provider",
                api_key="test-key",
                base_url=None,
                model="claude-test-model",
            ),
            database_url="postgresql://user:pass@localhost:5432/astral",
            memory_window_size=8,
            memory_summary_trigger=12,
        )

        with self.assertRaises(config_module.ConfigurationError) as context:
            config_module.validate_settings(settings)

        self.assertIn("LLM_PROVIDER", str(context.exception))

    def test_validate_settings_rejects_invalid_chat_base_url(self):
        config_module = importlib.import_module("app.core.config")
        importlib.reload(config_module)

        settings = config_module.Settings(
            chat_endpoint=config_module.ModelEndpointSettings(
                provider="anthropic",
                api_key="test-key",
                base_url="invalid-url",
                model="claude-test-model",
            ),
            database_url="postgresql://user:pass@localhost:5432/astral",
            memory_window_size=8,
            memory_summary_trigger=12,
        )

        with self.assertRaises(config_module.ConfigurationError) as context:
            config_module.validate_settings(settings)

        self.assertIn("LLM_BASE_URL", str(context.exception))


if __name__ == "__main__":
    unittest.main()
