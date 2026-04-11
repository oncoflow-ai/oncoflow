from __future__ import annotations


class Config:
    def __init__(self, filename: str | None = None) -> None:
        self.config_file_name = filename
        self._main_options: dict[str, str] = {}

    def set_main_option(self, key: str, value: str) -> None:
        self._main_options[key] = value

    def get_main_option(self, key: str, default: str | None = None) -> str | None:
        return self._main_options.get(key, default)
