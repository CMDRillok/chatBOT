
from __future__ import annotations

import json
import queue
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Any

import sounddevice as sd
from vosk import Model as VoskModel, KaldiRecognizer, SetLogLevel

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button, Checkbox, DirectoryTree, Footer, Header, Input, Label,
    Select, Static, TabbedContent, TabPane, TextArea,
)

APP_DIR = Path.home() / ".config" / "voice-assistant"
CONFIG_PATH = APP_DIR / "config.json"
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000  # 250 ms at 16 kHz

SetLogLevel(-1)

DEFAULTS: dict[str, Any] = {
    "vosk_model_path": "",
    "microphone": "default",
    "auto_send": True,
    "memory_pairs": 3,
    "tts_enabled": True,
    "tts_model_path": "",
    "tts_output_device": "auto",
    "llm_model_path": "",
    "llm_context_size": 4096,
    "llm_temperature": 0.7,
    "llm_max_tokens": 512,
    "llm_gpu_layers": 20,
    "disable_thinking": True,
    "system_prompt": "Ты полезный ассистент. Отвечай на русском языке кратко и по существу.",
    "voice_commands": [
        {"phrases": ["открой браузер", "запусти браузер"], "command": "firefox", "description": "Открыть браузер Firefox"},
        {"phrases": ["открой телеграм", "запусти телеграм"], "command": "telegram-desktop", "description": "Открыть Telegram Desktop"},
    ],
}


def load_settings() -> dict[str, Any]:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return {**DEFAULTS, **data}
    except (OSError, json.JSONDecodeError):
        return DEFAULTS.copy()


def save_settings(settings: dict[str, Any]) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def resample_pcm16(data: bytes, source_rate: int, target_rate: int) -> bytes:
    import numpy as np

    samples = np.frombuffer(data, dtype=np.int16)
    if samples.size == 0:
        return b""
    length = max(1, int(samples.size * target_rate / source_rate))
    positions = np.linspace(0, samples.size - 1, length)
    resampled = np.interp(positions, np.arange(samples.size), samples).astype(np.int16)
    return resampled.tobytes()


def apply_no_think_handler(llm) -> bool:
    """Force the model to skip its reasoning block when the chat template supports it."""
    from llama_cpp.llama_chat_format import (
        Jinja2ChatFormatter,
        chat_formatter_to_chat_completion_handler,
    )

    template = llm.metadata.get("tokenizer.chat_template")
    if not template:
        return False
    eos_id = llm.token_eos()
    bos_id = llm.token_bos()
    eos_token = llm.detokenize([eos_id], special=True).decode("utf-8", "replace") if eos_id != -1 else ""
    bos_token = llm.detokenize([bos_id], special=True).decode("utf-8", "replace") if bos_id != -1 else ""
    formatter = Jinja2ChatFormatter(
        template=template,
        eos_token=eos_token,
        bos_token=bos_token,
        stop_token_ids=[eos_id] if eos_id != -1 else None,
    )

    def no_think_formatter(**kwargs):
        return formatter(enable_thinking=False, **kwargs)

    llm.chat_handler = chat_formatter_to_chat_completion_handler(no_think_formatter)
    return True


class FilePicker(ModalScreen[str | None]):
    """Simple local path picker. Enter a path or choose a file in the tree."""

    CSS = """
    FilePicker {
        align: center middle;
        background: $background 70%;
    }
    #picker-dialog {
        width: 90%;
        height: 90%;
        border: solid #919191;
        background: #1c1c1c;
        padding: 1 2;
    }
    #picker-title { height: 2; color: #f5f5f5; }
    #picker-path { height: 3; }
    #picker-tree { height: 1fr; border: solid #4a4a4a; }
    #picker-buttons { height: 3; margin-top: 1; }
    """

    def __init__(self, start_path: str | None = None, purpose: str = "file") -> None:
        super().__init__()
        candidate = Path(start_path or Path.home()).expanduser()
        if candidate.is_file():
            candidate = candidate.parent
        self.start_path = str(candidate if candidate.exists() else Path.home())
        self.purpose = purpose

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-dialog"):
            yield Static(
                "Укажи путь или выбери файл. Для каталога модели Vosk введи путь к каталогу.",
                id="picker-title",
            )
            yield Input(value=self.start_path, id="picker-path")
            yield DirectoryTree(self.start_path, id="picker-tree")
            with Horizontal(id="picker-buttons"):
                yield Button("Выбрать путь", id="picker-use-path")
                yield Button("Отмена", id="picker-cancel")

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self.dismiss(str(event.path))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "picker-cancel":
            self.dismiss(None)
        elif event.button.id == "picker-use-path":
            value = self.query_one("#picker-path", Input).value.strip()
            path = Path(value).expanduser()
            if path.exists():
                self.dismiss(str(path))
            else:
                self.query_one("#picker-title", Static).update("Путь не существует.")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        path = Path(event.value.strip()).expanduser()
        if path.exists():
            self.dismiss(str(path))
        else:
            self.query_one("#picker-title", Static).update("Путь не существует.")


class VoiceAssistant(App):
    TITLE = "VOICE ASSISTANT"
    SUB_TITLE = "Vosk STT + local GGUF LLM"
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        ("ctrl+c", "quit", "Выход"),
        ("ctrl+l", "clear_chat", "Очистить чат"),
        ("ctrl+1", "show_chat", "Чат"),
        ("ctrl+2", "show_settings", "Настройки"),
    ]

    PICKER_FIELDS = {"vosk": "vosk-path", "llm": "llm-path", "tts": "tts-path"}

    def __init__(self) -> None:
        super().__init__()
        self.settings = load_settings()
        self.vosk_model: VoskModel | None = None
        self.llm = None
        self.mic_enabled = False
        self.audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=100)
        self.last_question = ""
        self.last_answer = ""
        self.conversation_history: list[dict[str, str]] = []
        self.tts_voice = None
        self._tts_loaded_path = ""
        self._tts_stop = threading.Event()
        self._tts_stream = None
        self._tts_lock = threading.Lock()
        self._vosk_loading = False
        self._llm_loading = False
        self._llm_lock = threading.Lock()

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="chat", id="tabs"):
            with TabPane("Chat", id="chat"):
                with Vertical(id="chat-layout"):
                    yield Static("Загрузи модель Vosk в настройках.", id="status")
                    with VerticalScroll(id="history"):
                        yield Static("Скажи вопрос в микрофон или введи его ниже.", classes="empty-state")
                    yield Static("Распознанный текст: —", id="recognized-text")
                    with Horizontal(id="input-row"):
                        yield Input(placeholder="Введи вопрос…", id="question-input")
                        yield Button("Send", id="send")
                        yield Button("Mic: OFF", id="mic")

            with TabPane("Settings", id="settings"):
                with VerticalScroll(id="settings-scroll"):
                    yield Static("РАСПОЗНАВАНИЕ РЕЧИ — VOSK", classes="section-title")
                    yield Label("Каталог локальной модели Vosk")
                    with Horizontal(classes="path-row"):
                        yield Input(
                            value=str(self.settings["vosk_model_path"]),
                            placeholder="/home/illok/Models/vosk-model-small-ru",
                            id="vosk-path",
                        )
                        yield Button("Выбрать…", id="pick-vosk")
                    yield Static(
                        "Выбери каталог распакованной модели Vosk, а не архив .zip.",
                        id="vosk-note",
                    )
                    yield Label("Микрофон")
                    yield Select(
                        self.get_microphone_options(),
                        value=self.valid_microphone_value(),
                        id="microphone",
                        allow_blank=False,
                    )
                    yield Checkbox(
                        "Автоматически отправлять распознанную фразу",
                        value=bool(self.settings["auto_send"]),
                        id="auto-send",
                    )
                    with Horizontal(id="vosk-actions"):
                        yield Button("Загрузить Vosk", id="load-vosk")
                        yield Button("Остановить микрофон", id="stop-mic")

                    yield Static("ГОЛОСОВЫЕ КОМАНДЫ", classes="section-title")
                    yield Static(
                        'JSON-список команд. У каждой команды есть фразы-триггеры и команда запуска. '
                        'Пример: {"phrases": ["открой браузер"], "command": "firefox", "description": "Браузер"}.',
                        id="commands-note",
                    )
                    yield TextArea(
                        json.dumps(self.settings.get("voice_commands", DEFAULTS["voice_commands"]), ensure_ascii=False, indent=2),
                        id="voice-commands",
                        language="json",
                        show_line_numbers=True,
                    )
                    yield Static(
                        "Срабатывает команда, если фраза-триггер содержится в распознанной речи. "
                        "Запуск выполняется без shell: указывай исполняемую программу и аргументы, например firefox или telegram-desktop.",
                        id="commands-safety-note",
                    )

                    yield Static("ЛОКАЛЬНАЯ LLM — llama.cpp / GGUF", classes="section-title")
                    yield Label("Файл модели GGUF")
                    with Horizontal(classes="path-row"):
                        yield Input(
                            value=str(self.settings["llm_model_path"]),
                            placeholder="/путь/к/модели.gguf",
                            id="llm-path",
                        )
                        yield Button("Выбрать…", id="pick-llm")
                    yield Label("Размер контекста (n_ctx)")
                    yield Input(value=str(self.settings["llm_context_size"]), id="llm-context")
                    yield Label("Температура (0–2)")
                    yield Input(value=str(self.settings["llm_temperature"]), id="llm-temperature")
                    yield Label("Максимум новых токенов")
                    yield Input(value=str(self.settings["llm_max_tokens"]), id="llm-max-tokens")
                    yield Label("Пар сообщений в памяти (0 — не помнить историю)")
                    yield Input(value=str(self.settings["memory_pairs"]), id="memory-pairs")
                    yield Label("Слои на GPU (0 = только CPU)")
                    yield Input(value=str(self.settings["llm_gpu_layers"]), id="llm-gpu-layers")
                    yield Checkbox(
                        "Отключить размышления модели (think) — отвечает быстрее",
                        value=bool(self.settings["disable_thinking"]),
                        id="disable-thinking",
                    )
                    yield Label("Системный промпт")
                    yield Input(value=str(self.settings["system_prompt"]), id="system-prompt")
                    yield Static(
                        "Модель GGUF загружается в память локально. Для GPU нужен llama-cpp-python, "
                        "собранный с поддержкой CUDA.",
                        id="llm-note",
                    )
                    with Horizontal(id="llm-actions"):
                        yield Button("Загрузить LLM", id="load-llm")
                        yield Button("Выгрузить LLM", id="unload-llm")

                    yield Static("ОЗВУЧКА ОТВЕТОВ — PIPER TTS", classes="section-title")
                    yield Checkbox(
                        "Озвучивать ответы модели",
                        value=bool(self.settings["tts_enabled"]),
                        id="tts-enabled",
                    )
                    yield Label("Файл голосовой модели (.onnx)")
                    with Horizontal(classes="path-row"):
                        yield Input(
                            value=str(self.settings["tts_model_path"]),
                            placeholder="~/Downloads/ru_RU-irina-medium.onnx",
                            id="tts-path",
                        )
                        yield Button("Выбрать…", id="pick-tts")
                    yield Label("Устройство вывода")
                    yield Select(
                        self.get_output_device_options(),
                        value=self.valid_output_device_value(),
                        id="tts-output",
                        allow_blank=False,
                    )
                    yield Static(
                        "Рядом с .onnx должен лежать .onnx.json. Голоса скачиваются командой "
                        "python -m piper.download_voices ru_RU-irina-medium. Синтез идёт локально. "
                        "«Системное» направляет звук через сервер (PipeWire/PulseAudio) в устройство по умолчанию.",
                        id="tts-note",
                    )
                    with Horizontal(id="tts-actions"):
                        yield Button("Проверить голос", id="test-tts")
                        yield Button("Стоп", id="stop-tts")

                    with Horizontal(id="settings-actions"):
                        yield Button("Сохранить настройки", id="save-settings", variant="primary")
        yield Footer()

    @staticmethod
    def get_microphone_options() -> list[tuple[str, str]]:
        options = [("Системный микрофон по умолчанию", "default")]
        try:
            for index, device in enumerate(sd.query_devices()):
                if device["max_input_channels"] > 0:
                    options.append((f"{index}: {device['name']}", str(index)))
        except Exception:
            pass
        return options

    def valid_microphone_value(self) -> str:
        value = str(self.settings.get("microphone", "default"))
        valid = [v for _, v in self.get_microphone_options()]
        return value if value in valid else "default"

    @staticmethod
    def get_output_device_options() -> list[tuple[str, str]]:
        options = [("Системное (через сервер звука)", "auto")]
        seen = {"auto"}
        try:
            for device in sd.query_devices():
                name = device["name"]
                if device["max_output_channels"] > 0 and name not in seen:
                    seen.add(name)
                    options.append((name, name))
        except Exception:
            pass
        return options

    def valid_output_device_value(self) -> str:
        value = str(self.settings.get("tts_output_device", "auto") or "auto")
        valid = [v for _, v in self.get_output_device_options()]
        return value if value in valid else "auto"

    @staticmethod
    def find_output_device(name: str) -> int | None:
        for index, device in enumerate(sd.query_devices()):
            if device["name"] == name and device["max_output_channels"] > 0:
                return index
        return None

    def tts_output_candidates(self) -> list[tuple[int | None, str]]:
        choice = str(self.settings.get("tts_output_device", "auto") or "auto")
        candidates: list[tuple[int | None, str]] = []
        if choice != "auto":
            index = self.find_output_device(choice)
            if index is not None:
                candidates.append((index, choice))
        for name in ("pulse", "pipewire", "default"):
            if name == choice:
                continue
            index = self.find_output_device(name)
            if index is not None:
                candidates.append((index, name))
        candidates.append((None, "устройство по умолчанию"))
        return candidates

    def on_mount(self) -> None:
        # Load configured models if their paths are already set.
        if self.settings.get("vosk_model_path"):
            self.load_vosk_model()
        if self.settings.get("llm_model_path"):
            self.load_llm_model()

    def set_status(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    def save_form_to_settings(self) -> bool:
        try:
            vosk_path = self.query_one("#vosk-path", Input).value.strip()
            llm_path = self.query_one("#llm-path", Input).value.strip()
            context_size = int(self.query_one("#llm-context", Input).value)
            temperature = float(self.query_one("#llm-temperature", Input).value)
            max_tokens = int(self.query_one("#llm-max-tokens", Input).value)
            memory_pairs = int(self.query_one("#memory-pairs", Input).value)
            gpu_layers = int(self.query_one("#llm-gpu-layers", Input).value)
            disable_thinking = self.query_one("#disable-thinking", Checkbox).value
            system_prompt = self.query_one("#system-prompt", Input).value.strip()
            tts_path = self.query_one("#tts-path", Input).value.strip()
            tts_enabled = self.query_one("#tts-enabled", Checkbox).value
            tts_output = str(self.query_one("#tts-output", Select).value or "auto")
            commands_raw = self.query_one("#voice-commands", TextArea).text
            voice_commands = json.loads(commands_raw)
            if not isinstance(voice_commands, list):
                raise ValueError("Список голосовых команд должен быть JSON-массивом.")
            for index, item in enumerate(voice_commands, start=1):
                if not isinstance(item, dict):
                    raise ValueError(f"Команда №{index} должна быть объектом JSON.")
                phrases = item.get("phrases")
                command = item.get("command")
                if not isinstance(phrases, list) or not phrases or not all(isinstance(x, str) and x.strip() for x in phrases):
                    raise ValueError(f"У команды №{index} должен быть непустой массив phrases.")
                if not isinstance(command, str) or not command.strip():
                    raise ValueError(f"У команды №{index} должно быть заполнено поле command.")
                try:
                    argv = shlex.split(command)
                except ValueError as error:
                    raise ValueError(f"Некорректная команда запуска №{index}: {error}") from error
                if not argv:
                    raise ValueError(f"Команда запуска №{index} пустая.")

            if vosk_path and not Path(vosk_path).expanduser().is_dir():
                raise ValueError("Каталог модели Vosk не найден.")
            tts_file = Path(tts_path).expanduser() if tts_path else None
            if tts_enabled:
                if tts_file is None or not tts_file.is_file():
                    raise ValueError("Файл голосовой модели TTS не найден.")
                if tts_file.suffix.lower() != ".onnx":
                    raise ValueError("Для Piper выбери файл голоса .onnx.")
                if not Path(f"{tts_file}.json").is_file():
                    raise ValueError("Рядом с .onnx должен лежать файл .onnx.json.")
            if tts_output not in [v for _, v in self.get_output_device_options()]:
                raise ValueError("Неизвестное устройство вывода звука.")
            if llm_path and not Path(llm_path).expanduser().is_file():
                raise ValueError("Файл LLM не найден.")
            if llm_path and Path(llm_path).suffix.lower() != ".gguf":
                raise ValueError("Для llama-cpp-python выбери файл .gguf.")
            if not 256 <= context_size <= 131072:
                raise ValueError("Контекст должен быть от 256 до 131072.")
            if not 0 <= temperature <= 2:
                raise ValueError("Температура должна быть от 0 до 2.")
            if not 1 <= max_tokens <= 32768:
                raise ValueError("Максимум токенов должен быть от 1 до 32768.")
            if not 0 <= memory_pairs <= 50:
                raise ValueError("Память должна быть от 0 до 50 пар сообщений.")
            if not 0 <= gpu_layers <= 200:
                raise ValueError("Число GPU-слоёв должно быть от 0 до 200.")

            new_tts_path = str(tts_file) if tts_file else ""
            if new_tts_path != self.settings.get("tts_model_path"):
                self.tts_voice = None
                self._tts_loaded_path = ""
            if not tts_enabled:
                self.stop_tts()
            self.settings.update({
                "vosk_model_path": str(Path(vosk_path).expanduser()) if vosk_path else "",
                "microphone": self.query_one("#microphone", Select).value,
                "auto_send": self.query_one("#auto-send", Checkbox).value,
                "tts_enabled": tts_enabled,
                "tts_model_path": new_tts_path,
                "tts_output_device": tts_output,
                "llm_model_path": str(Path(llm_path).expanduser()) if llm_path else "",
                "llm_context_size": context_size,
                "llm_temperature": temperature,
                "llm_max_tokens": max_tokens,
                "memory_pairs": memory_pairs,
                "llm_gpu_layers": gpu_layers,
                "disable_thinking": disable_thinking,
                "system_prompt": system_prompt or DEFAULTS["system_prompt"],
                "voice_commands": voice_commands,
            })
            self.trim_conversation_history()
            save_settings(self.settings)
            return True
        except (ValueError, TypeError) as error:
            self.set_status(f"Настройки не сохранены: {error}")
            return False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "send":
            self.send_question()
        elif button_id == "mic":
            self.toggle_microphone()
        elif button_id == "pick-vosk":
            self.open_picker("vosk")
        elif button_id == "pick-llm":
            self.open_picker("llm")
        elif button_id == "pick-tts":
            self.open_picker("tts")
        elif button_id == "test-tts":
            if self.save_form_to_settings():
                self.speak_text("Привет! Это проверка голоса ассистента.", force=True)
        elif button_id == "stop-tts":
            self.stop_tts()
        elif button_id == "save-settings":
            if self.save_form_to_settings():
                self.set_status(f"Настройки сохранены: {CONFIG_PATH}")
        elif button_id == "load-vosk":
            if self.save_form_to_settings():
                self.stop_microphone()
                self.vosk_model = None
                self.load_vosk_model()
        elif button_id == "load-llm":
            if self.save_form_to_settings():
                self.llm = None
                self.load_llm_model()
        elif button_id == "unload-llm":
            self.llm = None
            self.set_status("LLM выгружена из памяти.")
        elif button_id == "stop-mic":
            self.stop_microphone()

    def open_picker(self, target: str) -> None:
        field_id = self.PICKER_FIELDS[target]
        current = self.query_one(f"#{field_id}", Input).value.strip() or str(Path.home())
        self.push_screen(
            FilePicker(current, purpose=target),
            lambda path: self.apply_picked_path(target, path),
        )

    def apply_picked_path(self, target: str, path: str | None) -> None:
        if not path:
            return
        self.query_one(f"#{self.PICKER_FIELDS[target]}", Input).value = path

    @work(thread=True, exclusive=True)
    def load_vosk_model(self) -> None:
        if self._vosk_loading:
            return
        self._vosk_loading = True
        path = str(self.settings.get("vosk_model_path", "")).strip()
        try:
            if not path or not Path(path).expanduser().is_dir():
                raise ValueError("Укажи каталог распакованной модели Vosk в Settings.")
            model = VoskModel(str(Path(path).expanduser()))
            self.call_from_thread(self.vosk_model_loaded, model)
        except Exception as error:
            self.call_from_thread(self.set_status, f"Ошибка загрузки Vosk: {error}")
        finally:
            self._vosk_loading = False

    def vosk_model_loaded(self, model: VoskModel) -> None:
        self.vosk_model = model
        self.set_status("Vosk загружен. Можно включать микрофон.")

    @work(thread=True, exclusive=True)
    def load_llm_model(self) -> None:
        if self._llm_loading:
            return
        self._llm_loading = True
        settings = self.settings.copy()
        model_path = str(settings.get("llm_model_path", "")).strip()
        try:
            if not model_path or not Path(model_path).expanduser().is_file():
                raise ValueError("Укажи существующий файл модели .gguf в Settings.")
            from llama_cpp import Llama

            llm = Llama(
                model_path=str(Path(model_path).expanduser()),
                n_ctx=int(settings["llm_context_size"]),
                n_gpu_layers=int(settings["llm_gpu_layers"]),
                verbose=False,
            )
            if settings.get("disable_thinking", True):
                apply_no_think_handler(llm)
            self.call_from_thread(self.llm_model_loaded, llm)
        except Exception as error:
            self.call_from_thread(self.set_status, f"Ошибка загрузки LLM: {error}")
        finally:
            self._llm_loading = False

    def llm_model_loaded(self, llm) -> None:
        self.llm = llm
        self.set_status("Локальная LLM загружена и готова к ответам.")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "question-input":
            self.send_question()

    def send_question(self, question: str | None = None, *, check_commands: bool = True) -> None:
        input_widget = self.query_one("#question-input", Input)
        question = (question if question is not None else input_widget.value).strip()
        if not question:
            return
        input_widget.value = ""
        # Route configured commands before sending anything to the LLM.
        if check_commands and self.try_execute_voice_command(question):
            input_widget.value = ""
            return
        self.last_question = question
        self.last_answer = ""
        history = self.query_one("#history", VerticalScroll)
        history.remove_children()
        history.mount(Static(f"You: {question}", classes="chat-message"))
        self.set_status("Генерирую ответ локальной LLM…" if self.llm else "LLM не загружена.")
        self.generate_answer(question, list(self.conversation_history))


    @staticmethod
    def normalize_command_text(text: str) -> str:
        """Normalize case and punctuation while preserving word boundaries."""
        import re
        normalized = text.casefold().replace("ё", "е")
        normalized = re.sub(r"[^\w\s-]", " ", normalized, flags=re.UNICODE)
        return " ".join(normalized.split())

    def try_execute_voice_command(self, phrase: str) -> bool:
        """Run a matching configured command; return True when a trigger matched."""
        normalized_phrase = self.normalize_command_text(phrase)
        if not normalized_phrase:
            return False

        matches: list[tuple[int, dict[str, Any], str]] = []
        for item in self.settings.get("voice_commands", []):
            for trigger in item.get("phrases", []):
                normalized_trigger = self.normalize_command_text(trigger)
                if normalized_trigger and normalized_trigger in normalized_phrase:
                    # Prefer the most specific (longest) trigger if several match.
                    matches.append((len(normalized_trigger), item, trigger))
        if not matches:
            return False

        _, item, trigger = max(matches, key=lambda match: match[0])
        command_text = str(item.get("command", "")).strip()
        description = str(item.get("description", "")).strip() or f"Выполняю команду: {command_text}"
        try:
            argv = shlex.split(command_text)
            if not argv:
                raise ValueError("пустая команда запуска")
            # Intentionally no shell=True: voice phrases cannot inject shell syntax.
            subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
            self.set_status(f"Голосовая команда выполнена: {description} (триггер: «{trigger}»).")
            history = self.query_one("#history", VerticalScroll)
            history.remove_children()
            history.mount(Static(f"Команда: {description}", classes="chat-message"))
        except Exception as error:
            self.set_status(f"Не удалось выполнить команду «{command_text}»: {error}")
        return True

    @work(thread=True)
    def generate_answer(self, question: str, history: list[dict[str, str]]) -> None:
        llm = self.llm
        if llm is None:
            self.call_from_thread(
                self.show_answer,
                "LLM не загружена. Открой Settings, выбери GGUF-файл и нажми «Загрузить LLM».",
            )
            return
        settings = self.settings.copy()
        try:
            with self._llm_lock:
                result = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": settings["system_prompt"]},
                        *history,
                        {"role": "user", "content": question},
                    ],
                    temperature=float(settings["llm_temperature"]),
                    max_tokens=int(settings["llm_max_tokens"]),
                )
            answer = result["choices"][0]["message"]["content"].strip()
            if answer:
                self.call_from_thread(self.show_answer, answer, question)
            else:
                self.call_from_thread(self.show_answer, "(Модель вернула пустой ответ.)")
        except Exception as error:
            self.call_from_thread(self.set_status, f"Ошибка LLM: {error}")
            self.call_from_thread(self.show_answer, f"Ошибка генерации: {error}")

    def trim_conversation_history(self) -> None:
        max_messages = max(0, int(self.settings.get("memory_pairs", DEFAULTS["memory_pairs"]))) * 2
        if max_messages:
            del self.conversation_history[:-max_messages]
        else:
            self.conversation_history.clear()

    def show_answer(self, answer: str, question: str | None = None) -> None:
        self.last_answer = answer
        if question:
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": answer})
            self.trim_conversation_history()
        self.query_one("#history", VerticalScroll).mount(
            Static(f"Assistant: {answer}", classes="chat-message")
        )
        self.set_status("Готово. Можно задать следующий вопрос.")
        if question:
            self.speak_text(answer)

    def load_tts_voice(self, model_path: str):
        with self._tts_lock:
            if self.tts_voice is not None and self._tts_loaded_path == model_path:
                return self.tts_voice
            from piper import PiperVoice

            voice = PiperVoice.load(model_path)
            self.tts_voice = voice
            self._tts_loaded_path = model_path
            return voice

    def stop_tts(self) -> None:
        self._tts_stop.set()

    @work(thread=True, exclusive=True, group="tts")
    def speak_text(self, text: str, force: bool = False) -> None:
        if not force and not self.settings.get("tts_enabled", DEFAULTS["tts_enabled"]):
            return
        model_path = str(self.settings.get("tts_model_path", "")).strip()
        if not model_path:
            self.call_from_thread(self.set_status, "TTS: укажи файл голосовой модели .onnx в настройках.")
            return
        stop = threading.Event()
        self._tts_stop.set()
        self._tts_stop = stop
        stream = None
        try:
            voice = self.load_tts_voice(model_path)
            chunks = [chunk for chunk in voice.synthesize(text)]
            if not chunks or stop.is_set():
                return
            source = chunks[0]
            raw = b"".join(chunk.audio_int16_bytes for chunk in chunks)
            audio = raw
            rate = source.sample_rate
            last_error = None
            for device, _label in self.tts_output_candidates():
                if stop.is_set():
                    return
                info = sd.query_devices(device, kind="output")
                device_rate = int(info["default_samplerate"])
                attempts = [(raw, source.sample_rate)]
                if device_rate != source.sample_rate:
                    attempts.append((resample_pcm16(raw, source.sample_rate, device_rate), device_rate))
                for attempt_audio, attempt_rate in attempts:
                    try:
                        stream = sd.OutputStream(
                            device=device,
                            samplerate=attempt_rate,
                            channels=source.sample_channels,
                            dtype="int16",
                        )
                    except sd.PortAudioError as error:
                        last_error = error
                        continue
                    audio = attempt_audio
                    rate = attempt_rate
                    break
                if stream is not None:
                    break
            if stream is None:
                raise last_error or RuntimeError("не удалось открыть устройство вывода")
            import numpy as np

            samples = np.frombuffer(audio, dtype=np.int16)
            self._tts_stream = stream
            stream.start()
            step = max(1, rate // 10)
            for offset in range(0, samples.size, step):
                if stop.is_set():
                    break
                stream.write(samples[offset:offset + step])
        except Exception as error:
            self.call_from_thread(self.set_status, f"Ошибка TTS: {error}")
        finally:
            self._tts_stream = None
            if stream is not None:
                try:
                    if stop.is_set():
                        stream.abort()
                    else:
                        stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass

    def set_recognized_text(self, text: str, final: bool = False) -> None:
        self.query_one("#recognized-text", Static).update(
            f"{'Распознано' if final else 'Сейчас распознаётся'}: {text}"
        )
        if final:
            self.query_one("#question-input", Input).value = text
            # Voice commands are always checked first, independently of auto-send.
            if self.try_execute_voice_command(text):
                self.query_one("#question-input", Input).value = ""
                return
            if self.settings.get("auto_send", True):
                self.send_question(text, check_commands=False)

    def toggle_microphone(self) -> None:
        if self.mic_enabled:
            self.stop_microphone()
            return
        if self.vosk_model is None:
            self.set_status("Сначала выбери каталог Vosk и нажми «Загрузить Vosk» в Settings.")
            return
        if not self.save_form_to_settings():
            return
        self.mic_enabled = True
        self.query_one("#mic", Button).label = "Mic: ON"
        self.set_status("Микрофон активен. Говори.")
        self.listen_microphone()

    def stop_microphone(self) -> None:
        self.mic_enabled = False
        try:
            self.query_one("#mic", Button).label = "Mic: OFF"
        except Exception:
            pass
        self.set_status("Микрофон выключен.")

    @work(thread=True, exclusive=True)
    def listen_microphone(self) -> None:
        audio_queue = self.audio_queue
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break

        def callback(indata, frames, time_info, status):
            try:
                audio_queue.put_nowait(bytes(indata))
            except queue.Full:
                pass

        mic = self.settings.get("microphone", "default")
        device = None if mic in (None, "default") else int(mic)

        try:
            recognizer = KaldiRecognizer(self.vosk_model, SAMPLE_RATE)
            recognizer.SetWords(True)
            with sd.RawInputStream(
                device=device,
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                dtype="int16",
                channels=1,
                callback=callback,
            ):
                while self.mic_enabled:
                    try:
                        audio_bytes = audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    if recognizer.AcceptWaveform(audio_bytes):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "").strip()
                        if text and self.mic_enabled:
                            self.call_from_thread(self.set_recognized_text, text, True)
                    else:
                        partial = json.loads(recognizer.PartialResult()).get("partial", "").strip()
                        if partial:
                            self.call_from_thread(self.set_recognized_text, partial, False)
        except Exception as error:
            self.call_from_thread(self.set_status, f"Ошибка микрофона/Vosk: {error}")
            self.call_from_thread(self.mic_error)

    def mic_error(self) -> None:
        self.mic_enabled = False
        self.query_one("#mic", Button).label = "Mic: OFF"

    def action_clear_chat(self) -> None:
        history = self.query_one("#history", VerticalScroll)
        history.remove_children()
        history.mount(Static("История очищена", classes="empty-state"))
        self.last_question = ""
        self.last_answer = ""
        self.conversation_history.clear()

    def action_show_chat(self) -> None:
        self.query_one(TabbedContent).active = "chat"

    def action_show_settings(self) -> None:
        self.query_one(TabbedContent).active = "settings"


if __name__ == "__main__":
    VoiceAssistant().run()
