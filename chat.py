from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button, Checkbox, DirectoryTree, Footer, Header, Input, Label,
    Select, Static, TabbedContent, TabPane,
)

APP_DIR = Path.home() / ".config" / "voice-assistant"
CONFIG_PATH = APP_DIR / "config.json"

DEFAULTS: dict[str, Any] = {
    "whisper_model": "base",
    "whisper_device": "cpu",
    "compute_type": "int8",
    "language": "ru",
    "microphone": None,
    "volume_threshold": 0.015,
    "silence_seconds": 1.2,
    "auto_send": True,
    "llm_model_path": "",
    "llm_runtime": "not_connected",
    "history_enabled": False,
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


def ask_llm(question: str, settings: dict[str, Any]) -> str:
    # The selected LLM file is saved in settings, but inference is deliberately
    # not connected yet. A GGUF file requires a runtime such as llama.cpp.
    model_path = settings.get("llm_model_path", "").strip()
    if model_path:
        return (
            "LLM пока не подключена к генерации.\\n"
            f"Выбранный файл: {model_path}"
        )
    return "LLM пока не подключена. Выбери файл модели во вкладке Settings."


class FilePicker(ModalScreen[str | None]):
    """Small local file browser. Select a file; Escape cancels."""

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

    def __init__(self, start_path: str | None = None) -> None:
        super().__init__()
        candidate = Path(start_path or Path.home()).expanduser()
        if candidate.is_file():
            candidate = candidate.parent
        self.start_path = str(candidate if candidate.exists() else Path.home())

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-dialog"):
            yield Static(
                "Выбери файл в дереве. Для Whisper выбери config.json модели.",
                id="picker-title",
            )
            yield Input(value=self.start_path, id="picker-path")
            yield DirectoryTree(self.start_path, id="picker-tree")
            with Horizontal(id="picker-buttons"):
                yield Button("Выбрать путь", id="picker-use-path")
                yield Button("Отмена", id="picker-cancel")

    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
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
                self.query_one("#picker-title", Static).update(
                    "Путь не существует. Укажи существующий файл или каталог."
                )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "picker-path":
            value = event.value.strip()
            path = Path(value).expanduser()
            if path.exists():
                self.dismiss(str(path))
            else:
                self.query_one("#picker-title", Static).update(
                    "Путь не существует. Проверь путь и попробуй снова."
                )


class VoiceAssistant(App):
    TITLE = "VOICE ASSISTANT"
    SUB_TITLE = "локальный голосовой помошник"
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        ("ctrl+c", "quit", "Выход"),
        ("ctrl+l", "clear_chat", "Очистить чат"),
        ("ctrl+1", "show_chat", "Чат"),
        ("ctrl+2", "show_settings", "Настройки"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.settings = load_settings()
        self.model: WhisperModel | None = None
        self.mic_enabled = False
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self.model_lock = threading.Lock()
        self.last_question = ""
        self.last_answer = ""
        self._loading = False

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="chat", id="tabs"):
            with TabPane("Chat", id="chat"):
                with Vertical(id="chat-layout"):
                    yield Static("Инициализация...", id="status")
                    with VerticalScroll(id="history"):
                        yield Static(
                            "Скажи вопрос в микрофон или введи его ниже.",
                            id="empty-state",
                        )
                    yield Static("Распознанный текст: —", id="recognized-text")
                    with Horizontal(id="input-row"):
                        yield Input(placeholder="Введи вопрос…", id="question-input")
                        yield Button("Send", id="send")
                        yield Button("Mic: OFF", id="mic")
            with TabPane("Settings", id="settings"):
                with VerticalScroll(id="settings-scroll"):
                    yield Static("РАСПОЗНАВАНИЕ РЕЧИ", classes="section-title")
                    yield Label("Модель Whisper (имя или локальный каталог)")
                    with Horizontal(classes="path-row"):
                        yield Input(
                            value=str(self.settings["whisper_model"]),
                            placeholder="base или /путь/к/модели",
                            id="whisper-path",
                        )
                        yield Button("Выбрать…", id="pick-whisper")
                    yield Label("Язык распознавания")
                    yield Select(
                        [("Русский", "ru"), ("English", "en"), ("Автоопределение", "auto")],
                        value=self.settings["language"],
                        id="language",
                        allow_blank=False,
                    )
                    yield Label("Устройство вычислений")
                    yield Select(
                        [("CPU", "cpu"), ("CUDA / NVIDIA GPU", "cuda")],
                        value=self.settings["whisper_device"],
                        id="device",
                        allow_blank=False,
                    )
                    yield Label("Тип вычислений")
                    yield Select(
                        [
                            ("int8 (CPU, экономно)", "int8"),
                            ("float32 (CPU, точнее/медленнее)", "float32"),
                            ("float16 (GPU)", "float16"),
                            ("int8_float16 (GPU)", "int8_float16"),
                        ],
                        value=self.settings["compute_type"],
                        id="compute-type",
                        allow_blank=False,
                    )
                    yield Label("Микрофон")
                    mic_options = self.get_microphone_options()
                    current_mic = self.settings.get("microphone")
                    valid_values = [value for _, value in mic_options]
                    if current_mic not in valid_values:
                        current_mic = "default"
                    yield Select(
                        mic_options,
                        value=current_mic,
                        id="microphone",
                        allow_blank=False,
                    )
                    yield Label("Порог громкости (0.001–0.1)")
                    yield Input(
                        value=str(self.settings["volume_threshold"]),
                        placeholder="0.015",
                        id="threshold",
                    )
                    yield Label("Пауза до завершения фразы, секунды (0.3–4.0)")
                    yield Input(
                        value=str(self.settings["silence_seconds"]),
                        placeholder="1.2",
                        id="silence",
                    )
                    yield Checkbox(
                        "Автоматически отправлять распознанную фразу",
                        value=bool(self.settings["auto_send"]),
                        id="auto-send",
                    )
                    yield Static("ЛОКАЛЬНАЯ LLM", classes="section-title")
                    yield Label("Файл модели (например, GGUF для llama.cpp)")
                    with Horizontal(classes="path-row"):
                        yield Input(
                            value=str(self.settings["llm_model_path"]),
                            placeholder="/путь/к/модели.gguf",
                            id="llm-path",
                        )
                        yield Button("Выбрать…", id="pick-llm")
                    yield Static(
                        "Выбор файла сохраняется, но генерация LLM пока не подключена.",
                        id="llm-note",
                    )
                    yield Checkbox(
                        "Сохранять историю между запусками (пока не реализовано)",
                        value=bool(self.settings["history_enabled"]),
                        id="history-enabled",
                        disabled=True,
                    )
                    with Horizontal(id="settings-actions"):
                        yield Button("Сохранить настройки", id="save-settings", variant="primary")
                        yield Button("Перезагрузить Whisper", id="reload-model")
                        yield Button("В чат", id="back-chat")
        yield Footer()

    @staticmethod
    def get_microphone_options() -> list[tuple[str, str]]:
        options = [("Системный микрофон по умолчанию", "default")]
        try:
            devices = sd.query_devices()
            for index, device in enumerate(devices):
                if device["max_input_channels"] > 0:
                    name = str(device["name"])
                    options.append((f"{index}: {name}", str(index)))
        except Exception:
            pass
        return options

    def on_mount(self) -> None:
        self.load_model()

    def set_status(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    def save_form_to_settings(self) -> bool:
        try:
            threshold = float(self.query_one("#threshold", Input).value)
            silence = float(self.query_one("#silence", Input).value)
            if not 0.001 <= threshold <= 0.1:
                raise ValueError("Порог громкости должен быть от 0.001 до 0.1.")
            if not 0.3 <= silence <= 4.0:
                raise ValueError("Пауза должна быть от 0.3 до 4 секунд.")

            whisper_path = self.query_one("#whisper-path", Input).value.strip()
            llm_path = self.query_one("#llm-path", Input).value.strip()
            if whisper_path.startswith("~"):
                whisper_path = str(Path(whisper_path).expanduser())
            if whisper_path.startswith("/") and not Path(whisper_path).exists():
                raise ValueError("Каталог/путь Whisper не существует.")
            if llm_path and not Path(llm_path).expanduser().exists():
                raise ValueError("Файл LLM не существует.")

            self.settings.update({
                "whisper_model": whisper_path or "base",
                "language": self.query_one("#language", Select).value,
                "whisper_device": self.query_one("#device", Select).value,
                "compute_type": self.query_one("#compute-type", Select).value,
                "microphone": self.query_one("#microphone", Select).value,
                "volume_threshold": threshold,
                "silence_seconds": silence,
                "auto_send": self.query_one("#auto-send", Checkbox).value,
                "llm_model_path": str(Path(llm_path).expanduser()) if llm_path else "",
            })
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
        elif button_id == "save-settings":
            if self.save_form_to_settings():
                self.set_status(f"Настройки сохранены: {CONFIG_PATH}")
        elif button_id == "reload-model":
            if self.save_form_to_settings():
                self.stop_microphone()
                self.model = None
                self.load_model()
        elif button_id == "back-chat":
            self.query_one(TabbedContent).active = "chat"
        elif button_id == "pick-whisper":
            self.open_picker("whisper")
        elif button_id == "pick-llm":
            self.open_picker("llm")

    def open_picker(self, target: str) -> None:
        field_id = "whisper-path" if target == "whisper" else "llm-path"
        current = self.query_one(f"#{field_id}", Input).value.strip() or str(Path.home())
        self.push_screen(FilePicker(current), lambda path: self.apply_picked_path(target, path))

    def apply_picked_path(self, target: str, path: str | None) -> None:
        if not path:
            return
        selected = Path(path).expanduser()
        if target == "whisper":
            # Selecting a Hugging Face config.json selects its parent model folder.
            if selected.is_file() and selected.name == "config.json":
                selected = selected.parent
            field_id = "whisper-path"
        else:
            field_id = "llm-path"
        self.query_one(f"#{field_id}", Input).value = str(selected)

    @work(thread=True, exclusive=True)
    def load_model(self) -> None:
        if self._loading:
            return
        self._loading = True
        settings = self.settings.copy()
        try:
            model_ref = settings["whisper_model"]
            local_only = Path(str(model_ref)).expanduser().exists()
            model = WhisperModel(
                str(Path(model_ref).expanduser()) if local_only else str(model_ref),
                device=settings["whisper_device"],
                compute_type=settings["compute_type"],
                local_files_only=local_only,
            )
            self.call_from_thread(self.model_loaded, model)
        except Exception as error:
            self.call_from_thread(
                self.set_status,
                f"Ошибка загрузки Whisper: {error}",
            )
        finally:
            self._loading = False

    def model_loaded(self, model: WhisperModel) -> None:
        self.model = model
        self.set_status("Whisper готов. Настрой микрофон или введи вопрос.")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "question-input":
            self.send_question()

    def send_question(self, question: str | None = None) -> None:
        input_widget = self.query_one("#question-input", Input)
        question = (question if question is not None else input_widget.value).strip()
        if not question:
            return
        input_widget.value = ""
        self.last_question = question
        self.last_answer = ""
        history = self.query_one("#history", VerticalScroll)
        history.remove_children()
        history.mount(Static(f"You: {question}", classes="chat-message"))
        self.set_status("Обрабатываю вопрос…")
        self.generate_answer(question)

    @work(thread=True)
    def generate_answer(self, question: str) -> None:
        try:
            answer = ask_llm(question, self.settings.copy())
            self.call_from_thread(self.show_answer, answer)
        except Exception as error:
            self.call_from_thread(self.set_status, f"Ошибка обработки: {error}")

    def show_answer(self, answer: str) -> None:
        self.last_answer = answer
        self.query_one("#history", VerticalScroll).mount(
            Static(f"Assistant: {answer}", classes="chat-message")
        )
        self.set_status("Готово. Можно задать следующий вопрос.")

    def set_recognized_text(self, text: str) -> None:
        self.query_one("#recognized-text", Static).update(f"Распознано: {text}")
        self.query_one("#question-input", Input).value = text
        if self.settings.get("auto_send", True):
            self.send_question(text)

    def toggle_microphone(self) -> None:
        if self.mic_enabled:
            self.stop_microphone()
            return
        if self.model is None:
            self.set_status("Модель Whisper ещё не загружена.")
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
        settings = self.settings.copy()
        threshold = float(settings["volume_threshold"])
        silence_seconds = float(settings["silence_seconds"])
        block_seconds = BLOCK_SIZE / SAMPLE_RATE
        silence_limit = max(1, int(silence_seconds / block_seconds))
        recording: list[np.ndarray] = []
        speaking = False
        silent_blocks = 0

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        def callback(indata, frames, time_info, status):
            try:
                self.audio_queue.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

        mic = settings.get("microphone", "default")
        device = None if mic in (None, "default") else int(mic)

        try:
            with sd.InputStream(
                device=device,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=BLOCK_SIZE,
                callback=callback,
            ):
                while self.mic_enabled:
                    try:
                        audio = self.audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    volume = float(np.sqrt(np.mean(audio ** 2)))

                    if not speaking:
                        if volume >= threshold:
                            speaking = True
                            recording = [audio]
                            silent_blocks = 0
                        continue

                    recording.append(audio)
                    if volume < threshold:
                        silent_blocks += 1
                    else:
                        silent_blocks = 0

                    if silent_blocks >= silence_limit:
                        speaking = False
                        chunks = recording[:-silent_blocks]
                        recording = []
                        if not chunks:
                            continue

                        audio_data = np.concatenate(chunks)
                        try:
                            language = settings["language"]
                            segments, _ = self.model.transcribe(
                                audio_data,
                                language=None if language == "auto" else language,
                                beam_size=5,
                                vad_filter=True,
                            )
                            recognized = " ".join(
                                segment.text.strip() for segment in segments
                            ).strip()
                        except Exception as error:
                            self.call_from_thread(
                                self.set_status, f"Ошибка распознавания: {error}"
                            )
                            continue

                        if recognized and self.mic_enabled:
                            self.call_from_thread(
                                self.set_recognized_text, recognized
                            )
        except Exception as error:
            self.call_from_thread(self.set_status, f"Ошибка микрофона: {error}")
            self.call_from_thread(self.mic_error)

    def mic_error(self) -> None:
        self.mic_enabled = False
        self.query_one("#mic", Button).label = "Mic: OFF"

    def action_clear_chat(self) -> None:
        history = self.query_one("#history", VerticalScroll)
        history.remove_children()
        history.mount(Static("История очищена", id="empty-state"))
        self.last_question = ""
        self.last_answer = ""

    def action_show_chat(self) -> None:
        self.query_one(TabbedContent).active = "chat"

    def action_show_settings(self) -> None:
        self.query_one(TabbedContent).active = "settings"


if __name__ == "__main__":
    VoiceAssistant().run()
