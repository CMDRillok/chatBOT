
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any

import sounddevice as sd
from vosk import Model as VoskModel, KaldiRecognizer

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
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000  # 250 ms at 16 kHz

DEFAULTS: dict[str, Any] = {
    "vosk_model_path": "",
    "microphone": "default",
    "auto_send": True,
    "llm_model_path": "",
    "llm_context_size": 4096,
    "llm_temperature": 0.7,
    "llm_max_tokens": 512,
    "llm_gpu_layers": 0,
    "system_prompt": "Ты полезный ассистент для управления пк. отвечай коротко и по делу.",
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

    def __init__(self) -> None:
        super().__init__()
        self.settings = load_settings()
        self.vosk_model: VoskModel | None = None
        self.llm = None
        self.mic_enabled = False
        self.audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=100)
        self.last_question = ""
        self.last_answer = ""
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
                        yield Static("Скажи вопрос в микрофон или введи его ниже.", id="empty-state")
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
                    yield Label("Слои на GPU (0 = только CPU)")
                    yield Input(value=str(self.settings["llm_gpu_layers"]), id="llm-gpu-layers")
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
                    with Horizontal(id="settings-actions"):
                        yield Button("Сохранить настройки", id="save-settings", variant="primary")
                        yield Button("В чат", id="back-chat")
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
            gpu_layers = int(self.query_one("#llm-gpu-layers", Input).value)
            system_prompt = self.query_one("#system-prompt", Input).value.strip()

            if vosk_path and not Path(vosk_path).expanduser().is_dir():
                raise ValueError("Каталог модели Vosk не найден.")
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
            if not 0 <= gpu_layers <= 200:
                raise ValueError("Число GPU-слоёв должно быть от 0 до 200.")

            self.settings.update({
                "vosk_model_path": str(Path(vosk_path).expanduser()) if vosk_path else "",
                "microphone": self.query_one("#microphone", Select).value,
                "auto_send": self.query_one("#auto-send", Checkbox).value,
                "llm_model_path": str(Path(llm_path).expanduser()) if llm_path else "",
                "llm_context_size": context_size,
                "llm_temperature": temperature,
                "llm_max_tokens": max_tokens,
                "llm_gpu_layers": gpu_layers,
                "system_prompt": system_prompt or DEFAULTS["system_prompt"],
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
        elif button_id == "pick-vosk":
            self.open_picker("vosk")
        elif button_id == "pick-llm":
            self.open_picker("llm")
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
        elif button_id == "back-chat":
            self.query_one(TabbedContent).active = "chat"

    def open_picker(self, target: str) -> None:
        field_id = "vosk-path" if target == "vosk" else "llm-path"
        current = self.query_one(f"#{field_id}", Input).value.strip() or str(Path.home())
        self.push_screen(
            FilePicker(current, purpose=target),
            lambda path: self.apply_picked_path(target, path),
        )

    def apply_picked_path(self, target: str, path: str | None) -> None:
        if not path:
            return
        field_id = "vosk-path" if target == "vosk" else "llm-path"
        self.query_one(f"#{field_id}", Input).value = path

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
        self.set_status("Генерирую ответ локальной LLM…" if self.llm else "LLM не загружена.")
        self.generate_answer(question)

    @work(thread=True)
    def generate_answer(self, question: str) -> None:
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
                        {"role": "user", "content": question},
                    ],
                    temperature=float(settings["llm_temperature"]),
                    max_tokens=int(settings["llm_max_tokens"]),
                )
            answer = result["choices"][0]["message"]["content"].strip()
            self.call_from_thread(self.show_answer, answer or "(Модель вернула пустой ответ.)")
        except Exception as error:
            self.call_from_thread(self.set_status, f"Ошибка LLM: {error}")
            self.call_from_thread(self.show_answer, f"Ошибка генерации: {error}")

    def show_answer(self, answer: str) -> None:
        self.last_answer = answer
        self.query_one("#history", VerticalScroll).mount(
            Static(f"Assistant: {answer}", classes="chat-message")
        )
        self.set_status("Готово. Можно задать следующий вопрос.")

    def set_recognized_text(self, text: str, final: bool = False) -> None:
        self.query_one("#recognized-text", Static).update(
            f"{'Распознано' if final else 'Сейчас распознаётся'}: {text}"
        )
        if final:
            self.query_one("#question-input", Input).value = text
            if self.settings.get("auto_send", True):
                self.send_question(text)

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
        history.mount(Static("История очищена", id="empty-state"))
        self.last_question = ""
        self.last_answer = ""

    def action_show_chat(self) -> None:
        self.query_one(TabbedContent).active = "chat"

    def action_show_settings(self) -> None:
        self.query_one(TabbedContent).active = "settings"


if __name__ == "__main__":
    VoiceAssistant().run()
