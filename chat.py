import queue

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Static,
    TabbedContent,
    TabPane,
)


SAMPLE_RATE = 16000
BLOCK_SIZE = 1600
VOLUME_THRESHOLD = 0.015
SILENCE_SECONDS = 1.2
MODEL_SIZE = "base"


def ask_llm(question: str) -> str:
    """Заглушка. Позже здесь будет вызов локальной LLM."""
    return "LLM пока не подключена."


class ChatApp(App):
    TITLE = "VOICE ASSISTANT"
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        ("ctrl+c", "quit", "Выход"),
        ("ctrl+l", "clear_chat", "Очистить"),
    ]

    def __init__(self):
        super().__init__()
        self.model = None
        self.mic_enabled = False
        self.last_question = ""
        self.last_answer = ""
        self.audio_queue = queue.Queue()

    def compose(self) -> ComposeResult:
        yield Header()

        with TabbedContent(initial="chat"):
            with TabPane("Chat", id="chat"):
                with Vertical(id="chat-layout"):
                    yield Static(
                        "Голосовой ассистент готов к работе",
                        id="status",
                    )

                    with Vertical(id="history"):
                        yield Static(
                            "Здесь появится последний вопрос и ответ",
                            id="empty-state",
                        )

                    yield Static(
                        "Распознанный текст: —",
                        id="recognized-text",
                    )

                    with Horizontal(id="input-row"):
                        yield Input(
                            placeholder="Введи вопрос...",
                            id="question-input",
                        )
                        yield Button(
                            "Send",
                            id="send",
                            variant="default",
                        )
                        yield Button(
                            "Mic: OFF",
                            id="mic",
                            variant="default",
                        )

            with TabPane("Settings", id="settings"):
                with Vertical(id="settings-layout"):
                    yield Label("Настройки распознавания")
                    yield Label(f"Модель Whisper: {MODEL_SIZE}")
                    yield Label("Язык: русский")
                    yield Label("Устройство: CPU")
                    yield Label(
                        "Порог громкости регулируется "
                        "переменной VOLUME_THRESHOLD в chat.py"
                    )
                    yield Button("Назад в чат", id="back")

        yield Footer()

    def on_mount(self) -> None:
        self.load_model()

    @work(thread=True, exclusive=True)
    def load_model(self) -> None:
        try:
            model = WhisperModel(
                MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )
            self.call_from_thread(self.model_loaded, model)
        except Exception as error:
            self.call_from_thread(
                self.set_status,
                f"Ошибка загрузки Whisper: {error}",
            )

    def model_loaded(self, model: WhisperModel) -> None:
        self.model = model
        self.set_status("Whisper загружен. Можно говорить или печатать.")

    def set_status(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        if button_id == "send":
            self.send_question()

        elif button_id == "mic":
            if not self.mic_enabled:
                if self.model is None:
                    self.set_status("Модель ещё загружается.")
                    return

                self.mic_enabled = True
                event.button.label = "Mic: ON"
                self.set_status("Микрофон активен. Говори.")
                self.listen_microphone()
            else:
                self.mic_enabled = False
                event.button.label = "Mic: OFF"
                self.set_status("Микрофон выключен.")

        elif button_id == "back":
            self.query_one(TabbedContent).active = "chat"

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "question-input":
            self.send_question()

    def send_question(self) -> None:
        input_widget = self.query_one("#question-input", Input)
        question = input_widget.value.strip()

        if not question:
            return

        input_widget.value = ""
        self.last_question = question
        self.last_answer = ""

        history = self.query_one("#history", Vertical)
        history.remove_children()

        history.mount(
            Static(f"You: {question}", classes="chat-message")
        )

        self.set_status("Обрабатываю вопрос...")
        self.generate_answer(question)

    @work(thread=True)
    def generate_answer(self, question: str) -> None:
        try:
            answer = ask_llm(question)
            self.call_from_thread(self.show_answer, answer)
        except Exception as error:
            self.call_from_thread(
                self.set_status,
                f"Ошибка: {error}",
            )

    def show_answer(self, answer: str) -> None:
        self.last_answer = answer

        self.query_one("#history", Vertical).mount(
            Static(f"Assistant: {answer}", classes="chat-message")
        )
        self.set_status("Готово. Можно задать следующий вопрос.")

    @work(thread=True, exclusive=True)
    def listen_microphone(self) -> None:
        audio_queue = self.audio_queue
        block_seconds = BLOCK_SIZE / SAMPLE_RATE
        silence_limit = int(SILENCE_SECONDS / block_seconds)

        recording = []
        speaking = False
        silent_blocks = 0

        def callback(indata, frames, time_info, status):
            audio_queue.put(indata[:, 0].copy())

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=BLOCK_SIZE,
                callback=callback,
            ):
                while self.mic_enabled:
                    try:
                        audio = audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    volume = float(np.sqrt(np.mean(audio ** 2)))

                    if not speaking:
                        if volume >= VOLUME_THRESHOLD:
                            speaking = True
                            recording = [audio]
                            silent_blocks = 0
                        continue

                    recording.append(audio)

                    if volume < VOLUME_THRESHOLD:
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

                        segments, _ = self.model.transcribe(
                            audio_data,
                            language="ru",
                            beam_size=5,
                            vad_filter=True,
                        )

                        recognized = " ".join(
                            segment.text.strip()
                            for segment in segments
                        ).strip()

                        if recognized and self.mic_enabled:
                            self.call_from_thread(
                                self.set_recognized_text,
                                recognized,
                            )

        except Exception as error:
            self.call_from_thread(
                self.set_status,
                f"Ошибка микрофона: {error}",
            )
            self.call_from_thread(self.mic_error)

    def set_recognized_text(self, text: str) -> None:
        self.query_one("#recognized-text", Static).update(
            f"Распознано: {text}"
        )
        self.query_one("#question-input", Input).value = text

    def mic_error(self) -> None:
        self.mic_enabled = False
        self.query_one("#mic", Button).label = "Mic: OFF"

    def action_clear_chat(self) -> None:
        self.last_question = ""
        self.last_answer = ""
        self.query_one("#history", Vertical).remove_children()
        self.query_one("#history", Vertical).mount(
            Static("История очищена", id="empty-state")
        )


if __name__ == "__main__":
    ChatApp().run()
