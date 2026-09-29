
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, Button, Static
from textual.containers import Container, Vertical


class MainScreen(Screen):
    CSS = """
    MainScreen {
        align: center middle;
    }

    #menu {
        width: 50;
        height: auto;
        border: round #4a4a4a;
        padding: 2 3;
        background: #1c1c1c;
    }

    #title {
        text-align: center;
        color: #f5f5f5;
        text-style: bold;
        margin-bottom: 2;
    }

    Button {
        width: 100%;
        margin-bottom: 1;
        background: #242424;
        color: #f5f5f5;
        border: solid #4a4a4a;
    }

    Button:hover {
        background: #4a4a4a;
    }

    #mic-status {
        text-align: center;
        color: #919191;
        margin-top: 1;
    }

    #recognized-text {
        dock: bottom;
        height: 3;
        width: 100%;
        padding: 1 2;
        background: #242424;
        color: #f5f5f5;
        border-top: solid #4a4a4a;
}
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="menu"):
            yield Static("ГОЛОВНОЕ МЕНЮ", id="title")
            yield Button("Включить микрофон", id="mic")
            yield Button("Настройки", id="settings")
            yield Static("Микрофон выключен", id="mic-status")
            yield Static("Распознанный текст: ", id="recognized-text")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mic":
            self.app.toggle_microphone()
        elif event.button.id == "settings":
            self.app.push_screen(SettingsScreen())


class SettingsScreen(Screen):
    CSS = """
    SettingsScreen {
        align: center middle;
    }

    #settings-menu {
        width: 50;
        height: auto;
        border: round #4a4a4a;
        padding: 2 3;
        background: #1c1c1c;
        align: center middle;
    }

    #settings-title {
        text-align: center;
        color: #f5f5f5;
        text-style: bold;
        margin-bottom: 2;
    }

    Button {
        width: 100%;
        background: #1c1c1c;
        color: #f5f5f5;
        border: solid #4a4a4a;
    }

    Button:hover {
        background: #4a4a4a;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="settings-menu"):
            yield Static("НАСТРОЙКИ", id="settings-title")
            yield Button("На главный экран", id="back")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.app.pop_screen()


class MyTUI(App):
    TITLE = "chat"
    SUB_TITLE = "Голосовой интерфейс"

    SCREENS = {"main": MainScreen}

    CSS = """
    Screen {
        background: #1c1c1c;
    }
    """

    BINDINGS = [
    ("q", "quit", "Выход"),
    ("up", "focus_previous", "Предыдущая кнопка"),
    ("k", "focus_previous", "Предыдущая кнопка"),
    ("down", "focus_next", "Следующая кнопка"),
    ("j", "focus_next", "Следующая кнопка"),
    ]
    
    def action_focus_next(self) -> None:
        self.screen.focus_next()

    def action_focus_previous(self) -> None:
        self.screen.focus_previous()

    def __init__(self):
        super().__init__()
        self.microphone_enabled = False

    def on_mount(self) -> None:
        self.push_screen("main")

    def toggle_microphone(self) -> None:
        self.microphone_enabled = not self.microphone_enabled
    
        screen = self.get_screen("main")
        button = screen.query_one("#mic", Button)
        status = screen.query_one("#mic-status", Static)

        if self.microphone_enabled:
            button.label = "Выключить микрофон"
            status.update("Микрофон включён")
        else:
            button.label = "Включить микрофон"
            status.update("Микрофон выключен")
    
    def update_recognized_text(self, text: str) -> None:
        screen = self.get_screen("main")
        display = screen.query_one("#recognized-text", Static)
        display.update(f"Распознанный текст: {text}")

if __name__ == "__main__":
    MyTUI().run()
