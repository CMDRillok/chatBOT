#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

PROFILE=""
LLM_CHOICE=""
MODELS_DIR="$HOME/Models"
ASSUME_YES=0
DO_CUDA=""
INSTALL_DEPS=0
SKIP_DEPS=0
NO_LAUNCHER=0

LLM_NAME_14B="Qwen3-14B-Q6_K.gguf"
LLM_NAME_8B="Qwen3-8B-Q4_K_M.gguf"
VA_LLM_URL_14B="${VA_LLM_URL_14B:-https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/$LLM_NAME_14B}"
VA_LLM_URL_8B="${VA_LLM_URL_8B:-https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/$LLM_NAME_8B}"
VA_VOSK_URL="${VA_VOSK_URL:-https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip}"
VA_PIPER_URL="${VA_PIPER_URL:-https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx}"
VA_PIPER_JSON_URL="${VA_PIPER_JSON_URL:-$VA_PIPER_URL.json}"

GREEN=""; YELLOW=""; RED=""; BOLD=""; RESET=""
if [ -t 1 ]; then
    GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
fi

info()  { printf '%s==>%s %s\n' "$GREEN" "$RESET" "$*"; }
warn()  { printf '%s!!%s %s\n' "$YELLOW" "$RESET" "$*"; }
fail()  { printf '%sОшибка:%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

usage() {
    cat <<EOF
Установщик голосового ассистента (Arch Linux)

Использование: ./install.sh [опции]

Опции:
  --profile full|clean   full — приложение + модели, clean — только приложение
  --llm 14b|8b           LLM для полного профиля (по умолчанию спрашивает;
                         14b — Qwen3-14B-Q6_K ~12 ГБ, 8b — Qwen3-8B-Q4_K_M ~5 ГБ)
  --models-dir PATH      каталог моделей (по умолчанию ~/Models)
  --yes                  без вопросов, выбирать значения по умолчанию
  --cuda / --no-cuda     собрать llama-cpp-python с CUDA (без флага — спросит)
  --install-deps         разрешить установку системных пакетов через pacman
  --skip-system-deps     не проверять системные пакеты
  --no-launcher          не создавать лаунчер ~/.local/bin/voice-assistant
  -h, --help             эта справка
EOF
}

confirm() {
    local question="$1" default="${2:-y}" answer
    if [ "$ASSUME_YES" -eq 1 ]; then
        [ "$default" = "y" ]
        return
    fi
    if [ "$default" = "y" ]; then
        read -rp "$question [Y/n] " answer </dev/tty
        [ -z "$answer" ] || [[ "$answer" =~ ^[YyДд]$ ]]
    else
        read -rp "$question [y/N] " answer </dev/tty
        [[ "$answer" =~ ^[YyДд]$ ]]
    fi
}

ask_menu() {
    local prompt="$1"; shift
    local options=("$@") answer i
    echo ""
    for i in "${!options[@]}"; do
        printf '  %d) %s\n' "$((i + 1))" "${options[$i]}"
    done
    while true; do
        read -rp "$prompt " answer </dev/tty
        if [[ "$answer" =~ ^[0-9]+$ ]] && [ "$answer" -ge 1 ] && [ "$answer" -le "${#options[@]}" ]; then
            REPLY_INDEX=$((answer - 1))
            return
        fi
        warn "Введи число от 1 до ${#options[@]}."
    done
}

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --profile) PROFILE="${2:-}"; shift 2 ;;
            --llm) LLM_CHOICE="${2:-}"; shift 2 ;;
            --models-dir) MODELS_DIR="${2:-}"; shift 2 ;;
            --yes) ASSUME_YES=1; shift ;;
            --cuda) DO_CUDA=1; shift ;;
            --no-cuda) DO_CUDA=0; shift ;;
            --install-deps) INSTALL_DEPS=1; shift ;;
            --skip-system-deps) SKIP_DEPS=1; shift ;;
            --no-launcher) NO_LAUNCHER=1; shift ;;
            -h|--help) usage; exit 0 ;;
            *) fail "Неизвестная опция: $1 (см. ./install.sh --help)" ;;
        esac
    done
    case "$PROFILE" in
        ""|full|clean) ;;
        *) fail "Профиль должен быть full или clean." ;;
    esac
    case "$LLM_CHOICE" in
        ""|14b|8b) ;;
        *) fail "LLM должна быть 14b или 8b." ;;
    esac
}

check_arch() {
    command -v pacman >/dev/null 2>&1 || fail "Установщик поддерживает только Arch Linux (нужен pacman)."
    [ -f /etc/arch-release ] || warn "Файл /etc/arch-release не найден, продолжаю как на Arch."
}

check_base_tools() {
    for tool in python3 curl; do
        command -v "$tool" >/dev/null 2>&1 || fail "Не найдена команда $tool."
    done
    python3 - <<'PY' || fail "Нужен Python 3.10 или новее."
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
    python3 -c 'import venv' 2>/dev/null || fail "Модуль venv недоступен. Установи пакет python (sudo pacman -S python)."
    [ -f "$SCRIPT_DIR/requirements.txt" ] || fail "Рядом с install.sh нет requirements.txt."
}

check_system_deps() {
    if [ "$SKIP_DEPS" -eq 1 ]; then
        return 0
    fi
    local missing=()
    pacman -Qq python >/dev/null 2>&1 || missing+=(python)
    pacman -Qq portaudio >/dev/null 2>&1 || missing+=(portaudio)
    if [ "${#missing[@]}" -eq 0 ]; then
        return
    fi
    warn "Не хватает системных пакетов: ${missing[*]}"
    if [ "$INSTALL_DEPS" -eq 1 ] || { [ "$ASSUME_YES" -eq 0 ] && confirm "Установить их через sudo pacman -S?"; }; then
        sudo pacman -S --needed --noconfirm "${missing[@]}"
    else
        warn "Пропускаю установку пакетов. Установи вручную: sudo pacman -S ${missing[*]}"
    fi
}

create_venv() {
    if [ -x "$PY" ] && "$PY" -c 'import sys' 2>/dev/null; then
        info "Виртуальное окружение уже есть: $VENV_DIR"
    else
        info "Создаю виртуальное окружение..."
        python3 -m venv "$VENV_DIR"
    fi
}

install_python_deps() {
    info "Устанавливаю Python-зависимости (может занять пару минут)..."
    "$PIP" install --disable-pip-version-check -r "$SCRIPT_DIR/requirements.txt"
}

find_cuda_dir() {
    local nvcc
    nvcc="$(command -v nvcc || true)"
    if [ -z "$nvcc" ] && [ -x /opt/cuda/bin/nvcc ]; then
        nvcc=/opt/cuda/bin/nvcc
    fi
    if [ -z "$nvcc" ] && [ -x /usr/local/cuda/bin/nvcc ]; then
        nvcc=/usr/local/cuda/bin/nvcc
    fi
    if [ -n "$nvcc" ]; then
        dirname "$nvcc"
    fi
}

cuda_arch() {
    local cap
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' .' || true)"
    if [[ "$cap" =~ ^[0-9]+$ ]] && [ "$cap" -ge 50 ]; then
        printf '%s' "$cap"
    else
        printf 'native'
    fi
}

build_llama_cuda() {
    local cuda_dir="$1" arch ninja_missing=0
    arch="$(cuda_arch)"
    pacman -Qq ninja >/dev/null 2>&1 || ninja_missing=1
    if [ "$ninja_missing" -eq 1 ]; then
        if [ "$INSTALL_DEPS" -eq 1 ] || { [ "$ASSUME_YES" -eq 0 ] && confirm "Для сборки с CUDA нужен ninja. Установить?"; }; then
            sudo pacman -S --needed --noconfirm ninja
        else
            warn "Без ninja CUDA-сборка невозможна, остаюсь на CPU-версии."
            return
        fi
    fi
    info "Собираю llama-cpp-python с CUDA (архитектура $arch, займёт 10–20 минут)..."
    PATH="$cuda_dir:$PATH" CUDACXX="$cuda_dir/nvcc" \
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=$arch" \
        "$PIP" install --force-reinstall --no-cache-dir --no-binary llama-cpp-python llama-cpp-python==0.3.36
    "$PY" - <<'PY'
import sys
from llama_cpp import llama_supports_gpu_offload
print("GPU-оффлоад:", "доступен" if llama_supports_gpu_offload() else "недоступен")
if not llama_supports_gpu_offload():
    sys.exit(1)
PY
}

maybe_build_cuda() {
    local cuda_dir
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        if [ "$DO_CUDA" = "1" ]; then
            warn "NVIDIA GPU не найден, пропускаю CUDA-сборку."
        fi
        return 0
    fi
    cuda_dir="$(find_cuda_dir)"
    if [ -z "$cuda_dir" ]; then
        if [ "$DO_CUDA" = "1" ]; then
            warn "nvcc не найден (нужен пакет cuda), пропускаю CUDA-сборку."
        fi
        return 0
    fi
    if [ "$DO_CUDA" = "0" ]; then
        return
    fi
    if [ -z "$DO_CUDA" ] && [ "$ASSUME_YES" -eq 1 ]; then
        info "Режим --yes: пропускаю CUDA-сборку (для неё запусти с --cuda)."
        return
    fi
    if [ "$DO_CUDA" = "1" ] || confirm "Собрать llama-cpp-python с поддержкой CUDA? (~10–20 мин)"; then
        build_llama_cuda "$cuda_dir" || warn "CUDA-сборка не удалась, остаётся CPU-версия."
    fi
}

download() {
    local url="$1" target="$2" label="$3"
    if [ -s "$target" ]; then
        info "$label уже скачан: $target"
        return
    fi
    info "Скачиваю $label..."
    mkdir -p "$(dirname "$target")"
    if ! curl --fail --location --progress-bar --output "$target.part" "$url"; then
        rm -f "$target.part"
        fail "Не удалось скачать $label: $url. Проверь интернет и запусти install.sh ещё раз — готовые файлы он пропустит."
    fi
    mv "$target.part" "$target"
}

check_disk_space() {
    if [ "${VA_SKIP_DISK_CHECK:-0}" = "1" ]; then
        return 0
    fi
    local needed_mb available_mb
    if [ "$LLM_CHOICE" = "14b" ]; then
        needed_mb=19500
    else
        needed_mb=11200
    fi
    mkdir -p "$MODELS_DIR"
    available_mb="$(df -Pk "$MODELS_DIR" | awk 'NR==2 {print int($4 / 1024)}')"
    if [ "$available_mb" -lt "$needed_mb" ]; then
        fail "Не хватает места: нужно ~${needed_mb} МБ в $MODELS_DIR, свободно ${available_mb} МБ."
    fi
}

choose_profile() {
    if [ -z "$PROFILE" ]; then
        if [ "$ASSUME_YES" -eq 1 ]; then
            PROFILE=clean
        else
            ask_menu "Выбери вариант установки:" \
                "Полный — приложение и все модели, готово к запуску" \
                "Чистый — только приложение и зависимости (модели настроишь в приложении)"
            if [ "$REPLY_INDEX" -eq 0 ]; then PROFILE=full; else PROFILE=clean; fi
        fi
    fi
    info "Профиль установки: $PROFILE"
}

choose_llm() {
    if [ "$PROFILE" != "full" ]; then
        return 0
    fi
    if [ -z "$LLM_CHOICE" ]; then
        if [ "$ASSUME_YES" -eq 1 ]; then
            LLM_CHOICE=14b
        else
            ask_menu "Какую LLM скачать?" \
                "Qwen3-14B-Q6_K (~12 ГБ, нужно 8+ ГБ VRAM, рекомендовано)" \
                "Qwen3-8B-Q4_K_M (~5 ГБ, для слабых GPU и CPU)"
            if [ "$REPLY_INDEX" -eq 0 ]; then LLM_CHOICE=14b; else LLM_CHOICE=8b; fi
        fi
    fi
}

install_models() {
    if [ "$PROFILE" != "full" ]; then
        return 0
    fi
    choose_llm
    MODELS_DIR="${MODELS_DIR/#\~/$HOME}"
    check_disk_space

    local llm_url llm_name
    if [ "$LLM_CHOICE" = "14b" ]; then
        llm_url="$VA_LLM_URL_14B"; llm_name="$LLM_NAME_14B"
    else
        llm_url="$VA_LLM_URL_8B"; llm_name="$LLM_NAME_8B"
    fi
    download "$llm_url" "$MODELS_DIR/$llm_name" "LLM $llm_name"
    LLM_PATH="$MODELS_DIR/$llm_name"

    command -v unzip >/dev/null 2>&1 || fail "Не найдена команда unzip (sudo pacman -S unzip)."
    VOSK_DIR="$(find "$MODELS_DIR" -maxdepth 1 -mindepth 1 -type d -name 'vosk-*' 2>/dev/null | head -n1)"
    if [ -z "$VOSK_DIR" ]; then
        download "$VA_VOSK_URL" "$MODELS_DIR/vosk-model.zip" "модель Vosk"
        info "Распаковываю модель Vosk..."
        unzip -q -o "$MODELS_DIR/vosk-model.zip" -d "$MODELS_DIR"
        rm -f "$MODELS_DIR/vosk-model.zip"
        VOSK_DIR="$(find "$MODELS_DIR" -maxdepth 1 -mindepth 1 -type d -name 'vosk-*' 2>/dev/null | head -n1)"
        [ -n "$VOSK_DIR" ] || fail "Не удалось распаковать модель Vosk."
    else
        info "Модель Vosk уже установлена: $VOSK_DIR"
    fi

    local piper_name
    piper_name="$(basename "$VA_PIPER_URL")"
    download "$VA_PIPER_URL" "$MODELS_DIR/$piper_name" "голос Piper"
    download "$VA_PIPER_JSON_URL" "$MODELS_DIR/$piper_name.json" "конфиг голоса Piper"
    PIPER_PATH="$MODELS_DIR/$piper_name"
}

write_config() {
    if [ "$PROFILE" != "full" ]; then
        return 0
    fi
    local config_dir config_path
    config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/voice-assistant"
    config_path="$config_dir/config.json"
    if [ -f "$config_path" ] && [ "$ASSUME_YES" -eq 0 ]; then
        if ! confirm "Файл настроек уже существует. Обновить пути к моделям в нём?"; then
            warn "Настройки не изменены: $config_path"
            return
        fi
    fi
    VA_CFG_VOSK="$VOSK_DIR" VA_CFG_LLM="$LLM_PATH" VA_CFG_TTS="$PIPER_PATH" \
        "$PY" - "$config_path" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = {}
if path.exists():
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
data.update({
    "vosk_model_path": os.environ["VA_CFG_VOSK"],
    "llm_model_path": os.environ["VA_CFG_LLM"],
    "tts_model_path": os.environ["VA_CFG_TTS"],
    "tts_enabled": True,
    "tts_output_device": "auto",
    "tts_volume": data.get("tts_volume", 100),
    "disable_thinking": True,
    "llm_gpu_layers": data.get("llm_gpu_layers", 20),
})
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Настройки записаны: {path}")
PY
}

install_launcher() {
    if [ "$NO_LAUNCHER" -eq 1 ]; then
        return 0
    fi
    local bin_dir desktop_dir
    bin_dir="$HOME/.local/bin"
    desktop_dir="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    mkdir -p "$bin_dir" "$desktop_dir"
    cat > "$bin_dir/voice-assistant" <<EOF
#!/usr/bin/env bash
exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/chat.py" "\$@"
EOF
    chmod +x "$bin_dir/voice-assistant"
    cat > "$desktop_dir/voice-assistant.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Голосовой ассистент
Comment=Локальный голосовой ассистент: Vosk, llama.cpp, Piper
Exec=$VENV_DIR/bin/python $SCRIPT_DIR/chat.py
Path=$SCRIPT_DIR
Terminal=true
Categories=Utility;Audio;AudioVideo;
EOF
    info "Лаунчер: $bin_dir/voice-assistant"
    info "Ярлык: $desktop_dir/voice-assistant.desktop"
    case ":$PATH:" in
        *":$bin_dir:"*) ;;
        *) warn "Каталог $bin_dir не в PATH — запускай по полному пути или добавь его в PATH." ;;
    esac
}

summary() {
    echo ""
    info "${BOLD}Установка завершена.${RESET}"
    if [ "$PROFILE" = "full" ]; then
        echo "   Модели: $MODELS_DIR"
    else
        echo "   Модели не скачивались — укажи их в настройках приложения (Ctrl+2)."
    fi
    if [ "$NO_LAUNCHER" -eq 0 ]; then
        echo "   Запуск: voice-assistant"
    fi
    echo "   Или: $VENV_DIR/bin/python $SCRIPT_DIR/chat.py"
}

main() {
    parse_args "$@"
    check_arch
    check_base_tools
    choose_profile
    check_system_deps
    create_venv
    install_python_deps
    maybe_build_cuda
    install_models
    write_config
    install_launcher
    summary
}

main "$@"
