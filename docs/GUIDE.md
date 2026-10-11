# Запуск FSR 4.1.1 для Mac

По умолчанию используются **FSR 4.1.1 и Lossy**. Exact (эталонная
математика) и Exact Fast (та же сеть, быстрая математика PRE/POST, примерно
на 0,6 мс быстрее, разница с Exact глазом не видна) доступны в CLI,
параметрах запуска и панели. После переключения профиля история обновляется
с полного прохода сети.

## Требования и сборка

Нужны Apple Silicon, macOS 26+, Xcode 26+ с Metal Toolchain и Python 3.
Нативные INT8 tensor operations используются на M5 / Apple GPU family 10.
На более ранних Apple GPU выбирается scalar INT8; скорость на этих машинах
ещё не измерена. Подробности: [HARDWARE.md](HARDWARE.md).

```sh
xcodebuild -downloadComponent MetalToolchain
make -B
tools/bin/fsr4mac --backend-info
```

`--backend-info` сообщает выбранный по возможностям GPU путь. Реальная
компиляция модели выполняется при первом запуске кадров.

## Проверка на записи кадров

Формат `meta.json`, цвета, глубины и движения описан в
[TESTDATA.md](../TESTDATA.md), дополнения 4.1.1 — в
[FSR411_INPUT_CONTRACT.md](FSR411_INPUT_CONTRACT.md).

```sh
tools/bin/gen_testcase testcases/quality411 quality 1280 720 1920 1080 8
tools/bin/fsr4mac testcases/quality411 out/lossy --profile lossy
tools/bin/fsr4mac testcases/quality411 out/exact --profile exact
```

На выходе: `fNN_out.bin` (RGBA16F), `fNN_out.png` и `run.json`. В JSON
записаны модель, профиль, фактический backend, сбросы истории и число полных
проходов/повторно использованных результатов сети.

Для поиска ошибки можно выбрать `--reference` (импортированные HLSL),
`--scalar` (обычные целочисленные Metal-ядра) или `--i8mpp` (M5 tensor
operations). `--optimize 0` отключает оптимизации PRE и POST.
`--capture-stages pre,pass1,post` сохраняет выбранные промежуточные буферы,
`--dump-state` — историю, recurrent, scratch и экспозицию. Такие копии
замедляют прогон; для измерения скорости запускайте отдельно:

```sh
tools/bin/fsr4mac testcases/quality411 out/bench --profile lossy --bench 10
```

Одна последовательность прогревает GPU, следующие десять измеряются.
Каждая начинается с новой истории. Смотрите `effective_reset`, `skipped_inference` и
`gpu_ms` у каждого кадра: стоимость первого кадра и кадра с повторным
использованием сети различается. Параметр старого раннера `--reps` к 4.1.1
не относится. Методика и результаты — в [BENCHMARK.md](../BENCHMARK.md).

## CrossOver и переключение профилей

Полностью закройте игру и процессы её бутылки перед обновлением хука.
Установщик собирает локальную копию D3DMetal, хук и панель, проверяет их
на записи кадров и указывает бутылке путь к копии фреймворка.

```sh
python3 tools/crossover/crossover_fsr4.py install --panel --profile lossy
# Для конкретной бутылки:
python3 tools/crossover/crossover_fsr4.py install --panel --profile lossy --bottle "/path/to/cxbottle.conf"
# Только сборка и проверка dist/, без изменения бутылки:
python3 tools/crossover/crossover_fsr4.py install --no-bottle --profile lossy
```

Затем запустите игру и включите DLSS. В панели **FSR 4** выберите режим
**FSR 4** (прежняя модель 4.0.2), **FSR 4.1** (Exact Fast) или
**FSR 4.1 Lite** (Lossy, самый быстрый); переключение сбрасывает историю.
Битово-точный Exact задаётся только через `FSR4_PROFILE=exact`. Панель на
английском. Профиль сохраняется в `dist/crossover/panel.conf` и применяется
на следующем кадре; файл настроек перечитывается во время работы.

Для разбора шлейфа кнопка **Record diagnostics** через 1,5 секунды
сохраняет восемь последовательных кадров одного скейлера. Во время записи
подвигайте камеру и дождитесь сообщения «Done». В `dist/crossover/captures/`
появятся `capture.json`, входные color/depth/motion и выход FSR в float32.
Запись ограничена 512 MiB; при больших разрешениях число кадров уменьшается.
Без нажатия кнопки копии текстур не выполняются. Анализ движения:

```sh
python3 tools/diagnostics/analyze_motion_capture.py dist/crossover/captures/<каталог> --json motion-analysis.json
```

`dist/crossover/hook.log` записывает входные масштабы и jitter, активные
области текстур, отдельные скейлеры и заглушки, сбросы истории, профиль,
полные/повторно использованные результаты сети и длительность подготовки.
Подробные строки ограничены первыми 16 вызовами каждого скейлера,
сменой профиля и одним срезом каждые пять секунд.

Для запуска без файловых настроек задайте `FSR4_SETTINGS=none` и
`FSR4_PROFILE=lossy`, `exact-fast`, `exact` или `fsr4` в окружении бутылки. Иначе сохранённое
значение панели имеет приоритет над первоначальным значением окружения.

| Настройка | Значение |
| --- | --- |
| `FSR4_VERSION` | `4.1.1` по умолчанию; `4.0.2` закрепляет старую модель, режимы панели тогда не действуют. |
| `FSR4_PROFILE` | `lossy` по умолчанию, `exact-fast`, `exact` или `fsr4` (модель 4.0.2). |
| `FSR4_MODEL` | `auto`, `i8mpp`, `scalar`, `reference`. |
| `FSR4_PANEL` | `1` открывает панель; устанавливается через `--panel`. |
| `FSR4_SHARPNESS`, `FSR4_RCAS` | Настройки резкости; игровой путь по умолчанию включает RCAS с резкостью 0.4. |
| `FSR4_OPTIMIZE` | `1` по умолчанию; `0` для сравнения PRE/POST. |
| `FSR4_COMPILE` | `0` запрещает компиляцию отсутствующих вариантов. |

Режим Lossy изменяет модель и может влиять на мелкие детали и движение.
Он переиспользует результат сети только при подходящем jitter, масштабе и
валидном кэше, с полным обновлением после каждого такого кадра. Ultra
Performance имеет отдельную модель и выполняет сеть каждый кадр.

Входы уже преобразованы D3DMetal из DLSS в MetalFX. Хук не меняет знак
motion/jitter повторно; он учитывает размеры активных областей, exposure и
флаги дескриптора. Детали и диагностические сообщения — в
[CROSSOVER.md](CROSSOVER.md) и [FSR411_INPUT_CONTRACT.md](FSR411_INPUT_CONTRACT.md).

Откат настройки бутылки:

```sh
python3 tools/crossover/crossover_fsr4.py restore --bottle "/path/to/cxbottle.conf"
```

После этого перезапустите процессы бутылки. Исходный CrossOver.app
установщик не изменяет.

## Проверки и старый эталон

```sh
make test
make build-fsr411-tests
make test-fsr411-gpu
```

Проверяются арифметика INT8, Exact/Lossy, общая и UP-модель, преобразование
входов, прямые HLSL против Metal tensor/scalar, кэш и игровой мост.
Измерения на M5 не заменяют тестирование других Apple GPU и конкретной игры.

Windows-эталон FSR 4.0 относится к прежней модели и не используется для
подтверждения 4.1.1. Для старой модели передайте `--fsr-version 4.0.2`;
её команды и FP16 сохранены в [GUIDE_FSR402.md](GUIDE_FSR402.md).

Ошибка шлейфа из-за знаковой цветовой разности в Metal PRE описана в
[FSR411_GHOSTING_FIX.md](FSR411_GHOSTING_FIX.md). Её регрессионная проверка
`make test-fsr411-chroma` также входит в набор GPU-тестов.
