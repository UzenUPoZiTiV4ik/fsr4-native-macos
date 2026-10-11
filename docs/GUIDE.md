# Инструкция: FSR 4.1.1 на Mac

## 1. Что нужно

- Mac на Apple Silicon (быстрее всего на M5, подробнее: [HARDWARE.md](HARDWARE.md))
- macOS 26+, Xcode 26+, Python 3
- CrossOver, если хотите запускать игры

## 2. Сборка

```sh
xcodebuild -downloadComponent MetalToolchain
make -B
tools/bin/fsr4mac --backend-info
```

Последняя команда показывает, какой режим GPU будет использован.

## 3. Установка в CrossOver

1. Полностью закройте игру и её бутылку.
2. Выполните:

   ```sh
   python3 tools/crossover/crossover_fsr4.py install --panel --profile lossy
   ```

   Если бутылка одна, она выбирается сама. Если их несколько, добавьте
   `--bottle "Имя бутылки"` (или путь к `cxbottle.conf`).
3. Запустите игру и включите в настройках **DLSS**.
4. Появится панель **FSR 4**. Выберите режим:

| Режим | Что это |
| --- | --- |
| FSR 4.1 Lite | Самый быстрый (по умолчанию) |
| FSR 4.1 | Полное качество |
| FSR 4 | Старая модель 4.0.2, для сравнения |

Режим меняется сразу, перезапуск не нужен.

## 4. Как всё вернуть

```sh
python3 tools/crossover/crossover_fsr4.py restore --bottle "/path/to/cxbottle.conf"
```

Затем перезапустите бутылку. Сам CrossOver.app установщик не изменяет.

## 5. Если что-то не так

- Лог: `dist/crossover/hook.log`.
- Шлейфы при движении: нажмите в панели **Record diagnostics**, подвигайте
  камеру до сообщения «Done». Запись появится в `dist/crossover/captures/`.
  Анализ:

  ```sh
  python3 tools/diagnostics/analyze_motion_capture.py dist/crossover/captures/<каталог> --json motion-analysis.json
  ```

- Подробности: [CROSSOVER.md](CROSSOVER.md).

## Для продвинутых

### Настройки через окружение бутылки

Сохранённый выбор в панели важнее переменных. Чтобы использовать только
переменные, задайте `FSR4_SETTINGS=none`.

| Переменная | Значения |
| --- | --- |
| `FSR4_PROFILE` | `lossy` (по умолчанию), `exact-fast`, `exact`, `fsr4` |
| `FSR4_VERSION` | `4.1.1` (по умолчанию) или `4.0.2` |
| `FSR4_MODEL` | `auto`, `i8mpp`, `simd` (M1–M4, по умолчанию там), `scalar`, `reference` |
| `FSR4_SHARPNESS`, `FSR4_RCAS` | Резкость (по умолчанию RCAS 0.4) |
| `FSR4_OPTIMIZE` | `1` по умолчанию, `0` отключает оптимизации |
| `FSR4_COMPILE` | `0` запрещает компиляцию недостающих шейдеров |

`exact` (битово-точный эталон) доступен только через `FSR4_PROFILE`.

### Запуск без игры

```sh
tools/bin/gen_testcase testcases/quality411 quality 1280 720 1920 1080 8
tools/bin/fsr4mac testcases/quality411 out/lossy --profile lossy
tools/bin/fsr4mac testcases/quality411 out/bench --profile lossy --bench 10
```

Результат: PNG, `.bin` и `run.json`. Формат входа:
[TESTDATA.md](../TESTDATA.md), [FSR411_INPUT_CONTRACT.md](FSR411_INPUT_CONTRACT.md).
Методика замеров: [BENCHMARK.md](../BENCHMARK.md).

### Тесты

```sh
make test
make build-fsr411-tests
make test-fsr411-gpu
```

Старая модель 4.0.2: [GUIDE_FSR402.md](GUIDE_FSR402.md).
