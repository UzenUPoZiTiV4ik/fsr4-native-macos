> Historical FSR 4.0.2 documentation. Select `--fsr-version 4.0.2` for its CLI commands or `FSR4_VERSION=4.0.2` for the live bridge. Current defaults and instructions are in [GUIDE.md](GUIDE.md).

# Гайд по запуску fsr4-mac

Пошагово: от чистого Mac до FSR 4 в игре через CrossOver. Каждый шаг проверен
на свежем клоне репозитория: Apple M5, macOS 27.0, Xcode 27.0, CrossOver Preview.

Все команды выполняются **из корня репозитория**: `fsr4mac` ищет кэш шейдеров
в папке `build/` относительно текущего каталога.

## Содержание

1. [Что понадобится](#1-что-понадобится)
2. [Сборка](#2-сборка)
3. [Тестовые кадры](#3-тестовые-кадры)
4. [Запуск апскейлера](#4-запуск-апскейлера)
5. [Сравнение результатов](#5-сравнение-результатов)
6. [Сверка с эталоном AMD](#6-сверка-с-эталоном-amd)
7. [Замер скорости](#7-замер-скорости)
8. [FSR 4 в играх через CrossOver](#8-fsr-4-в-играх-через-crossover)
9. [Обновление](#9-обновление)
10. [Если что-то не работает](#10-если-что-то-не-работает)
11. [Справочник](#11-справочник)

---

## 1. Что понадобится

| Что | Зачем | Проверка |
|---|---|---|
| Mac на Apple Silicon, macOS 26 или новее | Проверено на M5 с macOS 27.0. Режим `auto` выбирает FP16 на M1–M4 и ускоренный INT8 на совместимых M5, включая Pro/Max/Ultra. Подробности и ограничения проверки — [HARDWARE.md](HARDWARE.md) | `sysctl -n machdep.cpu.brand_string`, `sw_vers` |
| Xcode (полный, не только Command Line Tools) | `clang++`, `xcrun metal`, `xcrun metallib` | `xcode-select -p` → путь внутри `Xcode.app` |
| Metal Toolchain | С Xcode 26 компилятор Metal ставится отдельным компонентом | `xcrun -f metal` |
| Python 3 и numpy | Скрипты генерации шейдеров; numpy нужен `tools/gen_i8mpp.py`, то есть `--i8mpp` и игровому пути | `python3 -c "import numpy; print(numpy.__version__)"` |
| GitHub CLI (`gh`) | Клонировать репозиторий и скачать эталонные кадры | `gh auth status` |

`dxc`, `spirv-cross`, `spirv-val` и `libdxcompiler.dylib` лежат в репозитории
(`tools/bin/`, `tools/lib/`), ставить их не нужно.

Для игр дополнительно:

| Что | Проверка |
|---|---|
| CrossOver Preview **в `/Applications`** — путь зашит в установщик | `ls "/Applications/CrossOver Preview.app"` |
| Rosetta 2 — Wine в CrossOver работает как x86_64 | `arch -x86_64 /usr/bin/true && echo ok` |
| Бутылка с игрой; по умолчанию установщик берёт бутылку «Игры» | `ls ~/Library/Application\ Support/CrossOver/Bottles/` |
| DLSS через MetalFX включён в бутылке | в `cxbottle.conf`, секция `[EnvironmentVariables]`: `"D3DM_ENABLE_METALFX" = "1"` |

Установка недостающего (Xcode — из App Store, `gh` — с
[cli.github.com](https://cli.github.com)):

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer   # если xcode-select -p показывает CommandLineTools
sudo xcodebuild -license accept
xcodebuild -downloadComponent MetalToolchain
python3 -m pip install --user numpy
gh auth login
softwareupdate --install-rosetta --agree-to-license               # только для игр
```

Место на диске: около 0,5 ГБ под тестовые кадры, результаты, кэш шейдеров и
копию D3DMetal. Эталонные кадры AMD (шаг 6) — ещё около 2,7 ГБ; архивы после
распаковки можно удалить.

## 2. Сборка

```bash
gh repo clone UzenUPoZiTiV4ik/fsr4-mac
cd fsr4-mac
make -B
```

`make -B` собирает `fsr4mac` и утилиты `gen_testcase`, `compare`, `bilin`,
`bin2png` в `tools/bin/` (около 30 секунд).

Почему `-B`: собранные бинарники лежат в git, и в свежем клоне они новее
исходников, поэтому обычный `make` отвечает `Nothing to be done` и ничего не
собирает. С `-B` бинарники гарантированно собраны из текущего кода вашим
Xcode. Дальше, после правок исходников, хватает обычного `make`. После
`make -B` git может показать изменения в `tools/bin/` — это пересобранные
бинарники.

Проверка GPU:

```bash
tools/bin/fsr4mac --caps
```

`automatic backend` показывает выбранный режим и причину. На M5 дополнительно
проверяется реальное INT8-умножение матриц: `INT8 matmul dispatch + exact result: ok`.
Успешная компиляция Metal 4 сама по себе не означает наличие GPU-ускорителей:
M1–M4 автоматически используют FP16. Проверить выбор без компиляции модели:
`tools/bin/fsr4mac --backend-info`. При отказе native INT8 оба пути переходят
на FP16; игровой энкодер повторяет тот же кадр.

## 3. Тестовые кадры

```bash
tools/bin/gen_testcase testcases/testcase_quality_720to1080 quality 1280 720 1920 1080 8
tools/bin/gen_testcase testcases/testcase_perf_540to1080 performance 960 540 1920 1080 8
```

Генератор детерминированный: синтетическая сцена (шахматка, градиент, мелкие
полосы, движущийся квадрат), по 8 кадров цвета, глубины и векторов движения и
`meta.json` с параметрами. Первый кейс — пресет quality, 1280×720 → 1920×1080;
второй — performance, 960×540 → 1920×1080. Формат описан в
[TESTDATA.md](../TESTDATA.md).

Аргументы: `gen_testcase <папка> <пресет> <ширина_вх> <высота_вх> <ширина_вых>
<высота_вых> <кадров>`. Пресеты: `native`, `quality`, `balanced`,
`performance`, `ultraperf`, `drs`.

## 4. Запуск апскейлера

```bash
tools/bin/fsr4mac testcases/testcase_quality_720to1080 out/ --auto
open out/f07_out.png
```

Первый запуск каждой модели занимает до минуты: шейдеры компилируются
(HLSL → SPIR-V → MSL → metallib) и кэшируются в `build/`. Повторные запуски
стартуют сразу. Для каждого кадра печатается время GPU, в конце — строка
`gpu_time_ms: … mean …`.

В `out/` появятся `f00_out.bin` … `f07_out.bin` (результат в RGBA16F,
1920×1080) и такие же `.png` для просмотра. Установщик на шаге 8 создаст собственный эталонный кадр для выбранного на этом Mac режима.

### Модели

| Ключ | Модель | Кадр quality 720→1080 на M5 | PSNR против эталона AMD |
|---|---|---|---|
| `--auto` | по умолчанию в CLI и играх: M1–M4 → FP16, M5 → INT8 | зависит от GPU | зависит от модели |
| `--i8mpp` | int8 на нейроускорителях M5; с проверкой возможностей и переходом на FP16 при отказе | ~5–8 мс | 42–49 dB |
| `--int8` | эталонная int8-модель на сгенерированных шейдерах; включается явно | ~48–60 мс | 42–49 dB |
| `--fp16` | веса заранее переведены в fp16 | ~13–17 мс | 39–43 dB |

```bash
tools/bin/fsr4mac testcases/testcase_quality_720to1080 out/int8/ --int8
tools/bin/fsr4mac testcases/testcase_quality_720to1080 out/fp16/ --fp16
```

Время указано для работы от сети. От батареи с включённым режимом
энергосбережения (Low Power Mode) частоты GPU ниже, и кадр считается заметно
дольше. Время GPU меняется и от запуска к запуску, поэтому сравнивайте цифры
только внутри одного сеанса.

Полезное:

- `--dump-passes` — сохранить промежуточный буфер сети после каждого прохода
  (`fNN_passPP_scratch.bin`); медленно.
- `MTL_DEBUG_LAYER=1 tools/bin/fsr4mac …` — проверка вызовов Metal API.
- `MTL_SHADER_VALIDATION=1` для сверки не используйте: эта валидация меняет
  результат даже на настоящем железе.

## 5. Сравнение результатов

Два прогона между собой — по каналам: максимальная и средняя разница, PSNR;
`diff_f07.png` — разница, усиленная в 16 раз:

```bash
tools/bin/compare out/f07_out.bin out/int8/f07_out.bin 1920 1080 out/diff_f07.png
```

Билинейное увеличение того же кадра — чтобы сравнить с FSR 4 на глаз:

```bash
tools/bin/bilin testcases/testcase_quality_720to1080/f07_color.bin 1280 720 1920 1080 out/f07_bilin.bin out/f07_bilin.png
open out/f07_bilin.png out/f07_out.png
```

## 6. Сверка с эталоном AMD

Эталон — выход оригинального DX12-провайдера FSR 4 от AMD, прогнанного на
Windows (программный адаптер WARP). Он лежит в релизе
[`golden-v0.1`](https://github.com/UzenUPoZiTiV4ik/fsr4-mac/releases/tag/golden-v0.1):
два архива по ~420 МБ, в распакованном виде ~0,9 ГБ каждый. Внутри — входные
кадры, эталонные выходы и дампы всех проходов сети. Архивы и релиз не
меняйте — это эталон.

```bash
mkdir -p testcases/golden
gh release download golden-v0.1 -R UzenUPoZiTiV4ik/fsr4-mac -p 'testcase_*.zip' -D testcases/golden
cd testcases/golden
unzip -q testcase_quality_720to1080.zip
unzip -q testcase_perf_540to1080.zip
cd ../..
```

Архивы собраны на Windows, и пути внутри записаны через `\`. `unzip`
раскладывает их по папкам правильно, но печатает `appears to use backslashes
as path separators` и завершается с кодом 1 — это нормально, поэтому команды
выше не склеены через `&&`. `ditto` пути не исправляет и создаёт файлы с `\` в
имени.

Прогон и сравнение. Первым аргументом `compare` идёт папка эталона: в ней
`meta.json` и эталонные `fNN_out.bin`.

```bash
tools/bin/fsr4mac testcases/golden/testcase_quality_720to1080 out/golden_quality/ --i8mpp
tools/bin/compare testcases/golden/testcase_quality_720to1080 out/golden_quality/

tools/bin/fsr4mac testcases/golden/testcase_perf_540to1080 out/golden_perf/ --i8mpp
tools/bin/compare testcases/golden/testcase_perf_540to1080 out/golden_perf/
```

Ожидаемый результат для `--i8mpp`: PSNR по кадрам около 42–47 dB (quality) и
42–49 dB (performance), итоговая строка `summary: … psnr` — около 43–44 dB.
Остаточная разница — округления на границах квантования int8, подробно в
[VERIFICATION.md](../VERIFICATION.md).

Папки `testcases/` и `out/` в git не попадают (они в `.gitignore`).

## 7. Замер скорости

```bash
tools/bin/fsr4mac testcases/testcase_quality_720to1080 /tmp/fsr4-bench --i8mpp --bench 3 --reps 10
```

Главные строки: `TOTAL(per-pass sum)` — сумма времени стадий, и
`fused frame gpu_ms` — весь кадр одним command buffer. В режиме `--bench` кадры
на диск не пишутся. Для замеров подключите питание и выключите Low Power Mode. Протокол замеров и таблицы — в [BENCHMARK.md](../BENCHMARK.md).

## 8. FSR 4 в играх через CrossOver

DLSS в CrossOver Preview реализован внутри D3DMetal через временной скейлер
MetalFX. Хук `libfsr4hook.dylib` подменяет у этого скейлера кодирование кадра
на FSR 4 — тот же пайплайн, что у `fsr4mac --i8mpp`, плюс резкость RCAS 0.4
по умолчанию (своей настройки резкости у DLSS нет). `CrossOver.app` не
меняется: установщик копирует D3DMetal в `dist/crossover/`, добавляет в копию
загрузку хука и через переменные окружения направляет на неё одну бутылку.
Устройство подробно — в [CROSSOVER.md](CROSSOVER.md).

### 8.1. Подготовка

1. Выполните сборку из шага 2. Установщик сам создаёт отсутствующий quality-тест
   и эталонный кадр с подходящим для GPU режимом; старый `out/f00_out.bin` не нужен.
2. В бутылке включён DLSS через MetalFX (`"D3DM_ENABLE_METALFX" = "1"` в
   `cxbottle.conf`).
3. Шейдеры для игры скомпилированы заранее, иначе первый кадр будет ждать
   компиляции:

   ```bash
   python3 tools/precompile_models.py
   ```

   Скрипт собирает варианты для пресетов quality, balanced, performance и
   ultraperf под выход до 1080p и до 2160p, примерно за полминуты. Можно только
   нужные: `python3 tools/precompile_models.py --presets balanced performance`.
4. CrossOver закрыт полностью (⌘Q), а не только окно игры: пока он запущен, в
   памяти остаются старые библиотеки.

### 8.2. Установка

```bash
python3 tools/crossover/crossover_fsr4.py install --panel
```

Около 30 секунд. Установщик:

1. собирает хук `dist/crossover/libfsr4hook.dylib` (x86_64, ad-hoc подпись);
2. копирует `D3DMetal.framework` из CrossOver Preview в `dist/crossover/` и
   добавляет в копию загрузку хука;
3. проверяет результат: подмена скейлера срабатывает, копия загружает хук,
   `live_test` считает кадр через хук и сравнивает его со свежим эталоном
   в `dist/crossover/reference-<режим>/f00_out.bin`;
4. собирает приложение `dist/crossover/FSR4 Panel.app`;
5. дописывает ключи в конец `cxbottle.conf` бутылки «Игры». Конфиг в том виде,
   какой был до первой установки, сохраняется в `dist/crossover/backups/`.

Успешная установка выводит среди прочего:

```
live vs fsr4mac  mean_abs=0.000000 max_abs=0.000000 …
live ok
original CrossOver.app was not modified
```

Строка `Error Domain=com.apple.appleneuralengine Code=26 … Bad argument error`
в середине вывода — сообщение MetalFX под Rosetta, на результат не влияет.
Признак успеха — `live ok`.

Другие варианты:

```bash
# другая бутылка
python3 tools/crossover/crossover_fsr4.py install --panel --bottle "$HOME/Library/Application Support/CrossOver/Bottles/<бутылка>/cxbottle.conf"
# только собрать dist/crossover/, конфиг бутылки не трогать
python3 tools/crossover/crossover_fsr4.py install --no-bottle
```

Без `--panel` всё то же самое, только панель не будет запускаться сама.

Проверьте, что ключи попали в секцию `[EnvironmentVariables]`. Установщик
пишет в конец файла; в бутылке «Игры» эта секция последняя, в другой бутылке
может быть иначе:

```bash
grep -n -E '^\[|fsr4-mac|FSR4_|D3DMETAL|D3DM_ENABLE_METALFX' "$HOME/Library/Application Support/CrossOver/Bottles/Игры/cxbottle.conf"
```

Последним заголовком секции в выводе должен быть `[EnvironmentVariables]`, а
после него — `D3DM_ENABLE_METALFX`, строка `;; fsr4-mac` и ключи
`D3DMETAL_FRAMEWORK_PATH`, `FSR4_HOOK_LOG`, `FSR4_ROOT`, `FSR4_REPLACE`,
`FSR4_PANEL`.

Папку репозитория после установки не переносите: в бутылку записаны
абсолютные пути. Если перенесли — повторите `install --panel` из нового места.

### 8.3. Запуск игры

1. Запустите CrossOver и игру из бутылки.
2. В графических настройках игры включите **DLSS**. Пресет FSR 4 выбирается по
   отношению выходного разрешения к входному по горизонтали, как в провайдере
   AMD: от 1,5× — quality, от 1,69× — balanced, от 1,99× — performance, от
   2,99× — ultraperf, меньше 1,5× — native.
3. С `--panel` в Dock появится приложение **FSR 4**. Оно показывает, работает
   ли FSR 4 (а если нет — почему), модель и пресет, разрешение вход → выход,
   время кадра игры, и даёт слайдер резкости RCAS. Окно висит поверх
   полноэкранной игры и не забирает у неё фокус; резкость меняется сразу и
   сохраняется. Открыть вручную: `open "dist/crossover/FSR4 Panel.app"`.
4. Лог хука:

   ```bash
   tail -f dist/crossover/hook.log
   ```

   Когда всё работает, в нём идут `loaded pid=…`, затем
   `swizzled … encodeToCommandBuffer:`, затем `encode mtl4 …`.

Первый кадр после смены разрешения или пресета может подвиснуть на несколько
секунд: компилируется вариант шейдера, которого ещё нет в кэше. Это
происходит один раз.

### 8.4. Настройки

Переменные задаются в `cxbottle.conf`, в секции `[EnvironmentVariables]`,
строками вида `"FSR4_SHARPNESS" = "0.6"`. После правки полностью перезапустите
CrossOver.

| Переменная | Значения | Что делает |
|---|---|---|
| `FSR4_REPLACE` | `1` (по умолчанию), `0` | `0` возвращает оригинальный MetalFX без удаления хука — удобно для сравнения |
| `FSR4_MODEL` | `auto` (по умолчанию), `f16`, `i8mpp` | выбор по возможностям GPU; `f16` принудительно включает FP16, `i8mpp` сохраняет проверки и запасной FP16 |
| `FSR4_SHARPNESS` | от `0` до `1`, по умолчанию `0.4` | резкость RCAS; `0` выключает |
| `FSR4_RCAS` | `0` | выключить RCAS |
| `FSR4_PRESET` | `native`, `quality`, `balanced`, `performance`, `ultraperf`, `drs` | задать пресет вручную |
| `FSR4_COLORSPACE` | `srgb`, `pq` | цветовое пространство входа; по умолчанию линейное |
| `FSR4_STUB` | `skip` (по умолчанию), `metalfx`, `run`, `reset` | что делать с вызовами, где вместо векторов движения заглушка 4×4 |
| `FSR4_JITTERED_MV`, `FSR4_LOW_RES_MV`, `FSR4_AUTOEXPOSURE` | `0`, `1` | переопределить флаги, которые игра передала скейлеру |
| `FSR4_QBARRIER` | `0` | убрать барьеры очереди Metal 4 вокруг FSR 4 (для отладки) |
| `FSR4_PANEL` | `1` | запускать панель FSR 4; ставится `install --panel` |
| `FSR4_SETTINGS` | путь, `none` | где лежат настройки панели (по умолчанию `dist/crossover/panel.conf`); `none` — не читать их |

Настройки из панели важнее `FSR4_SHARPNESS` и `FSR4_RCAS`. Ключи
`D3DMETAL_FRAMEWORK_PATH`, `FSR4_HOOK_LOG` и `FSR4_ROOT` пишет установщик —
вручную их не меняйте.

### 8.5. Откат

```bash
python3 tools/crossover/crossover_fsr4.py restore
# другая бутылка:
python3 tools/crossover/crossover_fsr4.py restore --bottle "$HOME/Library/Application Support/CrossOver/Bottles/<бутылка>/cxbottle.conf"
```

`restore` убирает из конфига строку `;; fsr4-mac` и пять ключей установщика
(`D3DMETAL_FRAMEWORK_PATH`, `FSR4_HOOK_LOG`, `FSR4_ROOT`, `FSR4_REPLACE`,
`FSR4_PANEL`), и бутылка снова берёт D3DMetal из CrossOver. Добавленные вручную
переменные вроде `FSR4_SHARPNESS` остаются, но без хука ни на что не влияют.
Файлы в `dist/crossover/` остаются на диске. После отката полностью
перезапустите CrossOver.

## 9. Обновление

Закройте CrossOver, затем:

```bash
git pull
make -B
python3 tools/precompile_models.py
python3 tools/crossover/crossover_fsr4.py install --no-bottle         # пересобрать хук; ключи бутылки уже записаны
```

Если `git pull` отказывается из-за изменений в `tools/bin/`, это ваши
пересобранные бинарники: `git restore tools/bin`, затем снова `git pull`.

После обновления самого CrossOver Preview повторите `install --no-bottle`,
чтобы копия D3DMetal соответствовала новой версии.

## 10. Если что-то не работает

| Симптом | Что делать |
|---|---|
| `unable to find utility "metal"` или сообщение об отсутствующем Metal Toolchain | `xcodebuild -downloadComponent MetalToolchain`. Если `xcode-select -p` показывает `CommandLineTools`: `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer` |
| `You have not agreed to the Xcode license agreements` | `sudo xcodebuild -license accept` |
| `No module named 'numpy'`, `gen_i8mpp failed` | `python3 -m pip install --user numpy` |
| `make: Nothing to be done for 'all'`, хотя код менялся | `make -B` |
| Правки в `patches/hlsl/**` или `tools/compile_variant.py` не действуют | кэш проверяет хеш исходников при создании сессии: перезапустите игру; после правок host-кода пересоберите хук по шагу 9 |
| `compare` пишет `cannot read …/meta.json` | в режиме папок первой идёт папка тесткейса с `meta.json`; два прогона сравнивайте по файлам: `compare a.bin b.bin 1920 1080` |
| Установщик падает на `live_test`, нет `live ok` | проверьте `dist/crossover/hook.log` и выбранный `backend`; установщик создаёт эталон автоматически, повторите `make -B` и установку после обновления |
| `missing /Applications/CrossOver Preview.app/…` | CrossOver Preview должен лежать именно в `/Applications` |
| После запуска игры в `hook.log` нет `loaded pid=` | CrossOver не был закрыт полностью; ключи не в `[EnvironmentVariables]`; выбрана не та бутылка; папку репозитория перенесли — повторите установку |
| `loaded pid=` есть, строк `encode …` нет | игра не использует DLSS: включите DLSS в игре и проверьте `"D3DM_ENABLE_METALFX" = "1"` в бутылке |
| Игра подвисает на первом кадре или после смены разрешения | компилируются шейдеры: `python3 tools/precompile_models.py`, дальше они берутся из кэша |
| Артефакты или чёрный экран | сравните с оригиналом через `"FSR4_REPLACE" = "0"`, попробуйте `"FSR4_MODEL" = "f16"`, смотрите `dist/crossover/hook.log` |
| Результат не сходится с эталоном при `MTL_SHADER_VALIDATION=1` | валидация шейдеров меняет результат даже на настоящем железе — сверяйте без неё |

## 11. Справочник

### fsr4mac

```
fsr4mac <testcase_dir> <out_dir> [--int8|--fp16|--i8mpp] [--dump-passes]
        [--bench N] [--reps R] [--sync barrier|encoder|none] [--builddir DIR]
        [--mpp] [--cnb]
fsr4mac --caps
fsr4mac --bench-tensorops
```

| Ключ | Что делает |
|---|---|
| `--auto` | автоматический выбор по GPU; по умолчанию |
| `--int8` | эталонная int8-модель; включается явно |
| `--fp16` | fp16-модель |
| `--i8mpp` | int8 на нейроускорителях M5 |
| `--dump-passes` | сохранять буфер сети после каждого прохода, константы и history; медленно |
| `--bench N` | N прогонов каждого кадра, время GPU по стадиям; файлы не пишутся |
| `--reps R` | вместе с `--bench`: каждая стадия R раз подряд, цифры стабильнее |
| `--sync` | синхронизация между проходами: `barrier` (по умолчанию), `encoder`, `none` (результат неверный, только для замера) |
| `--builddir DIR` | папка кэша шейдеров; по умолчанию `build` в текущем каталоге |
| `--mpp`, `--cnb` | экспериментальные ядра для fp16; медленнее обычных, оставлены для сравнения |
| `--caps` | устройство, семейства GPU, наличие Metal 4 и проверка tensor ops |
| `--bench-tensorops` | микробенч `matmul2d` против `simdgroup_matrix` |

Переменные окружения: `MTL_DEBUG_LAYER=1` — проверка вызовов Metal API;
`FSR4_SCRATCH_PRIVATE=1` — промежуточный буфер в private-памяти (для отладки).

### Утилиты

| Команда | Что делает |
|---|---|
| `tools/bin/gen_testcase <папка> <пресет> <rw> <rh> <uw> <uh> <кадров>` | синтетический тесткейс |
| `tools/bin/compare <a.bin> <b.bin> <w> <h> [diff.png]` | сравнить два кадра RGBA16F |
| `tools/bin/compare <папка_тесткейса> <папка_результата> [папка_diff]` | все кадры против эталона из тесткейса |
| `tools/bin/bilin <in.bin> <rw> <rh> <uw> <uh> <out.bin> [out.png]` | билинейное увеличение |
| `tools/bin/bin2png <in.bin> <w> <h> <out.png> [linear\|srgb]` | RGBA16F → PNG |
| `./build_shaders.sh [--all \| --preset P --res R]` | заранее собрать int8-шейдеры для CLI; необязательно, `fsr4mac` собирает их сам |
| `python3 tools/precompile_models.py [--presets …]` | заранее собрать шейдеры игрового пути |
| `python3 tools/crossover/crossover_fsr4.py install [--panel] [--bottle CONF \| --no-bottle]` | поставить FSR 4 в бутылку |
| `python3 tools/crossover/crossover_fsr4.py restore [--bottle CONF]` | откатить |

### Что где лежит

| Путь | Что там | В git |
|---|---|---|
| `tools/bin/` | собранные утилиты, `dxc`, `spirv-cross`, `spirv-val` | да |
| `build/` | кэш шейдеров и сгенерированных весов | нет |
| `testcases/` | тестовые и эталонные кадры | нет |
| `out/` | результаты | нет |
| `dist/crossover/` | копия D3DMetal с хуком, `FSR4 Panel.app`, `hook.log`, `status.conf`, `panel.conf`, `backups/` | нет |

Удалить всё сгенерированное: `rm -rf build out testcases dist`. Если хук
установлен в бутылку, сначала сделайте `restore`, иначе бутылка будет ссылаться
на удалённую копию D3DMetal.

### Документация

| Файл | О чём |
|---|---|
| [README.md](../README.md) | обзор, тулчейн шейдеров, устройство хоста (EN) |
| [CROSSOVER.md](CROSSOVER.md) | как устроена подмена DLSS |
| [STATE.md](STATE.md) | состояние проекта, замеры, отвергнутые подходы (EN) |
| [BENCHMARK.md](../BENCHMARK.md) | замеры скорости (EN) |
| [VERIFICATION.md](../VERIFICATION.md) | сверка с эталоном DX12 (EN) |
| [TESTDATA.md](../TESTDATA.md) | формат тестовых данных (EN) |
