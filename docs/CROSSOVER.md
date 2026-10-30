# FSR4 вместо DLSS в CrossOver

Текущий вариант по умолчанию — **FSR 4.1.2 INT8 Lite**. В панели можно
переключать **FSR 4.1.2 Lite / FSR 4.1.2 Exact / FSR 4.1.2 Lossy**; новая настройка сбрасывает историю на следующем
кадре. Краткие команды: [GUIDE.md](GUIDE.md). Преобразование входов:
[FSR411_INPUT_CONTRACT.md](FSR411_INPUT_CONTRACT.md).

Сборка CLI и шейдеры описаны в [README.md](../README.md) и [STATE.md](STATE.md), пошаговая установка и откат — в [GUIDE.md](GUIDE.md). Здесь только то, как FSR4 встаёт на место DLSS внутри CrossOver Preview, и какие файлы при этом меняются.

DLSS в CrossOver Preview — это не отдельный `nvngx.so`. D3DMetal (GPTK, каталог `lib64/apple_gptk` в CrossOver 26 и `lib/apple_gptk` в 24/25) сам реализует NGX и для временного апскейла вызывает MetalFX. Хук подменяет только `encodeToCommandBuffer:` у временного скейлера. Пространственный скейл, генерация кадров и денойз остаются на MetalFX. Путь DXMT не используется.

CrossOver 24/25: `CrossOver.app` не изменяется, бутылка указывает на копию
фреймворка в `dist/crossover/`. CrossOver 26: `libd3dshared.dylib` больше не
читает `D3DMETAL_FRAMEWORK_PATH` и грузит фреймворк рядом с собой, поэтому
правится собственный `D3DMetal` приложения — только `Versions/A/D3DMetal` и
`libfsr4hook.dylib` рядом с ним, оригинал сохраняется как
`D3DMetal.fsr4-original`. Каталог `dist/` в git не входит.

## Почему не `DYLD_INSERT_LIBRARIES`

Wine в CrossOver — x86_64 под Rosetta, с hardened runtime. Флага `allow-dyld-environment-variables` нет, поэтому `DYLD_INSERT_LIBRARIES` до процесса не доходит. Флаг `disable-library-validation` есть, поэтому ad-hoc подпись копии D3DMetal и хука принимается.

`libd3dshared.dylib` в CrossOver 24/25 читает `D3DMETAL_FRAMEWORK_PATH` и делает `dlopen` этого пути (`RTLD_NOW`). Если переменной нет, он открывает системный `D3DMetal.framework` или копию по rpath внутри приложения. Переменные секции `[EnvironmentVariables]` в `cxbottle.conf` CrossOver копирует в unix-процесс бутылки.

В CrossOver 26 переменной `D3DMETAL_FRAMEWORK_PATH` нет: `libd3dshared.dylib`
грузит `@rpath/D3DMetal.framework/D3DMetal`, а `LC_RPATH @loader_path`
резолвит его рядом с собой, внутри приложения. Поэтому на 26 патчится сам
фреймворк CrossOver, а не копия. Хук при этом загружается в D3D-процессы
всех бутылок, но до свизла выходит, если в бутылке нет `FSR4_ROOT`.

## Что создаёт установка

Источник фреймворка — найденный CrossOver, например:

```
/Applications/CrossOver.app/Contents/SharedSupport/CrossOver/lib64/apple_gptk/external/D3DMetal.framework
```

Порядок поиска: `lib64/apple_gptk`, `lib64/apple_gptk3`, затем `lib/apple_gptk`
и `lib/apple_gptk3` (CrossOver 24/25). Режим определяется по соседнему
`libd3dshared.dylib`: если в нём есть строка `D3DMETAL_FRAMEWORK_PATH` — это
24/25 (`redirect`), иначе 26+ (`in-place`). Установщик печатает
`D3DMetal: <путь> (redirect|in-place)`.

`python3 tools/crossover/crossover_fsr4.py install` делает следующее.

1. Собирает `dist/crossover/libfsr4hook.dylib` из `hook/fsr4_metalfx.m` и `src/fsr4_live.mm` (x86_64, ad-hoc подпись).
2. Копирует `D3DMetal.framework` в `dist/crossover/D3DMetal.framework`. Если каталог уже был, он удаляется и копируется заново; в режиме `in-place` источником служит `D3DMetal.fsr4-original`, если приложение уже пропатчено.
3. В копию бинарника `Versions/A/D3DMetal` дописывает одну load-команду `LC_LOAD_DYLIB` с именем `@loader_path/libfsr4hook.dylib`. Команда ложится в нулевой зазор заголовка Mach-O между концом load-команд и `__text`. Код `EncodeTemporallyScaleMTLFX` не патчится.
4. Кладёт `libfsr4hook.dylib` рядом с бинарником и подписывает оба файла ad-hoc (`codesign -s -`).
5. Прогоняет smoke (свизл создаётся, FSR4 ещё не подменяет кадр), `dlopen` копии и `live_test` по `testcases/testcase_quality_720to1080`.
6. Режим `redirect`: копия остаётся в `dist/`, приложение не трогается.
   Режим `in-place`: в `CrossOver.app/…/D3DMetal.framework/Versions/A/`
   кладётся `D3DMetal.fsr4-original` (исходный бинарник), затем подписанный
   бинарник из `dist/` копируется на место `D3DMetal`, а `libfsr4hook.dylib` —
   рядом. Установленные файлы заново **не** подписываются: их ad-hoc подписи
   уже верны, а `codesign` переписал бы `_CodeSignature/CodeResources`, и
   восстановление байт в байт перестало бы проходить проверку. После патча
   `dlopen` уже установленного бинарника проверяет, что хук грузится.
7. Если указана бутылка, дописывает переменные в конец `cxbottle.conf` и один раз сохраняет копию конфига. Ключ другого режима удаляется.

Запись в `CrossOver.app` на macOS 26 может требовать разрешения **App
Management**. Установщик проверяет запись до долгой сборки и при отказе
просит разрешить Терминал (или **FSR 4 Installer**) в
**Системные настройки → Конфиденциальность и безопасность → App Management**.

Установщик читает `BottlePath` из `CrossOver.conf` и `CX_BOTTLE_PATH`,
поэтому обнаруживает и бутылки на внешнем диске. Без этих настроек ищет в
`~/Library/Application Support/CrossOver/Bottles`. Если найдена одна бутылка,
выбирает её автоматически; при нескольких нужен `--bottle Steam` или полный
путь. Одинаковые имена в разных папках требуют полного пути:

```
python3 tools/crossover/crossover_fsr4.py install --bottle "/Volumes/SSD/Bottles/Steam"
CX_BOTTLE_PATH="/Volumes/SSD/Bottles" python3 tools/crossover/crossover_fsr4.py install
```

В графическом установщике кнопка **«Выбрать папку бутылки…»** позволяет
выбрать папку с `cxbottle.conf`, даже если её нет в настройках CrossOver.

Резервная копия конфига, снятая до первой записи:

```
dist/crossover/backups/Игры.cxbottle.conf
```

Ключи, которые пишет установщик (маркер `;; fsr4-mac`):

| Ключ | Режим | Значение |
| --- | --- | --- |
| `D3DMETAL_FRAMEWORK_PATH` | 24/25 | `dist/crossover/D3DMetal.framework/Versions/A/D3DMetal` |
| `FSR4_FRAMEWORK` | 26+ | `<CrossOver.app>/…/D3DMetal.framework/Versions/A/D3DMetal` |
| `FSR4_HOOK_LOG` | оба | `dist/crossover/hook.log` |
| `FSR4_ROOT` | оба | корень этого репозитория |
| `FSR4_REPLACE` | оба | `1` |
| `FSR4_VERSION` | оба | `4.1.2` по умолчанию |
| `FSR4_PROFILE` | оба | `int8-lite` по умолчанию, `lossy2` (FSR 4.1.2 Lossy, бывший Lite 2), `exact-fast` (FSR 4.1.2 Exact), `exact` или `fsr4` (модель 4.0.2) |

Они должны остаться внутри `[EnvironmentVariables]`. Установщик дописывает их в конец файла; в бутылке «Игры» эта секция как раз последняя.

Другие ключи бутылки, ярлыки и остальные файлы приложения не трогаются. Повторный `install` заново готовит фреймворк из CrossOver (в режиме `in-place` — из `D3DMetal.fsr4-original`) и заново накладывает только load-команду хука. Если CrossOver обновился, установщик обновляет и `.fsr4-original`.

Команды:

```bash
python3 tools/crossover/crossover_fsr4.py install --panel --profile lossy
python3 tools/crossover/crossover_fsr4.py install --bottle "/path/to/cxbottle.conf"
python3 tools/crossover/crossover_fsr4.py install --no-bottle   # только dist/, конфиг бутылки не писать
python3 tools/crossover/crossover_fsr4.py restore
```

`restore` удаляет управляемые ключи установщика, включая версию/профиль и
панель, и строку `;; fsr4-mac`. В режиме `redirect` копия в `dist/` остаётся
на диске, приложение не менялось. В режиме `in-place` оригинальный бинарник
возвращается на место из `D3DMetal.fsr4-original`, а `.fsr4-original` и
`libfsr4hook.dylib` удаляются; после этого `codesign --verify` у фреймворка
снова проходит. Ключ `D3DM_ENABLE_METALFX` restore не трогает — он мог быть
выставлен и до установки.

Если тот же пропатченный фреймворк ещё использует другая бутылка, restore
снимает ключи только у текущей, оставляет фреймворк пропатченным и печатает
имена бутылок, которые нужно откатить первыми. Оригинал возвращает последняя.
Если CrossOver обновился, пока стоял патч, restore видит непатченный бинарник
и просто удаляет устаревшие `.fsr4-original`/хук, не перезаписывая новый.

Пока CrossOver запущен, в память уже загружен старый dylib. После `install` или `restore` процесс нужно закрыть полностью, не только окно игры. Откатывайте патч перед обновлением CrossOver.

Где ищется CrossOver: сначала `CrossOver.app` и `CrossOver Preview.app` в `/Applications`, потом в `~/Applications`, потом любая переименованная копия (`CrossOver_patched.app`, `CrossOver Preview-20261006.app`) в этих папках или на уровень глубже, потом копии, которые знает Spotlight/LaunchServices (другие диски). CrossOver узнаётся по папке `Contents/SharedSupport/CrossOver`, а не по имени. Если копий несколько, в приложении-установщике есть кнопка **Сменить…**, а в скрипте — `--crossover /путь/к/CrossOver.app`. Выбирайте ту копию, через которую запускаете игру. Бутылки у копий общие.

## Что делает хук

Конструктор `libfsr4hook.dylib` первым делом проверяет `FSR4_ROOT`: без него
(бутылка без FSR 4) он сразу выходит, ничего не логируя и не свизля. Это
важно для CrossOver 26, где хук лежит в общем для всех бутылок фреймворке.

Дальше конструктор свизлит `MTLFXTemporalScalerDescriptor`:

- `newTemporalScalerWithDevice:`
- `newTemporalScalerWithDevice:compiler:`

У созданного объекта свизлится `encodeToCommandBuffer:`. На Metal 4 это класс `_M4FXTemporalScalingEffectV4`, на обычном command buffer — `_MFXTemporalScalingEffectV4`. Таблица свизлов на четыре класса.

`FSR4_REPLACE` по умолчанию включён. Значение `0` оставляет оригинальный MetalFX encode. Если `fsr4_encode_live` вернул 0 (нет текстур, не собрался шейдер, кодировщик уже занят), вызывается исходный IMP, и кадр не остаётся чёрным из-за раннего выхода.

Хук читает у скейлера цвет, глубину, motion, выход, exposure, `inputContentWidth/Height`, jitter, `motionVectorScale`, `preExposure`, `reset`, `depthReversed` и отдаёт их в `fsr4_encode_live`. Буфер команд не коммитится: его по-прежнему закрывает D3DMetal.

Лог пишется в stderr и в `FSR4_HOOK_LOG`. Рабочий запуск выглядит так: `loaded pid=…`, затем `swizzled … encodeToCommandBuffer:`, затем одна строка `encode mtl4 …` или `encode classic …`.

## Живой энкодер

`src/fsr4_live.mm` направляет версию 4.1.1 в `src/fsr411_live.mm`; тот
использует общий с CLI `fsr411::Runtime` в command buffer игры. По умолчанию
`FSR4_MODEL=auto`: M5 / family 10 выбирает native INT8 tensor operations,
другие Apple GPU — scalar INT8 той же модели. Для проверки есть явные
`i8mpp`, `scalar`, `reference`. При отказе FSR хук вызывает исходный MetalFX
encode и инвалидирует историю FSR. Подробности: [HARDWARE.md](HARDWARE.md).

После правок host-кода или `hook/` пересоберите библиотеку:
`python3 tools/crossover/crossover_fsr4.py install --no-bottle --profile lossy`.
Установщик создаёт smoke-reference с совпадающими версией, backend и профилем.
Это локальная проверка подключения хука; старый Windows-эталон 4.0 к ней не
относится. `compile_fsr411.py` собирает варианты по содержимому исходников,
а последующие кадры используют готовую сессию. `tools/precompile_models.py`
продолжает готовить старую 4.0.2; для неё требуется явный
`FSR4_VERSION=4.0.2` / `--fsr-version 4.0.2`.


Флаги варианта: `depthReversed` и флаги дескриптора со скейлера (см. выше), цветовое пространство линейное (`FSR4_COLORSPACE=srgb` или `pq`). У DLSS/MetalFX нет ручки резкости, поэтому RCAS в игровом пути включён с резкостью FSR **0.4** по умолчанию (`FSR4_SHARPNESS=<0..1>`; `0` или `FSR4_RCAS=0` выключает). Константы RCAS те же, что в провайдере (`FsrRcasCon(-2*sharpness+2)`, preExposure); post при этом пишет в промежуточную RGBA16F-текстуру и не делает обратный mu-law — его делает RCAS.

Игра на D3DMetal отдаёт MTL4 command buffer. У него нет `setBuffer:offset:atIndex:`; привязки идут через `MTL4ArgumentTable`. Классический command buffer (так устроен `live_test`) идёт обычным compute encoder.

Новые scratch/текстуры очищаются на GPU в command buffer перед PRE. Смена
профиля, модели, размера выхода или соглашений входа сбрасывает временную
историю. Изменение только входного разрешения сохраняет её. В работе
используются отдельные слоты констант и удержание ресурсов до завершения
кадра; библиотека не коммитит command buffer игры.

Пресет считается по отношению выход/вход по оси X, как в провайдере. Ярус выхода: ширина или высота `>1080` → 2160, `>2160` → 4320. Перекрыть пресет можно `FSR4_PRESET`.

Текстура motion берётся как есть из свойства скейлера. D3DMetal за один игровой кадр вызывает encode **больше одного раза и на разных объектах скейлера**: у одного привязано настоящее поле движения `RG16Float` размером с рендер, у другого заглушка `4×4 RGBA8` (так D3DMetal подставляет непривязанный ресурс). Раньше оба вызова шли через одну общую history: заглушка вмешивала текущий кадр в историю с нулевым движением, и за движущимися объектами оставался шлейф. Чем ниже входное разрешение, тем сильнее сеть доверяет истории, тем длиннее шлейф.

Сейчас:

- history, reprojection и recurrent живут **на каждый объект скейлера** (`view N for scaler …` в логе). Шейдерные варианты и scratch общие.
- encode с заглушкой вместо motion (текстура меньше `inputContent`) по умолчанию **не считается**: в выход копируется последний настоящий результат FSR4 (`FSR4_STUB=skip`). Другие режимы: `metalfx` (отдать вызов MetalFX), `run` (считать как есть), `reset` (считать с `reset=1`).
- на Metal 4 вокруг нашего энкодера ставятся очередные барьеры (`barrierAfterQueueStages` / `barrierAfterStages:beforeQueueStages`), чтобы следующий кадр не прочитал history раньше, чем `post` её записал. `FSR4_QBARRIER=0` выключает.
- флаги дескриптора читаются при создании скейлера и пишутся в лог (`scaler … created: … jitteredMV= outputResMV= autoExposure=`). Они выбирают вариант шейдера: `FFX_MLSR_JITTERED_MOTION_VECTORS`, `FFX_MLSR_LOW_RES_MV`, `FFX_MLSR_AUTOEXPOSURE_ENABLED` (SPD-проход перед `pre`, как в CLI). Перекрыть: `FSR4_JITTERED_MV`, `FSR4_LOW_RES_MV`, `FSR4_AUTOEXPOSURE`.
- константы совпадают с DX12-провайдером: `mv_scale = motionVectorScale / renderSize`, jitter без изменений, `reset` со скейлера **или** принудительно на первом кадре view и при смене варианта шейдера (провайдер делает так при пересоздании модели). Размер выхода для внутренних текстур, констант и dispatch'ей выравнивается до 8 (`AlignUp`, как в провайдере); текстура выхода игры остаётся своего размера. Пороги пресета те же, что в провайдере (по оси X: `≥1.50` quality, `≥1.69` balanced, `≥1.99` performance, `≥2.99` ultraperf).
- первые 60 encode и каждый 600-й пишутся в лог целиком (`enc#N … STUB …`), раз в 120 кадров GPU-проба: значение экспозиции, максимум |mv| в пикселях по сетке 16×16, глубина в центре. По этим строкам видно, что игра реально отдала.

Кэш `build/fsr4_<key>` не привязан к содержимому патчей: после правки `patches/hlsl/**` каталоги надо удалить (`rm -rf build/fsr4_*`), иначе энкодер загрузит старый metallib с другими биндингами.

## Панель «FSR 4» (`FSR4_PANEL=1`)

`dist/crossover/FSR4 Panel.app` (`tools/crossover/panel_app.mm`, arm64, собирается установщиком) — отдельное приложение со своей иконкой в Dock. Окно — плавающая неактивирующаяся HUD-панель поверх полноэкранной игры: слайдер и галочка работают мышью и не забирают у игры фокус. Показывает: активен ли FSR 4 и если нет, почему (нет текстур, заглушка motion, шейдер не собрался, `FSR4_REPLACE=0`, игра закрыта), модель и пресет, разрешение вход → выход, счётчики кадров и заглушек, среднее время кадра игры (интервал между апскейлами, EMA). Слайдер резкости и галочка RCAS действуют в игре сразу.

Связь с хуком — два маленьких файла в `dist/crossover/`: `status.conf` пишет хук (не чаще 2 раз в секунду, через атомарный rename), `panel.conf` (`sharpness=`, `rcas=`) пишет панель, а энкодер перечитывает его по mtime не чаще двух раз в секунду. Значение переживает перезапуск и имеет приоритет над `FSR4_SHARPNESS`/`FSR4_RCAS`; `FSR4_SETTINGS=none` игнорирует файл (так гоняется `live_test`), `FSR4_SETTINGS=<путь>` переносит его. Нагрузка на игру: запись 20 строк дважды в секунду и один `stat` дважды в секунду.

Запуск: с `FSR4_PANEL=1` в `[EnvironmentVariables]` бутылки (`install --panel` пишет ключ, `restore` убирает) хук на первом кадре сам запускает приложение через `open -g` (без перехвата фокуса); можно открыть и вручную — `open "dist/crossover/FSR4 Panel.app"`. Клик по иконке в Dock возвращает закрытое окно. Проверка без игры: `"dist/crossover/FSR4 Panel.app/Contents/MacOS/FSR4 Panel" --snapshot /tmp/panel.png`.

## Файлы, которых нет в описании CLI

| Путь | Роль |
| --- | --- |
| `hook/fsr4_metalfx.m` | конструктор и свизл временного скейлера |
| `src/fsr4_live.h`, `src/fsr4_live.mm` | fp16 encode в command buffer игры |
| `tools/crossover/crossover_fsr4.py` | копия D3DMetal, load-команда, бутылка, smoke |
| `tools/crossover/smoke.m` | свизл есть, `FSR4_REPLACE=0` |
| `tools/crossover/dlopen_test.c` | `dlopen` копии бинарника |
| `tools/crossover/live_test.mm` | кадр testcase через хук или напрямую, сравнение с эталоном выбранного режима |
| `src/fsr4_panel.mm` | сторона хука: публикация `status.conf`, запуск панели при `FSR4_PANEL=1` |
| `tools/crossover/panel_app.mm` | `FSR4 Panel.app`: Dock-приложение статуса/резкости |

`make` и `tools/bin/fsr4mac` этот путь не собирают. Игровой dylib появляется только из `crossover_fsr4.py install`.
