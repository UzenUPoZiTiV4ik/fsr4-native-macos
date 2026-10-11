# FSR4 вместо DLSS в CrossOver

Текущий вариант по умолчанию — **FSR 4.1.1 / Lossy**. В панели можно
переключать **Exact / Lossy**; новая настройка сбрасывает историю на следующем
кадре. Краткие команды: [GUIDE.md](GUIDE.md). Преобразование входов:
[FSR411_INPUT_CONTRACT.md](FSR411_INPUT_CONTRACT.md).

Сборка CLI и шейдеры описаны в [README.md](../README.md) и [STATE.md](STATE.md), пошаговая установка и откат — в [GUIDE.md](GUIDE.md). Здесь только то, как FSR4 встаёт на место DLSS внутри CrossOver Preview, и какие файлы при этом меняются.

DLSS в CrossOver Preview — это не отдельный `nvngx.so`. D3DMetal (GPTK, каталог `lib/apple_gptk`) сам реализует NGX и для временного апскейла вызывает MetalFX. Хук подменяет только `encodeToCommandBuffer:` у временного скейлера. Пространственный скейл, генерация кадров и денойз остаются на MetalFX. Путь DXMT не используется.

`CrossOver.app` не изменяется. Бутылка указывает на копию фреймворка в `dist/crossover/`. Каталог `dist/` в git не входит.

## Почему не `DYLD_INSERT_LIBRARIES`

Wine в CrossOver — x86_64 под Rosetta, с hardened runtime. Флага `allow-dyld-environment-variables` нет, поэтому `DYLD_INSERT_LIBRARIES` до процесса не доходит. Флаг `disable-library-validation` есть, поэтому ad-hoc подпись копии D3DMetal и хука принимается.

`libd3dshared.dylib` читает `D3DMETAL_FRAMEWORK_PATH` и делает `dlopen` этого пути (`RTLD_NOW`). Если переменной нет, он открывает системный `D3DMetal.framework` или копию по rpath внутри приложения. Переменные секции `[EnvironmentVariables]` в `cxbottle.conf` CrossOver копирует в unix-процесс бутылки.

## Что создаёт установка

Источник фреймворка зашит в установщик:

```
/Applications/CrossOver Preview.app/Contents/SharedSupport/CrossOver/lib/apple_gptk/external/D3DMetal.framework
```

`python3 tools/crossover/crossover_fsr4.py install` делает следующее.

1. Собирает `dist/crossover/libfsr4hook.dylib` из `hook/fsr4_metalfx.m` и `src/fsr4_live.mm` (x86_64, ad-hoc подпись).
2. Копирует `D3DMetal.framework` в `dist/crossover/D3DMetal.framework`. Если каталог уже был, он удаляется и копируется заново.
3. В копию бинарника `Versions/A/D3DMetal` дописывает одну load-команду `LC_LOAD_DYLIB` с именем `@loader_path/libfsr4hook.dylib`. Команда ложится в нулевой зазор заголовка Mach-O между концом load-команд и `__text`. Код `EncodeTemporallyScaleMTLFX` не патчится.
4. Кладёт `libfsr4hook.dylib` рядом с бинарником и подписывает оба файла ad-hoc (`codesign -s -`).
5. Прогоняет smoke (свизл создаётся, FSR4 ещё не подменяет кадр), `dlopen` копии и `live_test` по `testcases/testcase_quality_720to1080`.
6. Если указана бутылка, дописывает переменные в конец `cxbottle.conf` и один раз сохраняет копию конфига.

По умолчанию бутылка:

```
~/Library/Application Support/CrossOver/Bottles/Игры/cxbottle.conf
```

Резервная копия конфига, снятая до первой записи:

```
dist/crossover/backups/Игры.cxbottle.conf
```

Ключи, которые пишет установщик (маркер `;; fsr4-mac`):

| Ключ | Значение |
| --- | --- |
| `D3DMETAL_FRAMEWORK_PATH` | `dist/crossover/D3DMetal.framework/Versions/A/D3DMetal` |
| `FSR4_HOOK_LOG` | `dist/crossover/hook.log` |
| `FSR4_ROOT` | корень этого репозитория |
| `FSR4_REPLACE` | `1` |
| `FSR4_VERSION` | `4.1.1` по умолчанию |
| `FSR4_PROFILE` | `lossy` по умолчанию, `exact-fast`, `exact` или `fsr4` (модель 4.0.2, четвёртый режим панели) |

Они должны остаться внутри `[EnvironmentVariables]`. Установщик дописывает их в конец файла; в бутылке «Игры» эта секция как раз последняя.

Другие ключи бутылки, ярлыки и само приложение не трогаются. Повторный `install` заново копирует фреймворк из `/Applications` и заново накладывает только load-команду хука.

Команды:

```bash
python3 tools/crossover/crossover_fsr4.py install --panel --profile lossy
python3 tools/crossover/crossover_fsr4.py install --bottle "/path/to/cxbottle.conf"
python3 tools/crossover/crossover_fsr4.py install --no-bottle   # только dist/, конфиг бутылки не писать
python3 tools/crossover/crossover_fsr4.py restore
```

`restore` удаляет управляемые ключи установщика, включая версию/профиль и
панель, и строку `;; fsr4-mac`. Копия в `dist/` остаётся на диске. Следующий
запуск бутылки снова берёт D3DMetal из CrossOver.app.

Пока CrossOver запущен, в память уже загружен старый dylib. После `install` или `restore` процесс нужно закрыть полностью, не только окно игры.

Установщик смотрит только копию приложения в `/Applications`. В Spotlight бывает вторая иконка из `~/Applications`; бутылки у них общие, источник фреймворка для копии — всё равно `/Applications`.

## Что делает хук

Конструктор `libfsr4hook.dylib` свизлит `MTLFXTemporalScalerDescriptor`:

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
