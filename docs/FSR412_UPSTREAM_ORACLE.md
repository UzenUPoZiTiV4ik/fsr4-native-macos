# Oracle оригинального FSR 4.1.2 Lite Lossy

## Важное обновление проверки

Первичная повторяемость GPU replay не доказала правильность shader math. Дополнительная проверка original P2 integer DXIL SSA на CPU обнаружила **ошибку default-оптимизации Metal**: тот же MSL с `-O0` совпал с CPU SSA во всех 32 каналах в четырёх проверенных точках; default-optimized библиотека расходилась. Поэтому старые captures ниже являются **legacy diagnostics**, а их output hashes больше не служат сертификатами исходной сети.

Текущий correctness loader по умолчанию использует `-O0` для original pass2 и pass3. После этого full-class 4K original/native INT8-MPP A/B прошёл два кадра побитно: все CNB/DS/FNB/CT interiors, output, history, recurrent и reprojected совпали. Output SHA: `f79ac0b43b1ac3a5aa7e1a61429f70646d01337bee44cf78383ddedfa1984b3a`. Это ограниченная Metal/source-проверка, не Windows parity.

Также готов generic classifier и bundle emitter для **всех 384 комбинаций** tiers/presets/DI/LM/JM/CS/AE. Выбор каждой из шести стадий уникален либо доказанно byte-equivalent. Отдельный standalone fixture пока остаётся 2160-вариантом; generic production bundles выбирают остальные варианты без приближённой подмены.

## Что именно подтверждено

Цель: RX7000 `dll-2026-11-01/4.1.2l-lossy`.

- DLL SHA-256: `d9b0a1f5781c2d633a327272f6f1c8607057df7cafd0233cf38c7395c8f7dbc5`.
- Exact sibling: `56896078aef294ea875f4b977695701d827a9273a616d1f5c501b9963e559bd7`.
- AMD 4.1.1 provider для исходной таблицы: `d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46`.

Импортёр независимо читает PE, исходные записи `(size, pointer)` и соответствующие записи целевой DLL. Извлечены 639 именованных записей сети, включая 426 INT8-записей `cs_6_6`; дедупликация reference/lossy/exact даёт 778 шейдерных объектов. Это полная статическая карта замен, не наблюдение выбора permutation в Windows.

Upstream-рецепты прочитаны, но их Python и GPL-шейдерный код не скопированы и не выполнялись. Loader, проверки IO и контролируемые варианты написаны независимо. DXIL, SPIR-V, MSL, библиотеки и веса остаются только в игнорируемых `build/`-каталогах.

**Windows DLL не загружалась и не выполнялась.** GPU-проверки ниже относятся к оригинальным DXIL, переведённым в Metal на Apple M5. Форматы текстур и sampler заданы явно как условия replay, а не восстановлены из наблюдения Windows allocation.

## Файлы и запуск

- `tools/fsr412/upstream_oracle.py`: importer, таблица, reflection, IO-привязки, contract и выбранный bindings manifest.
- `tools/fsr412/oracle_io.py`: только проверки границ ресурсов, без политики FSR.
- `tools/fsr412/oracle_variants.py`: контролируемые ordered/isolated POST.
- `tools/fsr412/oracle_selection.py`: typed-DXIL классификация и выбор всех 384 option combinations.
- `tools/fsr412/oracle_bundle.py`: выбранные external shader bundles и production-like bindings.
- `tools/fsr412/oracle_pass1_border.py`: ветка original pass1 DS-border через фиксированный GroupID `(0,2,0)`, original skip gate сохранён.
- `tools/bench/fsr412_pass1_border_test.mm`: отдельная проверка exact border IO и отсутствия записей на skip.
- `tools/fsr412/oracle_compile_profile.py`: correctness `-O0` профиль без изменений shader source.
- `tools/fsr412/oracle_dxil_probe.py`: ограниченное CPU-выполнение integer P2 SSA из оригинального DXC listing.
- `tools/fsr412/oracle_p2_diagnostic.py`, `oracle_dot_variant.py`, `tools/bench/fsr412_p2_probe.mm`: отдельные диагностические варианты и GPU probe; baseline не заменяют.
- `tools/bench/fsr412_upstream_replay.h`: самостоятельный loader; публичные buffers, textures, kernels и `dispatch()` для neural A/B.
- `tools/bench/fsr412_upstream_replay.mm`: fixture и запись состояний всех шести проходов.

Импорт:

```bash
python3 tools/fsr412/upstream_oracle.py \
  /tmp/fsr4-lite-lossy-20261101/4.1.2l-lossy/amd_fidelityfx_upscaler_dx12.dll \
  --reference /tmp/fsr412-upstream-oracle/amd-original.dll \
  --exact /tmp/fsr4-lite-lossy-20261101/4.1.2l-exact/amd_fidelityfx_upscaler_dx12.dll
```

Для перевода передать уже собранный converter, не скачивать и не собирать второй:

```bash
python3 tools/fsr412/upstream_oracle.py \
  /tmp/fsr4-lite-lossy-20261101/4.1.2l-lossy/amd_fidelityfx_upscaler_dx12.dll \
  --reference /tmp/fsr412-upstream-oracle/amd-original.dll \
  --converter /tmp/fsr412-dxil-spirv/build/dxil-spirv \
  --translate SHA256
```

Повторить для шести SHA из таблицы ниже. Каждую `.metal` собрать:

```bash
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math \
  -c build/fsr412_upstream/SHA256.metal -o build/fsr412_upstream/SHA256.air
xcrun -sdk macosx metallib build/fsr412_upstream/SHA256.air \
  -o build/fsr412_upstream/SHA256.metallib
python3 tools/fsr412/oracle_compile_profile.py --stage pass2
python3 tools/fsr412/oracle_compile_profile.py --stage pass3
python3 tools/fsr412/oracle_variants.py
clang++ -std=c++17 -fobjc-arc -framework Foundation -framework Metal \
  tools/bench/fsr412_upstream_replay.mm -o build/fsr412_upstream/replay
```

Запуск устойчивого replay:

```bash
build/fsr412_upstream/replay --post-mode isolated \
  --recurrent-format u8 --frames 4 \
  --out build/fsr412_upstream-oracle/my-run
```

Режимы POST: `literal`, `ordered`, `isolated`. По умолчанию `literal`, чтобы изменение порядка или aliases не скрывалось. Recurrent: `u8` по умолчанию, `f16` только для явно выбранного другого fixture. Neural compiler profile: `correctness` по умолчанию (`-O0` для P2/P3), `legacy` только для воспроизведения старого compiler failure. Standalone fixture использует `2160 / normal / di1 lm1 jm0 cs0 ae0`; остальные permutations создаются generic emitter, а не подменяются этим набором.

Пример generic bundle, без GPU execution:

```bash
python3 tools/fsr412/upstream_oracle.py \
  /tmp/fsr4-lite-lossy-20261101/4.1.2l-lossy/amd_fidelityfx_upscaler_dx12.dll \
  --reference /tmp/fsr412-upstream-oracle/amd-original.dll \
  --select --tier 1080 --preset normal \
  --depth-inverted 0 --low-res-mv 1 --jittered-mv 1 \
  --colorspace 2 --auto-exposure 0 --post-mode isolated \
  --bundle build/fsr412_upstream/bundles/1080-di0-lm1-jm1-cs2-ae0 \
  --converter /tmp/fsr412-dxil-spirv/build/dxil-spirv --compile-selected
```

Три representative bundles успешно скомпилированы: 1080 normal/SRGB/jittered low-MV; 2160 UP/PQ/AE1/full-MV; 4320 normal/nonlinear/full-MV. GPU-проверки этих дополнительных permutations не выполнялись.

Classifier использует реальные resource handles, восемь depth min/max comparisons, используемые b1 lanes @56/60, коэффициенты SRGB/PQ/Mu-law, `r_input_exposure t6`, baked A matrix и quantizers. Полный index: `build/fsr412_upstream/permutation-index.json`. Активная Lite карта содержит 180 PRE и 48 изменённых non-debug INT8 POST, не 180 POST; debug/FP8 записи не выдаются за Lite.


После перевода:

```bash
python3 tools/fsr412/upstream_oracle.py \
  /tmp/fsr4-lite-lossy-20261101/4.1.2l-lossy/amd_fidelityfx_upscaler_dx12.dll \
  --reference /tmp/fsr412-upstream-oracle/amd-original.dll --bindings-only
```

`build/fsr412_upstream/bindings.json` содержит library/entry/SHA, `[Metal slot, original symbolic resource]`, размеры ресурсов, aliases, dispatch, условные biases и варианты POST. Это loader manifest, не интеграция в production runtime.

## Карта старых 27 проходов

| Старый индекс | Старый entry | Реальная замена |
|---|---|---|
| 0 | prepass | оригинальный Lite PRE |
| 1 | pass0_post | noop |
| 2 | pass1 | Lite CNB + downsample, группа `64x2x1` |
| 3 | pass1_post | noop |
| 4 | pass2 | Lite FNB, `64x1x1` |
| 5 | pass2_post | noop |
| 6 | pass3 | **fsr4_border, не noop** |
| 7 | pass3_post | noop |
| 8 | pass4 | Lite FNB + upsample, entry теперь pass3 |
| 9-25 | остальные model/padding | noop |
| 26 | postpass | оригинальный fused CNB/CT/temporal POST |

21 пустой проход, 126 INT8 table records, указывают на один noop:
`17afa4239aa66d37267373a5ef8d53f483e698d89a222d91fa3c1993fdc85316`.

Border выполняется даже на пропущенном neural-кадре. Только первая группа работает; остальные группы этого шейдера являются доказанными пустыми вызовами. Он чистит границы двух quarter-плоскостей `@0` и `@Q`. Чистку DS-границ `@2P` и `@2P+Q` делает отдельная ветка pass1 с group `(0,2,0)`.

Для native замены fused pass1 нужен helper `pass1_border`: он направляет оригинальный P1 shader в эту ветку, не копируя её stores/policy. Dispatch `1x1x1`, threads `64x2x1`, исходные T18/u11 aliases. **Original P1 сначала проверяет marker и возвращается на skip**, поэтому его DS-border ветка, в отличие от старого pass3 border, не выполняется на истинно skipped 1080/2160 frame. Helper сохраняет этот ранний выход; можно вызывать его каждый кадр. API standalone: `Replay.load_pass1_border()`, затем `dispatch("pass1_border")`. Generic/root manifests содержат library и этот gate отдельно.

Следовательно native run-порядок: CNB, DS, original P1 border helper, FNB, original old-pass3 border, CT, original POST. На skip helper P1 не меняет scratch; old-pass3 border всё равно выполняется.

Отдельный GPU test подтвердил helper: ровно 6008 class-border cells, 24032 uint words очищены на run, остальные logical words не изменены. На skip hash всего logical scratch сохранился. Команда: `build/fsr412_upstream/pass1_border_test build/fsr412_upstream`.

В исходной DLL также присутствуют FP8 и debug/permutation записи. Например 48 INT8 debug POST остаются исходными AMD. Они не входят в выбранный normal non-debug replay. Entry name или размер группы сами по себе не доказывают INT8 либо permutation.

## Реальные SHA выбранных шейдеров

| Роль | DXIL SHA-256 |
|---|---|
| PRE | `9523e223136343d56c1c3b2c89fbdea29adf511343bfcb7fb35ba7afbead4bba` |
| pass1 | `2638456df248932a18dfd82b556d2c06d8ae3ab0fcd6f67eeeb037782bf6a0ea` |
| pass2 | `6a5d98fc27a87938ab35d09af117f62238355cdab06aeaf3a9b26d91c434fb5e` |
| border | `779dd783525dbe328bf6bfa06509f813f928f9d1c59edf6993fa4bd278a502de` |
| pass3, старый pass4 | `4a8443a4c81abc7c08a64f4131e8a1aeb86e33c1f49c5be530c886d33507a238` |
| POST | `a69f41a0bb37376178001a7f73087d4726e21c0d531e7635653277053ac4b268` |

Для каждого сохранены `.translation.json`, container/header hashes, original table offsets, DXC listing hash, converter/SPIRV-Cross hashes и команды. Header hash DXBC является отдельным opaque identifier, не SHA-256.

## Buffers и dispatch

Полный `t18` равен **131072 байтам**, не 27392:

- normal: `621a99e22c4259c3200097513947d072633f67634eaf95365fb4123b4c56ac10`;
- UP: `f2d352fcfca0a334f0e67639735e19cc5b719be4ca3ba8f14246a82bc82ff264`.

Exact и Lossy содержат одинаковые полные Lite initializer buffers. Trained параметры находятся также в DXIL literal operands. Первые 27392 байта normal base имеют SHA `4b9898da719dc2dca5dc45fd71dec5f80bdb9886a580fe3038af59efe637d55e`. Native recovered training `c77c2d1a4d119ed76d94b8c23335490cc38280db3346c72396a1e1acb0523cfc` является отдельным представлением; его нельзя вместо полного `t18` подключать к оригинальным PRE/neural shaders.

- `b0`: shader declaration `uint4[9]`, минимум 144 байта. pass2/pass3 читают **slot 4**, quarter-размеры. MSL carrier объявлен `float4[9]`, но читает его через `as_type<uint4>`, не числовое float-to-uint преобразование. Upload должен содержать uint32 bits. Более длинный buffer допустим, если этот ABI сохранён.
- `b1`: 104 байта. MSL UBO округлён до 112, хвост нулевой. `jitter` @24, `width/height/reset/width_lr` @64/68/72/76, exposure @84/88.
- `u11`: один физический scratch. Все его scalar/vector Metal arguments должны ссылаться на тот же buffer. То же для всех aliases `t18`.
- Namespace-remap разделяет SRV/UAV/CBV/sampler. Иначе default converter создаёт collisions, например `t3/u3`. Read-only optimization не меняет D3D register class: above-4K POST читает UAV `u11`, который нельзя переименовывать в SRV `t11`. Class берётся из original DXIL reflection.

Для 2160: `S=30752`, `P=33273664`; quarter `Sq=15392`, `Q=8342464`.

| Выход/вход | Адрес |
|---|---|
| PRE input tensor | `0` |
| pass1 CNB / residual для pass3 | `P` |
| pass1 DS, две quarter-плоскости | `2P`, `2P+Q` |
| pass2 FNB, две quarter-плоскости | `0`, `Q` |
| pass3 upsample / neural input POST | `2P` |

Размер allocation изменён в `.text` DLL, функция VA `0x180008cc0`: 1080 = `25027392`, 2160 = `99820992`, 4320 = `398707392`, то есть `3P`. Старый 2160 размер `83232256` не подходит.

Dispatch:

- PRE: `ceil(W/16), ceil(H/16), 1`, `256x1x1`.
- pass1: provider-compatible `ceil((W/2)/64), H/2, 1`, `64x2x1`. Shader сам remap-ит groups.
- pass2: provider-compatible `ceil((W/2)/64),H/2,1`. Для 1080/2160 достаточно quarter ROI `ceil((W/4)/64),H/4,1`; для 4320 нельзя просто убрать invalid threads без сохранения sentinel-store эффекта.
- border: достаточно `1,1,1`, `64x1x1`; всегда.
- pass3: `ceil((W/4)/64), H/4, 1`, `64x1x1`.
- POST: `ceil(W/32), ceil(H/32),1`, `256x1x1`, четыре output pixels на thread.

Для 4320 есть отдельный край ABI: invalid-store sentinel `134217728` находится **внутри** `398707392`-байтового allocation. Dynamic-invalid P2 threads выбирают нулевой uint4 и записывают его туда. Уменьшение старого half-resolution overdispatch до quarter ROI может убрать эту запись в residual plane P. Для 1080/2160 тот же sentinel действительно вне allocation и запись отбрасывается. Точный Windows dispatch не наблюдался, поэтому conservative provider-compatible geometry и store effect отражены в generic manifest.

## Cadence, motion, persistence

Это не `frameIndex & 1`. PRE проверяет float **bits** jitter:

`negative nonzero jitter.x && reset==0 && (2*width_lr>=width || !(negative jitter.y && absbits(jitter.y)>1042983597))`.

Threshold float32 = `0.16666670143604279`. Negative zero X не пропускает кадр. На reset сеть выполняется. Marker words: `0x5a17c0de`, `0x0a110e57`. Для 2160 marker @`33273792`, сохранённый jitter @`33273800`. Для 1080 marker @`8342592`. Проверены все 60 PRE4320 permutations: skip-marker ветки нет.

Coarse motion/depth и owner plane используют свободные диапазоны того же scratch, не новые dense UAV:

- base `P+S = 33304416`, row stride `30752`;
- motion 8 байт на output 4x4 cell, parity ranges `192/16384`;
- owner 4 байта на cell, parity ranges `8192/24576`;
- две диагональные дырки: для row `r`, `d1=32*r+16`, `d2=d1+15392`; offsets сдвигаются на 32 байта после каждой дырки, второй тест после первого сдвига;
- owner2160: `depthKey[12 bits]<<20 | cellY<<10 | cellX`, unsigned atomic max;
- coarse displacement: round-to-nearest-even, clamp +/-254, signed int16 pair;
- fine motion: alpha PRE на even/even output pixel, `roundEven(dx)+64*roundEven(dy)+32` при +/-31, иначе `-2048`;
- alpha остальных skip pixels и всех run pixels сохраняет alpha **центрального Catmull-Rom history tap как half**, либо ноль, если tap не достигнут.

Owner plane очищается original POST на run-кадрах, не произвольным полным clear перед каждым PRE. Scratch persistent между кадрами. Менять эти aliases/clears/alpha на независимую приближённую политику нельзя при заявленной parity.

Даже `cs0/ae0` использует `t17` с current/previous exposure. `ae1` permutations также объявляют `r_input_exposure t6`. Exposure fixture с изменением значения выполнен отдельно, но не доказывает все AE permutations или Windows auto-exposure pipeline.

## Условные edge biases

Original POST не сводится к одному initializer:

- 16 A-biases имеют отдельные PHI literals, когда первый 3x3 tap не был прочитан;
- проверяется фактическая CNB coordinate, включая четыре motion/cache-coordinate копии, не просто край output;
- границы POST для этой проверки: integer `b1.width/2`, `b1.height/2`, не фиксированный максимум класса. Это отличается от class-sized raw halo в neural P3;
- пример channel0: primary `11`, missing-first-tap `12`; channel1 `-559/-562`;
- CT biases зависят от subpixel tap, порядок `[filter float4, recurrent float4]`.

CT taps 00/10/01/11:

```text
120 -303 -156 -92  -113 -152 -400 -149
117 -303 -157 -92  -113 -153 -398 -148
117 -303 -156 -92  -113 -154 -399 -148
120 -303 -156 -92  -113 -154 -399 -148
```

Все conditional records доступны в contract/bindings JSON. Нельзя объявлять parity только по SHA восстановленного общего initializer.

## savedCNB: реальное перекрытие и устойчивые варианты

Cache = **16 байт на half-resolution cell**, не четверть площади. Адрес в оригинале:

`49941264 + halfY*30752 + halfX*16`.

При 4K это 1920x1080 cells, 33177600 compact bytes, physical extent до `83153392`. Cache кодируется XOR `0xa5a5a5a5`; если любой из четырёх encoded words равен нулю, это miss. Hit использует сохранённый CNB; miss пересчитывает CNB и **тоже записывает cache даже на skipped frame**. Поэтому SAVE нельзя описывать как только `!skip`.

При cache Y=540 запись начинается в `2P+16`: перекрывает live neural input POST. В literal Metal replay:

- два 640x384 запуска совпали по всем 24 stage scratch hashes;
- в двух 4K запусках первые пять stages совпали полностью, различия начались только в POST;
- различия output frame0: 243696 half words, bbox X32..3809, Y0..1079.

Это согласуется с cross-workgroup alias race. **Наличие такой гонки в Windows не проверено.**

Контролируемые варианты сохраняют всю shader math:

1. **ordered**: группа Y задаётся дополнительным uint buffer slot4. Выполняется по одной строке groups сверху вниз, с completion между строками.
2. **isolated**: только пять доказанных cache-address definitions получают base `99821056`. Scratch extent `133094720`, отдельная appended plane. Перед POST span длиной `33273664` копируется из старого cache base, после POST копируется обратно. Таким образом следующий кадр видит тот же logical scratch, включая старое перекрытие; текущий POST читает immutable neural input.

В исходном literal шейдере ничего не заменено. Варианты имеют свои MSL hashes и отдельные receipts; это адаптации oracle, не literal DLL dispatch и не Windows runtime parity.

## GPU-проверки и результаты

**Ниже перечислены legacy captures с первоначальным default-optimized P2/P3.** Их повторяемость доказала dispatch/cache-IO свойства, но не numerical correctness P2. Для quality gates использовать correctness `-O0` и отдельный original/native A/B, описанный в начале документа.

Выполнялись все original шесть stages, без native substitutes. IO guard возвращает ноль для OOB reads и отбрасывает OOB stores/atomics, включая sentinel `134217728`; vector views проверяются отдельно по каждому 32-bit word. Integer texture IO также guarded. Sampling и model/temporal arithmetic не меняются.

Проверено:

- F16 recurrent fixture: small literal = ordered = isolated по всем stage scratch/images; 4K ordered = isolated, isolated repeat, по всем 12 stage scratch/images.
- U8 recurrent fixture (`RGBA8Unorm`): small literal = isolated, четыре кадра; 4K ordered = isolated, два кадра, по всем stage scratch/images.
- U8 small final SHA: `ae7bbb4ba77221fced85dd29a413b5c85d599942e2a611d2eebe6b140df2d9d2`.
- Устойчивый 4K final SHA: `c50a97ee6e344f6c61d15cbf969114aa6666e019852562e1b2448f503093d537`.
- Все output half values finite и nonzero в этих fixtures.

Главные captures:

- `build/fsr412_upstream-oracle/u8-smoke-literal/receipt.json`;
- `build/fsr412_upstream-oracle/u8-smoke-isolated/receipt.json`;
- `build/fsr412_upstream-oracle/u8-4k-ordered/receipt.json`;
- `build/fsr412_upstream-oracle/u8-4k-isolated/receipt.json`;
- `4k-a/4k-b`: сохранённые доказательства nondeterminism literal;
- `4k-isolated-a/4k-isolated-b`: повторяемость F16 fixture.

Captures содержат b0/b1, исходные изображения, compact tensors с original offsets/stride, marker, history/recurrent/output и hashes всего logical scratch. Full 4K captures объёмны: около 1.2 ГБ каждый.

## Что ещё не доказано

- Побитная Windows DLL/driver parity: Windows execution отсутствует.
- Физический Windows recurrent format и sampler allocation: shader reflection этого не доказывает. U8 default выбран для совпадения с существующим SDK/production contract; F16 fixture явно отдельный.
- Native M5/SIMD parity: нужен neural A/B на одинаковых PRE/stage snapshots. Loader предоставляет состояния, но сам не утверждает правильность optimized operators.
- Performance: этот recorder не benchmark; ordered режим намеренно медленный.
- Полное native A/B для partial viewport/DRS: маленький forced-2160 viewport показал CT-diff около right edge. Original reads используют class-sized raw halo (960x540 quarter в 2160), native convolution с input dimensions равными viewport зануляет этот halo. На следующих кадрах это может затрагивать P2 DS halo тоже. Dispatch/store ROI и class-sized input bounds нужно сохранять отдельно.
- GPU-проверка всех 384 permutations: статический выбор и representative compilation готовы, но hardware equivalence ещё не покрыта полной матрицей.

CPU tests:

```bash
python3 -m unittest discover -s tests -p test_fsr412_upstream_oracle.py -v
```
