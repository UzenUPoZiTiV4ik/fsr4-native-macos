# Original Lite Lossy: CPU-контракт геометрии P1/P2/P3

Проверено только по уже извлечённым текстам и CPU-перебором координат. DLL, GPU, Windows, network и бинарные контейнеры не открывались/не выполнялись. Это контракт выбранных shader variants и указанного provider-compatible dispatch, **не наблюдение Windows dispatch**.

## Источники

Имена ниже сокращены для ссылок; все пути относительно `build/fsr412_upstream/`:

- **P1**: `2638456df248932a18dfd82b556d2c06d8ae3ab0fcd6f67eeeb037782bf6a0ea.{ll,metal}`.
- **P2**: `6a5d98fc27a87938ab35d09af117f62238355cdab06aeaf3a9b26d91c434fb5e.{ll,metal}`.
- **P3**: `4a8443a4c81abc7c08a64f4131e8a1aeb86e33c1f49c5be530c886d33507a238.{ll,metal}`.
- **P2-4320**: `84e57a8c5f5e10a0fa04f69dbd0e0b257d36d76a1e1fe7704ca3a1f862a173b1.ll`; MSL в `bundles/4320-di1-lm0-jm0-cs1-ae0/`.
- **P1-4320**: `f01148e8802285c16f7582e7743893916a39860b16f0890af39f00d0c0f5c80f.ll`.

CPU fixture проверяет SHA-256 всех пяти `.ll`, а также P1/P2 MSL по `.translation.json`. Хеши текстов перечислены в `tests/test_fsr412_original_geometry.py:LISTING_HASHES`. Исходные DXIL SHA - имена файлов, не SHA текстового listing.

## Система координат и extent

Здесь `W,H` - полный output viewport, положительные числа, кратные 4, не больше выбранного класса. Для класса **2160**: full `3840x2160`, half `1920x1080`, quarter `960x540`. Все указанные write rectangles начинаются с logical `(0,0)`; размеры включают только interior.

Адрес logical cell `(x,y)` с 16 INT8 channels:

- half: `base + (y+1)*30752 + (x+1)*16`, plane `P=33273664`;
- quarter: `base + (y+1)*15392 + (x+1)*16`, plane `Q=8342464`;
- P1 input `base=0`, CNB `base=P`, DS `base=2P` и `2P+Q`;
- P2 FNB output `base=0` и `Q`;
- P3 upsample output `base=2P`.

Физический padded half plane `1922x1082`, quarter `962x542`. Logical extent для входных проверок определяется **классом**, не active ROI. Ни P1, ни P2 не устанавливают zero-border на границе partial viewport. Не записанная в текущем проходе область остаётся прежней.

## P1: точная запись и чтение

Dispatch groups: `ceil((W/2)/64), H/2, 1`; group size `64x2x1`.

Исходный remap, установленный независимо по SSA:

- neural groups только при `(GroupY & 2)==0`;
- один `GroupX` вместе с парой `GroupY=4k,4k+1` покрывает 64 колонки и 4 строки;
- `(GroupX,GroupY)=(0,2)` - отдельный border helper branch, не neural interior;
- остальные groups с `GroupY&2` не пишут.

Следовательно:

- CNB width: `min(classHalfW,64*ceil((W/2)/64))`;
- CNB height: `min(classHalfH,4*ceil((H/2)/4))`;
- DS width/height в **обеих** плоскостях: ровно половина CNB rectangle.

| Viewport | P1 groups | Active half / quarter | CNB writes | DS writes, обе plane | Extra CNB / DS cols,rows |
|---|---|---|---|---|---|
| 640x384 | 5x192 | 320x192 / 160x96 | 320x192 | 160x96 | 0,0 / 0,0 |
| 644x388 | 6x194 | 322x194 / 161x97 | **384x196** | **192x98** | **62,2 / 31,1** |
| 3840x2160 | 30x1080 | 1920x1080 / 960x540 | 1920x1080 | 960x540 | 0,0 / 0,0 |

Bottom nuance: при halfH=194 последняя neural pair `GroupY=192,193` пишет строки `192..195`. При aligned W=640,H=388 всё равно CNB `320x196`, DS `160x98`. Лишние строки не зависят от округления X.

P1 input - 3x3 taps вокруг **всего** CNB write rectangle, с per-tap class guard. In-bounds read union:

| Viewport | Logical input read union, включительно |
|---|---|
| 640x384 | X `0..320`, Y `0..192` |
| 644x388 | X `0..384`, Y `0..196` |
| full class | X `0..1919`, Y `0..1079` |

Отрицательные taps и taps за classHalf отсутствуют в convolution, то есть имеют zero contribution. На partial ROI, например `(320,192)` для640, это **реальное чтение**, а не ноль по ROI. Residual тоже берёт центральный P1 input cell.

**Доказательства:** P1 `.ll:70-101` group/lane remap; `.ll:158-166` fast class guard; `.ll:812-814,908,1005,1101,1398` fallback per-tap guards; `.metal:187-243` девять fast raw loads; `.metal:417-418,435-437` fallback top-left; `.ll:2439-2465` central residual; `.ll:3020-3029` CNB/DS class bounds; `.ll:3395-3411` CNB mask15 store и DS mask3 store. DS mask3 пишет 8 bytes на lane, две lanes дополняют один 16-byte cell каждой plane, без дублей/дырок.

## P1 helper и skip

Original `(0,2)` branch чистит **class quarter perimeter**, не partial ROI:

- raw rows `0,541`, X `0..961`;
- raw cols `0,961`, Y `1..540`;
- обе DS planes, 3004 cells/plane, 6008 cells всего, 24032 uint words.

В logical координатах это rows `-1,540`, cols `-1,960`. Соседние с active partial ROI interior cells не чистятся.

Marker `0x5a17c0de,0x0a110e57` @`33273792` проверяется **до** group branch. При skip нет ни CNB/DS записи, ни helper clear. `force_border_group()` меняет только GroupID на `(0,2,0)`; исходный skip gate сохраняется. Helper dispatch `1x1x1`, threads `64x2x1`. Нельзя заменять helper безусловным CPU clear.

**Доказательства:** P1 `.ll:56-66` early skip; `.ll:3417-3456` complete border loop; `.metal:106-144`; `tools/fsr412/oracle_pass1_border.py:16-30,41-50` coordinate-only substitution / preserved marker gate.

## P2: actual reads шире, stores только active quarter

Provider-compatible groups такие же, как P1, но threads `64x1x1`, без P1 remap. Class gate: `x<960 && y<540`. **Dynamic b0[4] проверяется только перед stores**, после вычислений и чтений.

Пусть `N=min(64*groupXCount,960)`, `M=min(H/2,540)`. Actual executing invocation rectangle `NxM`. В первой DS plane `@2P` читается 3x3, во второй `@2P+Q` - **только центр**, 16 bytes.

| Viewport | Executing invocations | Actual first-plane raw read union | Actual second-plane raw read union | FNB writes, обе plane |
|---|---|---|---|---|
| 640x384 | 320x192 | X `0..321`, Y `0..193` | X `1..320`, Y `1..192` | 160x96 |
| 644x388 | 384x194 | X `0..385`, Y `0..195` | X `1..384`, Y `1..194` | 161x97 |
| full class | 960x540 | X `0..961`, Y `0..541` | X `1..960`, Y `1..540` | 960x540 |

Raw coordinates - padded, logical = raw minus1. В первой plane raw top/left и full-class bottom/right halo реально читаются; нулевое содержимое обеспечено P1 border helper, не P2 class/ROI clipping.

Только dependency reads **valid output threads** уже:

- 640: first logical X `-1..160`, Y `-1..96`; second X `0..159`, Y `0..95`;
- 644: first logical X `-1..161`, Y `-1..97`; second X `0..160`, Y `0..96`.

Уменьшение P2 до quarter-grid не меняет valid output dependency footprint. Но actual read trace становится меньше; invalid-store effects надо проверять отдельно.

**Доказательства:** P2 `.ll:77-92` skip/class gate; `.ll:155-161,534-542,562-569,588-595,615-622,641-649,668-676,694-702,720-728` first-plane 3x3; `.ll:2565-2580` second-plane center; `.metal:354-357,1970-1977,3672-3682` center residuals; `.ll:7373-7375,7432-7459,7748,8042` b0 slot4 / dynamic select / stores. Active quarter bits записывает `tools/bench/fsr412_upstream_replay.h:74-75`; provider metadata `tools/fsr412/oracle_bundle.py:108`, standalone replay P2 quarter grid `.h:139-140`.

## P2 invalid sentinel: класс4320

Class-invalid invocation (`x>=classQuarterW` или `y>=classQuarterH`) возвращается без neural stores. **Dynamic-invalid, но class-valid** invocation записывает zero uint4 по byte address `134217728`, два раза для двух output planes. Это не halo clear и не output-ROI write.

- Для1080 allocation `25027392`, для2160 `99820992`: весь interval `[134217728,134217744)` вне allocation, guard отбрасывает записи.
- Для4320 allocation `398707392`: interval внутри residual plane `P=132902464`.
- `134217728-P = 21*61472 + 1522*16`. Это padded half `(1522,21)`, то есть **logical CNB/residual `(1521,20)`**, все16 channels обнуляются.

Provider640: 46080 dynamic-invalid invocations; provider644:58879. Все они дают одинаковый sentinel effect, не разные адреса. Для fullclass4320 `(7680x4320)` surviving class invocations все dynamic-valid: sentinel store отсутствует.

Важно: quarter-grid640/644 **уже сохраняет** sentinel, потому что X округлён до192quarter threads:3072 и3007 dynamic-invalid invocations соответственно. Эффект исчезнет при полном removal invalid threads, либо в quarter-grid с quarterW кратным64, например W256. Сокращение dispatch само по себе не всегда теряет sentinel.

**Доказательства:** P2-4320 `.ll:77-80` class gate; `.ll:7420-7447` selects zero, address sentinel; `.ll:7736,8030` два mask15 stores; `bundles/4320-di1-lm0-jm0-cs1-ae0/bindings.json:24-30,791,796-800` class geometry, dispatch, allocation/sentinel.

## P1 sentinel и savedCNB offsets

P1-4320 также содержит sentinel-select для **class-invalid CNB output coordinate**: `.ll:3014-3017`, stores `.ll:3394-3399`. Но при корректном provider dispatch, `W,H<=class`, все neural P1 coordinates остаются внутри classHalf: class width3840 кратна64, class height2160 кратна4. Это подтверждено exhaustive CPU-перебором полного4320. Top/left invalid convolution taps не означают invalid output store. Для заданных partial viewport и fullclass **P1 sentinel effect отсутствует**.

Произвольный расширенный P1 dispatch за class может выбрать этот адрес, но это **не доказанный zero-clear**: туда направляется вычисленный CNB uint4, включая возможные undef входы. Такой dispatch не входит в текущий контракт.

Layout offsets, также отражённые в `tools/fsr412/upstream_oracle.py:431-442` и bundle `bindings.json:31-34`:

| Tier | Marker P+128 | Saved jitter P+136 | savedCNB base |
|---|---|---|---|
|1080|8342592|8342600|12529104|
|2160|33273792|33273800|49941264|
|4320|отсутствует|отсутствует|отсутствует|

SavedCNB - POST cache, 16 encoded bytes на **half** cell. Base `3P/2+halfStride+16`, cell address `base+y*halfStride+x*16`. Для2160 cache rowY540 начинается в `2P+16` и перекрывает live neural input POST. Это не отдельный DS buffer и не unused region для clears. Skip marker находится в CNB padding до interior и не затрагивается P1 neural stores.

## P3: no extra writes при quarter overdispatch

P3 сначала читает class-sized quarter input, но затем делает **dynamic early-return по b0[4] до upsample stores**. Каждый valid quarter invocation пишет2x2 half cells; stores дополнительно guarded classHalf. Для данных viewport итог:

- 640: half output320x192;
- 644: half output322x194, **не384x194 и не384x196**;
- fullclass:1920x1080.

Quarter-grid X overdispatch, а также provider-compatible overdispatch, не пишут ни extra CNB cells, ни P2-style dynamic-invalid sentinel. P3 sentinel select существует только внутри2x2 valid-parent output branch для class-invalid child; при корректном viewport в пределах класса ни один child не invalid.

**Доказательства:** P3 `.ll:86-89` class gate; `.ll:1341-1351` b0[4] dynamic early return / doubled parent coordinates; `.ll:1393-1397` child `(2x+dx,2y+dy)`; `.ll:1866-1868,1891,2078` class child gate / store; `.metal:866-871,894-896,1208,1221-1225` те же условия.

## Что интегрировать и чем проверять

Разделять **input class extent**, **dispatch domain**, **write rectangle** и **active ROI**. Одинаковые width/height для этих четырёх ролей неверны. Не занулять всё за active ROI: scratch persistent и aliases сохраняют реальные данные. Class border helper, per-tap class zero contribution и physical-OOB zero/discard - три разные вещи.

Добавлены только новые файлы:

- `src/fsr412_original_geometry.h`: `fsr412_original_geometry::make(viewport,classHalf)` и adapter `Layout make_geometry(tier,W,H)`, чистые размеры/счётчики/offsets; native ABI и policies не трогает. Requested fields: `classHalf`, `classQuarter`, `CNBwritten`, `DSwritten`, `actualHalf`, `actualQuarter`, `originalProviderP2grid` (groups, Z1), `P2sentinelInAllocationNeedsClear`. Последний флаг означает наличие оригинального16-byte zero-store эффекта, а не реализует clear/dispatch policy.
- `tests/fsr412_original_geometry_test.cpp`: standalone CPU constexpr expectations, exhaustive CNB/DS/P3 coverage для2160 и4320, bottom194, sentinel address geometry.
- `tests/test_fsr412_original_geometry.py`: текстовые hashes, original SSA coordinate evaluator, marker-preservation, P2/P3 gate evidence, exhaustive CPU coverage.

Проверки:

```sh
clang++ -std=c++17 -O2 -Wall -Wextra -pedantic tests/fsr412_original_geometry_test.cpp -o /tmp/fsr412_original_geometry_test
/tmp/fsr412_original_geometry_test
python3 -m unittest discover -s tests -p test_fsr412_original_geometry.py -v
```

CPU coverage не выполняет shader math и не доказывает compiler/numerical/GPU/Windows parity. Для сравнения recurrent frame2 нужно сохранять class-sized backing и все оригинальные вне-ROI записи, а не только сверять interiors кадра0.
