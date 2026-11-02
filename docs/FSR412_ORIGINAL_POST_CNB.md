# Original Lite Lossy POST: точное ускорение CNB на run-кадрах

## Итог

**Рекомендуемый режим: native precomputed CNB + rewritten POST только на run. На skip - исходный isolated POST, без native precompute.**

MPP и SIMD прошли побитовый A/B для640x384,644x388 и3840x2160: каждый8 кадров,4skip, включая принудительные motion/cache-miss stress cases. Сверяются output,history,recurrent,reprojected, **весь logical scratch99820992 bytes** и isolated cache33273664 bytes.

Ускорение только run POST, native CNB включён в candidate time:

| Output | Original median | Native CNB + rewritten median | Ускорение |
|---|---:|---:|---:|
|1920x1080|7.7023мс|1.7615мс|4.373x|
|3840x2160|38.4355мс|5.8075мс|6.618x|

Это один interleaved11-sample замер с2 warmups, фиксированным pre-POST scratch, GPU timestamps; candidate состоит из двух command buffers, original из одного. CPU copies cache/scratch и чтение результатов не включены. Это bench fixture класса2160 на Apple M5, не прямой production timing и не Windows parity.

## Идентичность источников

External sources, не скопированные в repository:

- POST DXIL identity: `a69f41a0bb37376178001a7f73087d4726e21c0d531e7635653277053ac4b268`.
- Listing `build/fsr412_upstream/SHA.ll`: SHA256 `b447c87cfeeb2c1ca877d3af711c3c1ca2753f3fa56ab93e439fba9eb27269da`.
- Original isolated MSL `SHA.isolated.metal`: SHA256 `de56b08c7abaf362f3a913cd1d1dff002ff64e478746d3da3df20edc5807e498`.
- Rewritten `SHA.precomputed.metal`: SHA256 `b991352791e5cc5454a996aee04e7ec65f57feb426c0fccd65dc699eec36f6f8`.
- Rewritten metallib SHA256 `6cc9d7434a1c29999756a5d2432b77f61621113b4274a149e9f17a931b16ee95`.
- Original isolated metallib SHA256 `5913a870b7790dfe18b5d29d78e108f5bb957d1f0090b64645d427c0c90ae528`.

Далее `.metal`/`.ll` означают перечисленные original POST тексты. Диагностический ordered-isolated MSL имеет SHA256 `5c33c05c81787597a975436266ae2105bacef07e9fc23b3cf68e9bdbafa80c1c`; меняет только group-Y coordinate injection, не original math/cache.

## Четыре inline CNB блока

Активный half ROI: `hw=sdiv(b1.width,2)`, `hh=sdiv(b1.height,2)`. Базовый logical half-cell потока `(px,py)` берётся из GroupID/LocalID remap `.metal:219-224`, `.ll:215-227`.

POST marker @33273792 проверяет оба слова `0x5a17c0de,0x0a110e57`. На skip читается coarse motion из original scratch @33304416+row*30752 с diagonal-hole remap. Signed int16 X/Y clamp `[-254,254]`; **на run motion принудительно0**. Motion integer displacement measured in full pixels, half координата использует arithmetic shift right1, negative odd rounds toward negative infinity. Parity bits `dx&1`, `dy&1` определяют соседей.

Координаты:

- `bx=px+(dx>>1)`, `by=py+(dy>>1)`;
- `x0=clamp(bx,0,hw-1)`, `y0=clamp(by,0,hh-1)`;
- `x1=clamp(bx+1,0,hw-1)`, `y1=clamp(by+1,0,hh-1)`;
- сначала clamp каждой координаты независимо, а не прибавление1 к уже clamped x0/y0.

**Доказательства:** `.metal:683-715`; `.ll:1630-1695` motion, parity, signed clamp.

| Block | Half cell | Условие вычисления/lookup | Packed output |
|---|---|---|---|
|Primary|(x0,y0)|каждый valid half-cell; cache hit разрешён только наskip|`_107163.._107166`; miss-local `_26315,_26371,_26427,_26483`|
|X|(x1,y0)|`dx&1 !=0`; иначе reuse primary|`_107011.._107014`|
|Y|(x0,y1)|`dy&1 !=0`; иначе reuse primary|`_106859.._106862`|
|XY|(x1,y1)|обе parity!=0; иначе reuse primary|`_106707.._106710`|

Адрес cache: `99821056 + y*30752 + x*16` в isolated, original base49941264. Cache содержит4 uint words XOR `0xa5a5a5a5`. Hit: **все четыре ENCODED words ненулевые**. Нулевое decoded word является valid hit, если encoded word=`0xa5a5a5a5`. Любой encoded0 - miss.

Primary на run **всегда miss**, даже если cache содержит valid data. Primary miss вычисляет CNB и **пишет4 encoded words в cache на run и skip**. Наskip primary hit использует старый decoded CNB без пересчёта. Side X/Y/XY miss **не пишет cache**: только вычисляет packed output. В исходном POST всего один mask15 encoded-cache store site, в MSL4 scalar stores. Не надо дописывать новые side cache stores или заполнять cache целиком precomputed tensor.

**Доказательства:** `.metal:721-755,1750-1758,1773-1799,2780-2791,2797-2823,3804-3815,3821-3847,4828-4843`; `.ll:1697-1724` primary hit, `.ll:4247-4255` единственный cache store; `.ll:4298,6858,9417` side lookups. Owner clear `.metal:1760-1767` находится **вне** заменённых CNB scopes и сохранён.

## CNB math и точная граница чтения plane C

Каждый miss возвращает4 packed uint, то есть16signed INT8 channels:

1. 3x3 conv16->16 с первой quantization /256;
2. pointwise16->32, ReLU, quantization /256;
3. pointwise32->16, quantized contraction+central C residual, RNE /64 с residual shift5;
4. saturate signedINT8 и pack4bytes в каждое uint.

C logical cell address: `66547328 + (y+1)*30752 + (x+1)*16`. Row30752 фиксирован class2160, но **tap guards - active halfROI `hw,hh`**, не classHalf1920x1080 и не raw P2/P3 halo. Для tap `(x+tx-1,y+ty-1)` чтение выполняется только внутри `0<=tapX<hw,0<=tapY<hh`; остальные taps имеют zero contribution. Центральный residual всегда in-bounds из-за clamped parent coordinates.

Fast interior guard проверяет `(x-1)>=0,(y-1)>=0,(x+1)<hw,(y+1)<hh`; fast raw3x3 соответствует padded coordinates `(x..x+2,y..y+2)`. Fallback guards каждый tap отдельно. Особый center guard использует `GroupX<<4 | clampedY`; для допустимых положительных viewport всегдаtrue, не меняет центр.

**Доказательства:** `.metal:756-765,785-840` fast guard/loads; `.metal:995-1016,1040-1055,1060-1061,1134-1135,1212-1213,1306-1310,1371-1372,1448-1449,1525-1526` fallback tap guards; side equivalents `.metal:1790-1799,2029-2050,2814-2823,3838-3847`; `.ll:1728-1741,2394-2405,2494-2502,2593,2692,2789-2791,2995-2999` original integer guards.

### Missing-first-tap bias

Первый tap - `(x-1,y-1)`. Если он не прочитан, layerA берёт separate bias literals; это **не проверка общего краяoutput**, не fixed-class test и не изменение round mode. Для clamped coordinates условие сводится к `x==0 || y==0`. Например channel0primary bias11, missing-first-tap12; channel1 -559/-562.

**Доказательства:** `.ll:2494` PHI12; `.metal:995-1055` first-tap guard и missing literals; `tools/fsr412/native_model.py:92-101` native primary/missing biases; `src/fsr411_model.h:59-61,94-96` `kFirstTapBias` selection.

Имеющийся native API достаточен: `p.input.width=hw,height=hh` правильно задаёт **и tap bounds, и first-bias bounds**. Разделённый width/новый Epilogue flag не нужен. В rewritten math biases не остаются текстовыми literals: они перенесены в точное native CNB вычисление. Hit/miss/cache stores/owner/CT/temporal за пределами scopes сохранены byte-for-byte.

## Как runtime вызывать

Все перечисленные адреса - class2160 standalone fixture, **не автоматически универсальные для иных tiers/permutations**. Генератор fail-closed привязан к одному isolated shader SHA. Другие variants требуют отдельной верификации.

Precomputed layout default:

- base133094720 = isolatedCacheBase99821056 + cachePlane33273664;
- compact NHWC signedINT8,16bytes/cell;
- row stride30720 = classHalfW1920*16;
- size33177600 =30720*1080;
- scratch allocation166272320 = base+size;
- base/stride передаются generator `--base/--stride`, выравнивание16, stride>=30720. Native output обязано использовать те же значения.

На **run**, после originalP3 и перед rewrittenPOST:

1. `Runtime::prepare_integer_operators` с `Session.modelLite=true`,profileInt8LiteLossy,tier2160,`externalPostCnb=true`. Native trained initializer27392bytes; auxiliary pass **13**. Existing `kFirstTapBias` и `biases[3]` обязательны.
2. Параметры native13:
   - input.offset66578096 =2P+30752+16;
   - input.strideX16,strideY30752,channels16,bankStride0;
   - input.width=W/2,height=H/2 **active ROI**;
   - output.offset133094720,strideX16,strideY30720;
   - output.width=W/2,height=H/2,channels16,bankStride0;
   - weights/biases/q сохранить из native13 manifest без изменения.
3. Dispatch native13 по всему active half rectangle. Здесь весь rectangle нужен: motion наrun0, каждый halfcell primarymiss.
4. Обычный isolated cache-copy-in из original base49941264 в99821056, rewrittenPOST, прежний copy-back. Native precompute **не пишет original/isolated cache**.

На **skip**: native13 не выполнять; pipeline - **исходный isolated POST**; cache-copy-in/back и все bindings остаются прежними. Marker проверять по двум exactwords. Bench берёт skip из первогоword, поскольку originalPRE fixture гарантирует marker pair; production должен сохранить original exactgate.

Не менять original `OracleLoad/OracleStore` limits при расширении allocation: они остаются33273680uint words, precomputed reader адресует appended plane отдельно. Sentinel134217728 лежит внутри новой физической allocation, но original stores обязаны продолжать отбрасывать его по original guardedextent.

## Почему skip нельзя ускорять той же заменой

Наskip clamped motion mapping не взаимно однозначен: несколько threads/groups могут обращаться к одному cachecell. Primarymiss записывает4encoded words, другие primary/side hit ветки читают тот же mutablecache без межгрупповой синхронизации. Isolated adaptation убирает cache/C plane alias, **но не mutablecache read/write race внутриPOST**.

Факты GPU observation, не Windows:

1. All-frames precompute вариант ordinary8frames прошёл640/644/1080p/4K. Но forced signedmotion `(3,-3)` + encoded0 в16x16верхнем левом halfcache дал на4Kskipframe1 **93bytes recurrent_next mismatch**, первый byte476180, output pixel(5,31). Output/history/reprojected/scratch/cache совпали. Поэтому ordinary fixture недостаточен для general skip parity.
2. Сам **original isolated POST** сравнен с тем же original math/cache, но groupY строки выполняются по одной сcompletion между строками. Narrow16x16misses прошли normal/forward/reverse8frames.
3. Distributed misses every8half cells по всемуactiveROI пересекают границыgroups. На4Kskipframe1 результаты самогооригинала зависят от порядка:

| Originalnormal vs | Output differentbytes / firstpixel | History differentbytes / firstpixel | Recurrent differentbytes / firstpixel |
|---|---|---|---|
|Forwardrowgroups|2136 /(813,19)|1720 /(173,19)|38996 /(45,19)|
|Reverserowgroups|1993 /(813,19)|1637 /(349,19)|38915 /(173,19)|

Reprojected,весьlogicalScratch,финальнаяisolatedCacheplane совпали. Повторные counts немного меняются (предыдущие forward1904outputbytes,reverse2020), что дополнительно согласуется сschedule race. Нельзя утверждать конкретный Windows результат или устранять race заменой policies под видом parity.

Консервативный run-only вариант сохраняет originalskip behavior. Даже он не способен гарантировать равенство **двух отдельных запусков самогоoriginalskip** при произвольномschedule-stress: это ограничениеoracle, не nativeCNB math.

## Проверки и результаты

Default run-only + originalskip, narrowcache stress:

| Backend | Output sizes | Result |
|---|---|---|
|MPP|640x384,644x388,3840x2160|8frames/4skip каждый, fullbitwisepass|
|SIMD|640x384,644x388,3840x2160|8frames/4skip каждый, fullbitwisepass|

Stress finaloutput SHA:

-640: `0de6108bf66fa5189ba2549efbdf1e3a721582d8fb80204371f5dce2378e534b`;
-644: `3910b078ce1ab33cdcf6eb13b2de0124e4a00f104833974e166f4ef02cee489d`;
-4K: `38a36802cc0a0afb5e0d82b46a550698e3998b4f609dec722319392bf506ef50`.

Ordinary finaloutput SHA:

-1080p: `3525e03fe054944ece9b70e1d5660bd85816b9bab53db629db2c8cdf15c8d4f1`;
-4K: `358a2937a7bd96c4577d972f3453cf33f17b9742785c2416ba6d792f6e93c6d1`.

Scalar diagnostic6448frames allframes прошёл bitwise, ноCNB~2.63мс противMPP~0.04мс; не speed path.

## Новые файлы / запуск

- `tools/fsr412/original_post_cnb.py`: pinned fail-closed four math scopes replacement, custombase/stride, receipt, optional ordered-isolated diagnostic.
- `tests/test_fsr412_original_post_cnb.py`:9CPUtests: untouchedsource exactmatch, originalstores/XOR/gates, layout, failclosed, runonlypolicy, ordereddiagnostic.
- `tools/bench/fsr412_original_post_cnb_ab.mm`: Replay/native13A/B, defaultoriginalskipfallback, motion/cache stress, originalschedulecomparison, interleavedGPUtiming.

```sh
python3 tools/fsr412/original_post_cnb.py
python3 -m unittest discover -s tests -p test_fsr412_original_post_cnb.py -v
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -c build/fsr412_upstream/SHA.precomputed.metal -o build/fsr412_upstream/SHA.precomputed.air
xcrun -sdk macosx metallib build/fsr412_upstream/SHA.precomputed.air -o build/fsr412_upstream/SHA.precomputed.metallib
clang++ -std=c++17 -O2 -fobjc-arc -framework Foundation -framework Metal tools/bench/fsr412_original_post_cnb_ab.mm -o /tmp/fsr412_original_post_cnb_ab
/tmp/fsr412_original_post_cnb_ab ROOT --size 960 540 1920 1080 --frames 8 --time 11
/tmp/fsr412_original_post_cnb_ab ROOT --simd --size 388 228 644 388 --frames 8 --stress-cache
```

Вcompileкомандах заменитьSHA наPOSTidentityвначалеотчёта. Schedule diagnostic:

```sh
python3 tools/fsr412/original_post_cnb.py --ordered-diagnostic --out build/fsr412_upstream/SHA.isolated-row.metal
# compile .isolated-row.metal аналогично
/tmp/fsr412_original_post_cnb_ab ROOT --size 1920 1080 3840 2160 --frames 8 --stress-wide --ordered-original
/tmp/fsr412_original_post_cnb_ab ROOT --size 1920 1080 3840 2160 --frames 8 --stress-wide --reverse-original
```

`--all-frames` - **отрицательный диагностический режим**, не production recommendation. `--original-pair` сравниваетoriginalnormal сoriginalnormal. `--stress-motion-only` меняетmotionбезcachemiss injection. Всеgeneratedoriginal-derivedMSL/weights/receipts остаютсяв`build/`; `src/`,runtime/compiler/bundles не изменялись.
