# Профилирование FSR 4.1.1

`--profile-stages` добавляет GPU timestamps для PRE, проходов сети 1–12,
отдельного `post_cnb`, POST, AE, RCAS, padding и вспомогательных копирований
или очисток. Это диагностический режим CLI. Длительность обычного исполнения
измеряется отдельным запуском `--bench` без этого флага.

```sh
make tools/bin/fsr4mac
tools/bin/fsr4mac testcases/quality411 out/bench --i8mpp --profile lossy --bench 10
tools/bin/fsr4mac testcases/quality411 out/stages --i8mpp --profile lossy --bench 10 --profile-stages --no-compile
```

Используйте одну запись, одинаковые параметры модели, разрешения, RCAS,
питания и условия нагрузки. Запускайте сравниваемые варианты последовательно.
`--bench N` прогревает одну последовательность и измеряет следующие N;
записывается только `run.json`. `--no-compile` требует заранее собранного кэша.
Диагностические дампы текстур и scratch с `--bench` несовместимы.

Каждая последовательность создаёт новый `View`. Поэтому её первый кадр
включает начальные очистки, даже если исходная запись не запрашивала reset.
Для оценки обычных игровых кадров отдельно считайте строки с
`effective_reset=false`. В Lossy также разделяйте `skipped_inference=false`
и `true`: среднее всей последовательности зависит от доли полных проходов.

## Метод и единицы

На Apple M5 доступен timestamp counter set и sampling на границах encoder;
sampling на границах отдельных classic compute dispatch не поддерживается.
CLI помещает каждый dispatch в отдельный serial compute encoder внутри
одного command buffer и прикрепляет start/end timestamp samples. На GPU,
поддерживающем dispatch sampling, сохраняется один encoder, а samples
снимаются перед и после dispatch с sampling barriers.

Оба способа добавляют накладные расходы. Без `--profile-stages` runtime
сохраняет один compute encoder, не создаёт counter buffers и не записывает
timestamp samples. Профилирование не используется живым MetalFX-хуком;
представленный CLI-путь работает через classic Metal command buffer.

GPU ticks калибруются по двум парам `device.sampleTimestamps` до исполнения
и после завершения command buffer. Возвращаемый Metal CPU timestamp уже
измеряется в наносекундах:

```text
gpu_tick_ms = (cpu_after - cpu_before) / (gpu_after - gpu_before) / 1e6
stage_ms = (stage_end - stage_start) * gpu_tick_ms
```

`mach_timebase_info` здесь не применяется. Повторная конверсия CPU clock
на M5 ошибочно увеличивает результат примерно в 41,67 раза. CLI проверяет,
что охваченный samples интервал укладывается в полное GPU-время command
buffer с допуском на калибровку: максимум 0,1 мс или 10% его длительности.

## Поля `run.json`

`gpu_ms` в каждой строке `measurements` по-прежнему вычисляется из
`GPUEndTime - GPUStartTime`. При включённых counters это время измеренного
исполнения с инструментированием. `diagnostic_gpu_counters=true` и
`stage_profiling.production_timing=false` явно обозначают этот режим.

В `measurements[].stage_profile` находятся:

- `stages`: имя, длительность и исходные start/end GPU ticks каждого dispatch.
- `stage_sum_ms`: сумма отдельных длительностей.
- `sampled_span_ms`: время от самого раннего start до последнего end sample.
- `nonoverlapping_stage_ms`: длина объединения интервалов исполнения.
- `overlapped_stage_ms`: часть суммы, учтённая повторно из-за перекрытия.
- `inter_stage_gap_ms`: промежутки внутри охваченного интервала, не занятые samples.
- `gpu_tick_ms`, `calibration_cpu_ticks`, `calibration_gpu_ticks`: данные калибровки.

Независимые очистки и копирования в разных encoders могут перекрываться.
Поэтому `stage_sum_ms` может превышать полное GPU-время; объединение
неперекрывающихся интервалов проверяется по той же границе, что и span.
Промежутки включают sampling и scheduling, а не отдельную стоимость barriers.
`stage_profiling.summary` усредняет каждый проход только по кадрам, где он
исполнялся: отсутствующие NN-проходы в повторно используемых Lossy-кадрах
не добавляются в это среднее как нули.

Проверка профилирования на небольшой последовательности сверяет неизменность
выходных пикселей, расписание Lossy и согласованность timestamps:

```sh
python3 tests/run_fsr411_cli.py --stage-profiling
```
