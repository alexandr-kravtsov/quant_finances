# Research archive

Исследования разложены по направлениям, затем по отдельным экспериментам. Файлы notebook перенесены без изменения ячеек, outputs и metadata, кроме необходимых импортов и путей импорта к новому `src/`.

## Направления

- `afml/` — ранняя линия `JERICHON_mark`, AFML-эксперименты, feature lab и диагностика признаков.
- `nowcasting/` — MultiTicker NCA, Extremums/Reversal и Event-vs-Control.
- `tradeflow/` — последовательность сделок, TFI → price response, Stage 2 и Stage 3.
- `supercandles/` — SC01–SC08 и относящиеся к ним результаты.

## Правило хранения

Каждый `mark_N/` — отдельный эксперимент. Если к notebook есть сохранённые CSV/JSON/MD outputs, они лежат в его `results/`. Результаты не интерпретировались заново при переносе; для научной интерпретации будут созданы отдельные отчёты после проверки исходных результатов.

## AFML: последовательность JERICHON_mark

| Новая папка / файл | Исходный файл | Примечание |
|---|---|---|
| `afml/mark_1/JERICHON_mark_1.ipynb` | `data/for_ml/models/jerichon_mark_42.ipynb` | Первая версия по заданному порядку |
| `afml/mark_2/JERICHON_mark_2.ipynb` | `data/for_ml/models/jerichon_mark_42 #U2014 copied.ipynb` | Отдельная копия/вариант, сохранён как самостоятельный notebook |
| `afml/mark_3/JERICHON_mark_3.ipynb` | `data/for_ml/models/jerichon_mark_44.ipynb` | Продолжение Feature Lab |
| `afml/mark_4/JERICHON_mark_4.ipynb` | `data/for_ml/models/jerichon_mark_63.ipynb` | Causal feature dataset pipeline |
| `afml/mark_5/JERICHON_mark_5.ipynb` | `archive/jerichon_mark_63.ipynb` | Research 01: data quality → first baseline |
| `afml/mark_6/JERICHON_mark_6.ipynb` | `archive/JERICHON_mark_63.6.ipynb` | Research 02: режимы и аудит сделок |
| `afml/mark_7/JERICHON_mark_7.ipynb` | `archive/JERICHON_mark.63.7.ipynb` | Research 03: устойчивость режимов |

Файлы `JERICHON_mark_63.3` и `JERICHON_mark_63.3_CAUSAL_AUDIT_v2` сохранены отдельной веткой `afml/feature_lab/`, чтобы не менять заданную последовательность mark_1–mark_7.

## Классификация

`JERICHON_AFML_02_Binary_ModelZoo_Backtest_FIXED_BARRIERS.ipynb` перенесён в AFML, несмотря на прежнее расположение в `supercandles research/`: его заголовок и задача прямо обозначают AFML TP/SL model zoo. `XXX_Microstructure_Trade_Sequence.ipynb` перенесён в Trade Flow, поскольку он исследует последовательность сделок и поток, а не Super Candle event study.

## Исключённые каталоги

`archive/regime_research_outputs/` и `archive/regime_research_03_outputs/` намеренно не перенесены: происхождение и связь с конкретным экспериментом не подтверждены. Остальные явно относящиеся к AFML, Nowcasting, Trade Flow и Super Candles outputs сохранены рядом с соответствующими исследованиями.
