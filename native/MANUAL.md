# Руководство: запуск моделей на DGX Spark (GB10)

Это руководство для того, кто получил этот репозиторий и хочет запустить
одну из подготовленных здесь моделей на DGX Spark: что поставить, что
скачать, какой командой запустить, как проверить, что всё работает, и как
выжать из машины максимум. Подробности, замеры и обоснования каждого
значения лежат в [`README.md`](README.md) и
[`results/RESULTS.md`](results/RESULTS.md); здесь — порядок действий.

Все команды выполняются из каталога `native/` репозитория.

---

## 1. Что здесь есть

Два движка:

- **SGLang 0.5.20** (нативно, без Docker) — почти все модели. Одна
  команда `./serve.sh <профиль>`; профиль — это файл `models/<профиль>.sh`
  с путями к весам и флагами модели.
- **llama.cpp** (отдельная сборка в `llamacpp/`) — только GLM-5.3-Flash в
  GGUF, которую SGLang не обслуживает.

| Профиль | Модель | Веса на диске | Скорость, 1 поток | Состояние |
|---|---|---:|---:|---|
| `qwen3.8-27b` | Qwen3.8-27B + драфт DFlash2 | 22–29 ГБ + 3 ГБ | **94.2 ток/с** (NVFP4 RadixArk) | основная модель, измерена |
| `qwen3.8-27b-single` | то же, настроено на 1–2 пользователя | | 70.9 (драфт 16, замер до FP8-драфта) | измерена |
| `qwen3.8-27b-throughput` | то же, до 32 одновременных запросов | | 597.8 ток/с суммарно на 32 потоках | измерена |
| `qwen3.8-27b-longctx` | то же, 6 сессий по 262K | | | не проверялась целиком |
| `qwen3.8-27b-dspark` | то же с драфтом DSpark | +3.7 ГБ | | не измерена |
| `zen6` | Zen 6 (те же веса RadixArk + YaRN ×4, контекст 1M) | ~25 ГБ | 80.0 (драфт 11) | измерена, качество = базовой |
| `meerkat-triz` | Qwen3.8-27B + LoRA по ТРИЗ (слитая), FP8 при загрузке | ~54 ГБ | ~30 | измерена, HumanEval не хуже базы |
| `qwen3.8-flash-next` | Qwen3.8-Flash-Next 176B MoE, NVFP4 | 126 ГиБ | **65.8** | измерена |
| `qwen3.8-flash-next-abliterated` | то же, abliterated-веса Huihui | 126 ГиБ | | работает |
| `ornith-1.5-35b` | Ornith 1.5 35B-A3B FP8 + драфт DFlash | ~35 ГБ + драфт | **97.6** | измерена |
| `gemma4-31b` | Gemma 4 31B BF16 + MTP | ~62 ГБ | | не измерена |
| `llamacpp/` | GLM-5.3-Flash 320B, GGUF 3.0/3.5 бит | 117 / 137 ГБ | | на одном Spark — проверка, на двух — рабочий вариант |

Главное правило: **одна модель за раз**. Память GPU и оперативная память на
Spark — одна и та же (128 ГБ), и каждая модель занимает её почти всю.

---

## 2. Подготовка машины (один раз)

Нужен DGX Spark с DGX OS (Ubuntu 24.04, драйвер 580.x / CUDA 13.0).

```bash
git clone https://github.com/reproart/gb10-repo-modified.git
cd gb10-repo-modified/native

sudo apt install python3.12-venv python3.12-dev build-essential
sudo mkdir -p /models && sudo chown $USER /models

./scripts/00-check-host.sh     # драйвер, CUDA, Python, память, earlyoom
./serve.sh install             # venv ~/spark/venv-sglang-0.5.20 со SGLang 0.5.20 и hf CLI (~10 ГБ)
./scripts/00-check-host.sh     # ещё раз: теперь проверит, что torch видит GPU
```

`./serve.sh install` ставит пакеты строго по lock-файлу
`requirements/sglang-0.5.20-aarch64-py312.txt`: ничего не компилируется при
установке, версии совпадают с теми, на которых всё измерено. Утилита для
скачивания весов после этого лежит в `~/spark/venv/bin/hf`:

```bash
HF=~/spark/venv/bin/hf        # дальше в руководстве используется так
```

Веса всех моделей публичные, токен не нужен (кроме Gemma: см. ниже).
Скачивайте в `/models/...` — именно эти пути прописаны в профилях. Если
веса лежат в другом месте, путь передаётся переменной `MODEL_DIR=...`.

---

## 3. Как устроен запуск (общее для всех SGLang-моделей)

```bash
./serve.sh <профиль>                       # запуск в терминале, Ctrl-C — стоп
curl -s localhost:8888/v1/models           # сервер готов, когда отвечает
```

- **Адрес:** `http://<машина>:8888/v1`, совместим с OpenAI и Anthropic API.
  Имя модели для клиентов печатается при старте (`listen ... as "<имя>"`).
- **Первый запуск долгий.** Компилируются и подбираются ядра (от 5 минут до
  часа для первой FP8-модели). Не прерывайте «тихий» первый старт: всё
  кэшируется в `~/.cache/flashinfer` и `~/.triton`, следующие старты быстрее.
- **Строки при старте** — первое, что стоит читать: профиль, каталоги весов
  с ревизией (`revision unknown` — каталог без метаданных `hf download`),
  одна строка `model ...` со всеми включёнными оптимизациями.

**Настройки** бывают двух видов:

- **Машина** — в `serve.sh` (порт `PORT`, `HOST`, ключ `API_KEY`, привязка к
  ядрам `CPUSET`, `JIT_JOBS`, профиль по умолчанию `PROFILE`). Эти значения
  экспортируются как записаны в файле: меняйте их в самом `serve.sh`.
- **Модель** — в `models/<профиль>.sh`. Каждое значение вида
  `${VAR:-default}`, поэтому для разового эксперимента файл править не
  нужно:

  ```bash
  DRAFT_TOKENS=11 MAX_RUNNING=12 ./serve.sh qwen3.8-27b
  ```

  Постоянные изменения — правкой профиля (удобно держать их отдельным
  локальным коммитом поверх репозитория).

**Как службу** (запуск при загрузке, перезапуск при сбое):

```bash
./scripts/install-service.sh <профиль>     # просит sudo; одна служба — одна модель
journalctl -u gb10-sglang -f               # лог загрузки
sudo systemctl restart gb10-sglang         # после правки serve.sh или профиля
sudo systemctl stop gb10-sglang            # перед запуском другой модели вручную
```

Служба запускает профиль с его значениями по умолчанию: переменные из
командной строки на неё не действуют, только правка профиля.

**Перед запуском** `serve.sh` проверяет, что порт свободен и что свободной
памяти не меньше 40 ГиБ (`MIN_FREE_GIB`). Если нет — печатает, какие
процессы её держат: чаще всего это забытая служба `gb10-sglang` или
предыдущий эксперимент.

---

## 4. Модели

### 4.1. Qwen3.8-27B — основная модель

Плотная 27B с гибридным вниманием (GDN) и спекулятивным декодированием
DFlash2: маленький драфт предлагает до 15 токенов, большая модель проверяет
их за один шаг. Ответы при этом те же, что без драфта.

**Веса** (две цели на выбор + драфт):

```bash
# точнее (HumanEval 97.6% без размышления), медленнее:
$HF download Qwen/Qwen3.8-27B-FP8 --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --local-dir /models/Qwen3.8-27B-FP8
# быстрее (~40% на одном потоке, в 2.5 раза быстрее prefill), 93.9–95.1% без размышления:
$HF download RadixArk/Qwen3.8-27B-NVFP4 --revision 554ebba9b5f1b79dc11246341960360e6ef05ef4 \
  --local-dir /models/RadixArk/Qwen3.8-27B-NVFP4
# драфт (обязателен):
$HF download z-lab/Qwen3.8-27B-DFlash2 --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --local-dir /models/Qwen3.8-27B-DFlash2
```

С размышлением (`reasoning_effort` medium) обе цели ошибаются в HumanEval не
больше одного раза — тогда выгоднее быстрая NVFP4.

**Запуск:**

```bash
./serve.sh qwen3.8-27b                                              # цель FP8 по умолчанию
MODEL_DIR=/models/RadixArk/Qwen3.8-27B-NVFP4 ./serve.sh qwen3.8-27b # цель NVFP4
```

Чтобы NVFP4 была целью всегда — поменяйте `MODEL_DIR` в
`models/qwen3.8-27b.sh`; варианты `-single`, `-throughput`, `-longctx`,
`-dspark`, а также `zen6` берут остальное оттуда же.

**Варианты под нагрузку:**

| Профиль | Запросов одновременно | Драфт | Когда |
|---|---:|---:|---|
| `qwen3.8-27b` | до 12 | 15 | по умолчанию: один пользователь с кодом, агенты, до ~8 параллельных |
| `qwen3.8-27b-single` | до 16 | 16 | 1–2 пользователя |
| `qwen3.8-27b-throughput` | до 32 | 10 | много коротких параллельных запросов |
| `qwen3.8-27b-longctx` | 6 | 10 | несколько сессий по 262K токенов |
| `qwen3.8-27b-dspark` | до 12 | DSpark γ 7 | сравнение с другим драфтом (`$HF download RadixArk/Qwen3.8-27B-DSpark --local-dir /models/Qwen3.8-27B-DSpark`) |

**Главные ручки:**

- `DRAFT_TOKENS` — самый сильный рычаг одного потока. 15 (по умолчанию)
  даёт 94.2 ток/с на NVFP4 и выигрывает на 1–8 потоках; если обычно идёт
  10–12 запросов разом, `DRAFT_TOKENS=11` даёт +14% суммарно на 12.
- `MAX_RUNNING` — сколько запросов реально идёт одновременно. Резервирует
  память под состояние GDN заранее: больше, чем нужно, — меньше KV-кэша.
- `FP8_DRAFT=1` (по умолчанию) — драфт в FP8, ответы не меняет, +4%.
- `FP8_SIDE`, `FP4_GEMM_BACKEND` — измерены, выигрыша почти нет; оставьте
  как есть.
- `SPEC=mtp` — вместо драфта DFlash2 встроенная MTP-голова модели (NEXTN,
  3 шага, 4 токена; отдельный драфт не нужен). На Spark не измерена; скорее
  всего медленнее DFlash2 на одном потоке.

**Проверка:** в логе `FP8 side (DFlash draft): N linear layers ...`, затем
`curl`-вопрос и `python3 bench/perf.py`.

### 4.2. Zen 6 — та же 27B с контекстом 1M

`zenlm/zen6` — это веса RadixArk NVFP4 и драфт DFlash2 байт в байт, плюс
YaRN ×4 в конфиге (контекст 1 048 576 токенов).

```bash
$HF download zenlm/zen6 --local-dir /models/zenlm/zen6     # драфт внутри, в dflash2/
./serve.sh zen6
```

- Контекст берётся из YaRN в `config.json` сам; без YaRN профиль откажется
  ставить больше 262K.
- Скорость как у `qwen3.8-27b` на NVFP4. Качество на обычных задачах не
  хуже исходной модели (проверено), хотя YaRN статический и действует на
  все запросы.
- В логе посмотрите `KV Cache is allocated ... #tokens`: для одного запроса
  на 1M нужно не меньше 1 048 576. Если меньше —
  `MEM_FRACTION=0.85 MAX_RUNNING=6 ./serve.sh zen6`.
- Длинные промпты (100K+) нагревают GPU до 79–80 °C — это порог
  приостановки; следите за охлаждением.

### 4.3. Meerkat-TRIZ — 27B, дообученная на ТРИЗ

LoRA-адаптер, слитый с исходной BF16-моделью; при загрузке переводится в
FP8. Владеет методологией ТРИЗ заметно лучше базы, HumanEval не хуже.

```bash
$HF download Qwen/Qwen3.8-27B --local-dir /models/Qwen3.8-27B                       # ~54 ГБ, BF16
$HF download Meerkat-AI/Meerkat-TRIZ-v1-Qwen3.8-27B --local-dir /models/Meerkat-TRIZ-v1-Qwen3.8-27B
# сервер остановлен; Python из venv SGLang:
~/spark/venv-sglang-0.5.20/bin/python3 scripts/merge-lora.py /models/Qwen3.8-27B \
    /models/Meerkat-TRIZ-v1-Qwen3.8-27B /models/Meerkat-TRIZ-v1-Qwen3.8-27B-merged
./serve.sh meerkat-triz
```

- `merge-lora.py` должен напечатать `name mapping: 496 model.* ->
  model.language_model.*` и закончить на `496/496`. Он ничего не пишет, если
  хоть одна пара не нашла свой вес.
- Первый старт дольше: SGLang читает 54 ГБ BF16 и квантует в FP8.
- Сравнить с исходной моделью на одних и тех же вопросах — раздел 6.

### 4.4. Qwen3.8-Flash-Next — 176B MoE на одном Spark

MoE на 125B + таблица n-грамм 51B, 6B активных параметров, NVFP4, свой MTP-
драфт. Помещается в 121.6 ГиБ только потому, что таблица на 47.7 ГиБ
читается прямо с NVMe (патч `gb10_ple_mmap`), а не копируется при каждом
старте.

```bash
$HF download RadixArk/Qwen3.8-Flash-Next-NVFP4 --local-dir /models/RadixArk/Qwen3.8-Flash-Next-NVFP4
./serve.sh qwen3.8-flash-next
```

Загрузка весов ~7 минут. **65.8 ток/с** на одном потоке, 212–224 суммарно
на 8; HumanEval 97.6% без размышления, 99.4% с medium. Всё это уже
включено по умолчанию: FP8 для BF16-слоёв (`FP8_SIDE`), обеих голов
(`FP8_HEAD`, `FP8_DRAFT_HEAD`) и hyper-connections (`FP8_HC`), словарь
драфта на 65K токенов (`DRAFT_VOCAB`).

**При первом старте** в логе должно быть `PLE table: ... read in place ...,
nothing written` и `... shard tensors left in place`, затем строки `FP8 side`,
`FP8 HC`. Если ответы — мусор, `PLE_TABLE=file` даёт стоковое поведение для
сравнения.

**Abliterated-вариант** (без отказов, веса Huihui):

```bash
$HF download edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4 \
  --local-dir /models/edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4
./serve.sh qwen3.8-flash-next-abliterated
```

**Абляция векторами на лету** (без отдельных весов, с регулируемой силой):

```bash
$HF download Mambavtt/qwen3.8-flash-next-refusal-ablation-vectors \
  --local-dir /models/qwen3.8-flash-next-refusal-ablation-vectors
ABLATE=/models/qwen3.8-flash-next-refusal-ablation-vectors/refusal_directions-v4-late.npz \
  ./serve.sh qwen3.8-flash-next
```

`ABLATE_ALPHA` (1.0 — по умолчанию карточки, 0 — стоковая модель, 1.5 —
сильнее), `ABLATE_AT=output|input`, `ABLATE_STREAMS=each|mean`. В логе:
`GB10_ABLATE: 43 layers, alpha 1, ...`. Только для весов RadixArk; с
abliterated-весами смысла не имеет. Содержимое файла векторов можно
проверить до шестиминутной загрузки:

```bash
~/spark/venv-sglang-0.5.20/bin/python3 -c "import numpy as np; z=np.load('/models/qwen3.8-flash-next-refusal-ablation-vectors/refusal_directions-v4-late.npz'); print({k: z[k].shape for k in z.files})"
```

### 4.5. Ornith 1.5 35B-A3B — быстрая MoE с размышлением

Дообученная Qwen3.5-35B-A3B (256 экспертов, ~3B активных), FP8-чекпойнт
Ornith AI с их драфтом DFlash.

```bash
$HF download ornith-ai/Ornith-1.5-35B-A3B-FP8 --local-dir /models/Ornith-1.5-35B-A3B-FP8
$HF download ornith-ai/Ornith-1.5-35B-A3B-DFlash --local-dir /models/Ornith-1.5-35B-A3B-DFlash
./serve.sh ornith-1.5-35b
```

**97.6 ток/с** на одном потоке, ~345–360 суммарно на 16; HumanEval 98.2% с
medium. `SPEC=off` — без драфта (39.8), `SPEC=mtp` — встроенная MTP-голова.
`WEIGHTS=bf16` берёт исходные BF16-веса (`ornith-ai/Ornith-1.5-35B-A3B`,
67 ГБ) и квантует при загрузке. `WEIGHTS=w4a16` (NVFP4 r0b0tlab) на SGLang
0.5.20 выдаёт мусор — не используйте. Настройка ядер MoE под эту машину
(`MOE_TUNED=1`) описана в README; выигрыша не дала.

### 4.6. Gemma 4 31B — не измерена

```bash
$HF download google/gemma-4-31B-it --local-dir /models/gemma-4-31B-it
$HF download google/gemma-4-31B-it-assistant --local-dir /models/gemma-4-31B-it-assistant
./serve.sh gemma4-31b
```

BF16, ~62 ГБ: ожидайте низкую скорость. Репозиторий может быть закрыт
лицензией (ошибка 401/403 → принять лицензию на странице модели и
`hf auth login`). Размышление по умолчанию выключено.

### 4.7. GLM-5.3-Flash 320B на llama.cpp

Отдельный стек в `llamacpp/` (подробно — [`llamacpp/README.md`](llamacpp/README.md)).
Сервер SGLang перед этим остановите. Порт — 8889.

```bash
cd llamacpp
./build.sh                       # llama.cpp на нужном коммите + патч, ~10–20 мин
$HF download pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf \
  GLM-5.3-Flash-GSQ-RCO-3.5bit.gguf GLM-5.3-Flash-mmproj-BF16.gguf \
  --local-dir /models/GLM-5.3-Flash-GSQ-RCO-GGUF
```

**Один Spark — только проверка, что работает.** Влезает лишь 3.0-бит
(109.4 ГиБ), без картинок, с коротким контекстом:

```bash
./serve-single.sh                # откажется, если после весов останется < 6 ГиБ
```

**Два Spark — рабочий вариант** (3.5-бит + картинки, 32K на 4 слота).
Соедините машины одним QSFP-кабелем, задайте адреса на этом линке (пример
netplan — в `llamacpp/README.md`), соберите llama.cpp на обеих:

```bash
BIND=10.10.10.2 ./rpc-worker.sh        # на второй машине, первым
WORKER=10.10.10.2 ./serve-dual.sh      # на первой (там же лежат GGUF)
```

Второй кабель или агрегация линков ничего не дают: по сети на токен идут
килобайты, важна задержка. Два Spark дают память, а не скорость:
`SPLIT_MODE=tensor` — эксперимент, который может ускорить декод.

---

## 5. Как выжать максимум из машины

**Память.**

- `MEM_FRACTION` (0.80 для большинства, 0.85 для Flash-Next) — не рычаг
  скорости, а устойчивости. При 0.85 и выше earlyoom может убить сервер
  (код `-15`, без трейсбека; `journalctl -u earlyoom`). Не отключайте
  earlyoom — лучше снизьте долю.
- Чтобы освободить ~1–2 ГБ: выключите графический рабочий стол
  (`sudo systemctl isolate multi-user.target`, обратно —
  `graphical.target`).
- Одна модель за раз: перед другой моделью остановите службу.

**Параллельность.** `MAX_RUNNING` выставляйте по реальной нагрузке, а не «с
запасом»: каждый слот заранее занимает память под состояние GDN (у 27B это
5–6 слотов на запрос, ~0.07 ГиБ на слот).

**Длина драфта.** Чем больше `DRAFT_TOKENS`, тем быстрее один поток на
предсказуемом тексте (код) и тем хуже много параллельных потоков. Меряйте
под свою нагрузку: `perf.py` показывает и то и другое.

**Первая загрузка ядер.** `JIT_JOBS=2` в `serve.sh` ограничивает
параллельную компиляцию: при большем значении компиляторы убиваются по
памяти (`Ninja build failed`, код 137).

**Ядра процессора.** `CPUSET=5-9,15-19` закрепляет сервер на быстрых
ядрах Cortex-X5; пустое значение отключает привязку.

**Точность GDN** (необязательно). Патч из апстрима SGLang держит коэффициент
beta гибридного внимания во float32 вместо BF16 — на префилле всех гибридных
моделей здесь. Скорость не меняет, ответы — может, слегка:

```bash
~/spark/venv-sglang-0.5.20/bin/python scripts/patch-sglang-gdn-precision.py           # применить
~/spark/venv-sglang-0.5.20/bin/python scripts/patch-sglang-gdn-precision.py --revert  # откатить
```

Правит файлы в venv, только если они совпадают со стоковыми 0.5.20 по
sha256. После переустановки SGLang — применить заново.

**Температура.** На длинных prefill (32K+) GPU доходит до 79–84 °C. Порог
приостановки — 80 °C: свободное место вокруг корпуса важнее любой
настройки.

**Сравнение вариантов.** Разброс между перезапусками ~8% на одном потоке;
выигрыш меньше этого перепроверяйте на новом старте. Температура 0 тоже не
даёт побитово одинаковых ответов: ±2 задачи HumanEval — это шум.

---

## 6. Измерения и проверка качества

Все скрипты берут адрес из `GB10_BASE_URL` (по умолчанию
`http://127.0.0.1:8888/v1`), модель — из `/v1/models`.

```bash
python3 bench/perf.py                    # TTFT, один поток, 1–18 потоков, prefill до 101K
python3 bench/perf.py --only decode      # только один поток
python3 bench/longctx.py                 # несколько длинных контекстов разом
python3 bench/profile_decode.py          # куда уходит шаг декода (по группам ядер)
../scripts/run-humaneval.sh              # HumanEval без размышления
../scripts/run-humaneval.sh think-medium # с размышлением medium
```

HumanEval (`scripts/run-humaneval.sh` в корне репозитория) запускает
сгенерированный моделью код в изолированном контейнере, поэтому ему нужен
Docker; задачи он скачивает сам при первом запуске.

Для llama.cpp: `GB10_BASE_URL=http://127.0.0.1:8889/v1 python3 bench/perf.py`.

**Сравнить две модели на своих вопросах** (например, Meerkat-TRIZ и
исходную Qwen) — `bench/answers.py`: собирает ответы с каждого сервера и
кладёт их рядом в один Markdown-файл.

```bash
python3 bench/answers.py collect questions.jsonl --out runs/a.jsonl     # сервер с моделью A
python3 bench/answers.py collect questions.jsonl --out runs/b.jsonl     # перезапуск с моделью B
python3 bench/answers.py compare runs/a.jsonl runs/b.jsonl > compare.md
```

Понимает JSONL с `messages`, `conversations`, Alpaca
(`instruction`/`input`/`output`), `question`/`answer` и просто строки.
`--thinking --effort medium` — с размышлением.

---

## 7. Если что-то пошло не так

| Симптом | Причина | Что делать |
|---|---|---|
| `only N GiB of memory available` при старте | память держит другой процесс | остановить его (часто `sudo systemctl stop gb10-sglang`) |
| `CUDA error: out of memory` ещё до загрузки весов | то же | то же; если `nvidia-smi` не отвечает и после этого — перезагрузка |
| `something already serves :8888` | уже запущена служба или другая модель | `sudo systemctl stop gb10-sglang` |
| сервер молча умер, код `-15` | earlyoom | `MEM_FRACTION` ниже, `journalctl -u earlyoom` |
| `Ninja build failed`, `Killed` при первом старте | параллельная компиляция съела память | `JIT_JOBS=2` (по умолчанию); просто перезапустить — скомпилированное сохранится |
| `fatal error: Python.h` | нет заголовков Python | `sudo apt install python3.12-dev` |
| `DFLASH block size mismatch` | ожидаемо при драфте ≠ 8 | ничего, это предупреждение |
| ответ пустой, всё в `reasoning_content` | размышление включено, мал `max_tokens` | увеличить `max_tokens` или выключить размышление |
| мусор в ответах (Flash-Next) | неверно читается таблица n-грамм | сравнить с `PLE_TABLE=file` |
| `GB10_...: ... not found (written for SGLang 0.5.20)` | патч не узнал код SGLang (другая версия) | вернуть `SGLANG_VERSION=0.5.20` или выключить названный патч |
| первый старт идёт очень долго | компиляция и подбор ядер | ждать; повторные старты быстрее |

---

## 8. Что где лежит

```
serve.sh              запуск: настройки машины, профиль; ./serve.sh install / manifest
models/<профиль>.sh   одна модель: веса, версия SGLang, флаги и ручки с комментариями
scripts/              установка, проверка машины, служба, merge-lora.py, тюнер MoE
patches/              патчи SGLang, включаются профилями через переменные GB10_*:
  gb10_ple_mmap         таблица n-грамм Flash-Next читается на месте
  gb10_fp8_side         BF16-слои, головы и драфт DFlash → FP8 (Marlin)
  gb10_fp8_hc           hyper-connections Flash-Next → FP8
  gb10_marlin_lean      перепаковка MoE в Marlin без лишней копии
  gb10_ablate           абляция векторами (ABLATE=...)
  test_gb10_*.py        проверки на CPU: python3 patches/test_gb10_<имя>.py
bench/                perf.py, longctx.py, profile_decode.py, answers.py
moe-configs/          настроенные под GB10 конфиги ядер MoE (Ornith)
llamacpp/             сборка и запуск GLM-5.3-Flash на llama.cpp (1 или 2 Spark)
results/RESULTS.md    все замеры с датами и условиями
README.md             подробности и обоснования (англ.)
```

Новую модель проще всего добавить, скопировав ближайший профиль и поменяв
пути к весам, `SERVED_MODEL_NAME` и `model_args`.
