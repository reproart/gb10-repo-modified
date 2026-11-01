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

Движки:

- **SGLang 0.5.21** (нативно, без Docker; 27B и Flash-Next перешли с 0.5.20, `SGLANG_VERSION=0.5.20` возвращает) — почти все модели. Одна
  команда `./serve.sh <профиль>`; профиль — это файл `models/<профиль>.sh`
  с путями к весам и флагами модели.
- **llama.cpp** (отдельная сборка в `llamacpp/`) — только GLM-5.3-Flash в
  GGUF, которую SGLang не обслуживает.
- **vLLM в Docker** (`vllm-b12x/`) — Flash-Next FP4-Local от NVIDIA.
- **vLLM в Docker** (`vllm-flash-ar/`) — Flash-Next в int4 AutoRound от Intel
  (рецепт blazux → Saren-Arterius, ваша копия).
- **TensorFold в Docker** (`tensorfold-dsv41/`) — DeepSeek-V4.1-Flash на двух
  Spark.
- **Настроенный vLLM в Docker** (`vllm-dsv41/`) — та же DeepSeek на двух Spark,
  другой движок (для сравнения и для картинок).
- **TensorFold в Docker** (`tensorfold-glm53/`) — GLM-5.3-Flash на двух Spark
  (рецепт MiaAI-Lab).

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
| `mimo-v2.6-flash` | MiMo-V2.6-Flash 309B MoE (MXFP4) + DFlash, **на двух Spark**, SGLang 0.5.21 | ~165 ГиБ на каждом | | не измерена |
| `ling-3.0-flash` | Ling-3.0-flash 124B MoE (5.1B акт.), текст, MTP, FP8-голова, SGLang 0.5.21 | ~66 ГиБ | 54–103 у NVIDIA (на их ветке SGLang) | у нас не измерена |
| `ling-3.0-flash-vl` | Ling-3.0-flash-VL 125B MoE (5.1B акт.), картинки и видео, SGLang 0.5.21 | ~70 ГБ (оценка) | ~33 по рецепту SGLang (FP4) | рецепт проверен на Spark, у нас не измерена |
| `vllm-b12x/` | Qwen3.8-Flash-Next-FP4-Local (QAD от NVIDIA), vLLM в Docker | ~96 ГиБ | 61.0 (MTP 3) | измерена, качество не сравнивали |
| `vllm-flash-ar/` | Qwen3.8-Flash-Next, int4 AutoRound (Intel) + int8 голова + FP8, vLLM в Docker | ~67 + ~49 ГиБ (таблица) | 65.8 (MTP 3), 259 суммарно на 8 | измерена в рецепте, долго в работе |
| `tensorfold-dsv41/` | DeepSeek-V4.1-Flash-1017, EXL3 2.9 бит, контекст 1M, TensorFold в Docker, **на двух Spark** (TP=2) | 196 + 189 ГиБ на каждом | 92 код / 57 текст (по рецепту) | у нас не измерена |
| `vllm-dsv41/` | та же DeepSeek, те же веса, настроенный vLLM (sudoingX + styles01) в Docker, **на двух Spark**, 262K, картинки | те же папки + дерево из жёстких ссылок | 94 код / 52 текст (по рецепту) | у нас не измерена |
| `tensorfold-glm53/` | GLM-5.3-Flash 321B, EXL3 4 бит + DFlash2, контекст 1M, TensorFold в Docker, **на двух Spark** | ~205 ГБ на каждом | 114 коротко / 50 текст / 70 код (по замерам LimeChain) | у нас не измерена |
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
./serve.sh install             # venv ~/spark/venv-sglang-0.5.21 со SGLang 0.5.21 и hf CLI (~10 ГБ)
./scripts/00-check-host.sh     # ещё раз: теперь проверит, что torch видит GPU
./scripts/host-tuning.sh       # по желанию: память хоста, earlyoom, без swap (раздел 5); --undo откатывает
./scripts/mem-prep.sh sudoers  # по желанию: сброс кэша и уплотнение памяти перед стартом без пароля (раздел 5)
```

`./serve.sh install` ставит пакеты строго по lock-файлу
`requirements/sglang-<версия>-aarch64-py312.txt` (для версии без него, как
0.5.21, пока его lock не закоммичен, зависимости разрешаются при установке и
lock записывается туда же): ничего не компилируется при
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
# по умолчанию: быстрее (~40% на одном потоке, в 2.5 раза быстрее prefill), 93.9–95.1% без размышления:
$HF download RadixArk/Qwen3.8-27B-NVFP4 --revision 554ebba9b5f1b79dc11246341960360e6ef05ef4 \
  --local-dir /models/RadixArk/Qwen3.8-27B-NVFP4
# точнее (HumanEval 97.6% без размышления), медленнее:
$HF download Qwen/Qwen3.8-27B-FP8 --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --local-dir /models/Qwen3.8-27B-FP8
# драфт (обязателен):
$HF download z-lab/Qwen3.8-27B-DFlash2 --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --local-dir /models/Qwen3.8-27B-DFlash2
```

С размышлением (`reasoning_effort` medium) обе цели ошибаются в HumanEval не
больше одного раза — тогда выгоднее быстрая NVFP4.

**Запуск:**

```bash
./serve.sh qwen3.8-27b                                  # цель NVFP4 (RadixArk) по умолчанию
MODEL_DIR=/models/Qwen3.8-27B-FP8 ./serve.sh qwen3.8-27b # цель FP8 Qwen
```

Чтобы целью всегда была FP8 — поменяйте `MODEL_DIR` в
`models/qwen3.8-27b.sh`; варианты `-single`, `-throughput`, `-longctx`,
`-dspark`, а также `zen6` берут остальное оттуда же.

**Варианты под нагрузку:**

| Профиль | Запросов одновременно | Драфт | Когда |
|---|---:|---:|---|
| `qwen3.8-27b` | до 16 | 15 | по умолчанию: один пользователь с кодом, агенты, до ~16 параллельных; ~995K токенов KV — три сессии по 262K |
| `qwen3.8-27b-single` | до 16 | 16 | 1–2 пользователя |
| `qwen3.8-27b-throughput` | до 32 | 10 | много коротких параллельных запросов |
| `qwen3.8-27b-longctx` | 6 | 10 | несколько сессий по 262K токенов |
| `qwen3.8-27b-dspark` | до 16 | DSpark γ 7 | сравнение с другим драфтом (`$HF download RadixArk/Qwen3.8-27B-DSpark --local-dir /models/Qwen3.8-27B-DSpark`) |

**Главные ручки:**

- `DRAFT_TOKENS` — самый сильный рычаг одного потока. 15 (по умолчанию)
  даёт 94.2 ток/с на NVFP4 и выигрывает на 1–8 потоках; если обычно идёт
  10–12 запросов разом, `DRAFT_TOKENS=11` даёт +14% суммарно на 12.
- `MAX_RUNNING` — сколько запросов реально идёт одновременно (16 по
  умолчанию). Резервирует память под состояние GDN заранее: больше, чем
  нужно, — меньше KV-кэша.
- `CHUNKED_PREFILL=2048`, `PREFILL_DECODE_INTERVAL=1` (по умолчанию) — пока
  один запрос префиллит длинный промпт, остальные потоки замирают на ~1–2 с,
  а не на весь префилл (14 с на 32K при 8192 без интервала). Больше
  интервал — больше скорости остальным, дольше TTFT длинного промпта.
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
- Длинные промпты (100K+) нагревают GPU до 79–87 °C — это в пределах
  нормы; важен запас до собственного предела GPU (T.Limit), perf.py его
  показывает.

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

- **FP8-чекпойнт вместо перевода при каждой загрузке** (рекомендуется):
  тот же формат, что у официального `Qwen/Qwen3.8-27B-FP8` (блоки 128×128),
  ~29 ГБ вместо 54, загрузка быстрее, ядра — те же, что у FP8-цели 27B.
  Сначала `check`: квантует исходную BF16 Qwen и сравнивает с официальным
  FP8 побайтно (≈100% одинаковых байт — значит, метод тот же):

  ```bash
  PY=~/spark/venv-sglang-0.5.20/bin/python3     # сервер остановлен
  $PY scripts/fp8-like-reference.py check /models/Qwen3.8-27B /models/Qwen3.8-27B-FP8
  $PY scripts/fp8-like-reference.py convert /models/Meerkat-TRIZ-v1-Qwen3.8-27B-merged \
      /models/Qwen3.8-27B-FP8 /models/Meerkat-TRIZ-v1-Qwen3.8-27B-FP8
  ./serve.sh meerkat-triz      # сам возьмёт FP8-каталог, если он есть (WEIGHTS=bf16 — по-старому)
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
~/spark/venv-sglang-0.5.21/bin/python3 -c "import numpy as np; z=np.load('/models/qwen3.8-flash-next-refusal-ablation-vectors/refusal_directions-v4-late.npz'); print({k: z[k].shape for k in z.files})"
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

### 4.8. MiMo-V2.6-Flash 309B — на двух Spark (SGLang)

Xiaomi MiMo-V2.6-Flash: MoE 309B / 15B активных, эксперты в MXFP4,
остальное в блочном FP8, роутеры в BF16, драфт DFlash в `dflash/`. Чекпойнт
~165 ГиБ, на один Spark (~121 ГиБ) не помещается: профиль делит модель на
две машины, TP=2 (EP=2), по ~83 ГиБ на каждую. Нужен **SGLang 0.5.21**: в
нём есть загрузка MXFP4-экспертов MiMo (в 0.5.20 нет). На GB10 пока не
проверялась.

```bash
# на ОБЕИХ машинах (веса нужны целиком на каждой):
$HF download XiaomiMiMo/MiMo-V2.6-Flash --local-dir /models/XiaomiMiMo/MiMo-V2.6-Flash
sudo apt install ffmpeg                     # библиотеки FFmpeg для torchcodec
./serve.sh mimo-v2.6-flash install          # venv SGLang 0.5.21 + torchcodec

# Spark A (на нём будет API, :8888), 10.10.10.1 — его адрес на прямом линке:
NODE_RANK=0 HEAD_ADDR=10.10.10.1 ./serve.sh mimo-v2.6-flash
# Spark B (рабочий узел, API нет):
NODE_RANK=1 HEAD_ADDR=10.10.10.1 ./serve.sh mimo-v2.6-flash
```

Порядок запуска любой, узлы ждут друг друга. Линк тот же, что для
GLM на llama.cpp (QSFP-кабель, адреса из `llamacpp/README.md`). NCCL и Gloo
профиль привязывает к интерфейсу этого линка сам (`LINK_IF`, чтобы задать
вручную). Проверить, что NCCL пошёл по RDMA, а не по TCP: `NCCL_DEBUG=INFO`,
в логе должно быть `NET/IB`. Здесь, в отличие от llama.cpp, по линку на
каждом слое идёт all-reduce, так что задержка и RoCE важнее.

Отличия от рецепта SGLang (H200/B300): эксперты на Marlin (W4A16), внимание
на Triton (FA4 и DeepGEMM — только SM100), без DeepEP. Основные настройки:

- `DRAFT_TOKENS=8` — как в рецепте; на 27B лучшим оказалось 15, попробуйте
  12 и 16. `DFLASH=0` — без драфта.
- `MOE_RUNNER_BACKEND=flashinfer_mxfp4` — второй кандидат (MXFP8×MXFP4).
- `EP=1` — эксперты делятся по столбцам, а не целиком.
- `WEIGHTS=rl` — MiMo-V2.6-Flash-RL из `/models/XiaomiMiMo/MiMo-V2.6-Flash-RL`.
- `MEM_FRACTION` 0.80 — под KV остаётся ~14 ГиБ на машину; смотрите в логе
  `KV Cache is allocated ... #tokens`. `CONTEXT_LENGTH` по умолчанию 262144
  (модель умеет 1M, если хватит пула). `KV_CACHE_DTYPE=fp8_e4m3` вдвое
  уменьшает KV, но с этими ядрами не проверено.

Профиль заранее проверяет, что половина весов помещается в `MEM_FRACTION`
и что `HEAD_ADDR` — адрес этой машины (на A) или до него есть маршрут (на B).
Для службы systemd задайте `NODE_RANK` и `HEAD_ADDR` через `export` в
`serve.sh` на каждой машине.

### 4.9. Ling-3.0-flash 124B (текст, MTP) и Ling-3.0-flash-VL (картинки и видео)

**Текстовая Ling-3.0-flash** — та же основа (MoE 124B / 5.1B активных,
KDA + MLA), но со встроенным слоем MTP: черновая модель бесплатно, без
отдельной загрузки. Родной контекст 256K. NVIDIA на одном Spark намерила
36 ток/с по рецепту cookbook и 103 ток/с с тремя приёмами (ядра Humming,
FP8-голова, MTP) на синтетике; на GSM8K — 38 → 54 ток/с без потери точности.
Это было на их собственных ветках SGLang; профиль работает на релизе 0.5.21:

```bash
$HF download inclusionAI/Ling-3.0-flash-fp4 --local-dir /models/inclusionAI/Ling-3.0-flash-fp4
./serve.sh ling-3.0-flash install      # тот же venv, что у VL и Qwen
./serve.sh ling-3.0-flash
```

- **MTP** (3 шага, 4 токена) — по умолчанию; `MTP=0` выключает.
- **FP8-голова** (`FP8_HEAD=1`, по умолчанию) — наш патч
  `patches/gb10_fp8_side.py`, как у Flash-Next; черновая модель получает
  ту же FP8-голову, так что ловушка из статьи NVIDIA (приём падает до 1.00)
  здесь не возникает. С потерями в принципе: проверяйте ответы.
- **Humming и Marlin профиль не запускает.** У Ling в слоях 35–41
  активации экспертов обрезаются после SiLU (предел 4), а раннеры Humming
  (0.5.21 и 0.5.22) и Marlin этот предел теряют. Работают flashinfer_mxfp4
  (FP4, по умолчанию) и triton (`WEIGHTS=int4`). Поэтому и у VL-профиля
  `WEIGHTS=int4` теперь идёт через triton.
- **Сэмплинг.** В репозитории нет `generation_config.json`, и запрос без
  параметров идёт с temperature 1.0. Рекомендуемые 0.6 / 0.95 / 20 станут
  умолчанием сервера после одной команды (она в шапке профиля).
- **Первый старт** компилирует ядро MoE FlashInfer (один раз, потом из
  кэша). `JIT_JOBS=2` держит компиляцию в рамках памяти; у inclusionAI
  компиляция рядом с загрузкой весов вешала драйвер.

**Ling-3.0-flash-VL** — ниже.

inclusionAI Ling-3.0-flash-VL: MoE ~125B / 5.1B активных (гибрид линейного
внимания KDA и MLA), с энкодером для картинок и видео, размышлением и
вызовом инструментов. Драфта нет. В рецепте SGLang для неё есть **проверенные
ячейки DGX Spark**: FP4 — ~33 ток/с на одном потоке (TPOT 30 мс, вход 8K),
~1070 ток/с суммарно на 16 потоках; INT4 — ~23 ток/с. Нужен SGLang 0.5.21.

```bash
$HF download inclusionAI/Ling-3.0-flash-VL-fp4 --local-dir /models/inclusionAI/Ling-3.0-flash-VL-fp4
./serve.sh ling-3.0-flash-vl install
./serve.sh ling-3.0-flash-vl
```

- `WEIGHTS=int4` — GPTQ-версия (`/models/inclusionAI/Ling-3.0-flash-VL-int4`):
  медленнее, на бенчмарках чуть точнее (MMMU-Pro 77.5 против 76.2; мерили на
  B300, где MoE идёт через triton — так профиль и запускает её на Spark).
- `YARN=0` — родной контекст 128K без YaRN (по умолчанию 256K через YaRN ×2,
  как в рецепте; YaRN статический и действует и на короткие запросы).
- `MAX_RUNNING=16` — потолок одновременных запросов. Картинка 720p — ~880
  токенов, видеоклип — несколько тысяч.
- Размышление выключается в запросе:
  `"chat_template_kwargs": {"enable_thinking": false}`.

Картинки — `image_url`, видео — `video_url` в сообщении (формат OpenAI).
Аудио модель не принимает.

### 4.10. Qwen3.8-Flash-Next-FP4-Local на vLLM (Docker)

Версия Flash-Next от NVIDIA, квантованная с дообучением (QAD): по карточке
почти без потерь (GPQA 91.5 против 92.2 у BF16). SGLang её не загружает
(новый сжатый формат масштабов NVFP4 и таблица n-грамм в NVFP4), поэтому
запуск — в контейнере vLLM с ядрами b12x, как в карточке
(подробно — [`vllm-b12x/README.md`](vllm-b12x/README.md)). Порт тот же,
8888, так что `bench/` работает без изменений. Сервер SGLang перед этим
остановите.

```bash
cd vllm-b12x
$HF download nvidia/Qwen3.8-Flash-Next-FP4-Local --local-dir /models/nvidia/Qwen3.8-Flash-Next-FP4-Local
./pull.sh                  # контейнер, закрепляется по дайджесту в image.lock
./serve.sh                 # Ctrl-C останавливает
MAX_NUM_SEQS=4 MTP_TOKENS=4 ./serve.sh     # параметры — как у serve.sh (MTP 3 по умолчанию)
YARN=1 ./serve.sh          # контекст больше 262144: YaRN ×4, до 1 048 576
PLE_OFFLOAD=none ./serve.sh   # таблица n-грамм в памяти GPU (по умолчанию disk; ещё cpu)
./show-env.sh PLE          # какие переменные PLE понимает vLLM в контейнере
```

Таблица n-грамм (~29 ГБ в NVFP4): `PLE_OFFLOAD=disk` (по умолчанию, как в
карточке для Spark) читает её с диска, `cpu` держит в обычной памяти, `none`
загружает в память GPU вместе с весами. `none` на Spark на пределе: с MTP 4
и контекстом 262144 ядро убило vLLM по OOM, поэтому в этом режиме MTP по
умолчанию выключен, а контекст — 65536. Только для замера, не для работы.

YaRN строится из настроек RoPE самого чекпойнта и действует на все запросы,
включая короткие (возможна небольшая потеря качества). Включайте его, только
когда сессии нужно больше 262 144 токенов. Предел можно задать ниже:
`YARN=1 MAX_MODEL_LEN=524288`.

`image.lock` стоит закоммитить: тогда `serve.sh` всегда запускает ту же
сборку, даже если `:latest` на Docker Hub обновится. Перейти на новую
сборку — `rm image.lock && ./pull.sh`.


### 4.11. DeepSeek-V4.1-Flash-1017 на двух Spark (TensorFold)

Большая модель DeepSeek в квантовании EXL3 2.9 бит: 196 ГиБ весов плюс
189 ГиБ Engram-таблиц, которые читаются с NVMe. Работает сразу на **обеих**
машинах (TP=2 по кабелю ConnectX-7), контекст 1M. Движок — TensorFold
(Python-ветка) с портом V4.1 от bertholomus, из форка Elevatormusic: там
добавлены принудительные вызовы инструментов, JSON schema и regex вместе со
спекулятивным декодом, изоляция ошибок. Это рецепт Elevatormusic
(Apache-2.0), переложенный под наш проект: настройки в `config.sh` вида
`${VAR:-default}`, запуск — `./serve.sh`. Подробно —
[`tensorfold-dsv41/README.md`](tensorfold-dsv41/README.md).

По рецепту (у нас пока не мерили): декод 92 ток/с на коде и 57 на тексте,
до 4 запросов разом (140 ток/с суммарно), префилл ~2 000–2 200 ток/с до 128K,
промпт на 1M — 13.6 минуты до первого токена. Качество ответов рецепт не
проверяет. Обе машины заняты целиком: никакой другой модели рядом ни на
одной из них.

```bash
cd tensorfold-dsv41
cp config.local.sh.example config.local.sh   # адреса CX7, RDMA-устройства, ssh на вторую машину
./download.sh              # ~400 ГБ, докачивается; sha256; копия на вторую машину (или вручную, ниже)
./patch-template.py --worker   # шаблон чата по закреплённому коммиту + правка
./build.sh                 # образ движка, копия на вторую машину (~10 мин)
./host-tuning.sh           # один раз, sudo на обеих: без swap, sysctl памяти, earlyoom; --undo откатывает
../scripts/mem-prep.sh sudoers 10.100.80.2    # один раз (адрес второй машины): память перед стартом без пароля, раздел 5
./serve.sh                 # проверки, ранг 0 здесь, ранг 1 там; первый старт ~15 мин, дальше 2–4
./verify.sh                # пять настоящих запросов
./stop.sh                  # остановить оба ранга, логи — в logs/
MAX_RUNNING=2 ./serve.sh   # параметры — как у наших serve.sh (config.sh)
```

**Веса вручную** — две обычные папки `hf download`, как у остальных моделей:
их удобно качать по отдельности и переносить на вторую машину копированием.

```bash
HF=~/spark/venv/bin/hf
# веса EXL3 2.9 бит (196 ГиБ, 43 шарда)
$HF download LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw --revision afc7aa0aab98ee735c0f54e0a122019af1ac08f1 \
  --exclude 'assets/*' --local-dir /models/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw
# Engram-таблицы (189 ГиБ: два шарда базового репозитория, индекс, config.json)
$HF download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  model-00047-of-00048.safetensors model-00048-of-00048.safetensors model.safetensors.index.json config.json \
  --local-dir /models/deepseek-ai/DeepSeek-V4.1-Flash-1017-engram
cd tensorfold-dsv41 && ./download.sh verify   # sha256 всех 53 файлов по списку автора (~385 ГиБ чтения)
```

На вторую машину — те же пути (`cp -a`, `rsync`, диск), там тоже
`./download.sh verify`. Шарды Engram весят ~101 ГБ: файлы больше 50 ГБ хаб
отдаёт только через `hf_xet`, которому на Spark может не хватить памяти —
качайте, пока модели не запущены. Если `hf download` откажется,
`./download.sh engram` скачает те же файлы в ту же папку через `curl` с
докачкой. Шаблон чата `./patch-template.py --worker` кладёт в папку весов;
`preflight.sh` проверяет его sha256.

Оба ранга по умолчанию закреплены на быстрых ядрах Cortex-X925
(`CPUSET=5-9,15-19`, как у нашего `serve.sh`; `preflight.sh` сверяет их с
частотами на обеих машинах). Автор рецепта подозревал, что его медленные
прогоны (на 10–20%) — это главный поток на медленном ядре; `CPUSET=` для
сравнения запускает как у автора.

`host-tuning.sh` меняет настройки памяти системы целиком: они действуют и для
SGLang-профилей, пока не сделать `--undo`. У earlyoom пороги становятся 1 и
0.5 ГиБ (SIGTERM и SIGKILL). Никогда не делайте `docker exec` в ранг, пока он
работает. Не останавливайте дисплейный менеджер на Spark: на машинах автора
это выключило узел.

---

### 4.12. DeepSeek-V4.1-Flash-1017 на двух Spark (настроенный vLLM)

Та же модель и те же две папки весов, что в 4.11, но другой движок: vLLM от
sudoingX (22 патча и плагин для EXL3) с доработкой styles01 под выпуск 1017,
из их готового образа на GHCR (дайджест закрепляется в `image.lock`).
Подробно — [`vllm-dsv41/README.md`](vllm-dsv41/README.md). Адреса пары
берутся из `tensorfold-dsv41/config.local.sh`, порт 8888 и имя модели те же,
что у TensorFold, так что клиенты и `bench/` не меняются. Одновременно
работает только один из двух.

По их замерам (у нас пока не мерили), тем же клиентом `kit_bench`, что у
TensorFold: код 88–94 ток/с, текст 49–52, структурированный вывод 106–127.
Против TensorFold: код наравне, текст и JSON медленнее на 10–15%, контекст
262K вместо 1M, 2 запроса вместо 4. Зато картинки у них проверены.

```bash
cd vllm-dsv41
./pull.sh          # образ на обе машины, дайджест — в image.lock (закоммитьте)
./repack.sh        # дерево для движка на обеих машинах из папок весов (секунды, жёсткие ссылки)
./serve.sh         # проверки, ранг 1 на второй машине, ранг 0 здесь; ~13 мин
./status.sh / ./stop.sh
MAX_NUM_SEQS=1 GPU_UTIL=0.88 ./serve.sh   # параметры — как у наших serve.sh (config.sh)
```

**Перепаковка.** Движку нужен свой вид чекпойнта: другой `quant_method`, план
с черновой моделью, Engram-таблицы в общем индексе. `repack.py` — их скрипт с
двумя исправлениями: оригинал через жёсткие ссылки переписывал `config.json` и
индекс в папке Engram (от них зависит TensorFold и проверка sha256) и падал на
папке `.cache` от `hf download`. Наша версия строит то же дерево, исходные
папки не трогает.

**Таблица глубины черновика.** Глубина DSpark выбирается на каждом раунде по
таблице стоимости раунда (`ROUND_MS`). Таблица, снятая на чужой машине,
занижает глубину: у них структурированный вывод падал со 127 до 106 ток/с.
Переснять на своей паре:

```bash
DEPTH_PROBE=1 ./serve.sh && python3 depth-sweep.py   # печатает ROUND_MS=...
./stop.sh && ROUND_MS=<таблица> ./serve.sh           # или в config.local.sh
```

---

### 4.13. GLM-5.3-Flash на двух Spark (TensorFold)

Рецепт MiaAI-Lab (TensorFold 0.6.0 и 123 патча, Apache-2.0): эксперты в EXL3
4 бита, остальное BF16, черновик DFlash2, контекст 1M, картинки. Мы не
копируем его код, а держим их репозиторий на закреплённом коммите
(`KIT_COMMIT`) и запускаем их скрипты с нашими адресами пары, папкой весов и
портом 8888. Подробно — [`tensorfold-glm53/README.md`](tensorfold-glm53/README.md).

По сравнению LimeChain на одном железе (у нас пока не мерили): один короткий
запрос 114 ток/с (текст 50, код 70 без рассуждений), 8 запросов — 528 суммарно,
префил 128K — ~1700 ток/с. vLLM с NVFP4 на тех же машинах: 37 и 185. Нужно
~205 ГБ диска и ~110 ГиБ свободной памяти на каждой машине, поэтому вместе с
DeepSeek не запускается.

```bash
cd tensorfold-glm53
./setup.sh                    # один раз: их репозиторий, образ на обе машины, веса (~176 ГБ) и копия на вторую
./serve.sh                    # первый старт ~10 мин, дальше меньше минуты
./status.sh / ./stop.sh
PARALLEL=8 ./serve.sh restart # 8 запросов: так у LimeChain в работе (минимум памяти ~7 ГиБ на head)
```

Черновик DFlash2 — под CC BY-NC-ND 4.0, только некоммерческое использование.
`DRAFTER=mtp` — встроенная MTP-голова: медленнее и по одному запросу.
Уровень рассуждений по умолчанию `EFFORT=high`: у LimeChain с ним вызовы
инструментов 40/42 против 36/42 на «max» (их значение по умолчанию).

Веса лежат в кэше Hugging Face (`/models/huggingface`), один путь на обеих
машинах. Скачать вручную или перенести на другую машину:

```bash
HF_HOME=/models/huggingface hf download Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold \
  --revision 078455ffe6472f9a52fbc1139f58b9db2881b25c
HF_HOME=/models/huggingface hf download incoai/GLM-5.3-Flash-DFlash2 \
  --revision bf582e4eacc1810f76656d1811693ff6c6737d2a
rsync -a /models/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold \
  /models/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2  вторая-машина:/models/huggingface/hub/
```

---

### 4.14. Qwen3.8-Flash-Next в int4 AutoRound на vLLM (Docker)

Рецепт, который вы долго использовали (blazux → Saren-Arterius → ваша копия,
Apache-2.0), теперь в `vllm-flash-ar/` и запускается как остальные серверы
проекта. Подробно — [`vllm-flash-ar/README.md`](vllm-flash-ar/README.md).

- **Веса:** эксперты в int4 от Intel (AutoRound, ядра GPTQ-Marlin),
  lm_head в int8, боковые слои в FP8, таблица n-грамм в FP8 читается с NVMe
  через `mmap`. В памяти ~68 ГиБ.
- **Что умеет сверх нашего SGLang-профиля** `qwen3.8-flash-next`:
  исправленный кэш префиксов, закрепление длинного системного промпта
  (`PIN_PROMPT`: его KV не вытесняется никаким трафиком), детерминированный
  top-k в разреженном внимании, сайдкар Prometheus с метриками таблицы.
- **Скорость** (замеры рецепта): 65.8 ток/с на одном потоке (как наш SGLang),
  259 суммарно на 8 потоках (у SGLang 212–224), префил 2100–2300 ток/с (у
  SGLang 1450–1660). HumanEval без размышления 96.3% (у SGLang 97.6%).

```bash
cd vllm-flash-ar
./build.sh                 # образ: официальный vLLM для Flash-Next + патчи рецепта
./download.sh              # готовый чекпойнт (~67 ГиБ) и таблица (~49 ГиБ), закреплённые ревизии
./serve.sh                 # проверки, подготовка памяти, контейнер, ждёт, пока API ответит
./status.sh / ./stop.sh
MTP=2 SEQS=16 ./serve.sh   # любые параметры из config.sh
GB10_METRICS_PORT=18400 python3 ../bench/perf.py
python3 ../bench/consistency.py
```

**Что изменилось при переносе** (подробно — в README папки):

- **Патчи парсера инструментов 12–13 от blazux** — для этого же базового
  образа: пример вызова в блоке кода больше не превращается в вызов,
  съедающий остаток ответа. Нужно пересобрать образ (`./build.sh`); на
  старом образе запуск об этом предупреждает.
- **Порт 8888** и имя модели `qwen3.8-flash-next`; старое имя `qwen` тоже
  принимается. Старые клиенты на порту 8000 — `PORT=8000` в `config.local.sh`.
- **`RESTART=no`** по умолчанию: контейнер не поднимается сам после
  перезагрузки, как и остальные серверы проекта. Если это единственный
  сервер на машине, `RESTART=unless-stopped` в `config.local.sh` вернёт
  прежнее поведение.
- **Новые настройки:** `BATCHED_TOKENS` (чанк префила, был жёстко 8192),
  `MARLIN_ATOMIC_ADD` (был всегда включён; влияет только на плотные
  Marlin-слои, MoE суммирует без атомиков всегда), `CPUSET`, `EXTRA_ENV`.
- **`DET_MOE_ALIGN=1`** (по умолчанию включён с 2026-11-01, патч 15 в README папки):
  детерминированная раскладка токенов по экспертам перед MoE. Три холодных
  прогона одного жадного промпта расходились уже в первом токене
  (`../bench/consistency.py`, 0.16–0.45 nats), а у CUDA-ядра
  `moe_align_block_size` порядок маршрутов внутри эксперта зависит от
  порядка `atomicAdd`. Идея и реализация — prestonlogan (Flint). С ним холодные
  прогоны и попадания в кэш совпали полностью (разница 0.0000 на 4K и 32K).
  `DET_MOE_ALIGN_MIN_TOKENS=auto` (SEQS × (1 + MTP) + 1, то есть 33) включает его
  только на шагах с префилом; декод остаётся на штатном ядре. Цены не видно:
  декод одного потока 66.4 против 65.4 ток/с без него, в потоках разница
  меняет знак от уровня к уровню (шум одного прогона). `DET_MOE_ALIGN=0`
  возвращает штатное ядро.
- **`MOE_BACKEND=humming`** (по умолчанию пусто = Marlin): ядра экспертов
  Humming — JIT-библиотека vLLM, ядра собираются под GB10 при первом прогреве
  и кэшируются в `~/.cache/gb10/humming` (`HUMMING_CACHE`), так что компиляция
  только один раз. Раскладка токенов через тот же `moe_align_block_size`, так
  что `DET_MOE_ALIGN` действует и на неё. Измерено: не быстрее Marlin и не
  детерминирована — ядро делит сумму по K между блоками и складывает части
  атомиками в bf16 (Marlin в MoE складывает по порядку в fp32). Оставлена
  для экспериментов.
- **`--enable-prompt-tokens-details`:** в ответах видно, сколько токенов
  пришло из кэша префиксов.
- **`prepare.sh`:** относительные пути теперь работают (раньше он менял
  каталог до того, как их использовать); ревизии репозиториев Intel и Qwen
  закреплены.
- **Проверки проекта:** один сервер за раз (в обе стороны), подготовка
  памяти перед стартом, `build.sh` / `download.sh` / `prepare.sh` не
  работают рядом с занятым GPU.
- **Замеры:** копии наших `perf.py` / `longctx.py` / `common.py` убраны, есть
  `../bench/`; их `decode_bench.py`, `concurrency_bench.py`, `ppwatch.sh`
  остались.

Если настройки уже жили в старом `serve.sh` (пути, ключ, `PIN_PROMPT`),
перенесите их в `config.local.sh` (`config.local.sh.example`).

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
- **Настройки памяти хоста** (`./scripts/host-tuning.sh`): earlyoom на
  1 ГиБ вместо ~8 ГБ, сервер без swap, и профили 27B сами берут 0.85.
  Подробно — ниже, «Настройки памяти хоста».

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

**Температура.** На длинных prefill (32K+) GPU доходит до 79–87 °C. Это
в пределах нормы: `nvidia-smi -q -d TEMPERATURE` показывает в
`GPU T.Limit Temp` запас до собственного предела GPU (он сокращается
быстрее, чем растёт температура: на 101K префиле Flint — 8 °C при 81 °C),
счётчики
термального троттлинга на наших Spark — нули. Частоту под нагрузкой
ограничивает бюджет мощности SoC (SW Power Cap), а не температура; perf.py
показывает запас до T.Limit и предупреждает, когда он меньше 5 °C.
Свободное место вокруг корпуса всё равно полезно.

**Сравнение вариантов.** Разброс между перезапусками ~8% на одном потоке;
выигрыш меньше этого перепроверяйте на новом старте. Температура 0 тоже не
даёт побитово одинаковых ответов: ±2 задачи HumanEval — это шум.

### Настройки памяти хоста

Необязательно, но рекомендуется. Один раз на машине:

```bash
./scripts/host-tuning.sh --check   # текущие значения против наших, ничего не меняет (без sudo)
./scripts/host-tuning.sh           # применить (sudo); повторный запуск даёт тот же результат
./scripts/host-tuning.sh --undo    # вернуть то, что было до первого запуска
```

После применения перезапустите сервер (Ctrl-C и `./serve.sh` или
`sudo systemctl restart gb10-sglang`).

Что меняется:

| Что | Было (по умолчанию) | Станет |
|---|---|---|
| `vm.swappiness` | 60 | 10 |
| `vm.compaction_proactiveness` (фоновое уплотнение памяти) | 20 | 0 |
| `vm.extfrag_threshold` | 500 | 500 |
| `vm.min_free_kbytes` | задаётся при загрузке | 1 ГиБ |
| `vm.watermark_scale_factor` | 10 | 50 |
| earlyoom | ~8 ГБ, убивает самый большой процесс | 1 / 0.5 ГиБ (SIGTERM / SIGKILL), сначала сервер (`sglang::scheduler`, python, tensorfold, llama-server, vLLM), никогда sshd, systemd, dockerd и графику |
| swap для SGLang | разрешён | выключен |
| `MEM_FRACTION` у профилей 27B | 0.80 | 0.85 (KV ~1.15M токенов, четыре сессии по 262K вместо трёх) |

Значения sysctl записываются в `/etc/sysctl.d/99-gb10.conf`, прежние —
в `/var/lib/gb10-host-tuning/`, настройки earlyoom — в
`/etc/default/earlyoom` (старый файл сохраняется как `.pre-gb10`).

**Как выключается swap.** При ручном запуске `./serve.sh` поднимает SGLang в
пользовательском scope с `MemorySwapMax=0` (параметр `NOSWAP`, по умолчанию
`auto`). Это включается само, когда тюнинг применён. Перед этим
проверяется, что ограничение на этой машине действительно работает, иначе
печатается предупреждение. Что вышло, видно в строке `memory` при старте.
Служба `gb10-sglang` (`scripts/install-service.sh`) и так работает без swap.
`NOSWAP=0` запускает как раньше.

**Поведение и проверки.**

- Если применение сорвалось на полпути, машина откатывается в исходное
  состояние.
- `scripts/test-host-tuning.sh` прогоняет скрипт на заглушках `sysctl` и
  `systemctl`, ничего не меняя в системе (16 проверок).
- `./scripts/00-check-host.sh` показывает, применён ли тюнинг.
- Строки префилла в `bench/perf.py` показывают, сколько раз ядро
  останавливало поток на уплотнение и освобождение памяти и сколько ушло в
  swap (`vm: ...`).

**Что это даёт на деле** (A/B на 27B, по три полных прогона без и с,
`results/RESULTS.md`, «Host tuning»): префилл не изменился — SGLang берёт
всю память при старте, и ядро не останавливалось и без настройки. Выигрыш —
в устойчивости: earlyoom больше не убивает сервер при 0.85, сервер не
уходит в swap, а при нехватке памяти earlyoom снимает сервер, а не ssh. У
рецепта DeepSeek на двух Spark (Engram-таблицы читаются через страничный
кэш) медленных кусков префилла стало 2 из 248 вместо 46; для него есть свой
`tensorfold-dsv41/host-tuning.sh` с теми же значениями.

Настройки общие для всей машины: действуют для всех профилей, vLLM и
llama.cpp, пока не сделать `--undo`.

### Память перед стартом и копирование рядом с сервером

```bash
./scripts/mem-prep.sh sudoers            # один раз (спросит пароль): две записи ниже — без пароля
./scripts/mem-prep.sh sudoers 10.100.80.2   # то же здесь и на второй машине пары
./scripts/mem-prep.sh status [worker]    # свободная память, сколько её в блоках от 2 МиБ, аптайм
./scripts/mem-prep.sh idle [worker]      # код 1 и что именно, пока на GPU что-то работает
```

**Перед каждым стартом.** Все наши серверы (`./serve.sh`, `vllm-b12x/`, `vllm-flash-ar/`,
`tensorfold-dsv41/`, `vllm-dsv41/`, `tensorfold-glm53/`, `llamacpp/`), если
GPU свободен, сбрасывают страничный кэш и уплотняют свободную память, у пар
— на обеих машинах. В выводе при старте появляется строка: сколько памяти
свободно и сколько из неё в блоках от 2 МиБ. `MEM_PREP=0` — не трогать.

Зачем (замеры sudoingX, рецепт DeepSeek для двух Spark):
- выделения памяти CUDA на GB10 не вытесняют страничный кэш: 67 МБ не
  выделились при 45 ГБ «available»;
- за несколько дней аптайма свободная память дробится, и задача на GPU шла
  в ~2 раза медленнее, чем на соседней машине; на 16 загрузках одного
  сервера декод был тем быстрее, чем больше свободных блоков от 2 МиБ было
  при старте;
- сброс и уплотнение при свободном GPU занимают меньше секунды.

Возможно, отсюда и наш разброс ~8% между загрузками (`results/RESULTS.md`).
Чтобы это проверить, `bench/perf.py` теперь печатает в заголовке свободную
память, долю в блоках от 2 МиБ, аптайм и то, что было при старте сервера.
Фоновое уплотнение во время работы (`vm.compaction_proactiveness=0`) остаётся
выключенным: это разовое действие до старта.

Для сброса и уплотнения нужен sudo без пароля на две команды
(`tee /proc/sys/vm/drop_caches` и `tee /proc/sys/vm/compact_memory`):
`mem-prep.sh sudoers` кладёт ровно это правило в
`/etc/sudoers.d/gb10-mem-prep`, `sudoers --undo` убирает. Без него старт
идёт как раньше, строка памяти это пишет.

**Не копировать рядом с сервером.** У sudoingX Spark наглухо зависал (ни ssh,
ни ping) при копировании 25 ГБ рядом с задачей на GPU. Поэтому загрузки,
скачивание образов и копирование (`tensorfold-dsv41/download.sh`,
`build.sh`, `vllm-dsv41/pull.sh`, `vllm-b12x/pull.sh`,
`vllm-flash-ar/build.sh`, `download.sh`, `prepare.sh`,
`tensorfold-glm53/setup.sh`, `./serve.sh install`, `llamacpp/build.sh`)
отказываются работать, пока на GPU любой из затронутых машин что-то
запущено, и говорят что. `ALLOW_BUSY_GPU=1` — всё равно продолжить.
`hf download` вручную — тот же риск: сначала остановите сервер.
`scripts/test-mem-prep.sh` проверяет скрипт на заглушках (17 проверок).

---

### Новая версия SGLang

Каждая версия SGLang ставится в свой venv (`~/spark/venv-sglang-<версия>`)
рядом со старой, которую обновление не трогает. Откат — вернуть номер
версии. Профиль выбирает версию переменной `SGLANG_VERSION`.

```bash
sudo systemctl stop gb10-sglang                         # если работает служба
SGLANG_VERSION=0.5.21 ./serve.sh qwen3.8-27b install    # новый venv + lock-файл (~10 ГБ)
SGLANG_VERSION=0.5.21 ./serve.sh qwen3.8-27b            # пробный запуск
python3 bench/perf.py                                   # сравнить с замерами на 0.5.20
```

- `install` при первом запуске для новой версии сам разрешает зависимости
  (CUDA остаётся 13.0) и пишет `requirements/sglang-<версия>-aarch64-py312.txt`.
  Этот файл стоит закоммитить, когда версия прошла проверку.
- Переходить на новую версию по умолчанию (`SGLANG_VERSION` в профиле)
  стоит только после того, как модель на ней запустилась, ответила связно
  и не стала медленнее. Проверять каждую модель отдельно.
- Наши патчи (`GB10_*`) написаны под 0.5.20 и сами проверяют код SGLang:
  если что-то не совпадает, сервер останавливается с сообщением, какой
  патч выключить. Молча неправильно они не работают.
- Перед первым запуском на новой версии ядра компилируются заново —
  первый старт долгий: Flash-Next на 0.5.21 стартовал 31 минуту вместо ~10.
- Если профилю нужны пакеты сверх SGLang, он перечисляет их в
  `EXTRA_PACKAGES`, и `install` ставит их, не сдвигая версии из lock-файла
  (так MiMo получает `torchcodec`, которого SGLang на aarch64 не ставит).
- Патч точности GDN нужно применить и в новом venv (его python).
- Старый venv удаляется вручную, когда он больше не нужен:
  `rm -rf ~/spark/venv-sglang-0.5.20` (~8 ГБ).

## 6. Измерения и проверка качества

Все скрипты берут адрес из `GB10_BASE_URL` (по умолчанию
`http://127.0.0.1:8888/v1`), модель — из `/v1/models`.

```bash
python3 bench/perf.py                    # TTFT, один поток, 1–18 потоков, prefill до 101K
python3 bench/perf.py --only decode      # только один поток
python3 bench/perf.py --only interference  # декод других потоков, пока идёт префилл промпта ~32K
python3 bench/longctx.py                 # несколько длинных контекстов разом
python3 bench/profile_decode.py          # куда уходит шаг декода (по группам ядер)
python3 bench/consistency.py             # примеры вызовов инструментов, детерминизм, кэш префиксов
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

**Проверка правильности ответов** — `bench/consistency.py` (по мотивам
blazux/qwen3.8-Flash-DGX, где всё это нашли сломанным у Qwen3.8 на vLLM).
Работает с любым нашим сервером:

1. **Синтаксис вызова как текст.** С переданными инструментами модель
   повторяет блок ```` ```xml ```` с примером вызова и строку с
   `<function=NAME>` посреди фразы, потом `DONE`. Оба должны вернуться
   текстом, без вызова, в потоке и без. Контроль — «погода в Париже» —
   должен вернуться настоящим вызовом `get_weather`.
2. **Тот же жадный запрос ещё раз:** холодный, два попадания в кэш, снова
   холодный (после сброса кэша, а на vLLM — со своим `cache_salt`: те же
   токены без переиспользования), на 4K и 32K (гибридные модели кэшируют
   состояние только на границах блоков: у vLLM на Flash-Next — каждые 1600
   токенов, так что на ~2K попадания нет). Сравниваются текст и топ-5
   вероятностей первого токена. «Холодный против холодного» разошлись —
   недетерминированные ядра; «холодный против кэша» с другим первым
   токеном — кэш восстанавливает не то состояние; маленький разрыв и
   расхождение ближе к концу — численные мелочи.

Стоковый парсер `qwen3_coder` в SGLang (0.5.20–0.5.22, файл один и тот же)
принимает пример в блоке кода за настоящий вызов и выбрасывает остаток
ответа; в потоке любой `<function=NAME>` в тексте открывает вызов. Патч
`patches/gb10_tool_fences.py` оставляет такие маркеры текстом (блоки кода
по CommonMark, как патч 13 у blazux) и включён для всех профилей SGLang
(`TOOL_FENCES=1` в `serve.sh`, `0` — стоковый парсер). Проверка на CPU:
`python3 patches/test_gb10_tool_fences.py` прогоняет настоящий файл
детектора из venv без патча и с ним: ошибка воспроизводится без патча,
настоящие вызовы с ним разбираются так же, как без.

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
scripts/              установка, проверка машины, служба, merge-lora.py, тюнер MoE,
                      host-tuning.sh, mem-prep.sh (память перед стартом, проверка «GPU свободен»)
patches/              патчи SGLang, включаются профилями через переменные GB10_*:
  gb10_ple_mmap         таблица n-грамм Flash-Next читается на месте
  gb10_fp8_side         BF16-слои, головы и драфт DFlash → FP8 (Marlin)
  gb10_fp8_hc           hyper-connections Flash-Next → FP8
  gb10_marlin_lean      перепаковка MoE в Marlin без лишней копии
  gb10_ablate           абляция векторами (ABLATE=...)
  gb10_tool_fences      пример вызова инструмента в блоке кода остаётся текстом (TOOL_FENCES, вкл.)
  test_gb10_*.py        проверки на CPU: python3 patches/test_gb10_<имя>.py
bench/                perf.py, longctx.py, profile_decode.py, answers.py, consistency.py
moe-configs/          настроенные под GB10 конфиги ядер MoE (Ornith)
llamacpp/             сборка и запуск GLM-5.3-Flash на llama.cpp (1 или 2 Spark)
vllm-b12x/            Flash-Next FP4-Local от NVIDIA на vLLM (Docker)
tensorfold-dsv41/     DeepSeek-V4.1-Flash на двух Spark, TensorFold (Docker)
vllm-dsv41/           та же DeepSeek на двух Spark, настроенный vLLM (Docker)
tensorfold-glm53/     GLM-5.3-Flash на двух Spark, TensorFold по рецепту MiaAI-Lab (Docker)
vllm-flash-ar/        Flash-Next в int4 AutoRound на vLLM (Docker), рецепт blazux / Saren-Arterius
results/RESULTS.md    все замеры с датами и условиями
README.md             подробности и обоснования (англ.)
```

Новую модель проще всего добавить, скопировав ближайший профиль и поменяв
пути к весам, `SERVED_MODEL_NAME` и `model_args`.
