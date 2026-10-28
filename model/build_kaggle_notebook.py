"""Build model/kaggle_finetune.ipynb from model/finetune_mathwriting.py (run after editing the script)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
script = open(os.path.join(HERE, "finetune_mathwriting.py"), encoding="utf-8").read()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


cells = [
    md("""
# Дообучение распознавания для калькулятора

Модель CoMER дообучается на **MathWriting** (Google, ~230 тыс. рукописных формул) и на сгенерированных
строках калькулятора (×, ·, десятичная точка, x и y, степени, дроби). Картинки рисуются так же, как
их рисует приложение, поэтому модель учится ровно на том, что увидит в телефоне.

**Перед запуском** (справа, панель *Settings* / *Session options*):
1. *Accelerator* → **GPU T4 x2** (P100 новые версии PyTorch уже не поддерживают).
2. *Internet* → **On** (нужен подтверждённый телефоном аккаунт Kaggle).
3. Необязательно: *Add-ons → Secrets* → добавить секрет **`GH_TOKEN`** — GitHub-токен с правом
   *Contents: read and write* для репозитория `Des-fee/calculator`. Тогда веса сами загрузятся в GitHub
   и следующая сборка приложения возьмёт их автоматически.

**Запуск:** *Save Version* → **Save & Run All (Commit)**. Так ноутбук работает в фоне, браузер можно закрыть.
Обучение длится ~10 часов (лимит сессии Kaggle — 12). Результат — во вкладке *Output*:
`comer_mathwriting.ckpt` (веса) и `metrics.json` (точность до и после).

Если секрета нет: скачайте `comer_mathwriting.ckpt` из *Output* и пришлите его в чат.

**Ваш почерк:** в приложении меню ⋮ → «Мой почерк» → «Экспорт» сохраняет файл `calculator_samples.jsonl`
(все ваши исправления). Загрузите его сюда: *Add Input → Upload* (новый датасет из этого файла).
Ноутбук найдёт его сам, будет учиться на ваших примерах с повышенным весом и покажет точность на вашем почерке
до и после обучения.

**Продолжить обучение** после прошлого запуска: *Add Input → Your Work* → выбрать прошлую версию этого
ноутбука. Ноутбук сам найдёт там `comer_mathwriting.ckpt` и начнёт с него, а не с исходной модели.

**Новые символы.** Модель учится и символам сверх CROHME: ∂ ∇ ° % ≈ ≡ ∪ ∩ ∈ ∧ ∨ ¬ ⇒ ∬ ∭, матрицы в скобках,
\\begin{cases}, ln, exp, arcsin, недостающие заглавные буквы и греческие буквы. Словарь модели расширяется
автоматически, прошлые веса сохраняются. Лучше всего продолжить с прошлого запуска (см. выше): тогда старые
символы останутся такими же точными, а новые модель выучит за этот запуск.

Лицензия MathWriting — CC BY-NC-SA 4.0 (только некоммерческое использование).
"""),
    code("""
!nvidia-smi -L
!git clone -q --depth 1 https://github.com/Green-Wood/CoMER
!pip install -q einops "pytorch-lightning>=2.4" 2>&1 | tail -n 2
"""),
    md("Скрипт обучения (тот же, что `model/finetune_mathwriting.py` в репозитории):"),
    code("%%writefile finetune_mathwriting.py\n" + script),
    code("""
import glob, os, subprocess, sys
os.environ.update(COMER_DIR="CoMER", OUT_DIR="/kaggle/working", HOURS="10")
# continue from an earlier run: add its Output to this notebook (Add Input -> Your Work -> the notebook)
prev = sorted(glob.glob("/kaggle/input/**/comer_mathwriting.ckpt", recursive=True))
if prev:
    os.environ["INIT_CKPT"] = prev[0]
    print("продолжаю обучение с", prev[0])
p = subprocess.Popen([sys.executable, "-u", "finetune_mathwriting.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="")
assert p.wait() == 0, "training failed"
"""),
    md("Загрузка весов в GitHub (только если добавлен секрет `GH_TOKEN`):"),
    code("""
import json, os, urllib.request

REPO, TAG, NAME = "Des-fee/calculator", "model-weights", "comer_mathwriting.ckpt"
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GH_TOKEN")
except Exception:
    token = None

def gh(method, url, data=None, ctype="application/json"):
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": "Bearer " + token, "Accept": "application/vnd.github+json", "Content-Type": ctype})
    with urllib.request.urlopen(req) as r:
        body = r.read()
        return json.loads(body) if body else None

if not token:
    print("GH_TOKEN не задан: скачайте", NAME, "из вкладки Output и пришлите в чат")
else:
    api = "https://api.github.com/repos/" + REPO
    try:
        rel = gh("GET", api + "/releases/tags/" + TAG)
    except urllib.error.HTTPError:
        # a pre-release, so that it never becomes the "latest" release that the APK link points to
        rel = gh("POST", api + "/releases", json.dumps({"tag_name": TAG, "name": "Веса модели распознавания",
              "body": "Дообученная CoMER (MathWriting). Используется сборкой приложения.",
              "prerelease": True, "make_latest": "false"}).encode())
    for a in rel.get("assets", []):
        if a["name"] in (NAME, "metrics.json"):
            gh("DELETE", api + "/releases/assets/%d" % a["id"])
    upload = rel["upload_url"].split("{")[0]
    for name, ctype in ((NAME, "application/octet-stream"), ("metrics.json", "application/json")):
        with open("/kaggle/working/" + name, "rb") as f:
            gh("POST", upload + "?name=" + name, f.read(), ctype)
    print("загружено в", rel["html_url"], "— запустите сборку приложения (любой коммит или Run workflow)")
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
with open(os.path.join(HERE, "kaggle_finetune.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")
print("written model/kaggle_finetune.ipynb")
