import os
import json
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src.preprocessing import Preprocessor
from src.modeling import TrainedModel
from src.utils import load_json

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")

pre = Preprocessor.load(os.path.join(ARTIFACTS_DIR, "preprocessor.joblib"))
trained = TrainedModel.load(os.path.join(ARTIFACTS_DIR, "model.joblib"))
meta = load_json(os.path.join(ARTIFACTS_DIR, "metadata.json"))

inv_map = {int(k): v for k, v in meta["inv_label_map"].items()}

app = FastAPI(title="TripReason API", version="1.0")


class PredictRequest(BaseModel):
    records: list[dict] = Field(..., min_length=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        df = pd.DataFrame(req.records)
        X = pre.transform(df)

        proba = trained.predict_proba(X)  # probability for class 1
        pred_num = (proba >= trained.threshold).astype(int)
        pred_label = [inv_map[int(p)] for p in pred_num]

        return {
            "predictions": pred_label,
            "probabilities": [float(x) for x in proba],
            "threshold": float(trained.threshold),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def ui():
    # Frontend is embedded as a single HTML page (no extra dependencies).
    html = r"""
<!doctype html>
<html lang="fa" dir="rtl">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>TripReason Predictor</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    /* a tiny polish */
    .glass { background: rgba(255,255,255,.72); backdrop-filter: blur(10px); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
  </style>
</head>
<body class="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 text-slate-100">

  <div class="max-w-5xl mx-auto px-4 py-10">
    <header class="mb-8">
      <div class="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h1 class="text-3xl font-bold tracking-tight">🎫 TripReason Predictor</h1>
          <p class="text-slate-300 mt-2">
            فیلدهای دیفالت را پر کن، هر فیلد دلخواه هم خواستی اضافه کن، و نتیجه را بگیر.
          </p>
        </div>
        <div class="flex items-center gap-2">
          <a class="px-3 py-2 rounded-xl bg-white/10 hover:bg-white/15 border border-white/10"
             href="/docs" target="_blank">Swagger (/docs)</a>
          <button id="btnReset"
                  class="px-3 py-2 rounded-xl bg-white/10 hover:bg-white/15 border border-white/10">
            ریست
          </button>
        </div>
      </div>
    </header>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Form card -->
      <section class="glass rounded-2xl p-6 border border-white/10 shadow-xl">
        <h2 class="text-xl font-semibold mb-4">🧾 ورودی</h2>

        <form id="predictForm" class="space-y-6">
          <!-- Defaults -->
          <div>
            <div class="flex items-center justify-between mb-3">
              <h3 class="font-semibold text-slate-800">فیلدهای دیفالت</h3>
              <span class="text-xs text-slate-600 bg-slate-200 rounded-full px-2 py-1">پیشنهادی</span>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-slate-900">
              <div>
                <label class="block text-sm mb-1">TicketID</label>
                <input id="TicketID" type="number" class="w-full rounded-xl px-3 py-2 bg-white border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-400"
                       value="123"/>
              </div>

              <div>
                <label class="block text-sm mb-1">BillID</label>
                <input id="BillID" type="number" class="w-full rounded-xl px-3 py-2 bg-white border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-400"
                       value="999"/>
              </div>

              <div>
                <label class="block text-sm mb-1">Price</label>
                <input id="Price" type="number" class="w-full rounded-xl px-3 py-2 bg-white border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-400"
                       value="250000"/>
              </div>

              <div>
                <label class="block text-sm mb-1">CouponDiscount</label>
                <input id="CouponDiscount" type="number" class="w-full rounded-xl px-3 py-2 bg-white border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-400"
                       value="0"/>
              </div>

              <div>
                <label class="block text-sm mb-1">Vehicle</label>
                <input id="Vehicle" type="text" class="w-full rounded-xl px-3 py-2 bg-white border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-400"
                       value="Bus"/>
              </div>

              <div>
                <label class="block text-sm mb-1">VehicleClass</label>
                <input id="VehicleClass" type="text" class="w-full rounded-xl px-3 py-2 bg-white border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-400"
                       value="VIP"/>
              </div>

              <div>
                <label class="block text-sm mb-1">Created</label>
                <input id="Created" type="datetime-local" class="w-full rounded-xl px-3 py-2 bg-white border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-400"
                       value="2025-01-10T10:00"/>
              </div>

              <div>
                <label class="block text-sm mb-1">DepartureTime</label>
                <input id="DepartureTime" type="datetime-local" class="w-full rounded-xl px-3 py-2 bg-white border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-400"
                       value="2025-01-12T08:00"/>
              </div>

              <div class="md:col-span-2 grid grid-cols-1 sm:grid-cols-3 gap-3">
                <label class="flex items-center justify-between gap-3 rounded-xl bg-white border border-slate-200 px-4 py-3">
                  <span class="text-sm">Male</span>
                  <input id="Male" type="checkbox" class="w-5 h-5 accent-indigo-600" checked />
                </label>

                <label class="flex items-center justify-between gap-3 rounded-xl bg-white border border-slate-200 px-4 py-3">
                  <span class="text-sm">Domestic</span>
                  <input id="Domestic" type="checkbox" class="w-5 h-5 accent-indigo-600" checked />
                </label>

                <label class="flex items-center justify-between gap-3 rounded-xl bg-white border border-slate-200 px-4 py-3">
                  <span class="text-sm">Cancel</span>
                  <input id="Cancel" type="checkbox" class="w-5 h-5 accent-indigo-600" />
                </label>
              </div>
            </div>
          </div>

          <!-- Custom fields -->
          <div>
            <div class="flex items-center justify-between mb-3">
              <h3 class="font-semibold text-slate-800">فیلدهای دلخواه</h3>
              <button type="button" id="btnAddField"
                      class="px-3 py-2 rounded-xl bg-indigo-600 text-white hover:bg-indigo-500">
                ➕ افزودن فیلد
              </button>
            </div>

            <div id="customFields" class="space-y-3"></div>

            <p class="text-xs text-slate-600 mt-2">
              نکته: اگر فیلدی در مدل استفاده نشود، مشکلی نیست؛ مدل نادیده‌اش می‌گیرد.
            </p>
          </div>

          <!-- Actions -->
          <div class="flex items-center gap-3 flex-wrap">
            <button type="submit" id="btnPredict"
                    class="px-5 py-3 rounded-2xl bg-emerald-600 hover:bg-emerald-500 text-white font-semibold shadow-lg">
              🚀 Predict
            </button>

            <button type="button" id="btnCopyJson"
                    class="px-4 py-3 rounded-2xl bg-white/10 hover:bg-white/15 border border-white/10 text-slate-100">
              📋 کپی JSON
            </button>

            <span id="status" class="text-sm text-slate-200"></span>
          </div>
        </form>
      </section>

      <!-- Result card -->
      <section class="glass rounded-2xl p-6 border border-white/10 shadow-xl">
        <div class="flex items-center justify-between gap-3 flex-wrap">
          <h2 class="text-xl font-semibold text-slate-900">📌 نتیجه</h2>
          <span class="text-xs text-slate-700 bg-slate-200 rounded-full px-2 py-1">POST /predict</span>
        </div>

        <div class="mt-4 space-y-4">
          <div class="rounded-2xl bg-slate-950/90 border border-white/10 p-4">
            <div class="text-sm text-slate-300 mb-2">خروجی:</div>
            <pre id="resultBox" dir="ltr"
            class="mono text-xs text-slate-100 whitespace-pre-wrap text-left"
            style="direction:ltr; text-align:left; unicode-bidi:plaintext;">
            اینجا نتیجه نمایش داده می‌شود…
            </pre>
          </div>

          <div class="rounded-2xl bg-slate-950/60 border border-white/10 p-4">
            <div class="text-sm text-slate-300 mb-2">درخواست ارسالی (Payload):</div>
            <pre id="payloadBox" dir="ltr"
            class="mono text-xs text-slate-100 whitespace-pre-wrap text-left"
            style="direction:ltr; text-align:left; unicode-bidi:plaintext;"></pre>
          </div>
        </div>
      </section>
    </div>

    <footer class="mt-10 text-center text-xs text-slate-400">
      ساخته‌شده روی FastAPI — اگر چیزی خطا داد، متن خطا در همین صفحه نمایش داده می‌شود.
    </footer>
  </div>

<script>
  const customFieldsEl = document.getElementById("customFields");
  const payloadBox = document.getElementById("payloadBox");
  const resultBox  = document.getElementById("resultBox");
  const statusEl   = document.getElementById("status");

  function addCustomFieldRow(name="", value="", type="text") {
    const row = document.createElement("div");
    row.className = "grid grid-cols-1 md:grid-cols-12 gap-2 items-center";

    row.innerHTML = `
      <div class="md:col-span-4">
        <input placeholder="نام فیلد (مثلاً OriginCity)" value="${escapeHtml(name)}"
          class="w-full rounded-xl px-3 py-2 bg-white border border-slate-200 text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-400 field-name"/>
      </div>

      <div class="md:col-span-5">
        <input placeholder="مقدار" value="${escapeHtml(value)}"
          class="w-full rounded-xl px-3 py-2 bg-white border border-slate-200 text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-400 field-value"/>
      </div>

      <div class="md:col-span-2">
        <select class="w-full rounded-xl px-3 py-2 bg-white border border-slate-200 text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-400 field-type">
          <option value="text" ${type==="text"?"selected":""}>text</option>
          <option value="number" ${type==="number"?"selected":""}>number</option>
          <option value="bool" ${type==="bool"?"selected":""}>bool</option>
        </select>
      </div>

      <div class="md:col-span-1 flex justify-end">
        <button type="button" class="px-3 py-2 rounded-xl bg-rose-600 text-white hover:bg-rose-500 btn-remove">✕</button>
      </div>
    `;

    row.querySelector(".btn-remove").addEventListener("click", () => row.remove());
    customFieldsEl.appendChild(row);
  }

  function escapeHtml(str) {
    return String(str)
      .replaceAll("&","&amp;")
      .replaceAll("<","&lt;")
      .replaceAll(">","&gt;")
      .replaceAll('"',"&quot;")
      .replaceAll("'","&#039;");
  }

  function buildRecord() {
    const rec = {};

    // defaults
    const getVal = (id) => document.getElementById(id).value;
    const getBool = (id) => document.getElementById(id).checked;

    // numbers
    rec["TicketID"] = safeNumber(getVal("TicketID"));
    rec["BillID"] = safeNumber(getVal("BillID"));
    rec["Price"] = safeNumber(getVal("Price"));
    rec["CouponDiscount"] = safeNumber(getVal("CouponDiscount"));

    // strings
    rec["Vehicle"] = getVal("Vehicle");
    rec["VehicleClass"] = getVal("VehicleClass");

    // datetimes (string is fine, pandas can parse)
    const created = getVal("Created");
    const dep = getVal("DepartureTime");
    if (created) rec["Created"] = created;
    if (dep) rec["DepartureTime"] = dep;

    // bools
    rec["Male"] = getBool("Male");
    rec["Domestic"] = getBool("Domestic");
    rec["Cancel"] = getBool("Cancel");

    // custom fields
    const rows = customFieldsEl.querySelectorAll("div");
    rows.forEach((row) => {
      const name = row.querySelector(".field-name")?.value?.trim();
      const valRaw = row.querySelector(".field-value")?.value ?? "";
      const type = row.querySelector(".field-type")?.value ?? "text";
      if (!name) return;

      if (type === "number") rec[name] = safeNumber(valRaw);
      else if (type === "bool") rec[name] = parseBool(valRaw);
      else rec[name] = valRaw;
    });

    return rec;
  }

  function safeNumber(v) {
    const n = Number(v);
    return Number.isFinite(n) ? n : 0;
  }

  function parseBool(v) {
    const s = String(v).trim().toLowerCase();
    if (["true","1","yes","y","on"].includes(s)) return true;
    if (["false","0","no","n","off"].includes(s)) return false;
    // fallback: treat non-empty as true
    return Boolean(s);
  }

  function setStatus(text) {
    statusEl.textContent = text || "";
  }

  async function doPredict() {
    const payload = { records: [buildRecord()] };
    payloadBox.textContent = JSON.stringify(payload, null, 2);

    setStatus("در حال ارسال درخواست…");
    resultBox.textContent = "…";

    const res = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      setStatus("خطا!");
      resultBox.textContent = JSON.stringify(data, null, 2);
      return;
    }

    setStatus("✅ انجام شد");
    resultBox.textContent = JSON.stringify(data, null, 2);
  }

  // events
  document.getElementById("btnAddField").addEventListener("click", () => addCustomFieldRow());
  document.getElementById("btnCopyJson").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(payloadBox.textContent || "");
      setStatus("📋 JSON کپی شد");
      setTimeout(() => setStatus(""), 1200);
    } catch {
      setStatus("کپی نشد (اجازه مرورگر)");
    }
  });

  document.getElementById("btnReset").addEventListener("click", () => {
    // simple reload reset
    location.reload();
  });

  document.getElementById("predictForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    try { await doPredict(); }
    catch (err) {
      setStatus("خطای غیرمنتظره!");
      resultBox.textContent = String(err);
    }
  });

  // start with a couple of example custom fields (optional)
  addCustomFieldRow("OriginCity", "Tehran", "text");
  addCustomFieldRow("PassengerCount", "2", "number");

  // initial payload preview
  payloadBox.textContent = JSON.stringify({records:[buildRecord()]}, null, 2);
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)
    """
    return HTMLResponse(content=html)
    """