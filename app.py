import os
import json
import re
import traceback
from typing import List, Dict

from dotenv import load_dotenv
from flask import Flask, request, render_template_string, jsonify
import pytesseract
from PIL import Image
from openai import OpenAI

# ─────────── 环境 & 客户端 ───────────
load_dotenv()

DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("Missing DEEPSEEK_API_KEY in .env")

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
MODEL_NAME = "deepseek-chat"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────── 读取考纲数据 ───────────
with open("Syllabus_data.json", "r", encoding="utf-8") as f:
    SYLLABUS_DATA: List[Dict] = json.load(f)

# ─────────── 简易匹配工具 ───────────
STOP_WORDS = {
    "the","and","for","with","that","this","from","they","their","them","which",
    "such","into","also","been","were","have","has","had","are","was","but","not",
    "can","use","using","between","within","you","your","what","how","why","when","where","who"
}
_token_re = re.compile(r"[A-Za-z]+")

def tokenize(text: str) -> List[str]:
    tokens = _token_re.findall(text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

def search_syllabus(query: str, data: List[Dict]) -> List[Dict]:
    q_tokens = tokenize(query)
    hits = []
    for entry in data:
        weight = entry.get("weight", 1)
        e_tokens = tokenize(" ".join(entry.get("keywords", [])))
        score = sum(weight for t in q_tokens if t in e_tokens)

        matched_sub = []
        for sub in entry.get("subtopics", []):
            st = tokenize(" ".join(sub.get("keywords", [])))
            sub_score = sum(weight for t in q_tokens if t in st)
            if sub_score:
                score += sub_score
                matched_sub.append(sub.get("name"))

        if score:
            hits.append((score, entry, matched_sub))

    hits.sort(reverse=True, key=lambda x: x[0])

    result = []
    for score, entry, matched_sub in hits[:8]:
        e = entry.copy()
        e["score"] = score
        if matched_sub:
            e["matched_subtopics"] = matched_sub
        result.append(e)
    return result

# ─────────── HTML 模板（升级版 UI + MathJax + Markdown 渲染） ───────────
INDEX_HTML = """
<!doctype html>
<meta charset="utf-8" />
<title>A-Level 题目定位 · Upload</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root{
    --bg: #0b1020;
    --card: rgba(255,255,255,0.06);
    --glass: rgba(255,255,255,0.08);
    --stroke: rgba(255,255,255,0.12);
    --text: #e8ecf1;
    --muted:#9aa7b3;
    --accent:#7cf7ff;
    --accent2:#b388ff;
  }
  *{box-sizing:border-box}
  body{
    margin:0; color:var(--text);
    font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial;
    background:
      radial-gradient(1200px 500px at 10% -10%, #12204a 0%, transparent 60%),
      radial-gradient(900px 400px at 100% 10%, #3a1f6f 0%, transparent 60%),
      linear-gradient(180deg, #0a0f1f, #090d18);
    min-height:100vh;
  }
  .wrap{ max-width:980px; margin:48px auto; padding:0 20px; }
  .hero{
    background: var(--glass); border:1px solid var(--stroke);
    border-radius:20px; padding:28px; backdrop-filter: blur(10px);
    box-shadow: 0 20px 60px rgba(0,0,0,.35), inset 0 0 0 1px rgba(255,255,255,.03);
  }
  h1{ margin:0 0 8px; font-weight:800; letter-spacing:.3px; }
  .sub{ color:var(--muted); margin-bottom:20px }
  .upload{
    display:flex; gap:12px; align-items:center; flex-wrap:wrap;
    border:1px dashed var(--accent); border-radius:14px; padding:16px;
    background: rgba(124,247,255,0.06);
  }
  input[type=file]{ color:var(--text); }
  button{
    background:linear-gradient(135deg,var(--accent),var(--accent2));
    color:#05121a; border:0; border-radius:12px; padding:10px 16px;
    font-weight:700; cursor:pointer; box-shadow:0 10px 30px rgba(124,247,255,.25);
  }
  .tip{ color:var(--muted); font-size:13px; margin-left:6px }
</style>

<div class="wrap">
  <div class="hero">
    <h1>上传题目图片</h1>
    <div class="sub">本地 OCR + DeepSeek 解答 + 考纲定位（科技感玻璃风界面）</div>
    <form method="post" enctype="multipart/form-data" action="/upload" class="upload">
      <input type="file" name="image" accept="image/*" required>
      <button type="submit">开始识别</button>
      <span class="tip">支持拍照或图片；数学公式会自动排版。</span>
    </form>
  </div>
</div>
"""

RESULT_HTML = """
<!doctype html>
<meta charset="utf-8" />
<title>A-Level 题目定位 · Result</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<!-- MathJax：渲染 \\( ... \\) / \\[ ... \\] LaTeX 公式 -->
<script>
  window.MathJax = {
    tex: {inlineMath: [['\\(','\\)'],['$','$']], displayMath: [['\\[','\\]'],['$$','$$']]},
    svg: {fontCache: 'global'}
  };
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!-- marked.js：把 Markdown 转成 HTML -->
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

<style>
  :root{
    --bg: #0b1020; --card: rgba(255,255,255,0.06); --glass: rgba(255,255,255,0.08);
    --stroke: rgba(255,255,255,0.12); --text: #e8ecf1; --muted:#9aa7b3;
    --accent:#7cf7ff; --accent2:#b388ff; --good:#7CFF8C;
  }
  *{box-sizing:border-box}
  body{
    margin:0; color:var(--text);
    font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial;
    background:
      radial-gradient(1200px 500px at 10% -10%, #12204a 0%, transparent 60%),
      radial-gradient(900px 400px at 100% 10%, #3a1f6f 0%, transparent 60%),
      linear-gradient(180deg, #0a0f1f, #090d18);
    min-height:100vh;
  }
  .wrap{ max-width:1100px; margin:40px auto; padding:0 20px; }
  .grid{ display:grid; grid-template-columns: 1.1fr 0.9fr; gap:20px; }
  .card{
    background: var(--glass); border:1px solid var(--stroke);
    border-radius:18px; padding:20px; backdrop-filter: blur(10px);
    box-shadow: 0 20px 60px rgba(0,0,0,.35), inset 0 0 0 1px rgba(255,255,255,.03);
  }
  h1,h2{ margin:0 0 12px; }
  .badge{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; background:rgba(124,247,255,.12); border:1px solid rgba(124,247,255,.35); color:var(--accent); }
  .hit{ padding:12px 12px; border:1px solid var(--stroke); border-radius:14px; margin-bottom:10px; background:rgba(255,255,255,.03) }
  .meta{ color:var(--muted); font-size:13px }
  pre.ocr{ white-space:pre-wrap; background:#0f1729; color:#cfe7ff; padding:12px; border-radius:12px; border:1px solid var(--stroke); }
  details summary{ cursor:pointer; }
  .md h3{ margin-top:18px; }
  .md code, .md pre{ font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .md pre{ background:#0f1729; padding:12px; border-radius:12px; border:1px solid var(--stroke); overflow:auto }
  .muted{ color:var(--muted) }
  @media (max-width: 880px){ .grid{ grid-template-columns: 1fr; } }
</style>

<div class="wrap">
  <div class="grid">
    <!-- 左侧：考纲定位 -->
    <div class="card">
      <h2>对应考纲 <span class="badge">Top {{ syllabus_hits|length }}</span></h2>
      {% if syllabus_hits %}
        {% for hit in syllabus_hits %}
          <div class="hit">
            <div><strong>{{ hit.topic }}</strong> <span class="muted">({{ hit.syllabus_reference }})</span></div>
            <div class="meta">{{ hit.book }}，{{ hit.chapter }}，pp.{{ hit.page_range }}</div>
            <div class="meta">匹配分：<span style="color:var(--good);font-weight:700">{{ hit.score }}</span>
              ｜ 关键词：{{ ", ".join(hit.keywords[:4]) }}
              {% if hit.matched_subtopics %} ｜ 子条目：{{ ", ".join(hit.matched_subtopics) }}{% endif %}
            </div>
          </div>
        {% endfor %}
      {% else %}
        <p class="muted">⚠️ 未匹配到任何考纲条目。</p>
      {% endif %}
    </div>

    <!-- 右侧：AI 解答（Markdown + LaTeX 渲染） -->
    <div class="card">
      <details open>
        <summary><h2>▼ 查看 DeepSeek 解答与思路</h2></summary>
        <div id="ai" class="md"></div>
      </details>
      <h2 style="margin-top:18px;">OCR 识别的题干</h2>
      <pre class="ocr">{{ ocr_text }}</pre>
    </div>
  </div>
</div>

<!-- 把后端传来的原始文本当作 JSON 注入，再用 marked + MathJax 渲染 -->
<script>
  const AI_RAW = {{ deepseek_answer | tojson }};
  const container = document.getElementById('ai');
  container.innerHTML = marked.parse(AI_RAW);
  // 让 MathJax 对渲染后的 HTML 进行二次排版
  if (window.MathJax && window.MathJax.typesetPromise) {
    MathJax.typesetPromise([container]);
  }
</script>
"""

# ─────────── Flask 应用 ───────────
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "image" not in request.files:
            return "No file uploaded", 400
        f = request.files["image"]
        path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(path)

        # 1) OCR
        ocr_text = pytesseract.image_to_string(Image.open(path)).strip()

        # 2) DeepSeek
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert A-Level tutor."},
                {"role": "user",   "content": f"请针对这道题目给出答案和详细解题思路：\n{ocr_text}"}
            ],
            stream=False
        )
        deepseek_answer = resp.choices[0].message.content.strip()

        # 3) 考纲匹配
        syllabus_hits = search_syllabus(ocr_text, SYLLABUS_DATA)

        # 4) 渲染
        return render_template_string(
            RESULT_HTML,
            ocr_text=ocr_text,
            deepseek_answer=deepseek_answer,
            syllabus_hits=syllabus_hits
        )
    except Exception as e:
        traceback.print_exc()
        return f"Server Error: {e}", 500

@app.route("/api/upload", methods=["POST"])
def api_upload():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        f = request.files["image"]
        path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(path)

        ocr_text = pytesseract.image_to_string(Image.open(path)).strip()
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert A-Level tutor."},
                {"role": "user",   "content": f"请针对这道题目给出答案和详细解题思路：\n{ocr_text}"}
            ],
            stream=False
        )
        deepseek_answer = resp.choices[0].message.content.strip()
        syllabus_hits = search_syllabus(ocr_text, SYLLABUS_DATA)

        return jsonify({
            "ocr_text": ocr_text,
            "deepseek_answer": deepseek_answer,
            "syllabus_hits": syllabus_hits
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
