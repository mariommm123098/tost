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
load_dotenv()  # 读取 .env

DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

if not DEEPSEEK_API_KEY:
    raise RuntimeError("Missing DEEPSEEK_API_KEY. Create .env and put your key there.")

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
MODEL_NAME = "deepseek-chat"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────── 读取考纲数据 ───────────
with open("Syllabus_data.json", "r", encoding="utf-8") as f:
    SYLLABUS_DATA: List[Dict] = json.load(f)

# ─────────── 简易匹配工具 ───────────
STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "they",
    "their", "them", "which", "such", "into", "also", "been",
    "were", "have", "has", "had", "are", "was", "but", "not",
    "can", "use", "using", "between", "within", "you", "your",
    "what", "how", "why", "when", "where", "who"
}
_token_re = re.compile(r"[A-Za-z]+")

def tokenize(text: str) -> List[str]:
    tokens = _token_re.findall(text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

def search_syllabus(query: str, data: List[Dict]) -> List[Dict]:
    """
    关键词匹配：题干 token 与条目 keywords 命中一次计 weight 分（默认 1）。
    命中 subtopics 也累加（若存在），并记录 matched_subtopics。返回前 8 条。
    """
    q_tokens = tokenize(query)
    hits = []
    for entry in data:
        weight = entry.get("weight", 1)
        # 主关键词
        e_tokens = tokenize(" ".join(entry.get("keywords", [])))
        score = sum(weight for t in q_tokens if t in e_tokens)

        # 子条目（可选）
        matched_sub = []
        for sub in entry.get("subtopics", []):
            sub_tokens = tokenize(" ".join(sub.get("keywords", [])))
            sub_score  = sum(weight for t in q_tokens if t in sub_tokens)
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

# ─────────── HTML 模板 ───────────
INDEX_HTML = """
<!doctype html>
<title>OCR → 考纲定位 DEMO</title>
<h1>Upload A-Level Math Question</h1>
<form method="post" enctype="multipart/form-data" action="/upload">
  <input type="file" name="image" required>
  <input type="submit" value="Upload">
</form>
"""

RESULT_HTML = """
<!doctype html>
<title>A-Level 题目定位</title>

<h1>对应考纲 (Top {{ syllabus_hits|length }})</h1>
{% if syllabus_hits %}
  <ol>
  {% for hit in syllabus_hits %}
    <li style="margin-bottom:0.7em;">
      <strong>{{ hit.topic }} ({{ hit.syllabus_reference }})</strong><br>
      <small>
        {{ hit.book }}，{{ hit.chapter }}，pp.{{ hit.page_range }}<br>
        匹配分：{{ hit.score }} ｜ 关键词：{{ ", ".join(hit.keywords[:4]) }}
      </small>
      {% if hit.matched_subtopics %}
        <div style="font-size:0.9em; color:#666;">
          子条目：{{ ", ".join(hit.matched_subtopics) }}
        </div>
      {% endif %}
    </li>
  {% endfor %}
  </ol>
{% else %}
  <p>⚠️ 未匹配到任何考纲条目。</p>
{% endif %}

<details style="margin-top:2em;">
  <summary><strong>▼ 查看 DeepSeek 解答与思路</strong></summary>
  <pre style="white-space: pre-wrap;">{{ deepseek_answer }}</pre>
</details>

<h2 style="margin-top:2em;">OCR 识别的题干</h2>
<pre style="white-space: pre-wrap; background:#f8f8f8;">{{ ocr_text }}</pre>
"""

# ─────────── Flask 应用 ───────────
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # 调试时返回中文更友好

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

# 额外：JSON API，便于移动端或脚本调用
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
    # 开发阶段开启 debug；上生产请关闭
    app.run(debug=True)
