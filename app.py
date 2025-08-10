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

# Apple Silicon 上 tesseract 常见路径（找得到就用）
try:
    if os.path.exists("/opt/homebrew/bin/tesseract"):
        pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
except Exception:
    pass

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

# ─────────── HTML 模板（首页：拖拽/预览/骨架/粒子/主题） ───────────
INDEX_HTML = r"""
<!doctype html>
<meta charset="utf-8" />
<title>A-Level 题目定位 · Upload</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/gsap.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/lottie-web@5.12.2/build/player/lottie.min.js"></script>

<style>
  :root{
    --bg: #0b1020; --card: rgba(255,255,255,0.06); --glass: rgba(255,255,255,0.08);
    --stroke: rgba(255,255,255,0.14); --text: #e8ecf1; --muted:#9aa7b3;
    --accent:#7cf7ff; --accent2:#b388ff; --ring-bg: rgba(255,255,255,.14);
    --ok:#7CFF8C;
  }
  :root[data-theme="light"]{
    --bg:#f7f8fc; --card:#fff; --glass:#fff; --stroke:rgba(10,30,60,.12);
    --text:#182132; --muted:#5a6675; --accent:#0aa6ff; --accent2:#7b5cff; --ring-bg:#e6eef6;
  }
  *{box-sizing:border-box}
  body{
    margin:0; color:var(--text);
    font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial;
    background:
      radial-gradient(1200px 500px at 10% -10%, #12204a 0%, transparent 60%),
      radial-gradient(900px 400px at 100% 10%, #3a1f6f 0%, transparent 60%),
      linear-gradient(180deg, #0a0f1f, #090d18);
    min-height:100vh; overflow-x:hidden;
  }
  :root[data-theme="light"] body{
    background: linear-gradient(180deg,#f7f8fc,#eef3fb);
  }
  canvas#bg{ position:fixed; inset:0; width:100%; height:100%; z-index:-1; opacity:.55; }
  .wrap{ max-width:980px; margin:48px auto; padding:0 20px; }
  .topbar{ display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; }
  .theme-btn{
    border:1px solid var(--stroke); background:var(--glass); color:var(--text);
    border-radius:999px; padding:8px 12px; cursor:pointer;
  }
  .hero{
    background: var(--glass); border:1px solid var(--stroke);
    border-radius:20px; padding:28px; backdrop-filter: blur(10px);
    box-shadow: 0 20px 60px rgba(0,0,0,.15), inset 0 0 0 1px rgba(255,255,255,.03);
    transform: translateY(20px); opacity:0;
  }
  h1{ margin:0 0 8px; font-weight:800; letter-spacing:.3px; }
  .sub{ color:var(--muted); margin-bottom:22px }
  .upload{
    display:grid; grid-template-columns: 120px 1fr auto; gap:16px; align-items:center;
    border:1px dashed var(--accent); border-radius:16px; padding:16px;
    background: rgba(124,247,255,0.06); transition: transform .2s ease, box-shadow .2s ease, background .2s ease;
  }
  :root[data-theme="light"] .upload{ background: rgba(10,166,255,.06); }
  .upload.dragover{
    transform: scale(1.01);
    box-shadow: 0 10px 30px rgba(124,247,255,.18);
    background: rgba(124,247,255,0.12);
  }
  #lottieBox{ width:120px; height:120px; border-radius:12px; background:rgba(255,255,255,.04); border:1px solid var(--stroke); display:grid; place-items:center; }
  .preview{ display:flex; gap:12px; align-items:center; }
  .preview img{ width:80px; height:80px; object-fit:cover; border-radius:10px; border:1px solid var(--stroke); display:none; }
  input[type=file]{ display:none; }
  .fileLabel{
    display:inline-block; padding:10px 14px; border-radius:10px; border:1px solid var(--stroke);
    background:transparent; color:var(--text); cursor:pointer;
  }
  button.primary{
    background:linear-gradient(135deg,var(--accent),var(--accent2));
    color:#05121a; border:0; border-radius:12px; padding:10px 16px;
    font-weight:700; cursor:pointer; box-shadow:0 10px 30px rgba(124,247,255,.25);
  }
  .tip{ color:var(--muted); font-size:13px }

  /* 加载骨架 */
  #loading{
    position:fixed; inset:0; background:rgba(10,15,30,.75); backdrop-filter: blur(6px);
    display:none; place-items:center; z-index:9999;
  }
  .skeleton{
    width:min(680px, 92vw); background:var(--glass); border:1px solid var(--stroke);
    border-radius:16px; padding:18px;
    animation:pulse 1.2s infinite ease-in-out;
  }
  .sk-bar{ height:14px; border-radius:8px; background:rgba(255,255,255,.12); margin:10px 0; }
  .sk-big{ height:90px; border-radius:12px; }
  @keyframes pulse{ 0%{opacity:.6} 50%{opacity:1} 100%{opacity:.6} }
  @media (max-width: 760px){ .upload{ grid-template-columns: 1fr; } #lottieBox{ display:none; } }
</style>

<canvas id="bg"></canvas>

<div class="wrap">
  <div class="topbar">
    <div class="muted">A-Level OCR → DeepSeek → Syllabus</div>
    <div>
      <button class="theme-btn" id="toggleParticles">粒子背景：开/关</button>
      <button class="theme-btn" id="themeToggle">浅/深色</button>
    </div>
  </div>

  <section class="hero">
    <h1 id="title">上传题目图片</h1>
    <div class="sub">本地 OCR + DeepSeek 解答 + 考纲定位（拖拽上传，自动预览）</div>

    <form id="form" method="post" enctype="multipart/form-data" action="/upload" class="upload">
      <div id="lottieBox"><div id="lottie"></div></div>

      <div class="preview">
        <label for="file" class="fileLabel">选择图片</label>
        <input id="file" type="file" name="image" accept="image/*" required>
        <img id="thumb" alt="preview">
        <span class="tip">也可以把图片直接拖进这个框</span>
      </div>

      <button class="primary" type="submit" id="goBtn">开始识别</button>
    </form>
  </section>
</div>

<!-- 全屏加载骨架 -->
<div id="loading">
  <div class="skeleton">
    <div class="sk-bar" style="width:40%"></div>
    <div class="sk-bar" style="width:70%"></div>
    <div class="sk-big"></div>
    <div class="sk-bar" style="width:85%"></div>
    <div class="sk-bar" style="width:60%"></div>
  </div>
</div>

<script>
  // 主题 & 偏好
  const root = document.documentElement;
  const THEME_KEY = 'theme';
  function applyTheme(t){ root.setAttribute('data-theme', t); localStorage.setItem(THEME_KEY, t); }
  const saved = localStorage.getItem(THEME_KEY) || (matchMedia('(prefers-color-scheme: light)').matches? 'light':'dark');
  applyTheme(saved);
  document.getElementById('themeToggle').onclick = () => {
    applyTheme(root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
  };

  // 粒子背景（轻量 canvas）
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let particlesOn = !reduceMotion;
  const cvs = document.getElementById('bg');
  const ctx = cvs.getContext('2d');
  let W,H, P=[];
  function resize(){ cvs.width = W = window.innerWidth; cvs.height = H = window.innerHeight; }
  resize(); window.addEventListener('resize', resize);
  function make(n=60){
    P = Array.from({length:n},()=>({
      x: Math.random()*W, y: Math.random()*H,
      vx: (Math.random()-.5)*0.3, vy:(Math.random()-.5)*0.3,
      r: 0.6 + Math.random()*1.2
    }));
  }
  make();
  function draw(){
    if(!particlesOn) { ctx.clearRect(0,0,W,H); return; }
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    for(const p of P){
      p.x+=p.vx; p.y+=p.vy;
      if(p.x<0||p.x>W) p.vx*=-1;
      if(p.y<0||p.y>H) p.vy*=-1;
      ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2); ctx.fill();
    }
    requestAnimationFrame(draw);
  }
  if(particlesOn) draw();
  document.getElementById('toggleParticles').onclick = ()=>{
    particlesOn = !particlesOn; if(particlesOn) draw();
  };

  // GSAP 进场
  gsap.to('.hero',{duration:.5, y:0, opacity:1, ease:'power3.out'});

  // Lottie（若加载失败则静默）
  try{
    lottie.loadAnimation({
      container: document.getElementById('lottie'),
      renderer: 'svg', loop: true, autoplay: true,
      path: 'https://assets10.lottiefiles.com/packages/lf20_vnikrcia.json'
    });
  }catch(_){}

  // 拖拽上传 + 预览
  const form = document.getElementById('form');
  const file = document.getElementById('file');
  const thumb = document.getElementById('thumb');
  const uploadBox = document.querySelector('.upload');
  uploadBox.addEventListener('dragenter', e=>{e.preventDefault(); uploadBox.classList.add('dragover');});
  uploadBox.addEventListener('dragover', e=>{e.preventDefault();});
  uploadBox.addEventListener('dragleave', e=>{uploadBox.classList.remove('dragover');});
  uploadBox.addEventListener('drop', e=>{
    e.preventDefault(); uploadBox.classList.remove('dragover');
    const dt = e.dataTransfer;
    if(dt && dt.files && dt.files[0]) {
      file.files = dt.files;
      showPreview(dt.files[0]);
    }
  });
  file.addEventListener('change', e=>{
    if(file.files && file.files[0]) showPreview(file.files[0]);
  });
  function showPreview(f){
    const url = URL.createObjectURL(f);
    thumb.src = url; thumb.style.display='block';
  }

  // 提交时显示全屏骨架
  form.addEventListener('submit', ()=>{
    document.getElementById('loading').style.display='grid';
  });
</script>
"""

# ─────────── 结果页模板（转义修复 + Markdown/TeX 渲染 + 动效） ───────────
RESULT_HTML = r"""
<!doctype html>
<meta charset="utf-8" />
<title>A-Level 题目定位 · Result</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/gsap.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>
  // MathJax 配置
  window.MathJax = {
    tex: {inlineMath: [['\\(','\\)'],['$','$']], displayMath: [['\\[','\\]'],['$$','$$']]},
    svg: {fontCache: 'global'}
  };
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<style>
  :root{
    --bg: #0b1020; --card: rgba(255,255,255,0.06); --glass: rgba(255,255,255,0.08);
    --stroke: rgba(255,255,255,0.14); --text: #e8ecf1; --muted:#9aa7b3;
    --accent:#7cf7ff; --accent2:#b388ff; --good:#7CFF8C; --ring-bg: rgba(255,255,255,.14);
  }
  :root[data-theme="light"]{
    --bg:#f7f8fc; --card:#fff; --glass:#fff; --stroke:rgba(10,30,60,.12);
    --text:#182132; --muted:#5a6675; --accent:#0aa6ff; --accent2:#7b5cff; --ring-bg:#e6eef6; --good:#00b368;
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
  :root[data-theme="light"] body{
    background: linear-gradient(180deg,#f7f8fc,#eef3fb);
  }
  .wrap{ max-width:1100px; margin:24px auto 40px; padding:0 20px; }
  .topbar{ display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; }
  .theme-btn,.back-btn{
    border:1px solid var(--stroke); background:var(--glass); color:var(--text);
    border-radius:999px; padding:8px 12px; cursor:pointer; text-decoration:none;
  }
  .grid{ display:grid; grid-template-columns: 1.1fr 0.9fr; gap:20px; }
  .card{
    background: var(--glass); border:1px solid var(--stroke);
    border-radius:18px; padding:20px; backdrop-filter: blur(10px);
    box-shadow: 0 20px 60px rgba(0,0,0,.15), inset 0 0 0 1px rgba(255,255,255,.03);
  }
  h1,h2{ margin:0 0 12px; }
  .badge{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; background:rgba(124,247,255,.12); border:1px solid rgba(124,247,255,.35); color:var(--accent); }
  .hit{
    padding:12px; border:1px solid var(--stroke); border-radius:14px; margin-bottom:12px;
    background:rgba(255,255,255,.03); display:grid; grid-template-columns: 60px 1fr auto; gap:12px; align-items:center;
    opacity:0; transform: translateY(10px);
  }
  .meta{ color:var(--muted); font-size:13px }
  pre.ocr{ white-space:pre-wrap; background:#0f1729; color:#cfe7ff; padding:12px; border-radius:12px; border:1px solid var(--stroke); }
  :root[data-theme="light"] pre.ocr{ background:#f5f7fb; color:#1e2a3a; }
  details summary{ cursor:pointer; }
  .md h3{ margin-top:18px; }
  .md code, .md pre{ font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .md pre{ background:#0f1729; padding:12px; border-radius:12px; border:1px solid var(--stroke); overflow:auto }
  .muted{ color:var(--muted) }
  @media (max-width: 880px){ .grid{ grid-template-columns: 1fr; } .hit{ grid-template-columns: 60px 1fr } }

  /* 圆环进度条 */
  .ring{
    --p: 0; width:60px; height:60px; border-radius:50%;
    background: conic-gradient(var(--accent) calc(var(--p)*1%), var(--ring-bg) 0);
    display:grid; place-items:center; font-weight:800; color:var(--text);
    border:1px solid var(--stroke);
  }
  .ring span{ font-size:13px }

  /* 复制按钮与提示 */
  .copy-btn{
    border:1px solid var(--stroke); background:transparent; color:var(--text);
    border-radius:10px; padding:6px 10px; cursor:pointer; transition: transform .12s ease;
  }
  .copy-btn.copied{ color:var(--good); border-color: rgba(124,255,140,.65); transform: scale(1.03); }
  .toast{
    position: fixed; left:50%; transform: translateX(-50%);
    bottom: 24px; background: var(--text); color: #06111a;
    padding: 8px 12px; border-radius: 10px; font-weight:700; display:none;
  }
</style>

<div class="wrap">
  <div class="topbar">
    <a class="back-btn" href="/">← 返回上传页</a>
    <button class="theme-btn" id="themeToggle">浅/深色</button>
  </div>

  <div class="grid">
    <!-- 左侧：考纲定位 -->
    <div class="card">
      <h2>对应考纲 <span class="badge">Top {{ syllabus_hits|length }}</span></h2>
      {% if syllabus_hits %}
        {% for hit in syllabus_hits %}
          <div class="hit" data-topic="{{ hit.topic }}" data-ref="{{ hit.syllabus_reference }}" data-book="{{ hit.book }}" data-chapter="{{ hit.chapter }}" data-pages="{{ hit.page_range }}" data-score="{{ hit.score }}">
            <div class="ring"><span>0%</span></div>

            <div>
              <div><strong>{{ hit.topic }}</strong> <span class="muted">({{ hit.syllabus_reference }})</span></div>
              <div class="meta">{{ hit.book }}，{{ hit.chapter }}，pp.{{ hit.page_range }}</div>
              <div class="meta">关键词：{{ ", ".join(hit.keywords[:4]) }}
                {% if hit.matched_subtopics %} ｜ 子条目：{{ ", ".join(hit.matched_subtopics) }}{% endif %}
              </div>
            </div>

            <div>
              <button class="copy-btn">复制引用</button>
            </div>
          </div>
        {% endfor %}
      {% else %}
        <p class="muted">⚠️ 未匹配到任何考纲条目。</p>
      {% endif %}
    </div>

    <!-- 右侧：AI 解答（Markdown + LaTeX 渲染，先打字再完整渲染） -->
    <div class="card">
      <details open>
        <summary><h2>▼ DeepSeek 解答与思路</h2></summary>
        <div id="aiTyping" class="md" style="min-height:100px; opacity:.9; font-family:'JetBrains Mono', ui-monospace;"></div>
        <div id="ai" class="md" style="display:none;"></div>
      </details>
      <h2 style="margin-top:18px;">OCR 识别的题干</h2>
      <pre class="ocr">{{ ocr_text }}</pre>
    </div>
  </div>
</div>

<div class="toast" id="toast">已复制到剪贴板</div>

<script>
  // 主题记忆
  const root = document.documentElement;
  const KEY = 'theme';
  function applyTheme(t){ root.setAttribute('data-theme', t); localStorage.setItem(KEY, t); }
  const saved = localStorage.getItem(KEY) || (matchMedia('(prefers-color-scheme: light)').matches? 'light':'dark');
  applyTheme(saved);
  document.getElementById('themeToggle').onclick = () => {
    applyTheme(root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
  };

  // 命中卡片错峰浮现
  const rows = document.querySelectorAll('.hit');
  gsap.to(rows, {opacity:1, y:0, duration:.45, ease:'power3.out', stagger:.06});

  // 分数圆环动画：按最高分归一化
  const scores = [...rows].map(r => Number(r.dataset.score)||0);
  const max = Math.max(1, ...scores);
  rows.forEach((r,i)=>{
    const p = Math.round((Number(r.dataset.score)||0) / max * 100);
    const ring = r.querySelector('.ring');
    const label = ring.querySelector('span');
    gsap.fromTo(ring, {'--p': 0}, {'--p': p, duration:.8, ease:'power2.out',
      onUpdate(){ label.textContent = Math.round(parseFloat(getComputedStyle(ring).getPropertyValue('--p'))) + '%'; }
    });
  });

  // 复制引用
  const toast = document.getElementById('toast');
  function showToast(){ toast.style.display='block'; setTimeout(()=>toast.style.display='none', 1200); }
  document.querySelectorAll('.copy-btn').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      const c = btn.closest('.hit');
      const txt = `${c.dataset.topic} (${c.dataset.ref}) — ${c.dataset.book}, ${c.dataset.chapter}, pp.${c.dataset.pages}`;
      navigator.clipboard.writeText(txt).then(()=>{
        btn.classList.add('copied'); btn.textContent = '✓ 已复制';
        showToast();
        setTimeout(()=>{ btn.classList.remove('copied'); btn.textContent = '复制引用'; }, 1000);
      });
    });
  });

  // ====== 反斜杠/转义修复 + Markdown/TeX 渲染 ======
  const RAW = {{ deepseek_answer | tojson }};

  function normalizeMath(s){
    // 1) 压平多重转义（\\\( -> \()
    s = s
      .replace(/\\\\\\\(/g, '\\(').replace(/\\\\\\\)/g, '\\)')
      .replace(/\\\\\\\[/g, '\\[').replace(/\\\\\\\]/g, '\\]')
      .replace(/\\\\\(/g, '\\(').replace(/\\\\\)/g, '\\)')
      .replace(/\\\\\[/g, '\\[').replace(/\\\\\]/g, '\\]');

    // 2) 把 \(…\)/\[…\] 兜底转换为 $…$/$$…$$
    s = s.replace(/\\\(([^\n]+?)\\\)/g, '\$$1\$');         // inline
    s = s.replace(/\\\[((?:.|\n)+?)\\\]/g, '\$\$$1\$\$');  // block

    // 3) 去掉算符前多余的反斜杠（不影响 \alpha 等命令）
    s = s.replace(/\\([=+\-*/^()])/g, '$1');
    return s;
  }

  const RAW0 = normalizeMath(RAW);

  const typing = document.getElementById('aiTyping');
  const full = document.getElementById('ai');
  const snippet = RAW0.slice(0, Math.min(800, RAW0.length)); // 先打 800 字符

  // marked 配置：避免奇怪转义
  marked.setOptions({ gfm:true, breaks:true, headerIds:false, mangle:false });

  let i=0;
  const speed = 10;
  const timer = setInterval(()=>{
    typing.textContent = snippet.slice(0, i+=3);
    if(i >= snippet.length){
      clearInterval(timer);
      full.innerHTML = marked.parse(RAW0);
      typing.style.display='none';
      full.style.display='block';
      if (window.MathJax && window.MathJax.typesetPromise) {
        if (MathJax.typesetClear) MathJax.typesetClear();
        MathJax.typesetPromise([full]);
      }
    }
  }, speed);
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

        # 2) DeepSeek：提示词规范 Markdown/TeX，禁止转义
        system_prompt = (
            "You are an expert A-Level tutor. Always respond in the user's language. "
            "Output strictly in Markdown (no code fences). "
            "Use $...$ for inline math and $$...$$ for display math. "
            "Do NOT escape backslashes; write LaTeX commands normally (e.g., \\frac, \\sqrt). "
            "Structure the solution with concise headings and steps. "
            "Avoid unnecessary prose; keep math clean."
        )
        user_prompt = f"请针对这道题目给出**答案**与**详细解题思路**，按“问题重述 / 解题思路 / 详细解答 / 检查与总结 / 最终答案”的结构输出：\n\n{ocr_text}"

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
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

        system_prompt = (
            "You are an expert A-Level tutor. Always respond in the user's language. "
            "Output strictly in Markdown (no code fences). "
            "Use $...$ for inline math and $$...$$ for display math. "
            "Do NOT escape backslashes; write LaTeX commands normally."
        )
        user_prompt = f"请针对这道题目给出答案和详细解题思路：\n{ocr_text}"

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
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
