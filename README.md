# A-Level OCR → DeepSeek → Syllabus 定位（MVP）

## 功能
- 上传题图 → 本地 OCR（Tesseract）
- DeepSeek Chat 生成解答与思路
- 用 `Syllabus_data.json` 做关键词匹配，返回考纲条目（含教材页码）
- 网页优先展示考纲（分数+关键词），AI 解答用折叠，底部显示 OCR 原文
- 另有 JSON API：`POST /api/upload`

## 环境要求
- Python 3.10+
- macOS 建议先安装 Tesseract：
  ```bash
  brew install tesseract
