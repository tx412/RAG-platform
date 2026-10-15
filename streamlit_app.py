"""
RAG 文档知识库处理平台 — Cobalt 版
=================================
设计规范：Hallmark「Cobalt」主题（反 AI 味 · 开发者工具气质）
流程：文档上传 → 解析预览 → 文本切块 → 向量化入库 → 知识库问答
运行：streamlit run rag_platform.py
Hallmark · pre-emit critique: P5 H4 E4 S4 R5 V3
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
import tempfile
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()   # 本地开发读 .env

# DeepSeek 密钥：云端用 st.secrets，本地回退 .env
try:
    os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    pass

# 注意：重量级库（torch/chromadb 等）已改为「懒加载」——
# 在各自函数内部 import，页面打开时不再加载，大幅提速。

def sanitize_kb_name(name):
    """把知识库名称转成 Chroma 合法的 collection 名（仅 a-zA-Z0-9._-）"""
    clean = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    clean = clean.strip('._-')
    if not clean:
        clean = "kb"
    if not clean[0].isalnum():
        clean = "kb_" + clean
    return clean[:63]

st.set_page_config(
    page_title="RAG文档知识库处理平台",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== Cobalt 设计系统 ====================
def inject_css():
    st.markdown("""
    <style>
    :root {
        --color-paper:   oklch(98.5% 0.004 250);
        --color-paper-2: oklch(96% 0.006 250);
        --color-rule:    oklch(89% 0.006 250);
        --color-ink:     oklch(24% 0.02 258);
        --color-ink-2:   oklch(34% 0.018 257);
        --color-muted:   oklch(50% 0.012 255);
        --color-accent:  oklch(58% 0.20 256);
        --color-graphite: oklch(22% 0.016 260);
    }

    html, body, .stApp {
        background: var(--color-paper);
        color: var(--color-ink-2);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--color-ink);
        letter-spacing: -0.01em;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .block-container { padding-top: 1.4rem; padding-bottom: 1rem; max-width: 1180px; }

    /* 侧边栏：冷纸面 + 发丝线 */
    .stSidebar {
        background: var(--color-paper-2);
        border-right: 1px solid var(--color-rule);
    }
    .stSidebar [data-testid="stSidebar"] > div { background: transparent; }

    /* 发丝线分节 */
    .stSidebar hr { border-color: var(--color-rule); }

    /* 按钮：mono 大写 + 6px 圆角 + 描边 */
    .stButton > button {
        border-radius: 6px;
        border: 1px solid var(--color-rule);
        background: var(--color-paper);
        color: var(--color-ink);
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 0.76rem;
        transition: border-color 0.15s;
    }
    .stButton > button:hover { border-color: var(--color-accent); }

    /* 输入框：6px 圆角 + 描边 */
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        border-radius: 6px;
        border: 1px solid var(--color-rule);
        font-family: 'Inter', sans-serif;
    }

    /* Tab：mono 大写 + 发丝线 */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid var(--color-rule);
        gap: 2.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 0.78rem;
        color: var(--color-muted);
    }
    .stTabs [aria-selected="true"] { color: var(--color-accent); }

    /* 指标数字用 mono */
    .stMetric [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: var(--color-ink);
    }

    /* 眉题（eyebrow） */
    .eyebrow {
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        color: var(--color-accent);
        margin-bottom: 6px;
    }
    /* 展示标题 */
    .display-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1.9rem;
        color: var(--color-ink);
        letter-spacing: -0.02em;
        margin: 0;
    }
    .lede { color: var(--color-ink-2); font-size: 0.92rem; margin-top: 4px; }

    /* 发丝分割线 */
    .hairline { height: 1px; background: var(--color-rule); margin: 16px 0 20px 0; }

    /* 标题前的小色块（视觉锚点） */
    .tick {
        display: inline-block;
        width: 10px; height: 10px;
        background: var(--color-accent);
        margin-right: 8px;
        border-radius: 2px;
    }

    /* 全部文字统一正常字体（微软雅黑等，去掉楷体/等宽/衬线） */
    .stApp, .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown li,
    h1, h2, h3, h4, h5, .stButton > button,
    .stTextInput input, .stTextArea textarea, .stNumberInput input,
    .stTabs [data-baseweb="tab"], .stCaption, .stChatMessage, .lede,
    .eyebrow, .stMetric [data-testid="stMetricValue"], .display-title {
        font-family: 'Microsoft YaHei', '微软雅黑', 'PingFang SC', 'Segoe UI', sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ==================== 工具函数 ====================
def parse_document(uploaded_file):
    """解析单个文档，返回 Document 列表（自动识别 PDF/Word/TXT/MD）"""
    from langchain_core.documents import Document
    from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
    name = uploaded_file.name.lower()
    tmp = None
    try:
        if name.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded_file.getvalue()); tmp = f.name
            docs = PyMuPDFLoader(tmp).load()
        elif name.endswith((".docx", ".doc")):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f:
                f.write(uploaded_file.getvalue()); tmp = f.name
            docs = Docx2txtLoader(tmp).load()
        elif name.endswith((".txt", ".md")):
            text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            docs = [Document(page_content=text)]
        else:
            raise ValueError(f"不支持的文件格式：{name}")
        for d in docs:
            d.metadata["source"] = uploaded_file.name
        return docs
    except Exception as e:
        raise RuntimeError(f"解析失败：{e}")
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)

@st.cache_resource
def get_embeddings(model_choice, api_key=""):
    """根据选择的模型返回 embedding 对象（本地免费 / API 需密钥）"""
    # 本地模型：HuggingFace 下载（首次较慢，之后缓存）
    local_map = {
        "BGE-small 中文（本地，快）": "BAAI/bge-small-zh-v1.5",
        "BGE-large 中文（本地，准）": "BAAI/bge-large-zh-v1.5",
        "BGE-M3 多语言（本地）": "BAAI/bge-m3",
    }
    if model_choice in local_map:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=local_map[model_choice])
    # 通义千问（阿里 DashScope）
    if "通义" in model_choice:
        from langchain_community.embeddings import DashScopeEmbeddings
        return DashScopeEmbeddings(model="text-embedding-v3", dashscope_api_key=api_key)
    # 智谱 AI
    if "智谱" in model_choice:
        from langchain_community.embeddings import ZhipuAIEmbeddings
        return ZhipuAIEmbeddings(model="embedding-3", api_key=api_key)
    # 硅基流动（SiliconFlow，OpenAI 兼容）
    if "硅基流动" in model_choice:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="BAAI/bge-large-zh-v1.5", api_key=api_key,
                                base_url="https://api.siliconflow.cn/v1")
    # MiniMax
    if "MiniMax" in model_choice:
        from langchain_community.embeddings import MiniMaxEmbeddings
        return MiniMaxEmbeddings(model="embo-01", api_key=api_key)
    # OpenAI
    if "OpenAI" in model_choice:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    raise ValueError(f"未知的 embedding 模型：{model_choice}")

def split_documents(docs, chunk_size, chunk_overlap):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(docs)

def build_vectorstore(chunks, embeddings, kb_name):
    from langchain_chroma import Chroma
    import chromadb
    # 先删除同名旧集合：切换 embedding 模型后维度会变（如 512 → 2048），
    # 不删旧集合会报 "expecting dimension ... got ..."
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        client.delete_collection(kb_name)
    except Exception:
        pass
    return Chroma.from_documents(chunks, embeddings, collection_name=kb_name,
                                 persist_directory="./chroma_db")

def rag_ask(vectorstore, question, k=3):
    from langchain_deepseek import ChatDeepSeek
    from langchain_core.prompts import ChatPromptTemplate
    results = vectorstore.similarity_search_with_score(question, k=k)
    context = "\n\n".join(doc.page_content for doc, _ in results)
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是企业知识库问答助手。请严格依据资料回答，资料未提及就说明'资料中未提及'，不要编造。\n\n资料：\n{context}"),
        ("human", "{question}"),
    ])
    answer = (prompt | llm).invoke({"context": context, "question": question}).content
    sources = [(d.metadata.get("source", "未知"), d.page_content, round(1 - s, 4)) for d, s in results]
    return answer, sources

# ==================== 会话状态 ====================
if "parsed_docs" not in st.session_state:
    st.session_state.parsed_docs = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat" not in st.session_state:
    st.session_state.chat = []

# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown('<div class="eyebrow">Workspace</div>', unsafe_allow_html=True)
    kb_name = st.text_input("知识库名称", value="default_kb")

    st.markdown('<div class="eyebrow">Ingest</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "PDF / Word / TXT / Markdown（可多选）",
        type=["pdf", "docx", "doc", "txt", "md"],
        accept_multiple_files=True,
    )

    st.markdown('<div class="eyebrow">Chunking</div>', unsafe_allow_html=True)
    chunk_size = st.number_input("chunk_size", 50, 2000, 300, step=50)
    chunk_overlap = st.number_input("chunk_overlap", 0, 500, 50, step=10)

    st.markdown('<div class="eyebrow">Embedding</div>', unsafe_allow_html=True)
    embed_choice = st.selectbox("模型", [
        "BGE-small 中文（本地，快）",
        "BGE-large 中文（本地，准）",
        "BGE-M3 多语言（本地）",
        "通义 text-embedding-v3（阿里）",
        "智谱 embedding-3",
        "硅基流动 BGE-large-zh",
        "MiniMax embo-01",
        "OpenAI text-embedding-3-small",
    ])
    api_key = st.text_input("API 密钥（选 API 模型时填写对应平台的 key）", type="password")

    st.markdown('<div class="eyebrow">Pipeline</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    btn_parse = col_a.button("① 解析", use_container_width=True)
    btn_split = col_b.button("② 切块", use_container_width=True)
    btn_vec = st.button("③ 向量化", use_container_width=True)
    btn_clear = st.button("清空知识库", use_container_width=True)

# 侧边栏逻辑
if btn_parse:
    if not uploaded_files:
        st.sidebar.error("请先上传文档")
    else:
        with st.spinner("解析中…"):
            all_docs, errors = [], []
            for f in uploaded_files:
                try:
                    all_docs.extend(parse_document(f))
                except Exception as e:
                    errors.append(f"{f.name}: {e}")
            st.session_state.parsed_docs = all_docs
            if errors:
                st.sidebar.error("；".join(errors))
            if all_docs:
                st.sidebar.success(f"解析 {len(all_docs)} 片段")

if btn_split:
    if not st.session_state.parsed_docs:
        st.sidebar.error("请先解析文档")
    else:
        with st.spinner("切分中…"):
            st.session_state.chunks = split_documents(st.session_state.parsed_docs, chunk_size, chunk_overlap)
        st.sidebar.success(f"切分 {len(st.session_state.chunks)} 块")

if btn_vec:
    if not st.session_state.chunks:
        st.sidebar.error("请先执行切块")
    elif any(m in embed_choice for m in ["通义", "智谱", "硅基流动", "MiniMax", "OpenAI"]) and not api_key:
        st.sidebar.error("API 模型需填写对应平台的密钥")
    else:
        with st.spinner("向量化中…"):
            emb = get_embeddings(embed_choice, api_key)
            st.session_state.vectorstore = build_vectorstore(st.session_state.chunks, emb, sanitize_kb_name(kb_name))
        st.sidebar.success("向量化完成")

if btn_clear:
    if st.session_state.vectorstore:
        try:
            st.session_state.vectorstore.delete_collection()
        except Exception:
            pass
        st.session_state.vectorstore = None
        st.session_state.chunks = []
        st.session_state.parsed_docs = []
    st.sidebar.success("知识库已清空")

# ==================== 头部 ====================
st.markdown("""
<div style="background: var(--color-graphite); border-radius: 12px; padding: 30px 34px; margin-bottom: 22px;">
    <div class="eyebrow" style="color: var(--color-accent);">RAG · Document Knowledge Base</div>
    <h1 class="display-title" style="color: oklch(95% 0.006 80); margin: 6px 0;">文档知识库处理平台</h1>
    <p style="color: oklch(80% 0.01 250); font-size: 1rem; line-height: 1.7; margin: 10px 0 0 0;">
        上传企业文档，自动完成「解析 → 切块 → 向量化入库」，基于检索增强生成（RAG）实现精准问答与来源溯源。
        支持 PDF、Word、TXT、Markdown 多格式，向量库本地运行、零外部依赖。
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["上传与预览", "切块管理", "向量知识库", "问答"])

# ---------- Tab1 ----------
with tab1:
    if not st.session_state.parsed_docs:
        st.info("在左侧上传文档并点击「① 解析」。")
        with st.container(border=True):
            if st.button("加载内置示例文档"):
                from langchain_core.documents import Document
                sample = ("星云科技有限公司员工手册\n\n第一章 报销制度\n员工报销需在30天内提交并附发票。"
                          "单笔500元以内由部门主管审批，500至2000元由部门经理审批，超过2000元需总经理审批。\n\n"
                          "第二章 请假制度\n年假每年10天，病假需提供医院证明。\n\n"
                          "第三章 加班制度\n工作日加班可调休，周末1.5倍工资，节假日3倍工资，餐补每晚30元。")
                st.session_state.parsed_docs = [Document(page_content=sample, metadata={"source": "示例文档.txt"})]
                st.rerun()
    else:
        st.markdown('<div class="eyebrow">Parsed documents</div>', unsafe_allow_html=True)
        rows = [{"序号": i + 1, "来源文件": d.metadata.get("source", "未知"),
                 "片段长度": len(d.page_content)} for i, d in enumerate(st.session_state.parsed_docs)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown('<div class="eyebrow">Preview</div>', unsafe_allow_html=True)
        doc_idx = st.selectbox("选择片段", range(len(st.session_state.parsed_docs)),
                               format_func=lambda i: f"{i+1} · {st.session_state.parsed_docs[i].metadata.get('source','')}")
        st.text_area("原文", st.session_state.parsed_docs[doc_idx].page_content, height=300)

# ---------- Tab2 ----------
with tab2:
    if not st.session_state.chunks:
        st.info("尚未切分。")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("文档片段", len(st.session_state.parsed_docs))
        c2.metric("文本块", len(st.session_state.chunks))
        c3.metric("总字符", sum(len(c.page_content) for c in st.session_state.chunks))

        st.markdown('<div class="eyebrow">Chunks</div>', unsafe_allow_html=True)
        kw = st.text_input("筛选关键词")
        data = []
        for i, c in enumerate(st.session_state.chunks):
            if kw and kw not in c.page_content:
                continue
            data.append({"序号": i + 1, "片段文本": c.page_content, "长度": len(c.page_content)})
        st.dataframe(pd.DataFrame(data), use_container_width=True, height=400)

# ---------- Tab3 ----------
with tab3:
    vs = st.session_state.vectorstore
    if vs is None:
        st.info("向量库为空。")
    else:
        c1, c2 = st.columns(2)
        c1.metric("知识库", kb_name)
        c2.metric("向量数", vs._collection.count() if hasattr(vs, "_collection") else "—")

        st.markdown('<div class="eyebrow">Retrieval test</div>', unsafe_allow_html=True)
        query = st.text_input("查询文本", placeholder="例如：报销超过2000元需要谁审批")
        top_k = st.slider("Top N", 1, 8, 3)
        if st.button("检索") and query:
            with st.spinner("检索中…"):
                results = vs.similarity_search_with_score(query, k=top_k)
            for i, (d, s) in enumerate(results):
                st.markdown(f"**{i+1}** · 相似度 `{round(1-s,4)}` · {d.metadata.get('source','未知')}")
                st.caption(d.page_content[:200])
                st.divider()

        if st.button("清空知识库", type="secondary"):
            try:
                vs.delete_collection()
            except Exception:
                pass
            st.session_state.vectorstore = None
            st.session_state.chunks = []
            st.session_state.parsed_docs = []
            st.rerun()

# ---------- Tab4 ----------
with tab4:
    for q, a, srcs in st.session_state.chat:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
            if srcs:
                st.caption("引用来源")
                for s_name, s_text, s_score in srcs:
                    st.caption(f"· {s_name}（{s_score}）：{s_text[:60]}…")

    question = st.chat_input("输入问题…")
    if question:
        vs = st.session_state.vectorstore
        if vs is None:
            st.error("向量库为空，请先入库")
        else:
            with st.spinner("检索并生成…"):
                answer, sources = rag_ask(vs, question)
            st.session_state.chat.append((question, answer, sources))
            st.rerun()

    if st.session_state.chat and st.button("清空对话"):
        st.session_state.chat = []
        st.rerun()
