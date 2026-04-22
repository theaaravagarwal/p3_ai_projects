#!/usr/bin/env python3
"""
FastAPI web server for Assistant AP Physics 2 assistant
This provides a REST API endpoint for deployment
"""

import os
from typing import List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.openai_config import (
    get_active_client,
    get_active_model,
    get_completion_token_param,
    get_request_timeout_seconds,
    get_reasoning_effort,
    get_temperature_param,
    has_active_api_key,
    supports_reasoning_for_active_model,
    trim_conversation_history,
)
from src.system_prompt import SYSTEM_PROMPT
from src.security import (
    check_public_api_key,
    check_rate_limit,
    get_allowed_origins,
    validate_chat_payload,
)

PUBLIC_UI_ONLY = os.getenv("PUBLIC_UI_ONLY", "false").strip().lower() in {"1", "true", "yes", "on"}

# Initialize FastAPI app
app = FastAPI(
    title="Assistant",
    description="A low-cost AP Physics 2 assistant",
    version="1.0.0",
    docs_url=None if PUBLIC_UI_ONLY else "/docs",
    redoc_url=None if PUBLIC_UI_ONLY else "/redoc",
    openapi_url=None if PUBLIC_UI_ONLY else "/openapi.json",
)

# Add CORS middleware for web deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# Initialize OpenAI client
client = get_active_client()


@app.middleware("http")
async def public_api_controls(request: Request, call_next):
    if request.url.path in {"/chat", "/reset"}:
        check_public_api_key(request)
        check_rate_limit(request, scope=request.url.path)

    response = await call_next(request)
    return response

# Data models
class Message(BaseModel):
    """A single message in the conversation"""
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    """Request body for chat endpoint"""
    message: str
    conversation_history: List[Message] = []

class ChatResponse(BaseModel):
    """Response body for chat endpoint"""
    message: str
    conversation_history: List[Message]

# Chatbot configuration
CHATBOT_CONFIG = {
    "name": "Assistant",
    "avatar": "💬",
    "description": "A low-cost AP Physics 2 assistant",
    "version": "1.0.0"
}

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - returns the public chat UI."""
    docs_link = "" if PUBLIC_UI_ONLY else '<a href="/docs">OpenAPI Docs</a>'
    public_key_required = os.getenv("PUBLIC_API_KEY", "").strip() not in {"", "your_public_api_key_here", "change-me"}
    api_key_note = "Required" if public_key_required else "Optional"
    page = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Assistant</title>
    <style>
      :root {
        color-scheme: light;
        --bg: #edf1f7;
        --shell: rgba(248, 250, 252, 0.82);
        --panel: rgba(255, 255, 255, 0.92);
        --panel-soft: rgba(244, 247, 251, 0.88);
        --ink: #0f172a;
        --muted: #64748b;
        --subtle: #94a3b8;
        --border: #d6deea;
        --border-strong: #b8c4d6;
        --accent: #2563eb;
        --accent-soft: rgba(37, 99, 235, 0.12);
        --user: #2563eb;
        --assistant: #ffffff;
        --shadow: 0 20px 60px rgba(15, 23, 42, 0.08);
      }
      * { box-sizing: border-box; }
      html, body { height: 100%; }
      body {
        margin: 0;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background:
          radial-gradient(circle at top left, rgba(37, 99, 235, 0.12), transparent 28%),
          radial-gradient(circle at top right, rgba(15, 23, 42, 0.08), transparent 24%),
          linear-gradient(180deg, #f8fafc 0%, #e9eef6 100%);
        color: var(--ink);
      }
      .wrap {
        min-height: 100vh;
        max-width: 1600px;
        margin: 0 auto;
        padding: 20px;
      }
      .shell {
        min-height: calc(100vh - 40px);
        background: var(--shell);
        border: 1px solid var(--border);
        border-radius: 28px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(16px);
        overflow: hidden;
        display: grid;
        grid-template-columns: 280px minmax(0, 1fr) 320px;
      }
      .sidebar, .main, .debug {
        min-width: 0;
        background: var(--panel);
      }
      .sidebar {
        border-right: 1px solid var(--border);
        display: grid;
        grid-template-rows: auto auto 1fr auto;
        padding: 18px;
        gap: 16px;
      }
      .main {
        border-right: 1px solid var(--border);
        display: grid;
        grid-template-rows: auto 1fr auto;
        min-height: 0;
      }
      .debug {
        display: grid;
        grid-template-rows: auto 1fr;
        padding: 18px;
        gap: 16px;
      }
      .brand {
        display: flex;
        flex-direction: column;
        gap: 12px;
        padding-bottom: 6px;
      }
      .brand h1 {
        margin: 0;
        font-size: 1.45rem;
        line-height: 1.1;
        letter-spacing: -0.04em;
      }
      .brand p, .section p, .meta p {
        margin: 0;
        line-height: 1.5;
      }
      .pill {
        display: inline-flex;
        align-items: center;
        width: fit-content;
        gap: 8px;
        padding: 7px 11px;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-weight: 700;
        font-size: 0.82rem;
      }
      .card {
        border: 1px solid var(--border);
        border-radius: 18px;
        background: var(--panel-soft);
        padding: 14px;
      }
      .section {
        display: grid;
        gap: 10px;
      }
      .section h2, .section h3 {
        margin: 0;
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--muted);
      }
      .meta {
        display: grid;
        gap: 6px;
      }
      .meta-row {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        font-size: 0.93rem;
      }
      .meta-row span:last-child {
        color: var(--ink);
        text-align: right;
        word-break: break-word;
      }
      .link-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }
      .link-row a, .ghost, .session-item, .toolbar button {
        border: 1px solid var(--border);
        border-radius: 12px;
        background: #fff;
        color: var(--ink);
        text-decoration: none;
        font: inherit;
      }
      .link-row a {
        padding: 8px 10px;
      }
      .link-row a:hover, .ghost:hover, .session-item:hover, .toolbar button:hover {
        border-color: var(--border-strong);
      }
      .ghost, .toolbar button {
        padding: 9px 12px;
        cursor: pointer;
      }
      .sidebar-top, .composer-wrap, .debug-footer {
        display: grid;
        gap: 12px;
      }
      .sessions {
        display: grid;
        gap: 10px;
        overflow: auto;
        padding-right: 4px;
      }
      .session-item {
        width: 100%;
        text-align: left;
        padding: 12px;
        cursor: pointer;
        display: grid;
        gap: 4px;
        background: #fff;
      }
      .session-item.active {
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent);
      }
      .session-title {
        font-weight: 700;
        font-size: 0.95rem;
      }
      .session-meta {
        font-size: 0.82rem;
        color: var(--muted);
      }
      .main-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 16px;
        padding: 18px 20px;
        border-bottom: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.72);
      }
      .main-title {
        display: grid;
        gap: 6px;
      }
      .main-title h2 {
        margin: 0;
        font-size: 1.1rem;
        letter-spacing: -0.02em;
      }
      .main-title p {
        color: var(--muted);
      }
      .toolbar {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }
      .conversation {
        min-height: 0;
        overflow: auto;
        padding: 18px 20px;
        display: grid;
        align-content: start;
        gap: 12px;
      }
      .empty-state {
        margin: auto;
        max-width: 520px;
        width: 100%;
        text-align: center;
        padding: 40px 22px;
        border: 1px dashed var(--border-strong);
        border-radius: 20px;
        color: var(--muted);
        background: rgba(255, 255, 255, 0.55);
      }
      .bubble {
        max-width: min(760px, 88%);
        padding: 14px 16px;
        border-radius: 18px;
        line-height: 1.55;
        white-space: pre-wrap;
        box-shadow: 0 1px 0 rgba(15, 23, 42, 0.03);
        border: 1px solid transparent;
      }
      .bubble.user {
        margin-left: auto;
        background: linear-gradient(180deg, #2f6fed 0%, #2259cf 100%);
        color: #fff;
        border-top-right-radius: 8px;
      }
      .bubble.assistant {
        margin-right: auto;
        background: #fff;
        border-color: var(--border);
        border-top-left-radius: 8px;
      }
      .bubble.system {
        margin-right: auto;
        background: #f8fafc;
        border-color: #e2e8f0;
        color: var(--muted);
      }
      .composer-wrap {
        border-top: 1px solid var(--border);
        padding: 16px 20px 20px;
        background: rgba(255, 255, 255, 0.76);
      }
      .composer {
        display: grid;
        gap: 10px;
      }
      textarea, input {
        width: 100%;
        border-radius: 16px;
        border: 1px solid var(--border);
        background: #fff;
        color: var(--ink);
        font: inherit;
        padding: 13px 14px;
      }
      textarea {
        min-height: 88px;
        resize: vertical;
      }
      textarea:focus, input:focus, button:focus, .session-item:focus {
        outline: none;
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.14);
      }
      .submit-row {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        align-items: center;
      }
      .hint {
        color: var(--muted);
        font-size: 0.9rem;
      }
      .primary {
        border: none;
        background: var(--accent);
        color: white;
        font-weight: 700;
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.22);
      }
      .primary:hover {
        background: #1d4ed8;
      }
      .secondary {
        background: #fff;
      }
      .debug-log {
        min-height: 0;
        overflow: auto;
        display: grid;
        gap: 12px;
      }
      .debug-box {
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px;
        background: #fff;
      }
      .debug-box pre {
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        color: var(--ink);
        font-size: 0.88rem;
        line-height: 1.5;
      }
      .debug-title {
        margin-bottom: 8px;
        color: var(--muted);
        font-size: 0.8rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }
      .tiny {
        font-size: 0.86rem;
        color: var(--muted);
      }
      .divider {
        height: 1px;
        background: var(--border);
      }
      .kbd {
        border: 1px solid var(--border);
        border-bottom-width: 2px;
        border-radius: 8px;
        padding: 1px 6px;
        background: #fff;
        color: var(--muted);
        font-size: 0.82rem;
      }
      @media (max-width: 1180px) {
        .shell { grid-template-columns: 260px minmax(0, 1fr); }
        .debug { grid-column: 1 / -1; border-top: 1px solid var(--border); }
      }
      @media (max-width: 840px) {
        .wrap { padding: 0; }
        .shell {
          min-height: 100vh;
          border-radius: 0;
          grid-template-columns: 1fr;
        }
        .sidebar, .main, .debug {
          border-right: none;
          border-bottom: 1px solid var(--border);
        }
        .conversation {
          min-height: 48vh;
        }
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="shell">
        <aside class="sidebar">
          <div class="brand">
            <span class="pill">AP Physics 2 Assistant</span>
            <div>
              <h1>AP Physics 2 study chat</h1>
              <p class="tiny">Minimal UI. Sessions on the left, active chat in the center, debug on the right.</p>
            </div>
          </div>
          <div class="section card">
            <h2>Account</h2>
            <div class="meta">
              <div class="meta-row"><span>API key</span><span>__API_KEY_NOTE__</span></div>
              <div class="meta-row"><span>Model</span><span>gpt-5-nano</span></div>
              <div class="meta-row"><span>Max output</span><span>64 tokens</span></div>
            </div>
          </div>
          <div class="section">
            <h2>Chats</h2>
            <div class="toolbar">
              <button id="newChatBtn" class="ghost">New chat</button>
              <button id="exportBtn" class="ghost">Export</button>
            </div>
            <div id="sessions" class="sessions"></div>
          </div>
          <div class="section card">
            <h2>Links</h2>
            <div class="link-row">
              __DOCS_LINK__
              <a href="/info">Info JSON</a>
              <a href="/health">Health</a>
            </div>
          </div>
        </aside>

        <main class="main">
          <div class="main-header">
            <div class="main-title">
              <h2 id="chatTitle">New chat</h2>
              <p id="chatSubtitle">The thread history is local to this browser tab unless you export it.</p>
            </div>
            <div class="toolbar">
              <button id="renameBtn" class="ghost">Rename</button>
              <button id="clearBtn" class="ghost">Clear thread</button>
            </div>
          </div>
          <div id="conversation" class="conversation"></div>
          <div class="composer-wrap">
            <div class="composer">
              <textarea id="message" placeholder="Ask something..."></textarea>
              <div class="submit-row">
                <div class="hint">Send with <span class="kbd">Ctrl</span> + <span class="kbd">Enter</span></div>
                <div class="toolbar">
                  <button id="sendBtn" class="primary">Send</button>
                </div>
              </div>
            </div>
          </div>
        </main>

        <aside class="debug">
          <div class="section">
            <h2>Debug</h2>
            <div class="card meta">
              <div class="meta-row"><span>Status</span><span id="status">Ready</span></div>
              <div class="meta-row"><span>Session</span><span id="sessionCount">0 messages</span></div>
              <div class="meta-row"><span>Knowledge mode</span><span>system prompt + thread only</span></div>
              <div class="meta-row"><span>History sent</span><span>current thread only</span></div>
            </div>
          </div>
          <div class="debug-log">
            <div class="debug-box">
              <div class="debug-title">Request payload</div>
              <pre id="requestPayload">Waiting for a send action.</pre>
            </div>
            <div class="debug-box">
              <div class="debug-title">Response payload</div>
              <pre id="responsePayload">No response yet.</pre>
            </div>
            <div class="debug-box">
              <div class="debug-title">Context note</div>
              <pre>Each send call includes only system instructions and the current thread conversation history.</pre>
            </div>
          </div>
          <div class="debug-footer card">
            <div class="section">
              <h3>Public API</h3>
              <p class="tiny">Optional `X-API-Key` can be entered in the browser if the server expects one.</p>
            </div>
            <input id="apiKey" type="password" placeholder="PUBLIC_API_KEY" />
          </div>
        </div>
      </div>
    </div>
    <script>
      const sessionsEl = document.getElementById('sessions');
      const conversationEl = document.getElementById('conversation');
      const sendBtn = document.getElementById('sendBtn');
      const newChatBtn = document.getElementById('newChatBtn');
      const exportBtn = document.getElementById('exportBtn');
      const renameBtn = document.getElementById('renameBtn');
      const clearBtn = document.getElementById('clearBtn');
      const status = document.getElementById('status');
      const requestPayload = document.getElementById('requestPayload');
      const responsePayload = document.getElementById('responsePayload');
      const chatTitle = document.getElementById('chatTitle');
      const chatSubtitle = document.getElementById('chatSubtitle');
      const sessionCount = document.getElementById('sessionCount');
      const messageInput = document.getElementById('message');
      const apiKeyInput = document.getElementById('apiKey');
      const STORAGE_KEY = 'assistant.sessions.v1';
      const ACTIVE_KEY = 'assistant.activeSession.v1';
      const DEFAULT_NAME = 'New chat';
      let sessions = loadSessions();
      let activeSessionId = localStorage.getItem(ACTIVE_KEY) || (sessions[0] && sessions[0].id) || null;

      function nowLabel(ts) {
        return new Date(ts).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
      }

      function newSession(name = DEFAULT_NAME) {
        return {
          id: `chat-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
          name,
          createdAt: Date.now(),
          updatedAt: Date.now(),
          messages: [],
        };
      }

      function loadSessions() {
        try {
          const raw = localStorage.getItem(STORAGE_KEY);
          const parsed = raw ? JSON.parse(raw) : [];
          if (Array.isArray(parsed) && parsed.length) {
            return parsed;
          }
        } catch (err) {
          console.warn(err);
        }
        return [newSession('Start here')];
      }

      function saveSessions() {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
        if (activeSessionId) {
          localStorage.setItem(ACTIVE_KEY, activeSessionId);
        }
      }

      function activeSession() {
        let session = sessions.find((item) => item.id === activeSessionId);
        if (!session) {
          session = sessions[0];
          activeSessionId = session.id;
        }
        return session;
      }

      function setActiveSession(id) {
        activeSessionId = id;
        saveSessions();
        renderAll();
      }

      function formatCount(messages) {
        const messageCount = messages.length;
        if (messageCount === 0) return 'empty';
        const userTurns = messages.filter((message) => message.role === 'user').length;
        return `${messageCount} messages, ${userTurns} prompts`;
      }

      function renderSessions() {
        sessionsEl.innerHTML = '';
        const sorted = [...sessions].sort((a, b) => b.updatedAt - a.updatedAt);
        for (const session of sorted) {
          const button = document.createElement('button');
          button.className = `session-item${session.id === activeSessionId ? ' active' : ''}`;
          button.type = 'button';
          button.innerHTML = `
            <div class="session-title">${escapeHtml(session.name)}</div>
            <div class="session-meta">${formatCount(session.messages)} · ${nowLabel(session.updatedAt)}</div>
          `;
          button.addEventListener('click', () => setActiveSession(session.id));
          sessionsEl.appendChild(button);
        }
      }

      function renderConversation() {
        const session = activeSession();
        conversationEl.innerHTML = '';
        if (!session.messages.length) {
          conversationEl.innerHTML = `
            <div class="empty-state">
              <div class="pill" style="margin: 0 auto 12px;">Ready to chat</div>
              <h3 style="margin: 0 0 8px; font-size: 1.2rem; color: var(--ink);">Start a thread</h3>
              <p>This panel only shows the current chat. Use the sidebar to switch between local browser sessions.</p>
            </div>
          `;
          return;
        }
        for (const msg of session.messages) {
          const bubble = document.createElement('div');
          bubble.className = `bubble ${msg.role}`;
          bubble.textContent = msg.content;
          conversationEl.appendChild(bubble);
        }
        conversationEl.scrollTop = conversationEl.scrollHeight;
      }

      function renderDebug() {
        const session = activeSession();
        chatTitle.textContent = session.name;
        chatSubtitle.textContent = session.messages.length
          ? `This thread has ${formatCount(session.messages)}. History is only sent for the active session.`
          : 'This thread is empty. Messages only live in this browser unless exported.';
        status.textContent = session.messages.length ? 'Active' : 'Ready';
        sessionCount.textContent = formatCount(session.messages);
        requestPayload.textContent = JSON.stringify({
          message: messageInput.value.trim() || '',
          conversation_history: session.messages,
        }, null, 2);
      }

      function renderAll() {
        renderSessions();
        renderConversation();
        renderDebug();
      }

      function escapeHtml(value) {
        return String(value)
          .replaceAll('&', '&amp;')
          .replaceAll('<', '&lt;')
          .replaceAll('>', '&gt;')
          .replaceAll('"', '&quot;')
          .replaceAll("'", '&#39;');
      }

      function setStatus(text) {
        status.textContent = text;
      }

      function ensureSession() {
        if (!sessions.length) {
          const session = newSession();
          sessions = [session];
          activeSessionId = session.id;
        }
      }

      function startNewChat() {
        const session = newSession();
        sessions.push(session);
        activeSessionId = session.id;
        saveSessions();
        renderAll();
        messageInput.focus();
      }

      function clearActiveChat() {
        const session = activeSession();
        session.messages = [];
        session.updatedAt = Date.now();
        saveSessions();
        renderAll();
        setStatus('Thread cleared');
        responsePayload.textContent = 'Conversation cleared.';
      }

      function renameActiveChat() {
        const session = activeSession();
        const nextName = prompt('Rename this chat', session.name);
        if (!nextName) return;
        session.name = nextName.trim().slice(0, 48) || DEFAULT_NAME;
        session.updatedAt = Date.now();
        saveSessions();
        renderAll();
      }

      function exportChats() {
        const blob = new Blob([JSON.stringify(sessions, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'assistant-chats.json';
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }

      async function sendMessage() {
        const message = messageInput.value.trim();
        const apiKey = apiKeyInput.value.trim();
        if (!message) {
          setStatus('Enter a message first.');
          return;
        }
        ensureSession();
        const session = activeSession();
        setStatus('Sending...');
        requestPayload.textContent = JSON.stringify({
          message,
          conversation_history: session.messages,
        }, null, 2);
        responsePayload.textContent = 'Sending...';
        const headers = { 'Content-Type': 'application/json' };
        if (apiKey) headers['X-API-Key'] = apiKey;
        try {
          const res = await fetch('/chat', {
            method: 'POST',
            headers,
            body: JSON.stringify({ message, conversation_history: session.messages }),
          });
          const data = await res.json();
          if (!res.ok) {
            throw new Error(data.detail || 'Request failed');
          }
          session.messages = data.conversation_history || [];
          session.updatedAt = Date.now();
          if (session.name === DEFAULT_NAME && session.messages.length) {
            session.name = session.messages[0].content.slice(0, 28) || DEFAULT_NAME;
          }
          saveSessions();
          responsePayload.textContent = JSON.stringify(data, null, 2);
          messageInput.value = '';
          setStatus('Ready');
          renderAll();
        } catch (err) {
          responsePayload.textContent = String(err);
          setStatus('Error');
        }
      }

      newChatBtn.addEventListener('click', startNewChat);
      exportBtn.addEventListener('click', exportChats);
      renameBtn.addEventListener('click', renameActiveChat);
      sendBtn.addEventListener('click', sendMessage);
      clearBtn.addEventListener('click', () => {
        clearActiveChat();
      });
      messageInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
          event.preventDefault();
          sendMessage();
        }
      });
      ensureSession();
      renderAll();
    </script>
  </body>
</html>"""
    return (
        page.replace("__DOCS_LINK__", docs_link).replace("__API_KEY_NOTE__", api_key_note)
    )

@app.get("/info")
async def info():
    """Get chatbot information"""
    return CHATBOT_CONFIG


@app.get("/health")
async def health():
    """Simple health endpoint for deployment checks."""
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with Assistant
    
    Args:
        request: ChatRequest with message and conversation history
    
    Returns:
        ChatResponse with assistant message and updated history
    """
    validate_chat_payload(request.message, len(request.conversation_history))
    
    if not has_active_api_key():
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    try:
        # Convert Pydantic models to dicts for OpenAI API
        conversation = [{"role": m.role, "content": m.content} for m in request.conversation_history]
        conversation = trim_conversation_history(conversation)
        
        # Add user message
        conversation.append({"role": "user", "content": request.message.strip()})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(conversation)
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=get_active_model(),
            messages=messages,
            timeout=get_request_timeout_seconds(),
            **get_completion_token_param(get_active_model()),
            **get_temperature_param(get_active_model()),
            **(
                {"reasoning_effort": get_reasoning_effort()}
                if supports_reasoning_for_active_model(get_active_model())
                else {}
            ),
        )
        
        assistant_message = response.choices[0].message.content
        
        # Update conversation history
        conversation.append({"role": "assistant", "content": assistant_message})
        
        # Convert back to Message objects
        history = [Message(role=m["role"], content=m["content"]) for m in conversation]
        
        return ChatResponse(
            message=assistant_message,
            conversation_history=history
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/reset")
async def reset_conversation(request: Request):
    """Reset conversation history"""
    return {
        "status": "conversation reset",
        "message": "Start a new conversation with Assistant!"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Check for API key
    if not has_active_api_key():
        print("Error: OPENAI_API_KEY environment variable not set!")
        print("Please create a .env file with: OPENAI_API_KEY=your_key_here")
        exit(1)
    
    print(f"Starting {CHATBOT_CONFIG['name']} 💬 on http://0.0.0.0:8000")
    print("API docs available at http://0.0.0.0:8000/docs")
    
    # Run with: uvicorn src.api:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
