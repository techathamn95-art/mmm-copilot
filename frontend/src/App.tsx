import { useState } from 'react';
import { ChatPanel } from './components/ChatPanel';
import { DataSidebar } from './components/DataSidebar';
import { UploadPanel } from './components/UploadPanel';
import { streamChat, uploadCsv } from './lib/api';
import { ColumnMapping, Message, SummaryStats } from './lib/types';

function makeMessage(role: Message['role'], content: string, toolResults = []): Message {
  return {
    id: crypto.randomUUID(),
    role,
    content,
    createdAt: new Date().toISOString(),
    toolResults,
  };
}

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [summary, setSummary] = useState<SummaryStats | null>(null);
  const [messages, setMessages] = useState<Message[]>([
    makeMessage(
      'assistant',
      'Upload a CSV and I will fit a baseline marketing mix model, estimate ROAS by channel, and explore budget scenarios.',
    ),
  ]);
  const [isUploading, setIsUploading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([
    'Analyze my marketing spend',
    'Optimize my budget for $100K/month',
    'What if I double TikTok spend?',
  ]);

  async function handleUpload(file: File, mapping: ColumnMapping) {
    setIsUploading(true);
    try {
      const payload = await uploadCsv(file, mapping);
      setSessionId(payload.session_id);
      setSummary(payload.summary);
      setMessages((current) => [
        ...current,
        makeMessage(
          'system',
          `Loaded ${payload.file_name} with ${payload.summary.rows.toLocaleString()} rows across ${payload.summary.channels} channels.`,
        ),
      ]);
    } finally {
      setIsUploading(false);
    }
  }

  async function handleSendMessage(content: string) {
    if (!sessionId) return;
    setIsStreaming(true);
    const userMessage = makeMessage('user', content);
    const assistantMessageId = crypto.randomUUID();
    setMessages((current) => [...current, userMessage, { ...makeMessage('assistant', ''), id: assistantMessageId }]);

    await streamChat(sessionId, content, {
      onDelta: (chunk) => {
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantMessageId ? { ...message, content: `${message.content}${chunk}` } : message,
          ),
        );
      },
      onComplete: (payload) => {
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantMessageId
              ? { ...message, content: payload.content, toolResults: payload.tool_results }
              : message,
          ),
        );
        setSuggestions(payload.suggested_prompts);
      },
    }).finally(() => setIsStreaming(false));
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand__mark">MMM</span>
          <div>
            <h1>Marketing Mix Modeling</h1>
            <p>Upload, analyze, optimize, forecast.</p>
          </div>
        </div>
        <UploadPanel isUploading={isUploading} onUpload={handleUpload} />
        <DataSidebar summary={summary} />
      </aside>
      <main className="main-panel">
        <ChatPanel disabled={!sessionId || isStreaming} messages={messages} suggestions={suggestions} onSendMessage={handleSendMessage} />
      </main>
    </div>
  );
}
