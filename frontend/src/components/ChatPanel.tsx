import { SendHorizonal, Sparkles } from 'lucide-react';
import { FormEvent, useEffect, useRef, useState } from 'react';
import { Message, ToolResult } from '../lib/types';
import { ChartCard } from './ChartCard';

interface ChatPanelProps {
  disabled: boolean;
  messages: Message[];
  suggestions: string[];
  onSendMessage: (message: string) => Promise<void>;
}

export function ChatPanel({ disabled, messages, suggestions, onSendMessage }: ChatPanelProps) {
  const [draft, setDraft] = useState('');
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!draft.trim()) return;
    const next = draft;
    setDraft('');
    await onSendMessage(next);
  }

  return (
    <section className="chat-shell">
      <header className="chat-header">
        <div>
          <span className="eyebrow">AI Agent</span>
          <h2>MMM Copilot</h2>
        </div>
        <div className="header-badge">
          <Sparkles size={16} />
          Tool-aware analysis
        </div>
      </header>
      <div className="messages">
        {messages.map((message) => (
          <article key={message.id} className={`message message--${message.role}`}>
            <div className="message__bubble">
              <p>{message.content}</p>
            </div>
            {message.toolResults?.map((result) => (
              <ToolResultSection key={`${message.id}-${result.tool}`} result={result} />
            ))}
          </article>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="suggestions">
        {suggestions.map((suggestion) => (
          <button key={suggestion} className="suggestion-chip" onClick={() => void onSendMessage(suggestion)} disabled={disabled}>
            {suggestion}
          </button>
        ))}
      </div>

      <form className="composer" onSubmit={handleSubmit}>
        <textarea
          value={draft}
          disabled={disabled}
          onChange={(event) => setDraft(event.target.value)}
          placeholder={disabled ? 'Upload data to start chatting.' : 'Ask for an MMM analysis, budget optimization, or scenario forecast.'}
          rows={2}
        />
        <button className="button" type="submit" disabled={disabled || !draft.trim()}>
          <SendHorizonal size={16} />
          Send
        </button>
      </form>
    </section>
  );
}

function ToolResultSection({ result }: { result: ToolResult }) {
  const firstTableKey = Object.keys(result.tables)[0];
  const tableRows = firstTableKey ? result.tables[firstTableKey] : [];

  return (
    <div className="tool-result">
      <div className="tool-result__intro">
        <h3>{result.title}</h3>
        <p>{result.summary}</p>
      </div>
      {Object.keys(result.metrics).length > 0 && (
        <div className="tool-metrics">
          {Object.entries(result.metrics).map(([label, value]) => (
            <div key={label} className="tool-metric">
              <span>{label.replaceAll('_', ' ')}</span>
              <strong>{typeof value === 'number' ? value.toLocaleString() : value}</strong>
            </div>
          ))}
        </div>
      )}
      {result.charts.map((chart) => (
        <ChartCard key={chart.id} chart={chart} />
      ))}
      {tableRows.length > 0 && (
        <div className="tool-table">
          <table>
            <thead>
              <tr>
                {Object.keys(tableRows[0]).map((column) => (
                  <th key={column}>{column.replaceAll('_', ' ')}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableRows.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {Object.entries(row).map(([key, value]) => (
                    <td key={`${rowIndex}-${key}`}>{typeof value === 'number' ? value.toLocaleString() : value}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
