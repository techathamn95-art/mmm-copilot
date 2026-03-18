import { useState } from 'react';
import { Key, ChevronDown, ChevronUp, CheckCircle, XCircle, Loader2 } from 'lucide-react';

const PROVIDERS = [
  { id: 'google', name: 'Google Gemini', defaultModel: 'gemini-2.0-flash', keyHint: 'AIza...' },
  { id: 'openai', name: 'OpenAI', defaultModel: 'gpt-4o-mini', keyHint: 'sk-...' },
  { id: 'anthropic', name: 'Anthropic Claude', defaultModel: 'claude-sonnet-4-20250514', keyHint: 'sk-ant-...' },
  { id: 'groq', name: 'Groq', defaultModel: 'llama-3.3-70b-versatile', keyHint: 'gsk_...' },
] as const;

interface LLMSettingsProps {
  configured: boolean;
  provider: string | null;
  model: string | null;
  onConfigure: (provider: string, apiKey: string, model: string) => Promise<void>;
  onTest: (provider: string, apiKey: string, model: string) => Promise<{ ok: boolean; message: string }>;
}

export function LLMSettings({ configured, provider, model, onConfigure, onTest }: LLMSettingsProps) {
  const [open, setOpen] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState(provider || 'google');
  const [apiKey, setApiKey] = useState('');
  const [customModel, setCustomModel] = useState(model || '');
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null);

  const providerInfo = PROVIDERS.find(p => p.id === selectedProvider) || PROVIDERS[0];

  async function handleSave() {
    if (!apiKey.trim()) return;
    setSaving(true);
    try {
      await onConfigure(selectedProvider, apiKey.trim(), customModel || providerInfo.defaultModel);
      setOpen(false);
      setApiKey('');
    } finally {
      setSaving(false);
    }
  }

  async function handleTest() {
    if (!apiKey.trim()) return;
    setTesting(true);
    setTestResult(null);
    try {
      const result = await onTest(selectedProvider, apiKey.trim(), customModel || providerInfo.defaultModel);
      setTestResult(result);
    } finally {
      setTesting(false);
    }
  }

  return (
    <div className="sidebar-card">
      <button
        className="sidebar-card__toggle"
        onClick={() => setOpen(!open)}
      >
        <div className="sidebar-card__toggle-left">
          <div className={`sidebar-card__status ${configured ? 'connected' : ''}`} />
          <h3>AI Model</h3>
        </div>
        <div className="sidebar-card__toggle-right">
          {configured ? (
            <span className="sidebar-card__badge">{provider}</span>
          ) : (
            <span className="sidebar-card__hint">Configure</span>
          )}
          {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
      </button>

      {open && (
        <div className="settings-form">
          <label>
            <span>Provider</span>
            <select
              value={selectedProvider}
              onChange={(e) => {
                setSelectedProvider(e.target.value);
                setTestResult(null);
                const info = PROVIDERS.find(p => p.id === e.target.value);
                if (info) setCustomModel(info.defaultModel);
              }}
            >
              {PROVIDERS.map(p => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
          </label>

          <label>
            <span>API Key</span>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => { setApiKey(e.target.value); setTestResult(null); }}
              placeholder={providerInfo.keyHint}
            />
          </label>

          <label>
            <span>Model <small>(optional)</small></span>
            <input
              type="text"
              value={customModel}
              onChange={(e) => { setCustomModel(e.target.value); setTestResult(null); }}
              placeholder={providerInfo.defaultModel}
            />
          </label>

          {testResult && (
            <div className={`settings-test ${testResult.ok ? 'success' : 'error'}`}>
              {testResult.ok ? <CheckCircle size={14} /> : <XCircle size={14} />}
              <span>{testResult.message}</span>
            </div>
          )}

          <div className="settings-actions">
            <button
              className="button button--secondary"
              onClick={() => void handleTest()}
              disabled={testing || !apiKey.trim()}
            >
              {testing ? <Loader2 size={14} className="spin" /> : null}
              Test
            </button>
            <button
              className="button button--primary"
              onClick={() => void handleSave()}
              disabled={saving || !apiKey.trim()}
            >
              {saving ? <Loader2 size={14} className="spin" /> : null}
              {configured ? 'Update' : 'Connect'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
