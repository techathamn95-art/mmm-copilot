import { ColumnMapping, Message, ToolResult, UploadResponse } from './types';

const API_BASE = 'http://localhost:8000/api';

export async function uploadCsv(file: File, mapping: ColumnMapping): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('mapping_json', JSON.stringify(mapping));

  const response = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json();
}

export async function loadDemo(): Promise<UploadResponse> {
  const response = await fetch(`${API_BASE}/demo`);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export async function getLLMConfig(): Promise<{ configured: boolean; provider: string | null; model: string | null }> {
  const response = await fetch(`${API_BASE}/llm-config`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function setLLMConfig(provider: string, apiKey: string, model: string): Promise<void> {
  const response = await fetch(`${API_BASE}/llm-config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider, api_key: apiKey, model }),
  });
  if (!response.ok) throw new Error(await response.text());
}

export async function testLLM(provider: string, apiKey: string, model: string): Promise<{ ok: boolean; message: string }> {
  const response = await fetch(`${API_BASE}/llm-test`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider, api_key: apiKey, model }),
  });
  const data = await response.json();
  if (data.status === 'error') return { ok: false, message: data.detail };
  return { ok: true, message: data.response };
}

export async function streamChat(
  sessionId: string,
  message: string,
  handlers: {
    onDelta: (chunk: string) => void;
    onComplete: (payload: { content: string; tool_results: ToolResult[]; suggested_prompts: string[] }) => void;
  },
): Promise<void> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message }),
  });

  if (!response.ok || !response.body) {
    throw new Error(await response.text());
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop() ?? '';

    for (const eventBlock of events) {
      const lines = eventBlock.split('\n');
      const event = lines.find((line) => line.startsWith('event:'))?.replace('event:', '').trim();
      const dataLine = lines.find((line) => line.startsWith('data:'))?.replace('data:', '').trim();
      if (!event || !dataLine) continue;
      const payload = JSON.parse(dataLine);
      if (event === 'delta') {
        handlers.onDelta(payload.content);
      }
      if (event === 'message') {
        handlers.onComplete(payload);
      }
    }
  }
}

export async function getHistory(sessionId: string): Promise<Message[]> {
  const response = await fetch(`${API_BASE}/chat/${sessionId}/history`);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const payload = await response.json();
  return payload.messages.map((message: any) => ({
    id: message.id,
    role: message.role,
    content: message.content,
    createdAt: message.created_at,
    toolResults: message.tool_results,
  }));
}
