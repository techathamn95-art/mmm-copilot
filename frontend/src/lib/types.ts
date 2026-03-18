export type MessageRole = 'user' | 'assistant' | 'system';
export type ChartType = 'bar' | 'pie' | 'scatter' | 'line' | 'stacked_bar';

export interface ColumnMapping {
  date: string;
  channel: string;
  spend: string;
  revenue: string;
}

export interface SummaryStats {
  rows: number;
  date_range: { start: string; end: string };
  channels: number;
  total_spend: number;
  total_revenue: number;
  overall_roas: number;
  channel_summary: Array<{
    channel: string;
    spend: number;
    revenue: number;
    avg_daily_spend: number;
  }>;
}

export interface UploadResponse {
  session_id: string;
  file_name: string;
  columns: string[];
  mapping: ColumnMapping;
  summary: SummaryStats;
  preview: Array<Record<string, string | number>>;
}

export interface ChartSeries {
  key: string;
  label: string;
  color?: string | null;
}

export interface ChartPoint {
  label: string;
  value?: number | null;
  values?: Record<string, number>;
  color?: string | null;
  meta: Record<string, string | number>;
}

export interface ChartPayload {
  id: string;
  title: string;
  description: string;
  type: ChartType;
  data: ChartPoint[];
  series: ChartSeries[];
  x_key?: string | null;
  y_key?: string | null;
}

export interface ToolResult {
  tool: string;
  title: string;
  summary: string;
  metrics: Record<string, string | number>;
  tables: Record<string, Array<Record<string, string | number>>>;
  charts: ChartPayload[];
}

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: string;
  toolResults?: ToolResult[];
}
