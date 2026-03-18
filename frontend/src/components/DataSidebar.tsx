import { Activity, Database, DollarSign, Layers3 } from 'lucide-react';
import { SummaryStats } from '../lib/types';

interface DataSidebarProps {
  summary: SummaryStats | null;
}

export function DataSidebar({ summary }: DataSidebarProps) {
  if (!summary) {
    return (
      <div className="sidebar-card sidebar-card--empty">
        <p>Upload a CSV to inspect channel totals, date range, and modeled results.</p>
      </div>
    );
  }

  return (
    <div className="sidebar-stack">
      <div className="sidebar-grid">
        <MetricCard icon={<Database size={16} />} label="Rows" value={summary.rows.toLocaleString()} />
        <MetricCard icon={<Layers3 size={16} />} label="Channels" value={summary.channels.toString()} />
        <MetricCard icon={<DollarSign size={16} />} label="Spend" value={`$${summary.total_spend.toLocaleString()}`} />
        <MetricCard icon={<Activity size={16} />} label="ROAS" value={summary.overall_roas.toFixed(2)} />
      </div>
      <section className="sidebar-card">
        <h3>Coverage</h3>
        <p>
          {summary.date_range.start} to {summary.date_range.end}
        </p>
      </section>
      <section className="sidebar-card">
        <h3>Channel Summary</h3>
        <div className="channel-list">
          {summary.channel_summary.map((channel) => (
            <div key={channel.channel} className="channel-row">
              <div>
                <strong>{channel.channel}</strong>
                <span>${channel.avg_daily_spend.toFixed(0)}/day avg</span>
              </div>
              <div>
                <strong>${channel.spend.toLocaleString()}</strong>
                <span>${channel.revenue.toLocaleString()} revenue</span>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function MetricCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <article className="metric-card">
      <div className="metric-card__icon">{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}
