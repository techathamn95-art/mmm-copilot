import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { ChartPayload } from '../lib/types';

interface ChartCardProps {
  chart: ChartPayload;
}

function flattenData(chart: ChartPayload) {
  return chart.data.map((point) => ({
    label: point.label,
    value: point.value,
    color: point.color,
    ...point.values,
    ...point.meta,
  }));
}

export function ChartCard({ chart }: ChartCardProps) {
  const data = flattenData(chart);

  return (
    <section className="chart-card">
      <div className="chart-card__header">
        <div>
          <h4>{chart.title}</h4>
          <p>{chart.description}</p>
        </div>
      </div>
      <div className="chart-card__body">
        <ResponsiveContainer width="100%" height={280}>
          <>
            {chart.type === 'bar' && (
              <BarChart data={data} layout="vertical" margin={{ left: 18, right: 12 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.15)" />
                <XAxis type="number" stroke="rgba(235,235,245,0.3)" />
                <YAxis dataKey="label" type="category" stroke="rgba(235,235,245,0.45)" width={84} />
                <Tooltip />
                <Bar dataKey="value" radius={[0, 8, 8, 0]}>
                  {data.map((entry, index) => (
                    <Cell key={`${entry.label}-${index}`} fill={entry.color || '#0a84ff'} />
                  ))}
                </Bar>
              </BarChart>
            )}
            {chart.type === 'pie' && (
              <PieChart>
                <Tooltip />
                <Legend />
                <Pie data={data} dataKey="value" nameKey="label" innerRadius={52} outerRadius={96}>
                  {data.map((entry, index) => (
                    <Cell key={`${entry.label}-${index}`} fill={entry.color || '#0a84ff'} />
                  ))}
                </Pie>
              </PieChart>
            )}
            {chart.type === 'scatter' && (
              <ScatterChart margin={{ top: 12, right: 12, bottom: 12, left: 12 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.15)" />
                <XAxis type="number" dataKey="spend" stroke="rgba(235,235,245,0.3)" />
                <YAxis type="number" dataKey="revenue" stroke="rgba(235,235,245,0.3)" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter data={data} fill="#0a84ff" />
              </ScatterChart>
            )}
            {chart.type === 'line' && (
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.15)" />
                <XAxis dataKey="label" stroke="rgba(235,235,245,0.3)" />
                <YAxis stroke="rgba(235,235,245,0.3)" />
                <Tooltip />
                <Legend />
                {chart.series.map((series) => (
                  <Line
                    key={series.key}
                    type="monotone"
                    dataKey={series.key}
                    stroke={series.color || '#0a84ff'}
                    strokeWidth={3}
                    dot={{ r: 4 }}
                  />
                ))}
              </LineChart>
            )}
            {chart.type === 'stacked_bar' && (
              <BarChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.15)" />
                <XAxis dataKey="label" stroke="rgba(235,235,245,0.3)" />
                <YAxis stroke="rgba(235,235,245,0.3)" />
                <Tooltip />
                <Legend />
                {chart.series.map((series) => (
                  <Bar key={series.key} dataKey={series.key} stackId="budget" fill={series.color || '#0a84ff'} radius={6} />
                ))}
              </BarChart>
            )}
          </>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
