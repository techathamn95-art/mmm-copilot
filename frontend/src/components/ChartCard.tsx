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

const tooltipStyle = {
  backgroundColor: '#2c2c2e',
  border: '1px solid #38383a',
  borderRadius: 12,
  color: '#fff',
  fontSize: 13,
};

const legendStyle = { color: 'rgba(235,235,245,0.6)', fontSize: 12 };

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
        {chart.type === 'bar' && (
          <BarChart data={data} layout="vertical" width={580} height={220} margin={{ left: 18, right: 24, top: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.15)" />
            <XAxis type="number" stroke="rgba(235,235,245,0.3)" fontSize={11} />
            <YAxis dataKey="label" type="category" stroke="rgba(235,235,245,0.45)" width={84} fontSize={11} />
            <Tooltip contentStyle={tooltipStyle} />
            <Bar dataKey="value" radius={[0, 8, 8, 0]} barSize={24}>
              {data.map((entry, index) => (
                <Cell key={`${entry.label}-${index}`} fill={entry.color || '#0a84ff'} />
              ))}
            </Bar>
          </BarChart>
        )}
        {chart.type === 'pie' && (
          <PieChart width={580} height={220}>
            <Tooltip contentStyle={tooltipStyle} />
            <Legend formatter={(value) => <span style={legendStyle}>{value}</span>} />
            <Pie data={data} dataKey="value" nameKey="label" cx="50%" cy="50%" innerRadius={52} outerRadius={88}>
              {data.map((entry, index) => (
                <Cell key={`${entry.label}-${index}`} fill={entry.color || '#0a84ff'} />
              ))}
            </Pie>
          </PieChart>
        )}
        {chart.type === 'scatter' && (
          <ScatterChart width={580} height={220} margin={{ top: 12, right: 24, bottom: 12, left: 12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.15)" />
            <XAxis type="number" dataKey="spend" stroke="rgba(235,235,245,0.3)" name="Spend" fontSize={11} />
            <YAxis type="number" dataKey="revenue" stroke="rgba(235,235,245,0.3)" name="Revenue" fontSize={11} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={tooltipStyle} />
            <Scatter data={data} fill="#0a84ff">
              {data.map((entry, index) => (
                <Cell key={`${entry.label}-${index}`} fill={entry.color || '#0a84ff'} />
              ))}
            </Scatter>
          </ScatterChart>
        )}
        {chart.type === 'line' && (
          <LineChart data={data} width={580} height={220}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.15)" />
            <XAxis dataKey="label" stroke="rgba(235,235,245,0.3)" fontSize={11} />
            <YAxis stroke="rgba(235,235,245,0.3)" fontSize={11} />
            <Tooltip contentStyle={tooltipStyle} />
            <Legend formatter={(value) => <span style={legendStyle}>{value}</span>} />
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
          <BarChart data={data} width={580} height={220}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.15)" />
            <XAxis dataKey="label" stroke="rgba(235,235,245,0.3)" fontSize={11} />
            <YAxis stroke="rgba(235,235,245,0.3)" fontSize={11} />
            <Tooltip contentStyle={tooltipStyle} />
            <Legend formatter={(value) => <span style={legendStyle}>{value}</span>} />
            {chart.series.map((series) => (
              <Bar key={series.key} dataKey={series.key} stackId="budget" fill={series.color || '#0a84ff'} radius={6} />
            ))}
          </BarChart>
        )}
      </div>
    </section>
  );
}
