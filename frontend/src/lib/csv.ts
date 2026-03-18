import { ColumnMapping } from './types';

const COLUMN_CANDIDATES: Record<keyof ColumnMapping, string[]> = {
  date: ['date', 'day', 'ds'],
  channel: ['channel', 'source', 'platform', 'media_channel'],
  spend: ['spend', 'cost', 'investment', 'media_spend'],
  revenue: ['revenue', 'sales', 'conversion_value', 'income'],
};

export async function parseCsvPreview(file: File): Promise<{
  columns: string[];
  rows: Array<Record<string, string>>;
  mapping: ColumnMapping;
}> {
  const text = await file.text();
  const [headerLine, ...body] = text.trim().split(/\r?\n/);
  const columns = headerLine.split(',').map((cell) => cell.trim());
  const rows = body.slice(0, 8).map((line) => {
    const values = line.split(',').map((cell) => cell.trim());
    return Object.fromEntries(columns.map((column, index) => [column, values[index] ?? '']));
  });

  const mapping = Object.fromEntries(
    Object.entries(COLUMN_CANDIDATES).map(([target, candidates]) => {
      const match = columns.find((column) =>
        candidates.some((candidate) => column.toLowerCase() === candidate.toLowerCase()),
      );
      return [target, match ?? columns[0] ?? ''];
    }),
  ) as ColumnMapping;

  return { columns, rows, mapping };
}
