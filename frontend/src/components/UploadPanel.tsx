import { FileUp, Table2 } from 'lucide-react';
import { useMemo, useState } from 'react';
import { parseCsvPreview } from '../lib/csv';
import { ColumnMapping } from '../lib/types';

interface UploadPanelProps {
  isUploading: boolean;
  onUpload: (file: File, mapping: ColumnMapping) => Promise<void>;
}

export function UploadPanel({ isUploading, onUpload }: UploadPanelProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [previewRows, setPreviewRows] = useState<Array<Record<string, string>>>([]);
  const [mapping, setMapping] = useState<ColumnMapping>({
    date: '',
    channel: '',
    spend: '',
    revenue: '',
  });

  const mappingFields = useMemo(
    () => [
      { key: 'date', label: 'Date' },
      { key: 'channel', label: 'Channel' },
      { key: 'spend', label: 'Spend' },
      { key: 'revenue', label: 'Revenue' },
    ],
    [],
  );

  async function handleFile(file: File) {
    const parsed = await parseCsvPreview(file);
    setSelectedFile(file);
    setColumns(parsed.columns);
    setPreviewRows(parsed.rows);
    setMapping(parsed.mapping);
  }

  return (
    <section className="upload-panel">
      <div
        className="dropzone"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          const file = event.dataTransfer.files?.[0];
          if (file) {
            void handleFile(file);
          }
        }}
      >
        <FileUp size={20} />
        <div>
          <strong>Upload marketing data</strong>
          <p>Drag a CSV here or choose a file. V1 accepts date, channel, spend, and revenue.</p>
        </div>
        <label className="button button--secondary">
          Choose CSV
          <input
            type="file"
            accept=".csv"
            hidden
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) {
                void handleFile(file);
              }
            }}
          />
        </label>
      </div>

      {selectedFile && (
        <>
          <section className="mapping-panel">
            <div className="panel-title">
              <h3>Column mapping</h3>
              <p>{selectedFile.name}</p>
            </div>
            <div className="mapping-grid">
              {mappingFields.map((field) => (
                <label key={field.key}>
                  <span>{field.label}</span>
                  <select
                    value={mapping[field.key as keyof ColumnMapping]}
                    onChange={(event) =>
                      setMapping((current) => ({
                        ...current,
                        [field.key]: event.target.value,
                      }))
                    }
                  >
                    {columns.map((column) => (
                      <option key={column} value={column}>
                        {column}
                      </option>
                    ))}
                  </select>
                </label>
              ))}
            </div>
            <button className="button" onClick={() => void onUpload(selectedFile, mapping)} disabled={isUploading}>
              {isUploading ? 'Uploading...' : 'Validate and Load'}
            </button>
          </section>

          <section className="preview-panel">
            <div className="panel-title">
              <h3>Preview</h3>
              <Table2 size={16} />
            </div>
            <div className="preview-table">
              <table>
                <thead>
                  <tr>
                    {columns.map((column) => (
                      <th key={column}>{column}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {previewRows.map((row, index) => (
                    <tr key={index}>
                      {columns.map((column) => (
                        <td key={`${index}-${column}`}>{row[column]}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </section>
  );
}
