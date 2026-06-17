import React from 'react';

/**
 * FileDropzone — Carbon file-uploader. Dashed field, drag-active state,
 * supported-format helper text, and a selected-file chip.
 */
export function FileDropzone({
  accept = '.txt,.json,.csv',
  hint = 'TXT, JSON or CSV up to 200MB',
  title = 'Drag a Slack export here or click to browse',
  fileName = null,
  onFile,
  onClear,
  ...rest
}) {
  const [drag, setDrag] = React.useState(false);
  const inputRef = React.useRef(null);

  const pick = () => inputRef.current && inputRef.current.click();
  const handle = (file) => { if (file && onFile) onFile(file); };

  if (fileName) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        gap: 'var(--spacing-04)', padding: 'var(--spacing-04) var(--spacing-05)',
        background: 'var(--layer-02)', border: '1px solid var(--border-subtle)',
        borderLeft: '3px solid var(--green-60)',
      }} {...rest}>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 'var(--spacing-04)', minWidth: 0 }}>
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none" style={{ flex: '0 0 auto' }}>
            <path d="M11 2H5a1 1 0 00-1 1v14a1 1 0 001 1h10a1 1 0 001-1V7l-5-5z" stroke="var(--text-secondary)" strokeWidth="1.25" />
            <path d="M11 2v5h5" stroke="var(--text-secondary)" strokeWidth="1.25" />
          </svg>
          <span style={{ fontFamily: 'var(--font-sans)', fontSize: 'var(--type-body-01)', color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{fileName}</span>
        </span>
        <button type="button" onClick={onClear} aria-label="Remove file" style={{
          display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
          width: 24, height: 24, border: 'none', background: 'transparent',
          color: 'var(--text-secondary)', cursor: 'pointer',
        }}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="1.25" /></svg>
        </button>
      </div>
    );
  }

  return (
    <div
      onClick={pick}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); handle(e.dataTransfer.files[0]); }}
      style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        gap: 'var(--spacing-04)', textAlign: 'center',
        padding: 'var(--spacing-09) var(--spacing-06)',
        background: drag ? 'var(--blue-10)' : 'var(--layer-01)',
        border: `1px dashed ${drag ? 'var(--blue-60)' : 'var(--border-strong)'}`,
        cursor: 'pointer',
        transition: 'background var(--duration-base) var(--ease-productive), border-color var(--duration-base) var(--ease-productive)',
      }}
      {...rest}
    >
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <path d="M14 19V6M14 6l-5 5M14 6l5 5" stroke={drag ? 'var(--blue-60)' : 'var(--text-secondary)'} strokeWidth="1.4" />
        <path d="M5 19v2a1 1 0 001 1h16a1 1 0 001-1v-2" stroke={drag ? 'var(--blue-60)' : 'var(--text-secondary)'} strokeWidth="1.4" />
      </svg>
      <span style={{ fontFamily: 'var(--font-sans)', fontSize: 'var(--type-body-02)', color: 'var(--text-primary)' }}>{title}</span>
      <span style={{ fontFamily: 'var(--font-sans)', fontSize: 'var(--type-helper-01)', color: 'var(--text-helper)' }}>{hint}</span>
      <input ref={inputRef} type="file" accept={accept} style={{ display: 'none' }}
        onChange={(e) => handle(e.target.files[0])} />
    </div>
  );
}
