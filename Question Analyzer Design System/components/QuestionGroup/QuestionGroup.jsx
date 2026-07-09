import React from 'react';

/**
 * QuestionGroup — the analyzer's hero row. A ranked, expandable group of
 * semantically-similar questions: rank numeral, representative question,
 * frequency heat-bar, keyword tags, similarity, and the underlying questions.
 */
export function QuestionGroup({
  rank,
  question,
  count,
  maxCount = count,
  similarity = null,
  keywords = [],
  questions = [],
  defaultOpen = false,
}) {
  const [open, setOpen] = React.useState(defaultOpen);
  const [hover, setHover] = React.useState(false);
  const pct = Math.max(6, Math.round((count / Math.max(1, maxCount)) * 100));
  const heat = rank === 1 ? 'var(--blue-60)' : rank === 2 ? 'var(--blue-50)' : rank === 3 ? 'var(--blue-40)' : 'var(--gray-40)';

  return (
    <div style={{ background: 'var(--layer-02)', borderBottom: '1px solid var(--border-subtle)' }}>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        style={{
          width: '100%', display: 'grid',
          gridTemplateColumns: 'auto 1fr auto auto', alignItems: 'center',
          gap: 'var(--spacing-05)', padding: 'var(--spacing-05) var(--spacing-06)',
          background: hover ? 'var(--layer-hover)' : 'transparent',
          border: 'none', borderLeft: `3px solid ${open ? heat : 'transparent'}`,
          textAlign: 'left', cursor: 'pointer',
          transition: 'background var(--duration-base) var(--ease-productive)',
        }}
      >
        {/* Rank */}
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: 'var(--type-heading-03)',
          fontWeight: 'var(--weight-regular)', color: heat, width: '2.25rem',
          textAlign: 'right', fontVariantNumeric: 'tabular-nums',
        }}>{String(rank).padStart(2, '0')}</span>

        {/* Question + keywords */}
        <span style={{ minWidth: 0, display: 'flex', flexDirection: 'column', gap: 'var(--spacing-03)' }}>
          <span style={{
            fontFamily: 'var(--font-sans)', fontSize: 'var(--type-body-02)',
            color: 'var(--text-primary)', lineHeight: 'var(--lh-snug)',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: open ? 'normal' : 'nowrap',
          }}>{question}</span>
          {keywords.length ? (
            <span style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--spacing-02)' }}>
              {keywords.slice(0, 5).map((k, i) => (
                <span key={i} style={{
                  fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-helper)',
                  background: 'var(--gray-10)', padding: '1px 6px', borderRadius: 'var(--radius-sm)',
                }}>{k}</span>
              ))}
            </span>
          ) : null}
        </span>

        {/* Frequency heat bar + count */}
        <span style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-04)', width: '11rem' }}>
          <span style={{ flex: 1, height: 6, background: 'var(--gray-10)', position: 'relative' }}>
            <span style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${pct}%`, background: heat }} />
          </span>
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: 'var(--type-body-01)',
            fontWeight: 'var(--weight-medium)', color: 'var(--text-primary)',
            fontVariantNumeric: 'tabular-nums', width: '2.5rem', textAlign: 'right',
          }}>{count}×</span>
        </span>

        {/* Chevron */}
        <span style={{ display: 'inline-flex', color: 'var(--text-secondary)', transform: open ? 'rotate(180deg)' : 'none', transition: 'transform var(--duration-base) var(--ease-productive)' }}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M4 6l4 4 4-4" stroke="currentColor" strokeWidth="1.25" /></svg>
        </span>
      </button>

      {open ? (
        <div style={{ padding: '0 var(--spacing-06) var(--spacing-06) calc(var(--spacing-06) + 2.25rem + var(--spacing-05))' }}>
          {similarity != null ? (
            <div style={{
              fontFamily: 'var(--font-sans)', fontSize: 'var(--type-label-01)',
              color: 'var(--text-helper)', marginBottom: 'var(--spacing-04)',
            }}>
              Avg. similarity <strong style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>{similarity}</strong> · {questions.length} occurrences
            </div>
          ) : null}
          <ul style={{ listStyle: 'none', margin: 0, padding: 0, borderLeft: '1px solid var(--border-subtle)' }}>
            {questions.map((q, i) => (
              <li key={i} style={{ display: 'flex', justifyContent: 'space-between', gap: 'var(--spacing-05)', padding: 'var(--spacing-03) var(--spacing-05)' }}>
                <span style={{ fontFamily: 'var(--font-sans)', fontSize: 'var(--type-body-01)', color: 'var(--text-secondary)', lineHeight: 'var(--lh-normal)' }}>{q.text}</span>
                {q.date ? <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--type-code-01)', color: 'var(--text-placeholder)', whiteSpace: 'nowrap' }}>{q.date}</span> : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
