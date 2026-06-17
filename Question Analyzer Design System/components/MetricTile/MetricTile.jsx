import React from 'react';

/**
 * MetricTile — Carbon "big number" stat. Light-weight numeral, uppercase
 * label, optional delta. Used in the analysis summary row.
 */
export function MetricTile({
  label,
  value,
  unit = null,
  delta = null,
  accent = 'var(--blue-60)',
  ...rest
}) {
  const positive = typeof delta === 'string' ? delta.trim().startsWith('+') : (delta > 0);
  return (
    <div
      style={{
        background: 'var(--layer-02)',
        borderLeft: `3px solid ${accent}`,
        padding: 'var(--spacing-05) var(--spacing-06)',
        display: 'flex', flexDirection: 'column', gap: 'var(--spacing-03)',
        minWidth: 0,
      }}
      {...rest}
    >
      <span style={{
        fontFamily: 'var(--font-sans)', fontSize: 'var(--type-label-01)',
        letterSpacing: 'var(--tracking-caps)', textTransform: 'uppercase',
        color: 'var(--text-helper)', fontWeight: 'var(--weight-medium)',
      }}>{label}</span>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--spacing-03)' }}>
        <span style={{
          fontFamily: 'var(--font-sans)', fontSize: 'var(--type-heading-05)',
          fontWeight: 'var(--weight-light)', lineHeight: 1, color: 'var(--text-primary)',
          letterSpacing: 'var(--tracking-display)',
        }}>{value}</span>
        {unit ? <span style={{ fontSize: 'var(--type-body-01)', color: 'var(--text-secondary)' }}>{unit}</span> : null}
        {delta != null ? (
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: 'var(--type-code-01)',
            color: positive ? 'var(--green-60)' : 'var(--red-60)',
          }}>{delta}</span>
        ) : null}
      </div>
    </div>
  );
}
