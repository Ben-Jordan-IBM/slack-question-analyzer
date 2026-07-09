import React from 'react';

/**
 * Tag — Carbon pill tag for keywords, categories, and counts.
 * Color pairs follow Carbon (soft fill + deep text). Optional dismiss / dot.
 */
export function Tag({
  children,
  color = 'gray',
  size = 'md',
  outline = false,
  dot = false,
  onDismiss = null,
  ...rest
}) {
  const pairs = {
    gray:    { bg: 'var(--gray-20)',   fg: 'var(--gray-100)', line: 'var(--gray-50)' },
    blue:    { bg: 'var(--blue-20)',   fg: 'var(--blue-80)',  line: 'var(--blue-60)' },
    green:   { bg: '#a7f0ba',          fg: '#044317',         line: 'var(--green-60)' },
    red:     { bg: '#ffd7d9',          fg: '#750e13',         line: 'var(--red-60)' },
    purple:  { bg: '#e8daff',          fg: '#491d8b',         line: 'var(--purple-60)' },
    teal:    { bg: '#9ef0f0',          fg: '#004144',        line: 'var(--teal-60)' },
    magenta: { bg: '#ffd6e8',          fg: '#740937',        line: 'var(--magenta-60)' },
    cyan:    { bg: '#bae6ff',          fg: '#00539a',        line: 'var(--cyan-50)' },
  };
  const p = pairs[color] || pairs.gray;
  const heights = { sm: '1.125rem', md: '1.5rem' };

  const style = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 'var(--spacing-02)',
    height: heights[size] || heights.md,
    padding: size === 'sm' ? '0 var(--spacing-03)' : '0 var(--spacing-04)',
    borderRadius: 'var(--radius-pill)',
    fontFamily: 'var(--font-sans)',
    fontSize: 'var(--type-label-01)',
    fontWeight: 'var(--weight-regular)',
    lineHeight: 1,
    whiteSpace: 'nowrap',
    background: outline ? 'transparent' : p.bg,
    color: outline ? p.fg : p.fg,
    border: outline ? `1px solid ${p.line}` : '1px solid transparent',
  };

  return (
    <span style={style} {...rest}>
      {dot ? <span style={{ width: 6, height: 6, borderRadius: '50%', background: p.line, flex: '0 0 auto' }} /> : null}
      <span>{children}</span>
      {onDismiss ? (
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss"
          style={{
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
            width: 16, height: 16, marginRight: -4, padding: 0, border: 'none',
            background: 'transparent', color: p.fg, cursor: 'pointer', borderRadius: '50%',
          }}
        >
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none"><path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="1.25" /></svg>
        </button>
      ) : null}
    </span>
  );
}
