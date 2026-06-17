import React from 'react';

/**
 * Card — Carbon layered surface. Sharp corners, 1px subtle border,
 * optional left accent bar and hover elevation.
 */
export function Card({
  children,
  padding = 'var(--spacing-06)',
  accent = null,
  interactive = false,
  selected = false,
  onClick,
  style = {},
  ...rest
}) {
  const [hover, setHover] = React.useState(false);
  const base = {
    position: 'relative',
    background: 'var(--layer-02)',
    border: `1px solid ${selected ? 'var(--blue-60)' : 'var(--border-subtle)'}`,
    borderRadius: 'var(--radius-none)',
    padding,
    boxShadow: interactive && hover ? 'var(--shadow-md)' : 'none',
    cursor: interactive ? 'pointer' : 'default',
    transition: 'box-shadow var(--duration-base) var(--ease-productive), border-color var(--duration-base) var(--ease-productive)',
    ...style,
  };
  return (
    <div
      style={base}
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      {...rest}
    >
      {accent ? (
        <span style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 3, background: accent }} />
      ) : null}
      {children}
    </div>
  );
}
