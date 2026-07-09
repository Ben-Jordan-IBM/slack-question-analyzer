import React from 'react';

/**
 * Button — IBM Carbon-style action trigger.
 * Sharp corners, asymmetric padding when an icon is present (Carbon hallmark:
 * label left, icon pinned right). Variants: primary, secondary, tertiary, ghost, danger.
 */
export function Button({
  children,
  variant = 'primary',
  size = 'lg',
  icon = null,
  fullWidth = false,
  disabled = false,
  onClick,
  type = 'button',
  ...rest
}) {
  const heights = { sm: '2rem', md: '2.5rem', lg: '3rem' };

  const palettes = {
    primary:   { bg: 'var(--button-primary)',   bgHover: 'var(--button-primary-hover)',   color: 'var(--text-on-color)', border: 'transparent' },
    secondary: { bg: 'var(--button-secondary)',  bgHover: 'var(--button-secondary-hover)', color: 'var(--text-on-color)', border: 'transparent' },
    tertiary:  { bg: 'transparent',              bgHover: 'var(--blue-60)',                color: 'var(--blue-60)',       border: 'var(--blue-60)', colorHover: 'var(--text-on-color)' },
    ghost:     { bg: 'transparent',              bgHover: 'var(--layer-hover)',            color: 'var(--blue-60)',       border: 'transparent' },
    danger:    { bg: 'var(--button-danger)',     bgHover: 'var(--button-danger-hover)',   color: 'var(--text-on-color)', border: 'transparent' },
  };
  const p = palettes[variant] || palettes.primary;
  const [hover, setHover] = React.useState(false);
  const hasIcon = !!icon;

  const style = {
    appearance: 'none',
    display: 'inline-flex',
    alignItems: 'center',
    // Centered label (and icon beside it): the Carbon asymmetric padding
    // left-aligned labels and pinned icons to the far edge, which read as
    // off-center text in short buttons
    justifyContent: 'center',
    gap: 'var(--spacing-04)',
    width: fullWidth ? '100%' : 'auto',
    minWidth: variant === 'ghost' ? 0 : '6rem',
    height: heights[size] || heights.lg,
    padding: '0 var(--spacing-06)',
    fontFamily: 'var(--font-sans)',
    fontSize: 'var(--type-body-01)',
    fontWeight: 'var(--weight-regular)',
    lineHeight: 1,
    letterSpacing: '0.01em',
    textAlign: 'center',
    cursor: disabled ? 'not-allowed' : 'pointer',
    border: `1px solid ${p.border}`,
    borderRadius: 'var(--radius-none)',
    background: disabled ? 'var(--layer-active)' : (hover ? p.bgHover : p.bg),
    color: disabled ? 'var(--text-disabled)' : (hover && p.colorHover ? p.colorHover : p.color),
    transition: 'background var(--duration-base) var(--ease-productive), color var(--duration-base) var(--ease-productive)',
    outline: 'none',
    position: 'relative',
  };

  return (
    <button
      type={type}
      style={style}
      disabled={disabled}
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      {...rest}
    >
      <span>{children}</span>
      {icon ? <span style={{ display: 'inline-flex', flex: '0 0 auto' }}>{icon}</span> : null}
    </button>
  );
}
