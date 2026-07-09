import React from 'react';

/**
 * Slider — Carbon range control. Thin rail, filled progress, round thumb,
 * and a live numeric readout. Used for the similarity threshold.
 */
export function Slider({
  value,
  min = 0,
  max = 100,
  step = 1,
  onChange,
  label = null,
  format = (v) => v,
  disabled = false,
  ...rest
}) {
  const pct = ((value - min) / (max - min)) * 100;
  const [active, setActive] = React.useState(false);

  return (
    <div style={{ width: '100%' }} {...rest}>
      {label ? (
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
          marginBottom: 'var(--spacing-04)',
        }}>
          <span style={{
            fontFamily: 'var(--font-sans)', fontSize: 'var(--type-label-01)',
            color: 'var(--text-secondary)', letterSpacing: 'var(--tracking-label)',
          }}>{label}</span>
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: 'var(--type-body-01)',
            color: 'var(--text-primary)', fontWeight: 'var(--weight-medium)',
          }}>{format(value)}</span>
        </div>
      ) : null}

      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-04)' }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--type-code-01)', color: 'var(--text-helper)' }}>{format(min)}</span>
        <div style={{ position: 'relative', flex: 1, height: 16, display: 'flex', alignItems: 'center' }}>
          {/* rail */}
          <div style={{ position: 'absolute', left: 0, right: 0, height: 2, background: 'var(--gray-30)' }} />
          {/* filled */}
          <div style={{ position: 'absolute', left: 0, width: `${pct}%`, height: 2, background: disabled ? 'var(--text-disabled)' : 'var(--text-primary)' }} />
          {/* thumb */}
          <div style={{
            position: 'absolute', left: `${pct}%`, transform: 'translateX(-50%)',
            width: 14, height: 14, borderRadius: '50%',
            background: disabled ? 'var(--text-disabled)' : 'var(--text-primary)',
            boxShadow: active ? '0 0 0 3px var(--blue-20)' : 'none',
            transition: 'box-shadow var(--duration-fast) var(--ease-productive)',
            pointerEvents: 'none',
          }} />
          <input
            type="range"
            min={min} max={max} step={step} value={value} disabled={disabled}
            onChange={(e) => onChange && onChange(Number(e.target.value))}
            onMouseDown={() => setActive(true)}
            onMouseUp={() => setActive(false)}
            onFocus={() => setActive(true)}
            onBlur={() => setActive(false)}
            style={{
              position: 'absolute', left: 0, right: 0, width: '100%', margin: 0,
              height: 16, opacity: 0, cursor: disabled ? 'not-allowed' : 'pointer',
            }}
          />
        </div>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--type-code-01)', color: 'var(--text-helper)' }}>{format(max)}</span>
      </div>
    </div>
  );
}
