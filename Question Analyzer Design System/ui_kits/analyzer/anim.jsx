// ============================================================
// Animation toolkit — count-up, reveal, animated bars, and a
// polished SVG area chart with draw-on, gridlines, hover crosshair.
// ============================================================

// Count a number up on mount (easeOutCubic).
const QA_REDUCED = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

function useCountUp(target, opts = {}) {
  const { duration = 1100, decimals = 0, start = 0 } = opts;
  const [val, setVal] = React.useState(QA_REDUCED ? target : start);
  React.useEffect(() => {
    if (QA_REDUCED) { setVal(target); return; }
    let raf; const t0 = performance.now();
    const tick = (t) => {
      const p = Math.min(1, (t - t0) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setVal(start + (target - start) * eased);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target]);
  return Number(val.toFixed(decimals)).toLocaleString();
}

function CountUp({ to, decimals = 0, duration = 1100, prefix = '', suffix = '' }) {
  const v = useCountUp(to, { decimals, duration });
  return <React.Fragment>{prefix}{v}{suffix}</React.Fragment>;
}

// Fade + slide children in after `delay` ms.
function Reveal({ children, delay = 0, y = 14, dur = 520, style = {} }) {
  const [shown, setShown] = React.useState(QA_REDUCED);
  React.useEffect(() => { if (QA_REDUCED) return; const id = setTimeout(() => setShown(true), delay); return () => clearTimeout(id); }, []);
  return (
    <div style={{
      opacity: shown ? 1 : 0,
      transform: shown ? 'none' : `translateY(${y}px)`,
      transition: `opacity ${dur}ms var(--ease-entrance), transform ${dur}ms var(--ease-entrance)`,
      ...style,
    }}>{children}</div>
  );
}

// Bar whose fill animates from 0 → pct%.
function Bar({ pct, color, height = 8, bg = 'var(--field)', delay = 0, duration = 900, radius = 0 }) {
  const [w, setW] = React.useState(QA_REDUCED ? pct : 0);
  React.useEffect(() => { if (QA_REDUCED) { setW(pct); return; } const id = setTimeout(() => setW(pct), delay + 40); return () => clearTimeout(id); }, [pct, delay]);
  return (
    <span style={{ display: 'block', height, background: bg, position: 'relative', overflow: 'hidden', borderRadius: radius }}>
      <span style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${w}%`, background: color, transition: `width ${duration}ms var(--ease-productive)` }} />
    </span>
  );
}

// Smooth Catmull-Rom → cubic bezier path.
function smoothPath(pts) {
  if (pts.length < 2) return '';
  let d = `M ${pts[0].x} ${pts[0].y}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || p2;
    const c1x = p1.x + (p2.x - p0.x) / 6, c1y = p1.y + (p2.y - p0.y) / 6;
    const c2x = p2.x - (p3.x - p1.x) / 6, c2y = p2.y - (p3.y - p1.y) / 6;
    d += ` C ${c1x.toFixed(1)} ${c1y.toFixed(1)}, ${c2x.toFixed(1)} ${c2y.toFixed(1)}, ${p2.x} ${p2.y}`;
  }
  return d;
}

// Polished animated area chart.
function AreaChart({ data, labels, width = 720, height = 240, accent = 'var(--blue-60)', marker = null, onPointClick = null }) {
  const padL = 16, padR = 16, padT = 30, padB = 28;
  const plotW = width - padL - padR, plotH = height - padT - padB;
  const max = Math.max(...data) * 1.18, min = Math.min(...data) * 0.6;
  // A one-point series has no span to divide by — center the point.
  // (A topic first seen this week is the most common history shape.)
  const span = Math.max(data.length - 1, 1);
  const range = (max - min) || 1;
  const x = (i) => data.length === 1 ? padL + plotW / 2 : padL + (i / span) * plotW;
  const y = (v) => padT + plotH - ((v - min) / range) * plotH;
  const pts = data.map((v, i) => ({ x: x(i), y: y(v) }));
  const line = smoothPath(pts);
  const area = `${line} L ${pts[pts.length - 1].x} ${padT + plotH} L ${pts[0].x} ${padT + plotH} Z`;
  const grid = [0, 0.25, 0.5, 0.75, 1].map((f) => padT + plotH - f * plotH);

  const lineRef = React.useRef(null);
  const [drawn, setDrawn] = React.useState(QA_REDUCED);
  const [hover, setHover] = React.useState(null);
  const svgRef = React.useRef(null);

  React.useEffect(() => {
    const path = lineRef.current; if (!path) return;
    if (QA_REDUCED) { setDrawn(true); return; }
    const len = path.getTotalLength();
    path.style.transition = 'none';
    path.style.strokeDasharray = len;
    path.style.strokeDashoffset = len;
    // force reflow then animate
    path.getBoundingClientRect();
    requestAnimationFrame(() => {
      path.style.transition = 'stroke-dashoffset 1500ms var(--ease-productive)';
      path.style.strokeDashoffset = '0';
    });
    const id = setTimeout(() => setDrawn(true), 700);
    return () => clearTimeout(id);
  }, []);

  const indexFromEvent = (e) => {
    const r = svgRef.current.getBoundingClientRect();
    const mx = ((e.clientX - r.left) / r.width) * width;
    const idx = Math.round((mx - padL) / (plotW / span));
    return Math.max(0, Math.min(data.length - 1, idx));
  };
  const onMove = (e) => setHover(indexFromEvent(e));

  const last = data.length - 1;
  const uid = React.useMemo(() => 'ac' + Math.random().toString(36).slice(2, 8), []);

  return (
    <svg ref={svgRef} viewBox={`0 0 ${width} ${height}`}
      style={{ width: '100%', height: 'auto', display: 'block', cursor: onPointClick && hover != null ? 'pointer' : 'default' }}
      onMouseMove={onMove} onMouseLeave={() => setHover(null)}
      // Index from the CLICK coordinates, not hover state: a touch tap
      // fires click without any mousemove, and must still navigate
      onClick={(e) => { if (onPointClick) onPointClick(indexFromEvent(e)); }}
      tabIndex={onPointClick ? 0 : undefined}
      role={onPointClick ? 'group' : undefined}
      aria-label={onPointClick ? 'Weekly trend chart. Use arrow keys to pick a week, Enter to open it.' : undefined}
      onKeyDown={(e) => {
        if (!onPointClick) return;
        if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
          e.preventDefault();
          const base = hover != null ? hover : data.length - 1;
          const next = e.key === 'ArrowLeft' ? Math.max(0, base - 1) : Math.min(data.length - 1, base + 1);
          setHover(next);
        } else if (e.key === 'Enter' && hover != null) {
          onPointClick(hover);
        }
      }}>
      <defs>
        <linearGradient id={uid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={accent} stopOpacity="0.20" />
          <stop offset="100%" stopColor={accent} stopOpacity="0" />
        </linearGradient>
      </defs>
      {/* gridlines */}
      {grid.map((gy, i) => (
        <line key={i} x1={padL} y1={gy} x2={width - padR} y2={gy} stroke="var(--border-subtle)" strokeWidth="1"
          strokeDasharray={i === grid.length - 1 ? '0' : '2 4'} shapeRendering="crispEdges" />
      ))}
      {/* area */}
      <path d={area} fill={`url(#${uid})`} style={{ opacity: drawn ? 1 : 0, transition: 'opacity 700ms var(--ease-entrance)' }} />
      {/* line */}
      <path ref={lineRef} d={line} fill="none" stroke={accent} strokeWidth="2.5"
        strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
      {/* value labels */}
      {pts.map((p, i) => (
        <text key={i} x={p.x} y={p.y - 12} textAnchor="middle"
          style={{ fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: i === last ? 600 : 400,
            fill: i === last ? accent : 'var(--text-placeholder)',
            opacity: drawn ? 1 : 0, transition: `opacity 400ms ${200 + i * 80}ms var(--ease-entrance)` }}>{data[i]}</text>
      ))}
      {/* x labels */}
      {pts.map((p, i) => (
        <text key={i} x={p.x} y={height - 8} textAnchor="middle"
          style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fill: i === last ? 'var(--text-secondary)' : 'var(--text-placeholder)' }}>{labels[i]}</text>
      ))}
      {/* hover crosshair */}
      {hover != null ? (
        <g>
          <line x1={pts[hover].x} y1={padT - 6} x2={pts[hover].x} y2={padT + plotH} stroke="var(--gray-40)" strokeWidth="1" strokeDasharray="2 3" />
          <circle cx={pts[hover].x} cy={pts[hover].y} r="6" fill="var(--layer-02)" stroke={accent} strokeWidth="2.5" />
        </g>
      ) : null}
      {/* points */}
      {pts.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={i === last ? 4.5 : 3}
          fill={i === last ? accent : 'var(--layer-02)'} stroke={accent} strokeWidth="2"
          style={{ opacity: drawn ? 1 : 0, transition: `opacity 360ms ${300 + i * 70}ms var(--ease-entrance)` }} />
      ))}
      {/* live pulse on latest */}
      {drawn ? <circle cx={pts[last].x} cy={pts[last].y} r="4.5" fill="none" stroke={accent} strokeWidth="2" className="qa-pulse" /> : null}
      {/* optional event marker (e.g. "FAQ published") at a data point */}
      {marker && pts[marker.index] ? (
        <g>
          <line x1={pts[marker.index].x} y1={padT - 4} x2={pts[marker.index].x} y2={padT + plotH}
            stroke="var(--teal-60, #009d9a)" strokeWidth="1.5" strokeDasharray="4 3" />
          <text x={pts[marker.index].x + 4} y={padT + 6}
            style={{ fontFamily: 'var(--font-mono)', fontSize: 9.5, fill: 'var(--teal-60, #009d9a)' }}>
            {marker.label}
          </text>
        </g>
      ) : null}
    </svg>
  );
}

// NEW / ▲2 / ▼1 movement marker.
function MovementBadge({ movement }) {
  if (movement === 'new') {
    return <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '.5px', color: 'var(--link)', background: 'var(--chip-active-bg)', padding: '2px 7px' }}>NEW</span>;
  }
  if (typeof movement === 'number' && movement !== 0) {
    const up = movement > 0;
    return <span style={{ display: 'inline-flex', alignItems: 'center', gap: 2, fontFamily: 'var(--font-mono)', fontSize: 12, color: up ? 'var(--green-60)' : 'var(--gray-50)' }}>
      <Icon name={up ? 'arrow-up' : 'arrow-down'} size={12} />{Math.abs(movement)}
    </span>;
  }
  return <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-placeholder)' }}>—</span>;
}

Object.assign(window, { useCountUp, CountUp, Reveal, Bar, AreaChart, MovementBadge });
