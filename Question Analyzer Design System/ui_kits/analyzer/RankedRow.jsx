// Animated, expandable ranked question row — shared by Dashboard & Week.
function RankedRow({ rank, question, count, maxCount, keywords = [], movement = null,
  similarity = null, questions = null, index = 0, defaultOpen = false,
  topic = null, summary = null, seenIn = 0, onRenameTopic = null, aiConfirmed = false,
  theme = null, answered = 0, needsReview = false, dateRange = null,
  onInspect = null, onFaqTools = null }) {
  const [open, setOpen] = React.useState(defaultOpen);
  const [hover, setHover] = React.useState(false);
  // Reduced-motion users get rows immediately (matches Reveal/Bar/CountUp)
  const [shown, setShown] = React.useState(QA_REDUCED);
  const [editing, setEditing] = React.useState(false);
  const [editName, setEditName] = React.useState('');
  const bodyRef = React.useRef(null);
  const [bodyH, setBodyH] = React.useState(0);
  const expandable = !!(questions && questions.length);

  // Cap the entrance stagger so long lists don't take seconds to appear
  const stagger = Math.min(index, 12);
  React.useEffect(() => { if (QA_REDUCED) return; const id = setTimeout(() => setShown(true), 80 + stagger * 70); return () => clearTimeout(id); }, []);
  React.useEffect(() => { if (bodyRef.current) setBodyH(bodyRef.current.scrollHeight); }, [open, questions]);

  // Color by relative count, not rank position: tied groups must look equal
  const pct = Math.max(6, Math.round((count / Math.max(1, maxCount)) * 100));
  const heat = pct >= 90 ? 'var(--blue-60)' : pct >= 60 ? 'var(--blue-50)' : pct >= 30 ? 'var(--blue-40)' : 'var(--gray-40)';

  return (
    <div style={{
      borderBottom: '1px solid var(--border-subtle)',
      background: hover ? 'var(--layer-hover)' : 'transparent',
      borderLeft: `3px solid ${open ? heat : 'transparent'}`,
      transition: 'background var(--duration-base) var(--ease-productive), border-left-color var(--duration-base), opacity 480ms var(--ease-entrance), transform 480ms var(--ease-entrance)',
      opacity: shown ? 1 : 0, transform: shown ? 'none' : 'translateY(10px)',
    }}>
      <div onClick={() => expandable && setOpen(!open)}
        onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}
        className={movement != null ? 'qa-rr-grid qa-rr-6' : 'qa-rr-grid qa-rr-5'}
        style={{
          display: 'grid', gridTemplateColumns: movement != null ? '30px 52px 1fr 168px 46px 22px' : '34px 1fr 168px 46px 22px',
          alignItems: 'center', gap: 16, padding: '15px 20px', cursor: expandable ? 'pointer' : 'default',
        }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 18, color: heat, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{String(rank).padStart(2, '0')}</span>
        {movement != null ? <span><MovementBadge movement={movement} /></span> : null}
        <span style={{ minWidth: 0 }}>
          {needsReview ? (
            <div style={{ marginBottom: 3 }}>
              <StatusChip kind="warn" title="No category fits this cluster — it sits in the review pile (a recurring review cluster means a category is missing)">needs review</StatusChip>
            </div>
          ) : null}
          {topic ? (
            <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '.04em', textTransform: 'uppercase', color: heat, marginBottom: 3, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {theme ? <span title="Broad theme this topic falls under" style={{ color: 'var(--text-helper)', fontWeight: 500 }}>{theme} · </span> : null}
              {editing ? (
                <input autoFocus value={editName}
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) => setEditName(e.target.value)}
                  onKeyDown={(e) => {
                    e.stopPropagation();
                    if (e.key === 'Enter') {
                      const clean = editName.trim();
                      if (clean && clean !== topic) {
                        onRenameTopic(clean);
                        if (window.QA_TOAST) window.QA_TOAST('Topic renamed');
                      }
                      setEditing(false);
                    } else if (e.key === 'Escape') { setEditing(false); }
                  }}
                  onBlur={() => setEditing(false)}
                  style={{ fontFamily: 'var(--font-sans)', fontSize: 12, fontWeight: 600, letterSpacing: '.04em', textTransform: 'uppercase', color: 'var(--text-primary)', background: 'var(--field)', border: '1px solid var(--blue-60)', outline: 'none', padding: '1px 6px', width: 200 }} />
              ) : topic}
              {!editing && seenIn > 1 ? <StatusChip title={`This topic has come up in ${seenIn} analyses`} style={{ marginLeft: 8, textTransform: 'none', letterSpacing: 0, fontWeight: 500 }}>recurring ×{seenIn}</StatusChip> : null}
              {!editing && onRenameTopic ? (
                <button title="Rename this topic (updates the learned bank)" aria-label="Rename topic"
                  onClick={(e) => { e.stopPropagation(); setEditName(topic); setEditing(true); }}
                  style={{ marginLeft: 6, background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--text-helper)', padding: 0, verticalAlign: 'middle', display: 'inline-flex',
                    // Revealed by ROW hover — an always-faint 11px pencil was invisible
                    opacity: hover ? 1 : 0.25, transition: 'opacity var(--duration-base) var(--ease-productive)' }}
                  onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--blue-60)'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-helper)'; }}>
                  <Icon name="pencil" size={13} />
                </button>
              ) : null}
            </div>
          ) : null}
          <div title={open ? undefined : question} style={{ fontSize: 15, color: 'var(--text-primary)', lineHeight: 1.3, whiteSpace: open ? 'normal' : 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{question}</div>
          {keywords.length ? (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginTop: 6 }}>
              {keywords.slice(0, 4).map((k, i) => (
                <span key={i} style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-helper)', background: 'var(--field)', padding: '1px 7px' }}>{k}</span>
              ))}
            </div>
          ) : null}
        </span>
        <Bar pct={pct} color={heat} height={8} delay={stagger * 70} duration={1000} />
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 15, fontWeight: 500, textAlign: 'right', color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>{count}×</span>
        <span style={{ display: 'inline-flex', justifyContent: 'center', color: 'var(--text-secondary)', opacity: expandable ? 1 : 0, transform: open ? 'rotate(180deg)' : 'none', transition: 'transform var(--duration-moderate) var(--ease-productive)' }}>
          <Icon name="chevron-down" size={16} />
        </span>
      </div>

      {expandable ? (
        <div style={{ maxHeight: open ? bodyH : 0, overflow: 'hidden', transition: 'max-height var(--duration-slow) var(--ease-productive)' }}>
          <div ref={bodyRef} style={{ padding: '0 20px 18px', marginLeft: movement != null ? 98 : 50 }}>
            {summary ? <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: 10, fontStyle: 'italic' }}>{summary}</div> : null}
            {similarity && similarity !== '—' ? <div style={{ fontSize: 12, color: 'var(--text-helper)', marginBottom: 10 }}>{questions.length} occurrence{questions.length === 1 ? '' : 's'} · <span title="How closely the phrasings in this group match each other">wording match <b style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>{similarity}</b></span>{aiConfirmed ? <span title="This group sat near the grouping boundary, so the AI double-checked that one answer really covers every member"> · verified by AI</span> : ''}{answered > 0 ? <span title="Occurrences whose thread replies actually answered the question" style={{ color: 'var(--tag-ok-fg)' }}> · {answered} answered</span> : null}{dateRange && dateRange.first_asked ? <span title="When this was first and most recently asked"> · first asked {dateRange.first_asked}{dateRange.last_asked && dateRange.last_asked !== dateRange.first_asked ? ` · last ${dateRange.last_asked}` : ''}</span> : null}</div> : null}
            {onInspect ? (
              <button onClick={(e) => { e.stopPropagation(); onInspect(); }}
                style={{ display: 'inline-flex', alignItems: 'center', gap: 6, marginBottom: 10, padding: 0, background: 'transparent', border: 'none', cursor: 'pointer', fontFamily: 'var(--font-sans)', fontSize: 12.5, color: 'var(--link)' }}>
                View full history in Dashboard <Icon name="arrow-right" size={13} />
              </button>
            ) : null}
            {onFaqTools ? (
              <button onClick={(e) => { e.stopPropagation(); onFaqTools(); }}
                title="Open Learned topics: save an approved answer, mark the FAQ published, and track whether asks fall afterward"
                style={{ display: 'inline-flex', alignItems: 'center', gap: 6, marginBottom: 10, marginRight: 16, padding: 0, background: 'transparent', border: 'none', cursor: 'pointer', fontFamily: 'var(--font-sans)', fontSize: 12.5, color: 'var(--link)' }}>
                <Icon name="book-marked" size={13} /> FAQ tools: curate the answer, track the published doc
              </button>
            ) : null}
            <ul style={{ listStyle: 'none', margin: 0, padding: 0, borderLeft: '1px solid var(--border-subtle)' }}>
              {questions.map((q, i) => (
                <li key={i} style={{
                  display: 'flex', justifyContent: 'space-between', gap: 16, padding: '8px 16px',
                  opacity: open ? 1 : 0, transform: open ? 'none' : 'translateX(-6px)',
                  transition: `opacity 360ms ${i * 70}ms var(--ease-entrance), transform 360ms ${i * 70}ms var(--ease-entrance)`,
                }}>
                  <span style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                    {q.answered === true ? <StatusChip kind="ok" title="A thread reply answered this question" style={{ marginRight: 8 }}>answered</StatusChip> : null}
                    {q.answered === false ? <StatusChip kind="err" title="This question has a thread but no reply actually answered it" style={{ marginRight: 8 }}>unanswered</StatusChip> : null}
                    {q.text}
                  </span>
                  {q.date ? <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-placeholder)', whiteSpace: 'nowrap' }}>{q.date}</span> : null}
                </li>
              ))}
            </ul>
          </div>
        </div>
      ) : null}
    </div>
  );
}
window.RankedRow = RankedRow;
