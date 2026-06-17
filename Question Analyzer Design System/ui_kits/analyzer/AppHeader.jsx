// App header: brand, animated Dashboard | Week in Review toggle,
// Upload transcript, and the account / sign-in avatar.
function AppHeader({ view, setView, onUpload, onHistory, onTopics, onSettings, version }) {
  const segs = [{ key: 'dashboard', label: 'Dashboard' }, { key: 'week', label: 'Week in Review' }];
  const activeIdx = segs.findIndex((s) => s.key === view);

  const iconBtn = {
    width: 32, height: 32, display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
    background: 'transparent', color: 'var(--gray-30)', border: '1px solid var(--gray-70)',
    cursor: 'pointer', transition: 'background var(--duration-base) var(--ease-productive), color var(--duration-base)',
  };
  const hover = (e) => { e.currentTarget.style.background = 'var(--gray-80)'; e.currentTarget.style.color = '#fff'; };
  const unhover = (e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--gray-30)'; };

  const seg = (s, i) => ({
    position: 'relative', zIndex: 1, height: 32, padding: '0 18px',
    display: 'inline-flex', alignItems: 'center', whiteSpace: 'nowrap',
    fontFamily: 'var(--font-sans)', fontSize: 13, cursor: 'pointer', border: 'none',
    background: 'transparent',
    color: view === s.key ? '#fff' : 'var(--gray-30)',
    transition: 'color var(--duration-moderate) var(--ease-productive)',
  });

  return (
    <header style={{ height: 48, background: 'var(--gray-100)', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 16px', flex: '0 0 auto', zIndex: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        {/* Animated segmented toggle */}
        <span style={{ position: 'relative', display: 'inline-flex', border: '1px solid var(--gray-80)' }}>
          <span style={{
            position: 'absolute', top: 0, bottom: 0, left: 0, width: `${100 / segs.length}%`,
            background: 'var(--blue-60)', transform: `translateX(${activeIdx * 100}%)`,
            transition: 'transform var(--duration-moderate) var(--ease-productive)', zIndex: 0,
          }} />
          {segs.map((s, i) => (
            <button key={s.key} style={seg(s, i)} onClick={() => setView(s.key)}>{s.label}</button>
          ))}
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        {version ? (
          <span title="Backend version — if this doesn't match what you just pulled, restart the server and hard-refresh (Ctrl+F5)"
            style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--gray-50)' }}>
            v{version}
          </span>
        ) : null}
        <button onClick={onUpload} style={{
          height: 32, padding: '0 14px', display: 'inline-flex', alignItems: 'center', gap: 8,
          background: 'transparent', color: '#fff', border: '1px solid var(--gray-70)',
          fontFamily: 'var(--font-sans)', fontSize: 13, cursor: 'pointer', whiteSpace: 'nowrap',
          transition: 'background var(--duration-base) var(--ease-productive), border-color var(--duration-base)',
        }}
          onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--gray-80)'; e.currentTarget.style.borderColor = 'var(--gray-50)'; }}
          onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.borderColor = 'var(--gray-70)'; }}>
          <Icon name="upload" size={15} /> Upload transcript
        </button>

        <button onClick={onHistory} title="Analysis history" aria-label="Analysis history"
          style={iconBtn} onMouseEnter={hover} onMouseLeave={unhover}>
          <Icon name="history" size={16} />
        </button>

        <button onClick={onTopics} title="Learned topics" aria-label="Learned topics"
          style={iconBtn} onMouseEnter={hover} onMouseLeave={unhover}>
          <Icon name="book-marked" size={16} />
        </button>

        <button onClick={onSettings} title="Analysis settings" aria-label="Analysis settings"
          style={iconBtn} onMouseEnter={hover} onMouseLeave={unhover}>
          <Icon name="settings" size={16} />
        </button>
      </div>
    </header>
  );
}
window.AppHeader = AppHeader;
