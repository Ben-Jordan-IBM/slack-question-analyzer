// Week in Review — "Pulse": animated trend hero + ranked rows with movement.
function WeekView({ onInspect }) {
  // Real weekly stats from the latest saved analysis; demo data only when
  // nothing has been analyzed yet. undefined = still loading.
  const [weekly, setWeekly] = React.useState(undefined);
  // null = the latest week; a 'YYYY-MM-DD' Monday reviews that calendar
  // week (chart dots and the Latest-week button drive this)
  const [selectedWeek, setSelectedWeek] = React.useState(null);
  // A busy week can rank hundreds of rows (each singleton is a row):
  // paginate like the dashboard does instead of animating them all at once
  const [visibleCount, setVisibleCount] = React.useState(50);
  // Week navigation keeps the current view on screen and dims it while
  // the next week loads — resetting to the full-page spinner tore down
  // the chart mid-click and replayed every entrance animation
  const [refreshing, setRefreshing] = React.useState(false);
  React.useEffect(() => {
    let cancelled = false;
    if (!window.QA_API) { setWeekly(null); return; }
    setRefreshing(true);
    setVisibleCount(50);
    window.QA_API.latestWeekly(selectedWeek).then((w) => {
      if (!cancelled) { setWeekly(w); setRefreshing(false); }
    });
    return () => { cancelled = true; };
  }, [selectedWeek]);

  if (weekly === undefined) {
    return (
      <div className="qa-page" style={{ maxWidth: 1040, margin: '0 auto', padding: '60px 40px', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10, color: 'var(--text-secondary)', fontSize: 14 }}>
        <Icon name="loader" size={16} /> Loading weekly trends…
      </div>
    );
  }

  const d = weekly;

  if (!d) {
    // Backend-driven empty state: no mock data, ever
    return (
      <div className="qa-page" style={{ maxWidth: 1040, margin: '0 auto', padding: '70px 40px 80px', width: '100%', textAlign: 'center' }}>
        <h2 style={{ fontSize: 32, fontWeight: 300, marginBottom: 16 }}>Week in Review</h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: 16, maxWidth: 460, margin: '0 auto' }}>
          Weekly trends appear here once you have analyzed a transcript with dated questions.
        </p>
      </div>
    );
  }

  const onLatestWeek = d.week === d.latestWeek;
  const max = d.groups.length ? d.groups[0].count : 1;
  // deltaPct === null means there is no prior week to compare against
  const hasBaseline = d.deltaPct !== null && d.deltaPct !== undefined;
  const rising = hasBaseline && d.deltaPct >= 0;
  const deltaColor = rising ? 'var(--green-60)' : 'var(--red-60)';
  // Both comparison bars scale against the bigger week — hardcoding "this
  // week" to 100% made a falling week look flat
  const weekMax = Math.max(1, d.totalThisWeek, d.totalLastWeek);

  return (
    <div className="qa-page" style={{ maxWidth: 1040, margin: '0 auto', padding: '36px 40px 80px', width: '100%',
      opacity: refreshing ? 0.55 : 1, transition: 'opacity 160ms var(--ease-productive)' }}>
      <Reveal>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-helper)', fontWeight: 500 }}>
              Week in review
            </div>
            <div style={{ fontSize: 22, fontWeight: 300, letterSpacing: '-.01em', marginTop: 4 }}>{d.weekLabel}</div>
          </div>
          <button onClick={() => setSelectedWeek(null)} disabled={onLatestWeek}
            title={onLatestWeek ? 'Showing the newest week of your data' : 'Back to the newest week'}
            style={{ display: 'inline-flex', alignItems: 'center', gap: 6, height: 32, padding: '0 12px', fontFamily: 'var(--font-sans)',
              border: `1px solid ${onLatestWeek ? 'var(--border-subtle)' : 'var(--blue-60)'}`,
              background: 'var(--layer-02)', fontSize: 13,
              color: onLatestWeek ? 'var(--text-secondary)' : 'var(--link)',
              cursor: onLatestWeek ? 'default' : 'pointer' }}>
            <Icon name="calendar" size={14} /> {onLatestWeek ? 'Latest week' : 'Back to latest week'}
          </button>
        </div>
      </Reveal>

      {/* Trend hero */}
      <Reveal delay={80}>
        <div className="qa-hero" style={{ display: 'grid', gridTemplateColumns: '1fr 320px', border: '1px solid var(--border-subtle)', marginBottom: 34, background: 'var(--layer-02)' }}>
          <div style={{ padding: '22px 26px 14px' }}>
            <div style={{ fontSize: 11, color: 'var(--text-helper)', fontWeight: 500, marginBottom: 4 }}>Weekly question volume</div>
            <AreaChart data={d.trend} labels={d.trendLabels} width={560} height={232}
              selected={d.trendWeeks ? (d.trendWeeks.indexOf(d.week) === -1 ? null : d.trendWeeks.indexOf(d.week)) : null}
              onPointClick={(i) => { if (d.trendWeeks && d.trendWeeks[i]) setSelectedWeek(d.trendWeeks[i]); }} />
            <div style={{ fontSize: 11, color: 'var(--text-placeholder)', marginTop: 2 }}>
              Calendar weeks (Mon – Sun), labeled by their Monday. Click a dot to review that week.
            </div>
          </div>
          <div style={{ padding: '22px 26px', display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 20, borderLeft: '1px solid var(--border-subtle)', background: 'var(--field)' }}>
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-helper)', fontWeight: 500, marginBottom: 8 }}>Vs. last week</div>
              {hasBaseline ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Icon name={rising ? 'trending-up' : 'trending-down'} size={24} color={deltaColor} />
                  <span style={{ fontSize: 42, fontWeight: 300, fontFamily: 'var(--font-mono)', lineHeight: 1, color: deltaColor }}>{rising ? '+' : '−'}<CountUp to={Math.abs(d.deltaPct)} duration={1300} />%</span>
                </div>
              ) : (
                <div style={{ fontSize: 15, color: 'var(--text-secondary)' }}>
                  {onLatestWeek ? 'First week of data — trends appear next week'
                    : 'No questions in the week before this one to compare against'}
                </div>
              )}
            </div>

            {/* last vs this week comparison */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text-helper)', marginBottom: 5 }}><span>Last week</span><span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>{d.totalLastWeek}</span></div>
                <Bar pct={(d.totalLastWeek / weekMax) * 100} color="var(--gray-40)" bg="var(--layer-hover)" height={8} delay={220} />
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text-primary)', marginBottom: 5 }}><span style={{ fontWeight: 500 }}>This week</span><span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{d.totalThisWeek}</span></div>
                <Bar pct={(d.totalThisWeek / weekMax) * 100} color="var(--blue-60)" bg="var(--layer-hover)" height={8} delay={360} />
              </div>
            </div>

            <div style={{ height: 1, background: 'var(--border-subtle)' }} />
            <div style={{ display: 'flex', gap: 30 }}>
              <div><div style={{ fontSize: 26, fontWeight: 300, color: 'var(--text-primary)' }}><CountUp to={d.newQuestionTypes} /></div><div style={{ fontSize: 11, color: 'var(--text-helper)', marginTop: 2 }}>new topics</div></div>
              <div><div style={{ fontSize: 26, fontWeight: 300, color: 'var(--teal-60)' }}><CountUp to={d.answered} /></div><div style={{ fontSize: 11, color: 'var(--text-helper)', marginTop: 2 }}>answered</div></div>
              {d.feedback > 0 ? <div><div style={{ fontSize: 26, fontWeight: 300, color: 'var(--purple-60, #8a3ffc)' }}><CountUp to={d.feedback} /></div><div style={{ fontSize: 11, color: 'var(--text-helper)', marginTop: 2 }}>feature requests</div></div> : null}
            </div>
          </div>
        </div>
      </Reveal>

      {/* Ranked rows */}
      <Reveal delay={160}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
          <span style={{ fontSize: 13, color: 'var(--text-helper)', fontWeight: 500 }}>
            Questions this week, by frequency
          </span>
        </div>
      </Reveal>

      <div style={{ borderTop: '2px solid var(--text-primary)', borderBottom: '1px solid var(--border-subtle)', background: 'var(--layer-02)' }}>
        {d.groups.length === 0 ? (
          <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-helper)', fontSize: 13.5 }}>
            No questions in this week of your data.
          </div>
        ) : null}
        {d.groups.slice(0, visibleCount).map((g, i) => (
          <RankedRow key={g.rank} rank={g.rank} index={i} question={g.question} count={g.count}
            maxCount={max} keywords={g.keywords} movement={g.movement}
            topic={g.topic} theme={g.theme} summary={g.summary} seenIn={g.seenIn}
            aiConfirmed={g.aiConfirmed} needsReview={g.needsReview} answered={g.answered}
            similarity={g.similarity} questions={g.questions}
            onInspect={onInspect ? () => onInspect(g.topic || g.question) : null} />
        ))}
      </div>
      {d.groups.length > visibleCount ? (
        <button onClick={() => setVisibleCount(visibleCount + 50)}
          style={{ display: 'block', width: '100%', padding: '12px 0', marginTop: 2, background: 'var(--layer-02)', border: '1px solid var(--border-subtle)', cursor: 'pointer', fontSize: 13, color: 'var(--text-secondary)' }}>
          Show 50 more ({d.groups.length - visibleCount} remaining)
        </button>
      ) : null}
    </div>
  );
}
window.WeekView = WeekView;
