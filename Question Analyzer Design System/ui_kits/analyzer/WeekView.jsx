// Week in Review — "Pulse": animated trend hero + ranked rows with movement.
function WeekView() {
  // Real weekly stats from the latest saved analysis; demo data only when
  // nothing has been analyzed yet. undefined = still loading.
  const [weekly, setWeekly] = React.useState(undefined);
  React.useEffect(() => {
    let cancelled = false;
    if (!window.QA_API) { setWeekly(null); return; }
    window.QA_API.latestWeekly().then((w) => { if (!cancelled) setWeekly(w); });
    return () => { cancelled = true; };
  }, []);

  if (weekly === undefined) {
    return (
      <div style={{ maxWidth: 1040, margin: '0 auto', padding: '60px 40px', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10, color: 'var(--text-secondary)', fontSize: 14 }}>
        <Icon name="loader" size={16} /> Loading weekly trends…
      </div>
    );
  }

  const d = weekly;

  if (!d || !d.groups || d.groups.length === 0) {
    // Backend-driven empty state: no mock data, ever
    return (
      <div style={{ maxWidth: 1040, margin: '0 auto', padding: '70px 40px 80px', width: '100%', textAlign: 'center' }}>
        <h2 style={{ fontSize: 32, fontWeight: 300, marginBottom: 16 }}>Week in Review</h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: 16, maxWidth: 460, margin: '0 auto' }}>
          {d ? 'No questions in the most recent week of your latest analysis — upload a newer transcript to see trends.'
             : 'Weekly trends appear here once you have analyzed a transcript with dated questions.'}
        </p>
      </div>
    );
  }

  const max = d.groups[0].count;
  const rising = d.deltaPct >= 0;
  const deltaColor = rising ? 'var(--green-60)' : 'var(--red-60)';

  return (
    <div style={{ maxWidth: 1040, margin: '0 auto', padding: '36px 40px 80px', width: '100%' }}>
      <Reveal>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-helper)', fontWeight: 500 }}>
              Week in review
            </div>
            <div style={{ fontSize: 22, fontWeight: 300, letterSpacing: '-.01em', marginTop: 4 }}>{d.weekLabel}</div>
          </div>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6, height: 32, padding: '0 12px', border: '1px solid var(--border-subtle)', background: '#fff', fontSize: 13, color: 'var(--text-secondary)' }}>
            <Icon name="calendar" size={14} /> Latest week
          </div>
        </div>
      </Reveal>

      {/* Trend hero */}
      <Reveal delay={80}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', border: '1px solid var(--border-subtle)', marginBottom: 34, background: '#fff' }}>
          <div style={{ padding: '22px 26px 14px' }}>
            <div style={{ fontSize: 11, color: 'var(--text-helper)', fontWeight: 500, marginBottom: 4 }}>Weekly question volume</div>
            <AreaChart data={d.trend} labels={d.trendLabels} width={560} height={232} />
          </div>
          <div style={{ padding: '22px 26px', display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 20, borderLeft: '1px solid var(--border-subtle)', background: 'var(--gray-10)' }}>
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-helper)', fontWeight: 500, marginBottom: 8 }}>Vs. last week</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Icon name={rising ? 'trending-up' : 'trending-down'} size={24} color={deltaColor} />
                <span style={{ fontSize: 42, fontWeight: 300, fontFamily: 'var(--font-mono)', lineHeight: 1, color: deltaColor }}>{rising ? '+' : '−'}<CountUp to={Math.abs(d.deltaPct)} duration={1300} />%</span>
              </div>
            </div>

            {/* last vs this week comparison */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text-helper)', marginBottom: 5 }}><span>Last week</span><span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>{d.totalLastWeek}</span></div>
                <Bar pct={d.totalThisWeek > 0 ? Math.min(100, (d.totalLastWeek / d.totalThisWeek) * 100) : 100} color="var(--gray-40)" bg="var(--gray-20)" height={8} delay={220} />
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text-primary)', marginBottom: 5 }}><span style={{ fontWeight: 500 }}>This week</span><span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{d.totalThisWeek}</span></div>
                <Bar pct={100} color="var(--blue-60)" bg="var(--gray-20)" height={8} delay={360} />
              </div>
            </div>

            <div style={{ height: 1, background: 'var(--border-subtle)' }} />
            <div style={{ display: 'flex', gap: 30 }}>
              <div><div style={{ fontSize: 26, fontWeight: 300, color: 'var(--text-primary)' }}><CountUp to={d.newQuestionTypes} /></div><div style={{ fontSize: 11, color: 'var(--text-helper)', marginTop: 2 }}>new topics</div></div>
              <div><div style={{ fontSize: 26, fontWeight: 300, color: 'var(--teal-60)' }}><CountUp to={d.answered} /></div><div style={{ fontSize: 11, color: 'var(--text-helper)', marginTop: 2 }}>answered</div></div>
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

      <div style={{ borderTop: '2px solid var(--gray-100)', borderBottom: '1px solid var(--border-subtle)', background: '#fff' }}>
        {d.groups.map((g, i) => (
          <RankedRow key={g.rank} rank={g.rank} index={i} question={g.question} count={g.count}
            maxCount={max} keywords={g.keywords} movement={g.movement}
            topic={g.topic} similarity={g.similarity} questions={g.questions} />
        ))}
      </div>
    </div>
  );
}
window.WeekView = WeekView;
