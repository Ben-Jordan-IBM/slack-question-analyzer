// Shared Lucide icon helper for the UI kit.
function Icon({ name, size = 16, stroke = 2, color = 'currentColor', style = {} }) {
  const ref = React.useRef(null);
  React.useEffect(() => {
    if (ref.current && window.lucide) {
      // setAttribute, never string-interpolated innerHTML: every current
      // caller passes a literal, but this stays safe if an icon name is
      // ever wired from data
      ref.current.textContent = '';
      const el = document.createElement('i');
      el.setAttribute('data-lucide', name);
      ref.current.appendChild(el);
      window.lucide.createIcons({ attrs: { width: size, height: size, 'stroke-width': stroke }, nameAttr: 'data-lucide' });
    }
  }, [name, size, stroke]);
  return <span ref={ref} style={{ display: 'inline-flex', color, ...style }} />;
}
window.Icon = Icon;
