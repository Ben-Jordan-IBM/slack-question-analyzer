// Shared Lucide icon helper for the UI kit.
function Icon({ name, size = 16, stroke = 2, color = 'currentColor', style = {} }) {
  const ref = React.useRef(null);
  React.useEffect(() => {
    if (ref.current && window.lucide) {
      ref.current.innerHTML = `<i data-lucide="${name}"></i>`;
      window.lucide.createIcons({ attrs: { width: size, height: size, 'stroke-width': stroke }, nameAttr: 'data-lucide' });
    }
  }, [name, size, stroke]);
  return <span ref={ref} style={{ display: 'inline-flex', color, ...style }} />;
}
window.Icon = Icon;
