/* @ds-bundle: {"format":3,"namespace":"QuestionAnalyzerDesignSystem_03a921","components":[{"name":"Button","sourcePath":"components/Button/Button.jsx"},{"name":"Card","sourcePath":"components/Card/Card.jsx"},{"name":"FileDropzone","sourcePath":"components/FileDropzone/FileDropzone.jsx"},{"name":"MetricTile","sourcePath":"components/MetricTile/MetricTile.jsx"},{"name":"QuestionGroup","sourcePath":"components/QuestionGroup/QuestionGroup.jsx"},{"name":"Slider","sourcePath":"components/Slider/Slider.jsx"},{"name":"Tag","sourcePath":"components/Tag/Tag.jsx"}],"sourceHashes":{"components/Button/Button.jsx":"33bd71e0fcdb","components/Card/Card.jsx":"b1e7fb9aff3d","components/FileDropzone/FileDropzone.jsx":"76c7a5fe2c66","components/MetricTile/MetricTile.jsx":"cfdd0ef9f91b","components/QuestionGroup/QuestionGroup.jsx":"3f8ab9213ace","components/Slider/Slider.jsx":"f6c12e0dbb3a","components/Tag/Tag.jsx":"7c89450bc3ad"},"inlinedExternals":[],"unexposedExports":[]} */

(() => {

const __ds_ns = (window.QuestionAnalyzerDesignSystem_03a921 = window.QuestionAnalyzerDesignSystem_03a921 || {});

const __ds_scope = {};

(__ds_ns.__errors = __ds_ns.__errors || []);

// components/Button/Button.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * Button — IBM Carbon-style action trigger.
 * Sharp corners, asymmetric padding when an icon is present (Carbon hallmark:
 * label left, icon pinned right). Variants: primary, secondary, tertiary, ghost, danger.
 */
function Button({
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
  const heights = {
    sm: '2rem',
    md: '2.5rem',
    lg: '3rem'
  };
  const palettes = {
    primary: {
      bg: 'var(--button-primary)',
      bgHover: 'var(--button-primary-hover)',
      color: 'var(--text-on-color)',
      border: 'transparent'
    },
    secondary: {
      bg: 'var(--button-secondary)',
      bgHover: 'var(--button-secondary-hover)',
      color: 'var(--text-on-color)',
      border: 'transparent'
    },
    tertiary: {
      bg: 'transparent',
      bgHover: 'var(--blue-60)',
      color: 'var(--blue-60)',
      border: 'var(--blue-60)',
      colorHover: 'var(--text-on-color)'
    },
    ghost: {
      bg: 'transparent',
      bgHover: 'var(--layer-hover)',
      color: 'var(--blue-60)',
      border: 'transparent'
    },
    danger: {
      bg: 'var(--button-danger)',
      bgHover: 'var(--button-danger-hover)',
      color: 'var(--text-on-color)',
      border: 'transparent'
    }
  };
  const p = palettes[variant] || palettes.primary;
  const [hover, setHover] = React.useState(false);
  const hasIcon = !!icon;
  const style = {
    appearance: 'none',
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: hasIcon && !fullWidth ? 'space-between' : 'center',
    gap: 'var(--spacing-05)',
    width: fullWidth ? '100%' : 'auto',
    minWidth: variant === 'ghost' ? 0 : '6rem',
    height: heights[size] || heights.lg,
    // Carbon asymmetric padding: roomy right when icon sits at the edge
    padding: hasIcon ? '0 var(--spacing-05) 0 var(--spacing-05)' : '0 var(--spacing-07) 0 var(--spacing-05)',
    paddingRight: hasIcon && !fullWidth ? 'var(--spacing-09)' : undefined,
    fontFamily: 'var(--font-sans)',
    fontSize: 'var(--type-body-01)',
    fontWeight: 'var(--weight-regular)',
    lineHeight: 1,
    letterSpacing: '0.01em',
    textAlign: 'left',
    cursor: disabled ? 'not-allowed' : 'pointer',
    border: `1px solid ${p.border}`,
    borderRadius: 'var(--radius-none)',
    background: disabled ? 'var(--gray-20)' : hover ? p.bgHover : p.bg,
    color: disabled ? 'var(--text-disabled)' : hover && p.colorHover ? p.colorHover : p.color,
    transition: 'background var(--duration-base) var(--ease-productive), color var(--duration-base) var(--ease-productive)',
    outline: 'none',
    position: 'relative'
  };
  return /*#__PURE__*/React.createElement("button", _extends({
    type: type,
    style: style,
    disabled: disabled,
    onClick: onClick,
    onMouseEnter: () => setHover(true),
    onMouseLeave: () => setHover(false),
    onFocus: e => {
      e.target.style.boxShadow = 'var(--focus-ring-inset)';
    },
    onBlur: e => {
      e.target.style.boxShadow = 'none';
    }
  }, rest), /*#__PURE__*/React.createElement("span", null, children), icon ? /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'inline-flex',
      flex: '0 0 auto'
    }
  }, icon) : null);
}
Object.assign(__ds_scope, { Button });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/Button/Button.jsx", error: String((e && e.message) || e) }); }

// components/Card/Card.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * Card — Carbon layered surface. Sharp corners, 1px subtle border,
 * optional left accent bar and hover elevation.
 */
function Card({
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
    ...style
  };
  return /*#__PURE__*/React.createElement("div", _extends({
    style: base,
    onClick: onClick,
    onMouseEnter: () => setHover(true),
    onMouseLeave: () => setHover(false)
  }, rest), accent ? /*#__PURE__*/React.createElement("span", {
    style: {
      position: 'absolute',
      left: 0,
      top: 0,
      bottom: 0,
      width: 3,
      background: accent
    }
  }) : null, children);
}
Object.assign(__ds_scope, { Card });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/Card/Card.jsx", error: String((e && e.message) || e) }); }

// components/FileDropzone/FileDropzone.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * FileDropzone — Carbon file-uploader. Dashed field, drag-active state,
 * supported-format helper text, and a selected-file chip.
 */
function FileDropzone({
  accept = '.txt,.json,.csv',
  hint = 'TXT, JSON or CSV up to 200MB',
  title = 'Drag a Slack export here or click to browse',
  fileName = null,
  multiple = false,
  onFile,
  onClear,
  ...rest
}) {
  const [drag, setDrag] = React.useState(false);
  const inputRef = React.useRef(null);
  const pick = () => inputRef.current && inputRef.current.click();
  // onFile fires once per file so single-file callers keep working;
  // without `multiple`, extra files in one drop are ignored (not silently
  // kept in the input)
  const handle = fileList => {
    if (!onFile) return;
    const files = Array.from(fileList || []).slice(0, multiple ? undefined : 1);
    files.forEach(f => onFile(f));
  };
  if (fileName) {
    return /*#__PURE__*/React.createElement("div", _extends({
      style: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 'var(--spacing-04)',
        padding: 'var(--spacing-04) var(--spacing-05)',
        background: 'var(--layer-02)',
        border: '1px solid var(--border-subtle)',
        borderLeft: '3px solid var(--green-60)'
      }
    }, rest), /*#__PURE__*/React.createElement("span", {
      style: {
        display: 'inline-flex',
        alignItems: 'center',
        gap: 'var(--spacing-04)',
        minWidth: 0
      }
    }, /*#__PURE__*/React.createElement("svg", {
      width: "20",
      height: "20",
      viewBox: "0 0 20 20",
      fill: "none",
      style: {
        flex: '0 0 auto'
      }
    }, /*#__PURE__*/React.createElement("path", {
      d: "M11 2H5a1 1 0 00-1 1v14a1 1 0 001 1h10a1 1 0 001-1V7l-5-5z",
      stroke: "var(--text-secondary)",
      strokeWidth: "1.25"
    }), /*#__PURE__*/React.createElement("path", {
      d: "M11 2v5h5",
      stroke: "var(--text-secondary)",
      strokeWidth: "1.25"
    })), /*#__PURE__*/React.createElement("span", {
      style: {
        fontFamily: 'var(--font-sans)',
        fontSize: 'var(--type-body-01)',
        color: 'var(--text-primary)',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap'
      }
    }, fileName)), /*#__PURE__*/React.createElement("button", {
      type: "button",
      onClick: onClear,
      "aria-label": "Remove file",
      style: {
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: 24,
        height: 24,
        border: 'none',
        background: 'transparent',
        color: 'var(--text-secondary)',
        cursor: 'pointer'
      }
    }, /*#__PURE__*/React.createElement("svg", {
      width: "16",
      height: "16",
      viewBox: "0 0 16 16",
      fill: "none"
    }, /*#__PURE__*/React.createElement("path", {
      d: "M4 4l8 8M12 4l-8 8",
      stroke: "currentColor",
      strokeWidth: "1.25"
    }))));
  }
  return /*#__PURE__*/React.createElement("div", _extends({
    onClick: pick,
    onDragOver: e => {
      e.preventDefault();
      setDrag(true);
    },
    onDragLeave: () => setDrag(false),
    onDrop: e => {
      e.preventDefault();
      setDrag(false);
      handle(e.dataTransfer.files);
    },
    style: {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 'var(--spacing-04)',
      textAlign: 'center',
      padding: 'var(--spacing-09) var(--spacing-06)',
      background: drag ? 'var(--blue-10)' : 'var(--layer-01)',
      border: `1px dashed ${drag ? 'var(--blue-60)' : 'var(--border-strong)'}`,
      cursor: 'pointer',
      transition: 'background var(--duration-base) var(--ease-productive), border-color var(--duration-base) var(--ease-productive)'
    }
  }, rest), /*#__PURE__*/React.createElement("svg", {
    width: "28",
    height: "28",
    viewBox: "0 0 28 28",
    fill: "none"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M14 19V6M14 6l-5 5M14 6l5 5",
    stroke: drag ? 'var(--blue-60)' : 'var(--text-secondary)',
    strokeWidth: "1.4"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M5 19v2a1 1 0 001 1h16a1 1 0 001-1v-2",
    stroke: drag ? 'var(--blue-60)' : 'var(--text-secondary)',
    strokeWidth: "1.4"
  })), /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-sans)',
      fontSize: 'var(--type-body-02)',
      color: 'var(--text-primary)'
    }
  }, title), /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-sans)',
      fontSize: 'var(--type-helper-01)',
      color: 'var(--text-helper)'
    }
  }, hint), /*#__PURE__*/React.createElement("input", {
    ref: inputRef,
    type: "file",
    accept: accept,
    multiple: multiple,
    style: {
      display: 'none'
    },
    onChange: e => {
      handle(e.target.files);
      e.target.value = '';
    }
  }));
}
Object.assign(__ds_scope, { FileDropzone });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/FileDropzone/FileDropzone.jsx", error: String((e && e.message) || e) }); }

// components/MetricTile/MetricTile.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * MetricTile — Carbon "big number" stat. Light-weight numeral, uppercase
 * label, optional delta. Used in the analysis summary row.
 */
function MetricTile({
  label,
  value,
  unit = null,
  delta = null,
  accent = 'var(--blue-60)',
  ...rest
}) {
  const positive = typeof delta === 'string' ? delta.trim().startsWith('+') : delta > 0;
  return /*#__PURE__*/React.createElement("div", _extends({
    style: {
      background: 'var(--layer-02)',
      borderLeft: `3px solid ${accent}`,
      padding: 'var(--spacing-05) var(--spacing-06)',
      display: 'flex',
      flexDirection: 'column',
      gap: 'var(--spacing-03)',
      minWidth: 0
    }
  }, rest), /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-sans)',
      fontSize: 'var(--type-label-01)',
      letterSpacing: 'var(--tracking-caps)',
      textTransform: 'uppercase',
      color: 'var(--text-helper)',
      fontWeight: 'var(--weight-medium)'
    }
  }, label), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      alignItems: 'baseline',
      gap: 'var(--spacing-03)'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-sans)',
      fontSize: 'var(--type-heading-05)',
      fontWeight: 'var(--weight-light)',
      lineHeight: 1,
      color: 'var(--text-primary)',
      letterSpacing: 'var(--tracking-display)'
    }
  }, value), unit ? /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: 'var(--type-body-01)',
      color: 'var(--text-secondary)'
    }
  }, unit) : null, delta != null ? /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-mono)',
      fontSize: 'var(--type-code-01)',
      color: positive ? 'var(--green-60)' : 'var(--red-60)'
    }
  }, delta) : null));
}
Object.assign(__ds_scope, { MetricTile });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/MetricTile/MetricTile.jsx", error: String((e && e.message) || e) }); }

// components/QuestionGroup/QuestionGroup.jsx
try { (() => {
/**
 * QuestionGroup — the analyzer's hero row. A ranked, expandable group of
 * semantically-similar questions: rank numeral, representative question,
 * frequency heat-bar, keyword tags, similarity, and the underlying questions.
 */
function QuestionGroup({
  rank,
  question,
  count,
  maxCount = count,
  similarity = null,
  keywords = [],
  questions = [],
  defaultOpen = false
}) {
  const [open, setOpen] = React.useState(defaultOpen);
  const [hover, setHover] = React.useState(false);
  const pct = Math.max(6, Math.round(count / Math.max(1, maxCount) * 100));
  const heat = rank === 1 ? 'var(--blue-60)' : rank === 2 ? 'var(--blue-50)' : rank === 3 ? 'var(--blue-40)' : 'var(--gray-40)';
  return /*#__PURE__*/React.createElement("div", {
    style: {
      background: 'var(--layer-02)',
      borderBottom: '1px solid var(--border-subtle)'
    }
  }, /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: () => setOpen(!open),
    onMouseEnter: () => setHover(true),
    onMouseLeave: () => setHover(false),
    style: {
      width: '100%',
      display: 'grid',
      gridTemplateColumns: 'auto 1fr auto auto',
      alignItems: 'center',
      gap: 'var(--spacing-05)',
      padding: 'var(--spacing-05) var(--spacing-06)',
      background: hover ? 'var(--layer-hover)' : 'transparent',
      border: 'none',
      borderLeft: `3px solid ${open ? heat : 'transparent'}`,
      textAlign: 'left',
      cursor: 'pointer',
      transition: 'background var(--duration-base) var(--ease-productive)'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-mono)',
      fontSize: 'var(--type-heading-03)',
      fontWeight: 'var(--weight-regular)',
      color: heat,
      width: '2.25rem',
      textAlign: 'right',
      fontVariantNumeric: 'tabular-nums'
    }
  }, String(rank).padStart(2, '0')), /*#__PURE__*/React.createElement("span", {
    style: {
      minWidth: 0,
      display: 'flex',
      flexDirection: 'column',
      gap: 'var(--spacing-03)'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-sans)',
      fontSize: 'var(--type-body-02)',
      color: 'var(--text-primary)',
      lineHeight: 'var(--lh-snug)',
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      whiteSpace: open ? 'normal' : 'nowrap'
    }
  }, question), keywords.length ? /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'flex',
      flexWrap: 'wrap',
      gap: 'var(--spacing-02)'
    }
  }, keywords.slice(0, 5).map((k, i) => /*#__PURE__*/React.createElement("span", {
    key: i,
    style: {
      fontFamily: 'var(--font-mono)',
      fontSize: '11px',
      color: 'var(--text-helper)',
      background: 'var(--gray-10)',
      padding: '1px 6px',
      borderRadius: 'var(--radius-sm)'
    }
  }, k))) : null), /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'flex',
      alignItems: 'center',
      gap: 'var(--spacing-04)',
      width: '11rem'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      flex: 1,
      height: 6,
      background: 'var(--gray-10)',
      position: 'relative'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      position: 'absolute',
      left: 0,
      top: 0,
      bottom: 0,
      width: `${pct}%`,
      background: heat
    }
  })), /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-mono)',
      fontSize: 'var(--type-body-01)',
      fontWeight: 'var(--weight-medium)',
      color: 'var(--text-primary)',
      fontVariantNumeric: 'tabular-nums',
      width: '2.5rem',
      textAlign: 'right'
    }
  }, count, "\xD7")), /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'inline-flex',
      color: 'var(--text-secondary)',
      transform: open ? 'rotate(180deg)' : 'none',
      transition: 'transform var(--duration-base) var(--ease-productive)'
    }
  }, /*#__PURE__*/React.createElement("svg", {
    width: "16",
    height: "16",
    viewBox: "0 0 16 16",
    fill: "none"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M4 6l4 4 4-4",
    stroke: "currentColor",
    strokeWidth: "1.25"
  })))), open ? /*#__PURE__*/React.createElement("div", {
    style: {
      padding: '0 var(--spacing-06) var(--spacing-06) calc(var(--spacing-06) + 2.25rem + var(--spacing-05))'
    }
  }, similarity != null ? /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: 'var(--font-sans)',
      fontSize: 'var(--type-label-01)',
      color: 'var(--text-helper)',
      marginBottom: 'var(--spacing-04)'
    }
  }, "Avg. similarity ", /*#__PURE__*/React.createElement("strong", {
    style: {
      color: 'var(--text-secondary)',
      fontFamily: 'var(--font-mono)'
    }
  }, similarity), " \xB7 ", questions.length, " occurrences") : null, /*#__PURE__*/React.createElement("ul", {
    style: {
      listStyle: 'none',
      margin: 0,
      padding: 0,
      borderLeft: '1px solid var(--border-subtle)'
    }
  }, questions.map((q, i) => /*#__PURE__*/React.createElement("li", {
    key: i,
    style: {
      display: 'flex',
      justifyContent: 'space-between',
      gap: 'var(--spacing-05)',
      padding: 'var(--spacing-03) var(--spacing-05)'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-sans)',
      fontSize: 'var(--type-body-01)',
      color: 'var(--text-secondary)',
      lineHeight: 'var(--lh-normal)'
    }
  }, q.text), q.date ? /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-mono)',
      fontSize: 'var(--type-code-01)',
      color: 'var(--text-placeholder)',
      whiteSpace: 'nowrap'
    }
  }, q.date) : null)))) : null);
}
Object.assign(__ds_scope, { QuestionGroup });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/QuestionGroup/QuestionGroup.jsx", error: String((e && e.message) || e) }); }

// components/Slider/Slider.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * Slider — Carbon range control. Thin rail, filled progress, round thumb,
 * and a live numeric readout. Used for the similarity threshold.
 */
function Slider({
  value,
  min = 0,
  max = 100,
  step = 1,
  onChange,
  label = null,
  format = v => v,
  disabled = false,
  ...rest
}) {
  const pct = (value - min) / (max - min) * 100;
  const [active, setActive] = React.useState(false);
  return /*#__PURE__*/React.createElement("div", _extends({
    style: {
      width: '100%'
    }
  }, rest), label ? /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'baseline',
      marginBottom: 'var(--spacing-04)'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-sans)',
      fontSize: 'var(--type-label-01)',
      color: 'var(--text-secondary)',
      letterSpacing: 'var(--tracking-label)'
    }
  }, label), /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-mono)',
      fontSize: 'var(--type-body-01)',
      color: 'var(--text-primary)',
      fontWeight: 'var(--weight-medium)'
    }
  }, format(value))) : null, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      alignItems: 'center',
      gap: 'var(--spacing-04)'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-mono)',
      fontSize: 'var(--type-code-01)',
      color: 'var(--text-helper)'
    }
  }, format(min)), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'relative',
      flex: 1,
      height: 16,
      display: 'flex',
      alignItems: 'center'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      height: 2,
      background: 'var(--gray-30)'
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      width: `${pct}%`,
      height: 2,
      background: disabled ? 'var(--gray-40)' : 'var(--gray-100)'
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: `${pct}%`,
      transform: 'translateX(-50%)',
      width: 14,
      height: 14,
      borderRadius: '50%',
      background: disabled ? 'var(--gray-40)' : 'var(--gray-100)',
      boxShadow: active ? '0 0 0 3px var(--blue-20)' : 'none',
      transition: 'box-shadow var(--duration-fast) var(--ease-productive)',
      pointerEvents: 'none'
    }
  }), /*#__PURE__*/React.createElement("input", {
    type: "range",
    min: min,
    max: max,
    step: step,
    value: value,
    disabled: disabled,
    onChange: e => onChange && onChange(Number(e.target.value)),
    onMouseDown: () => setActive(true),
    onMouseUp: () => setActive(false),
    onFocus: () => setActive(true),
    onBlur: () => setActive(false),
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      width: '100%',
      margin: 0,
      height: 16,
      opacity: 0,
      cursor: disabled ? 'not-allowed' : 'pointer'
    }
  })), /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-mono)',
      fontSize: 'var(--type-code-01)',
      color: 'var(--text-helper)'
    }
  }, format(max))));
}
Object.assign(__ds_scope, { Slider });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/Slider/Slider.jsx", error: String((e && e.message) || e) }); }

// components/Tag/Tag.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * Tag — Carbon pill tag for keywords, categories, and counts.
 * Color pairs follow Carbon (soft fill + deep text). Optional dismiss / dot.
 */
function Tag({
  children,
  color = 'gray',
  size = 'md',
  outline = false,
  dot = false,
  onDismiss = null,
  ...rest
}) {
  const pairs = {
    gray: {
      bg: 'var(--gray-20)',
      fg: 'var(--gray-100)',
      line: 'var(--gray-50)'
    },
    blue: {
      bg: 'var(--blue-20)',
      fg: 'var(--blue-80)',
      line: 'var(--blue-60)'
    },
    green: {
      bg: '#a7f0ba',
      fg: '#044317',
      line: 'var(--green-60)'
    },
    red: {
      bg: '#ffd7d9',
      fg: '#750e13',
      line: 'var(--red-60)'
    },
    purple: {
      bg: '#e8daff',
      fg: '#491d8b',
      line: 'var(--purple-60)'
    },
    teal: {
      bg: '#9ef0f0',
      fg: '#004144',
      line: 'var(--teal-60)'
    },
    magenta: {
      bg: '#ffd6e8',
      fg: '#740937',
      line: 'var(--magenta-60)'
    },
    cyan: {
      bg: '#bae6ff',
      fg: '#00539a',
      line: 'var(--cyan-50)'
    }
  };
  const p = pairs[color] || pairs.gray;
  const heights = {
    sm: '1.125rem',
    md: '1.5rem'
  };
  const style = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 'var(--spacing-02)',
    height: heights[size] || heights.md,
    padding: size === 'sm' ? '0 var(--spacing-03)' : '0 var(--spacing-04)',
    borderRadius: 'var(--radius-pill)',
    fontFamily: 'var(--font-sans)',
    fontSize: 'var(--type-label-01)',
    fontWeight: 'var(--weight-regular)',
    lineHeight: 1,
    whiteSpace: 'nowrap',
    background: outline ? 'transparent' : p.bg,
    color: outline ? p.fg : p.fg,
    border: outline ? `1px solid ${p.line}` : '1px solid transparent'
  };
  return /*#__PURE__*/React.createElement("span", _extends({
    style: style
  }, rest), dot ? /*#__PURE__*/React.createElement("span", {
    style: {
      width: 6,
      height: 6,
      borderRadius: '50%',
      background: p.line,
      flex: '0 0 auto'
    }
  }) : null, /*#__PURE__*/React.createElement("span", null, children), onDismiss ? /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: onDismiss,
    "aria-label": "Dismiss",
    style: {
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center',
      width: 16,
      height: 16,
      marginRight: -4,
      padding: 0,
      border: 'none',
      background: 'transparent',
      color: p.fg,
      cursor: 'pointer',
      borderRadius: '50%'
    }
  }, /*#__PURE__*/React.createElement("svg", {
    width: "12",
    height: "12",
    viewBox: "0 0 16 16",
    fill: "none"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M4 4l8 8M12 4l-8 8",
    stroke: "currentColor",
    strokeWidth: "1.25"
  }))) : null);
}
Object.assign(__ds_scope, { Tag });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/Tag/Tag.jsx", error: String((e && e.message) || e) }); }

__ds_ns.Button = __ds_scope.Button;

__ds_ns.Card = __ds_scope.Card;

__ds_ns.FileDropzone = __ds_scope.FileDropzone;

__ds_ns.MetricTile = __ds_scope.MetricTile;

__ds_ns.QuestionGroup = __ds_scope.QuestionGroup;

__ds_ns.Slider = __ds_scope.Slider;

__ds_ns.Tag = __ds_scope.Tag;

})();
