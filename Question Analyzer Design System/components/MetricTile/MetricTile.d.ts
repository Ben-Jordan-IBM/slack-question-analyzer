import * as React from 'react';

/**
 * Carbon "big number" summary stat with light numeral and uppercase label.
 * @startingPoint section="Data" subtitle="Summary metric tile" viewport="700x160"
 */
export interface MetricTileProps {
  label: string;
  value: React.ReactNode;
  /** Small trailing unit, e.g. "%". */
  unit?: string;
  /** Delta string ("+12") or number; colored green/red. */
  delta?: string | number;
  /** Left accent bar color. @default "var(--blue-60)" */
  accent?: string;
}

export function MetricTile(props: MetricTileProps): JSX.Element;
