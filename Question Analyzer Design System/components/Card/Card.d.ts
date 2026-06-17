import * as React from 'react';

/**
 * Carbon layered surface. Sharp corners, subtle border, optional accent + hover lift.
 * @startingPoint section="Core" subtitle="Layered content surface" viewport="700x200"
 */
export interface CardProps {
  children: React.ReactNode;
  /** CSS padding value. @default "var(--spacing-06)" */
  padding?: string;
  /** Left accent bar color (e.g. "var(--blue-60)"). */
  accent?: string;
  /** Enables pointer cursor + hover elevation. */
  interactive?: boolean;
  /** Blue selected border. */
  selected?: boolean;
  onClick?: (e: React.MouseEvent<HTMLDivElement>) => void;
  style?: React.CSSProperties;
}

export function Card(props: CardProps): JSX.Element;
