import * as React from 'react';

/**
 * Carbon range slider with min/max rail labels and live numeric readout.
 * @startingPoint section="Core" subtitle="Range / threshold control" viewport="700x140"
 */
export interface SliderProps {
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange?: (value: number) => void;
  /** Caption above the rail. */
  label?: string;
  /** Formats value + rail labels, e.g. v => `${v}%`. */
  format?: (value: number) => React.ReactNode;
  disabled?: boolean;
}

export function Slider(props: SliderProps): JSX.Element;
