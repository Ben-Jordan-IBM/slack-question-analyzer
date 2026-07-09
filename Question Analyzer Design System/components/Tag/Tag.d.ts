import * as React from 'react';

/**
 * Carbon pill tag for keywords, categories, and status. Soft-fill or outline.
 * @startingPoint section="Core" subtitle="Keyword & category tags" viewport="700x140"
 */
export interface TagProps {
  children: React.ReactNode;
  /** Color pair. @default "gray" */
  color?: 'gray' | 'blue' | 'green' | 'red' | 'purple' | 'teal' | 'magenta' | 'cyan';
  /** @default "md" */
  size?: 'sm' | 'md';
  /** Outline treatment instead of soft fill. */
  outline?: boolean;
  /** Leading status dot. */
  dot?: boolean;
  /** When provided, renders a dismiss (×) button. */
  onDismiss?: () => void;
}

export function Tag(props: TagProps): JSX.Element;
