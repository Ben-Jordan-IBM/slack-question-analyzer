import * as React from 'react';

/**
 * IBM Carbon-style button. Sharp corners; asymmetric padding with a right-pinned icon.
 * @startingPoint section="Core" subtitle="Primary / secondary / ghost / danger actions" viewport="700x200"
 */
export interface ButtonProps {
  children: React.ReactNode;
  /** Visual treatment. @default "primary" */
  variant?: 'primary' | 'secondary' | 'tertiary' | 'ghost' | 'danger';
  /** Control height. @default "lg" */
  size?: 'sm' | 'md' | 'lg';
  /** Optional icon node, pinned to the right edge (Carbon hallmark). */
  icon?: React.ReactNode;
  fullWidth?: boolean;
  disabled?: boolean;
  onClick?: (e: React.MouseEvent<HTMLButtonElement>) => void;
  type?: 'button' | 'submit' | 'reset';
}

export function Button(props: ButtonProps): JSX.Element;
