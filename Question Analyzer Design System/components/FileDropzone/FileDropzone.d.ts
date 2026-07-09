import * as React from 'react';

/**
 * Carbon file-uploader dropzone with drag-active state and selected-file chip.
 * @startingPoint section="Data" subtitle="Upload / drag-drop field" viewport="700x240"
 */
export interface FileDropzoneProps {
  /** Accepted extensions. @default ".txt,.json,.csv" */
  accept?: string;
  /** Helper text under the title. */
  hint?: string;
  /** Prompt shown in the empty state. */
  title?: string;
  /** When set, renders the selected-file chip instead of the dropzone. */
  fileName?: string | null;
  /** Accept several files per drop/pick; onFile fires once per file. @default false */
  multiple?: boolean;
  onFile?: (file: File) => void;
  onClear?: () => void;
}

export function FileDropzone(props: FileDropzoneProps): JSX.Element;
