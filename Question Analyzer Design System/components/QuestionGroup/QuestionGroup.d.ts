import * as React from 'react';

interface QuestionInstance {
  text: string;
  date?: string;
}

/**
 * Ranked, expandable group of semantically-similar questions — the core
 * artifact of the analyzer. Shows rank, frequency heat-bar, keywords, and
 * the underlying question instances on expand.
 * @startingPoint section="Data" subtitle="Ranked question group row" viewport="700x120"
 */
export interface QuestionGroupProps {
  /** 1-based rank by frequency. */
  rank: number;
  /** Representative question for the group. */
  question: string;
  /** Number of occurrences. */
  count: number;
  /** Highest count across all groups, for bar scaling. */
  maxCount?: number;
  /** Average similarity, pre-formatted (e.g. "91%"). */
  similarity?: string | null;
  keywords?: string[];
  questions?: QuestionInstance[];
  defaultOpen?: boolean;
}

export function QuestionGroup(props: QuestionGroupProps): JSX.Element;
