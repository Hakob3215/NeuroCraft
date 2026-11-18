import { CONFIG } from '../config';
import { SubtaskStatus } from '../types';

export type PrefetchTriggerCallback = (completedOrNearDoneSubtask: SubtaskStatus) => void;

/**
 * Speculative Pipeline Prefetcher.
 * Monitors the active System 1 subtask progress and triggers lookahead planning
 * when progress reaches >= 75% (or < 1.5s estimated remaining duration),
 * ensuring zero idle pause between sequential subtasks.
 */
export class SpeculativePrefetcher {
  private prefetchedSubtaskIds = new Set<string>();

  constructor(
    private readonly onTriggerPrefetch: PrefetchTriggerCallback,
    private readonly progressThreshold = CONFIG.speculative.prefetchProgressThreshold,
    private readonly remainingMsThreshold = CONFIG.speculative.prefetchRemainingMsThreshold
  ) {}

  /**
   * Evaluates whether the currently running subtask has crossed the speculative lookahead threshold.
   */
  public shouldPrefetch(subtask: SubtaskStatus): boolean {
    if (subtask.status !== 'RUNNING') return false;
    if (this.prefetchedSubtaskIds.has(subtask.id)) return false;

    const elapsedMs = Date.now() - subtask.startedAt;
    const estimatedRemainingMs = Math.max(0, subtask.estimatedDurationMs - elapsedMs);

    const crossedProgress = subtask.progress >= this.progressThreshold;
    const crossedRemainingTime =
      subtask.progress >= 0.65 && estimatedRemainingMs <= this.remainingMsThreshold;

    return crossedProgress || crossedRemainingTime;
  }

  public onSubtaskUpdate(subtask: SubtaskStatus): void {
    if (this.shouldPrefetch(subtask)) {
      this.prefetchedSubtaskIds.add(subtask.id);
      this.onTriggerPrefetch(subtask);
    }
  }
}
