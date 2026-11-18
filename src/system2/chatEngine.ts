import { CONFIG } from '../config';

/**
 * Calculates realistic human typing duration in milliseconds for a chat string
 * based on average Words Per Minute (default 68 WPM, where 1 word = 5 characters)
 * plus brief cognitive initiation latency (~280ms).
 */
export function calculateHumanTypingDelayMs(
  message: string,
  wpm = CONFIG.kinematics.typingWpm
): number {
  const charCount = message.trim().length;
  if (charCount === 0) return 0;

  const charsPerMinute = wpm * 5;
  const charsPerMs = charsPerMinute / 60000;
  const rawTypingMs = charCount / charsPerMs;

  // Add slight jitter (+/- 12%) and clamp to [300ms, 2400ms] so chat feels snappy yet human
  const jitter = 0.88 + Math.random() * 0.24;
  return Math.min(2400, Math.max(300, Math.round(250 + rawTypingMs * jitter)));
}

/**
 * Humanized Chat Engine that prevents instantaneous 0ms bot chat packets
 * and coordinates speech with the PIANO Cognitive Controller.
 */
export class HumanizedChatEngine {
  private lastChatSentAt = 0;

  constructor(private readonly botProvider: () => any) {}

  public async sayHumanized(message: string): Promise<void> {
    const bot = this.botProvider();
    if (!bot?.chat || !message.trim()) return;

    const delayMs = calculateHumanTypingDelayMs(message);
    await new Promise((r) => setTimeout(r, delayMs));

    // Prevent server anti-spam kicks by ensuring at least 450ms between consecutive chat packets
    const sinceLast = Date.now() - this.lastChatSentAt;
    if (sinceLast < 450) {
      await new Promise((r) => setTimeout(r, 450 - sinceLast));
    }

    try {
      bot.chat(message);
      this.lastChatSentAt = Date.now();
      console.log(`[Chat -> World] <${bot.username}> ${message}`);
    } catch {
      // Ignore chat error if disconnected
    }
  }
}
