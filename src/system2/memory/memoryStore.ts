import * as fs from 'fs';
import * as path from 'path';
import { EpisodicMemoryEntry, Vector3D } from '../../types';
import { CONFIG } from '../../config';

/**
 * Tokenizes text into normalized lowercase stems for fast semantic/keyword cosine similarity.
 */
export function tokenizeText(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 1);
}

/**
 * Computes cosine similarity between two token sets.
 */
export function computeTokenSimilarity(tokensA: string[], tokensB: string[]): number {
  if (tokensA.length === 0 || tokensB.length === 0) return 0;
  const freqA = new Map<string, number>();
  const freqB = new Map<string, number>();

  for (const t of tokensA) freqA.set(t, (freqA.get(t) || 0) + 1);
  for (const t of tokensB) freqB.set(t, (freqB.get(t) || 0) + 1);

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (const [k, v] of freqA.entries()) {
    normA += v * v;
    if (freqB.has(k)) {
      dot += v * freqB.get(k)!;
    }
  }
  for (const v of freqB.values()) {
    normB += v * v;
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
}

/**
 * Persistent Episodic & Working Memory Store.
 * Saves landmarks, chest locations, player interactions, and synthesized skill records
 * inside `data/episodic_memory.json` within the project folder.
 */
export class EpisodicMemoryStore {
  private entries: EpisodicMemoryEntry[] = [];
  private chatHistory: Array<{ speaker: string; message: string; timestamp: number }> = [];

  constructor(private readonly filePath = CONFIG.paths.memoryFile) {
    this.loadFromDisk();
  }

  public addMemory(
    type: EpisodicMemoryEntry['type'],
    summary: string,
    tags: string[] = [],
    position?: Vector3D
  ): EpisodicMemoryEntry {
    const entry: EpisodicMemoryEntry = {
      id: `mem_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
      type,
      summary,
      position,
      timestamp: Date.now(),
      tags: [...new Set([...tags, ...tokenizeText(summary)])],
    };

    this.entries.push(entry);
    if (this.entries.length > 300) {
      this.entries.shift();
    }
    this.saveToDisk();
    return entry;
  }

  public recordChat(speaker: string, message: string): void {
    this.chatHistory.push({ speaker, message, timestamp: Date.now() });
    if (this.chatHistory.length > 30) {
      this.chatHistory.shift();
    }
  }

  public getRecentChat(limit = 10): Array<{ speaker: string; message: string; timestamp: number }> {
    return this.chatHistory.slice(-limit);
  }

  /**
   * Semantic/Vector similarity retrieval over stored episodic memories.
   */
  public queryRelevantMemories(query: string, topK = 5): EpisodicMemoryEntry[] {
    const queryTokens = tokenizeText(query);
    if (queryTokens.length === 0) {
      return this.entries.slice(-topK);
    }

    const scored = this.entries.map((entry) => ({
      entry,
      score: computeTokenSimilarity(queryTokens, entry.tags),
    }));

    scored.sort((a, b) => b.score - a.score || b.entry.timestamp - a.entry.timestamp);
    return scored
      .filter((s) => s.score > 0)
      .slice(0, topK)
      .map((s) => s.entry);
  }

  private loadFromDisk(): void {
    try {
      if (fs.existsSync(this.filePath)) {
        const raw = fs.readFileSync(this.filePath, 'utf8');
        this.entries = JSON.parse(raw);
      }
    } catch {
      this.entries = [];
    }
  }

  private saveToDisk(): void {
    try {
      const dir = path.dirname(this.filePath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      fs.writeFileSync(this.filePath, JSON.stringify(this.entries, null, 2), 'utf8');
    } catch {
      // Ignore transient write errors
    }
  }
}
