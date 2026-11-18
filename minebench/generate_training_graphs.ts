/**
 * NeuroCraft Training Graph & Decomposed 4-Panel Dashboard Generator
 * =================================================================
 * Reads `training_state.json` (plus `rl_replay_buffer.jsonl` backfill for domain bars) and renders:
 * 1. `minebench/results/training_curves.svg` (Decomposed 3-row vector chart)
 * 2. `minebench/results/training_dashboard.html` (Interactive 4-panel web dashboard)
 * 3. `minebench/results/training_history.csv` (Tabular telemetry export)
 */

import * as fs from 'fs';
import * as path from 'path';

export interface DomainMetricsSnapshot {
  passed: number;
  total: number;
  passRate: number;
  meanScore: number;
}

export interface SurvivalCampaignSnapshot {
  goalId: string;
  goalTitle: string;
  techTierReached: string;
  advancementsUnlocked: string[];
  survivalSteps: number;
  score: number;
}

export interface IterationRecord {
  iteration: number;
  phase?: string;
  timestamp: string;
  meanScore: number;
  passRate: number;
  tasksPassed: number;
  totalTasks: number;
  rawReward?: number;
  durationSeconds?: number;
  slmInitialLoss?: number;
  slmFinalLoss?: number;
  system1Loss?: number;
  actAimErrorDeg?: number;
  actDiscreteAccPct?: number;
  domainBreakdown?: Record<string, DomainMetricsSnapshot>;
  survivalCampaign?: SurvivalCampaignSnapshot;
  dragonSurvivalScore?: number;
  dragonMilestones?: number;
  milestone?: string;
}

export interface PersistentTrainingState {
  sessionCount: number;
  currentIteration: number;
  totalEpisodesPlayed: number;
  bestPassRate: number;
  bestMeanScore: number;
  bestIteration: number;
  latestMeanScore: number;
  latestPassRate: number;
  latestReward: number;
  lastSavedAt: string;
  latestCheckpoint?: string;
  bestCheckpoint?: string;
  history?: IterationRecord[];
}

const DOMAIN_BUCKETS = [
  'Tactical Combat',
  'Crafting & Tech',
  'Agility & Parkour',
  'Building & 3D',
  'Survival & Affordances',
  'Puzzles & Tooling',
  'Multiplayer',
] as const;

const DOMAIN_COLORS: Record<string, string> = {
  'Tactical Combat': '#ef4444',
  'Crafting & Tech': '#f59e0b',
  'Agility & Parkour': '#10b981',
  'Building & 3D': '#8b5cf6',
  'Survival & Affordances': '#3b82f6',
  'Puzzles & Tooling': '#06b6d4',
  'Multiplayer': '#ec4899',
};

function classifyReplayRecordDomain(rec: any): string {
  const tid = String(rec?.taskId || '').toLowerCase();
  const dom = String(rec?.domain || '').toUpperCase();
  if (tid.startsWith('omni_craft_') || tid.startsWith('prog_')) return 'Crafting & Tech';
  if (tid.startsWith('omni_mob_') || dom === 'TACTICAL_COMBAT') return 'Tactical Combat';
  if (dom === 'AGILITY_PARKOUR') return 'Agility & Parkour';
  if (dom === 'ARCHITECTURE_BUILDING') return 'Building & 3D';
  if (dom === 'MULTIPLAYER_ARENA') return 'Multiplayer';
  if (dom === 'PUZZLE_ENGINEERING') return 'Puzzles & Tooling';
  return 'Survival & Affordances';
}

function backfillHistoricalDomains(history: IterationRecord[], replayPath: string): void {
  if (!fs.existsSync(replayPath)) return;
  const needsBackfill = history.some((h) => !h.domainBreakdown);
  if (!needsBackfill) return;

  try {
    const lines = fs.readFileSync(replayPath, 'utf-8').split(/\r?\n/);
    const agg: Record<number, Record<string, { passed: number; total: number; scoreSum: number }>> = {};

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const rec = JSON.parse(line);
        const iter = Number(rec.iteration);
        if (!Number.isFinite(iter) || iter <= 0) continue;
        if (String(rec.taskId || '').startsWith('campaign_')) continue;

        const bucket = classifyReplayRecordDomain(rec);
        if (!agg[iter]) agg[iter] = {};
        if (!agg[iter][bucket]) agg[iter][bucket] = { passed: 0, total: 0, scoreSum: 0 };
        agg[iter][bucket].total++;
        if (rec.passed) agg[iter][bucket].passed++;
        agg[iter][bucket].scoreSum += Number(rec.score || 0);
      } catch {}
    }

    for (const h of history) {
      if (!h.domainBreakdown && agg[h.iteration]) {
        const bd: Record<string, DomainMetricsSnapshot> = {};
        for (const [dom, st] of Object.entries(agg[h.iteration])) {
          bd[dom] = {
            passed: st.passed,
            total: st.total,
            passRate: st.total > 0 ? Number(((st.passed / st.total) * 100).toFixed(1)) : 0,
            meanScore: st.total > 0 ? Number((st.scoreSum / st.total).toFixed(1)) : 0,
          };
        }
        h.domainBreakdown = bd;
      }
    }
  } catch {}
}

export function generateTrainingVisuals(): void {
  const resultsDir = path.resolve(process.cwd(), 'minebench', 'results');
  const statePath = path.join(resultsDir, 'training_state.json');
  const replayPath = path.join(resultsDir, 'rl_replay_buffer.jsonl');

  if (!fs.existsSync(statePath)) {
    console.error(`[GraphGen] Error: training_state.json not found at ${statePath}`);
    return;
  }

  const state: PersistentTrainingState = JSON.parse(fs.readFileSync(statePath, 'utf-8'));
  const history: IterationRecord[] = state.history || [];

  if (history.length === 0) {
    console.log(`[GraphGen] Rendering fresh Loop 0 baseline dashboard (0 completed iterations).`);
  }

  backfillHistoricalDomains(history, replayPath);

  // 1. Export CSV
  const csvPath = path.join(resultsDir, 'training_history.csv');
  const csvRows = [
    'iteration,phase,timestamp,mean_score,pass_rate_pct,tasks_passed,total_tasks,slm_initial_loss,slm_final_loss,act_aim_err_deg,survival_tier,survival_advancements,dragon_survival_score,milestone',
  ];
  for (const h of history) {
    csvRows.push(
      `${h.iteration},"${h.phase || ''}","${h.timestamp}",${h.meanScore},${h.passRate},${h.tasksPassed},${h.totalTasks},${h.slmInitialLoss ?? ''},${h.slmFinalLoss ?? ''},${h.actAimErrorDeg ?? ''},"${h.survivalCampaign?.techTierReached ?? ''}",${h.survivalCampaign?.advancementsUnlocked?.length ?? ''},${h.dragonSurvivalScore ?? ''},"${h.milestone || ''}"`
    );
  }
  fs.writeFileSync(csvPath, csvRows.join('\n'), 'utf-8');
  console.log(`[GraphGen] Saved CSV: ${csvPath}`);

  // 2. Render SVG Vector Graphic (Decomposed 3-panel SVG)
  const svgPath = path.join(resultsDir, 'training_curves.svg');
  const svgContent = renderSvgChart(history, state);
  fs.writeFileSync(svgPath, svgContent, 'utf-8');
  console.log(`[GraphGen] Saved SVG Chart: ${svgPath}`);

  // 3. Render Interactive 4-Panel HTML Dashboard
  const htmlPath = path.join(resultsDir, 'training_dashboard.html');
  const htmlContent = renderHtmlDashboard(history, state);
  fs.writeFileSync(htmlPath, htmlContent, 'utf-8');
  console.log(`[GraphGen] Saved Interactive Dashboard: ${htmlPath}`);

  // 4. Mirror to conversation artifact directory if active
  const artifactDir = path.resolve(
    process.env.USERPROFILE || 'C:\\Users\\hataj',
    '.gemini',
    'antigravity',
    'brain',
    '5863baac-97fc-45ed-a450-9e2ceeaf79c1'
  );
  if (fs.existsSync(artifactDir)) {
    try {
      fs.copyFileSync(htmlPath, path.join(artifactDir, 'training_dashboard.html'));
      fs.copyFileSync(svgPath, path.join(artifactDir, 'training_curves.svg'));
      fs.copyFileSync(csvPath, path.join(artifactDir, 'training_history.csv'));
      fs.copyFileSync(statePath, path.join(artifactDir, 'training_state.json'));
    } catch {}
  }
}

function renderSvgChart(history: IterationRecord[], state: PersistentTrainingState): string {
  const width = 1200;
  const height = 940;
  const margin = { top: 75, right: 60, bottom: 55, left: 70 };

  const plotW = width - margin.left - margin.right;
  const plotH1 = 220; // Top: Standardized MINEbench Exam
  const gap = 65;
  const plotH2 = 220; // Middle: Per-Domain Bar Breakdown (Recent Increments)
  const plotH3 = 180; // Bottom: Dual-Brain Loss & Aim Error

  const y1 = margin.top;
  const y2 = y1 + plotH1 + gap;
  const y3 = y2 + plotH2 + gap;

  const minIter = history.length > 0 ? Math.min(0, ...history.map((h) => h.iteration)) : 0;
  const maxIter = history.length > 0 ? Math.max(10, ...history.map((h) => h.iteration)) : 10;
  const getX = (iter: number) => margin.left + ((iter - minIter) / Math.max(1, maxIter - minIter)) * plotW;
  const getYScore = (score: number) => y1 + plotH1 - (Math.max(0, Math.min(100, score)) / 100) * plotH1;

  const validLosses = history.map((h) => h.slmFinalLoss).filter((l): l is number => typeof l === 'number');
  const maxLoss = validLosses.length > 0 ? Math.max(0.35, ...validLosses) : 0.4;
  const getYLoss = (loss: number) => y3 + plotH3 - (Math.max(0, Math.min(maxLoss, loss)) / maxLoss) * plotH3;

  const scorePath = history
    .map((h, i) => `${i === 0 ? 'M' : 'L'} ${getX(h.iteration).toFixed(1)} ${getYScore(h.meanScore).toFixed(1)}`)
    .join(' ');
  const passPath = history
    .map((h, i) => `${i === 0 ? 'M' : 'L'} ${getX(h.iteration).toFixed(1)} ${getYScore(h.passRate).toFixed(1)}`)
    .join(' ');

  const lossPoints = history.filter((h) => typeof h.slmFinalLoss === 'number');
  const lossPath = lossPoints
    .map((h, i) => `${i === 0 ? 'M' : 'L'} ${getX(h.iteration).toFixed(1)} ${getYLoss(h.slmFinalLoss!).toFixed(1)}`)
    .join(' ');

  // Middle Panel: Latest Domain Bar Breakdown
  const latestWithDomains =
    history.length > 0 ? [...history].reverse().find((h) => h.domainBreakdown) || history[history.length - 1] : undefined;
  const bd = latestWithDomains?.domainBreakdown || {};
  const barWidth = Math.floor(plotW / DOMAIN_BUCKETS.length) - 24;
  const barSvg = DOMAIN_BUCKETS.map((dom, idx) => {
    const snap = bd[dom];
    const passVal = snap ? snap.passRate : 0;
    const scoreVal = snap ? snap.meanScore : 0;
    const bx = margin.left + idx * (plotW / DOMAIN_BUCKETS.length) + 12;
    const hPass = (Math.max(0, Math.min(100, passVal)) / 100) * (plotH2 - 35);
    const hScore = (Math.max(0, Math.min(100, scoreVal)) / 100) * (plotH2 - 35);
    const byPass = y2 + plotH2 - 20 - hPass;
    const byScore = y2 + plotH2 - 20 - hScore;
    const color = DOMAIN_COLORS[dom] || '#3b82f6';
    const subW = Math.floor(barWidth / 2) - 4;
    return `
      <rect x="${bx}" y="${byScore}" width="${subW}" height="${hScore}" fill="${color}" fill-opacity="0.35" rx="3"/>
      <rect x="${bx + subW + 4}" y="${byPass}" width="${subW}" height="${hPass}" fill="${color}" rx="3"/>
      <text x="${bx + barWidth / 2}" y="${Math.min(byScore, byPass) - 5}" fill="#e2e8f0" font-size="10" font-weight="bold" text-anchor="middle">${passVal.toFixed(0)}%</text>
      <text x="${bx + barWidth / 2}" y="${y2 + plotH2 - 5}" fill="#94a3b8" font-size="10" text-anchor="middle">${dom.split(' ')[0]}</text>
    `;
  }).join('');

  const latestSurvivalTier = history.length > 0 ? history[history.length - 1]?.survivalCampaign?.techTierReached || 'Active' : 'Ready (Loop 0)';

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" style="background-color: #0b1120; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  <text x="${margin.left}" y="34" fill="#f8fafc" font-size="20" font-weight="bold">NeuroCraft Decomposed Dual-Track Training Telemetry</text>
  <text x="${margin.left}" y="54" fill="#94a3b8" font-size="12">Iterations: ${history.length} | Peak Score: ${state.bestMeanScore}/100 | Open-World Tier: ${latestSurvivalTier}</text>

  <!-- Panel 1: Standardized Exam -->
  <rect x="${margin.left}" y="${y1}" width="${plotW}" height="${plotH1}" fill="#1e293b" rx="6"/>
  <text x="${margin.left + 12}" y="${y1 + 20}" fill="#f8fafc" font-size="13" font-weight="bold">1. Standardized MINEbench Exam (Pass Rate % &amp; Mean Score)</text>
  ${[0, 25, 50, 75, 100].map((v) => {
    const yy = getYScore(v);
    return `<line x1="${margin.left}" y1="${yy}" x2="${margin.left + plotW}" y2="${yy}" stroke="#334155" stroke-width="1" stroke-dasharray="2,2"/><text x="${margin.left - 8}" y="${yy + 4}" fill="#64748b" font-size="10" text-anchor="end">${v}</text>`;
  }).join('')}
  ${passPath ? `<path d="${passPath}" fill="none" stroke="#10b981" stroke-width="2.5" stroke-dasharray="5,4"/>` : ''}
  ${scorePath ? `<path d="${scorePath}" fill="none" stroke="#3b82f6" stroke-width="3"/>` : `<text x="${margin.left + plotW / 2}" y="${y1 + plotH1 / 2}" fill="#94a3b8" font-size="14" text-anchor="middle">Loop 0 Initialized — Ready for Iteration #1</text>`}

  <!-- Panel 2: Domain Bar Breakdown -->
  <rect x="${margin.left}" y="${y2}" width="${plotW}" height="${plotH2}" fill="#1e293b" rx="6"/>
  <text x="${margin.left + 12}" y="${y2 + 20}" fill="#f8fafc" font-size="13" font-weight="bold">2. Latest Per-Domain Skill Breakdown (Iter #${latestWithDomains?.iteration ?? state.currentIteration}: Dim=Score, Solid=Pass%)</text>
  ${barSvg}

  <!-- Panel 3: SLM Loss -->
  <rect x="${margin.left}" y="${y3}" width="${plotW}" height="${plotH3}" fill="#1e293b" rx="6"/>
  <text x="${margin.left + 12}" y="${y3 + 20}" fill="#f8fafc" font-size="13" font-weight="bold">3. System 2 Qwen2.5-Coder SLM Cross-Entropy Loss</text>
  ${lossPath ? `<path d="${lossPath}" fill="none" stroke="#f43f5e" stroke-width="3"/>` : `<text x="${margin.left + plotW / 2}" y="${y3 + plotH3 / 2}" fill="#94a3b8" font-size="13" text-anchor="middle">No QLoRA updates recorded yet (awaiting Iteration #1)</text>`}
  ${history.map((h, i) => {
    if (i % Math.max(1, Math.ceil(history.length / 15)) === 0 || i === history.length - 1) {
      return `<text x="${getX(h.iteration)}" y="${y3 + plotH3 + 22}" fill="#94a3b8" font-size="10" text-anchor="middle">Iter ${h.iteration}</text>`;
    }
    return '';
  }).join('')}
</svg>`;
}

function renderHtmlDashboard(history: IterationRecord[], state: PersistentTrainingState): string {
  const dragonRuns = history.filter((h) => typeof h.dragonSurvivalScore === 'number');
  const bestDragonScore = dragonRuns.length > 0 ? Math.max(...dragonRuns.map((h) => h.dragonSurvivalScore!)) : 0;
  const bestDragonMilestones = dragonRuns.length > 0 ? Math.max(...dragonRuns.map((h) => h.dragonMilestones || 0)) : 0;

  const validScores = history.map((h) => h.meanScore);
  const bestScore = validScores.length > 0 ? Math.max(...validScores) : state.bestMeanScore;
  const bestScoreIter = history.find((h) => h.meanScore === bestScore)?.iteration ?? state.bestIteration;
  const validPassRates = history.map((h) => h.passRate);
  const bestPass = validPassRates.length > 0 ? Math.max(...validPassRates) : state.bestPassRate;
  const validLosses = history.map((h) => h.slmFinalLoss).filter((l): l is number => typeof l === 'number');
  const minLoss = validLosses.length > 0 ? Math.min(...validLosses) : 0;
  const initialLoss = validLosses.length > 0 ? validLosses[0] : 0;

  const latestRecord = history[history.length - 1];
  const latestSurvival = [...history].reverse().find((h) => h.survivalCampaign)?.survivalCampaign;

  // Sample up to 10 evenly spaced time increments t (plus latest) for readable grouped bar visualization across time
  const domainHistory = history.filter((h) => h.domainBreakdown);
  const sampledDomainIters: IterationRecord[] = [];
  if (domainHistory.length <= 10) {
    sampledDomainIters.push(...domainHistory);
  } else {
    const step = (domainHistory.length - 1) / 9;
    for (let i = 0; i < 9; i++) sampledDomainIters.push(domainHistory[Math.round(i * step)]);
    sampledDomainIters.push(domainHistory[domainHistory.length - 1]);
  }

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NeuroCraft AI Agent — Decomposed 4-Panel Training Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg: #0b1120;
      --card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #3b82f6;
      --green: #10b981;
      --pink: #f43f5e;
      --amber: #f59e0b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
    body { background-color: var(--bg); color: var(--text); padding: 2rem; }
    header { margin-bottom: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 1.5rem; }
    h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.75rem; }
    .badge { font-size: 0.75rem; background: #1e3a8a; color: #93c5fd; padding: 0.2rem 0.6rem; border-radius: 9999px; }
    p.subtitle { color: var(--muted); font-size: 0.95rem; }
    .grid-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
    .card-stat { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1.25rem; }
    .stat-title { color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
    .stat-val { font-size: 1.8rem; font-weight: 800; color: var(--text); }
    .stat-val.highlight { color: var(--green); }
    .stat-sub { font-size: 0.75rem; color: var(--muted); margin-top: 0.25rem; }
    .grid-charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
    .chart-container { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; height: 390px; }
    table { width: 100%; border-collapse: collapse; margin-top: 1rem; background: var(--card); border-radius: 8px; overflow: hidden; font-size: 0.85rem; }
    th, td { padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }
    th { background: #0f172a; color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 0.75rem; }
    tr:hover { background: rgba(255,255,255,0.02); }
    .tag { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
    .tag-pass { background: rgba(16, 185, 129, 0.15); color: var(--green); }
    .tag-amber { background: rgba(245, 158, 11, 0.15); color: var(--amber); }
  </style>
</head>
<body>

  <header>
    <h1>NeuroCraft Decomposed Dual-Track Training Studio <span class="badge">Live RTX 5060 + Mojang 1.20.4</span></h1>
    <p class="subtitle">Isolated Standardized Exam, Per-Domain Bar Breakdowns, Open-World Survival Campaign, & Dual-Brain Neural Convergence.</p>
  </header>

  <div class="grid-stats">
    <div class="card-stat">
      <div class="stat-title">Completed Iterations</div>
      <div class="stat-val">#${state.currentIteration}</div>
      <div class="stat-sub">${state.totalEpisodesPlayed} Total Episodes (Session #${state.sessionCount})</div>
    </div>
    <div class="card-stat">
      <div class="stat-title">Exam Best Score</div>
      <div class="stat-val highlight">${bestScore} / 100</div>
      <div class="stat-sub">Best Pass Rate: ${bestPass}% (Iter #${bestScoreIter})</div>
    </div>
    <div class="card-stat">
      <div class="stat-title">Open-World Survival Tier</div>
      <div class="stat-val" style="color: #10b981;">${latestSurvival?.techTierReached || 'Ready'}</div>
      <div class="stat-sub">${latestSurvival ? `${latestSurvival.advancementsUnlocked.length} Advancements | ${latestSurvival.goalTitle}` : 'Starts on next loop'}</div>
    </div>
    <div class="card-stat">
      <div class="stat-title">Vanilla Dragon Survival</div>
      <div class="stat-val" style="color: #f59e0b;">${bestDragonScore.toFixed(0)}%</div>
      <div class="stat-sub">${bestDragonMilestones} / 8 Milestones Passed</div>
    </div>
    <div class="card-stat">
      <div class="stat-title">SLM Loss & ACT Aim</div>
      <div class="stat-val" style="color: #60a5fa;">${minLoss.toFixed(4)}</div>
      <div class="stat-sub">Init: ${initialLoss.toFixed(4)} | ACT Err: ${typeof latestRecord?.actAimErrorDeg === 'number' ? latestRecord.actAimErrorDeg + '°' : 'Ready'}</div>
    </div>
  </div>

  <div class="grid-charts">
    <div class="chart-container">
      <canvas id="scoreChart"></canvas>
    </div>
    <div class="chart-container">
      <canvas id="domainBarChart"></canvas>
    </div>
    <div class="chart-container">
      <canvas id="survivalChart"></canvas>
    </div>
    <div class="chart-container">
      <canvas id="neuralChart"></canvas>
    </div>
  </div>

  <h2 style="margin-bottom: 0.5rem; font-size: 1.2rem;">Training Iteration Telemetry Logs</h2>
  <table>
    <thead>
      <tr>
        <th>Iter</th>
        <th>Mean Score</th>
        <th>Pass Rate</th>
        <th>Crafting%</th>
        <th>Combat%</th>
        <th>SLM Loss</th>
        <th>ACT Aim Err</th>
        <th>Survival Campaign</th>
      </tr>
    </thead>
    <tbody>
      ${
        history.length === 0
          ? `<tr><td colspan="8" style="text-align:center;color:#94a3b8;padding:1.5rem;">Loop 0 Initialized — Ready for Iteration #1</td></tr>`
          : history.slice().reverse().map((h) => {
        const craftRate = h.domainBreakdown?.['Crafting & Tech']?.passRate;
        const combatRate = h.domainBreakdown?.['Tactical Combat']?.passRate;
        return `
        <tr>
          <td><strong>#${h.iteration}</strong></td>
          <td><strong>${h.meanScore}</strong> / 100</td>
          <td><span class="tag ${h.passRate >= 40 ? 'tag-pass' : ''}">${h.passRate}%</span> (${h.tasksPassed}/${h.totalTasks})</td>
          <td>${craftRate !== undefined ? craftRate + '%' : '-'}</td>
          <td>${combatRate !== undefined ? combatRate + '%' : '-'}</td>
          <td style="color: #f43f5e; font-family: monospace;">${h.slmFinalLoss !== undefined ? h.slmFinalLoss.toFixed(4) : '-'}</td>
          <td style="color: #38bdf8; font-family: monospace;">${h.actAimErrorDeg !== undefined ? h.actAimErrorDeg.toFixed(2) + '°' : '-'}</td>
          <td>${h.survivalCampaign ? `<span class="tag tag-amber">${h.survivalCampaign.techTierReached} (${h.survivalCampaign.advancementsUnlocked.length} Adv)</span>` : (h.milestone ? `<span class="tag tag-amber">${h.milestone}</span>` : '-')}</td>
        </tr>
      `;
      }).join('')}
    </tbody>
  </table>

  <script>
    const historyData = ${JSON.stringify(history)};
    const sampledDomains = ${JSON.stringify(sampledDomainIters)};
    const domainBuckets = ${JSON.stringify(DOMAIN_BUCKETS)};
    const domainColors = ${JSON.stringify(DOMAIN_COLORS)};

    const labels = historyData.map(h => 'Iter ' + h.iteration);

    // 1. Standardized Exam Trajectory
    new Chart(document.getElementById('scoreChart'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Exam Mean Score (0-100)',
            data: historyData.map(h => h.meanScore),
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.12)',
            borderWidth: 3,
            fill: true,
            tension: 0.2,
          },
          {
            label: 'Exam Pass Rate (%)',
            data: historyData.map(h => h.passRate),
            borderColor: '#10b981',
            borderWidth: 2,
            borderDash: [5, 5],
            tension: 0.2,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: { display: true, text: 'Panel 1: Standardized MINEbench Exam over Time (t)', color: '#f8fafc', font: { size: 15 } },
          legend: { labels: { color: '#94a3b8' } }
        },
        scales: {
          y: { min: 0, max: 100, grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
          x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
        }
      }
    });

    // 2. Per-Domain Skill Breakdown Grouped Bar Chart at Time Increments t
    new Chart(document.getElementById('domainBarChart'), {
      type: 'bar',
      data: {
        labels: sampledDomains.map(h => 'Iter ' + h.iteration),
        datasets: domainBuckets.map(dom => ({
          label: dom,
          data: sampledDomains.map(h => h.domainBreakdown?.[dom]?.passRate ?? 0),
          backgroundColor: domainColors[dom] || '#64748b',
          borderRadius: 3,
        }))
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: { display: true, text: 'Panel 2: Per-Domain Pass Rate (%) Grouped Bars across Increments (t)', color: '#f8fafc', font: { size: 15 } },
          legend: { labels: { color: '#94a3b8', boxWidth: 12 } }
        },
        scales: {
          y: { min: 0, max: 100, grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
          x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
        }
      }
    });

    // 3. Open-World Survival Campaign & Advancements Tracker
    const tierScale = { Raw: 10, Wood: 25, Stone: 45, Iron: 65, Diamond: 80, Nether: 92, End: 100 };
    new Chart(document.getElementById('survivalChart'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Tech Tier Depth Score (0-100)',
            data: historyData.map(h => h.survivalCampaign ? (tierScale[h.survivalCampaign.techTierReached] ?? h.survivalCampaign.score) : (h.dragonSurvivalScore ?? null)),
            backgroundColor: 'rgba(16, 185, 129, 0.65)',
            borderColor: '#10b981',
            borderWidth: 1,
          },
          {
            label: 'Advancements Unlocked (Count x 10)',
            data: historyData.map(h => h.survivalCampaign ? (h.survivalCampaign.advancementsUnlocked.length * 10) : null),
            type: 'line',
            borderColor: '#f59e0b',
            backgroundColor: '#f59e0b',
            borderWidth: 3,
            spanGaps: true,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: { display: true, text: 'Panel 3: Persistent Open-World Survival & Campaign Progress', color: '#f8fafc', font: { size: 15 } },
          legend: { labels: { color: '#94a3b8' } }
        },
        scales: {
          y: { min: 0, max: 100, grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
          x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
        }
      }
    });

    // 4. Dual-Brain Neural Telemetry (SLM Loss + ACT Aim Error)
    new Chart(document.getElementById('neuralChart'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'System 2 SLM Loss',
            data: historyData.map(h => h.slmFinalLoss ?? null),
            borderColor: '#f43f5e',
            backgroundColor: 'rgba(244, 63, 94, 0.1)',
            borderWidth: 2.5,
            fill: true,
            yAxisID: 'yLoss',
            tension: 0.2,
          },
          {
            label: 'System 1 ACT Aim Error (deg)',
            data: historyData.map(h => h.actAimErrorDeg ?? null),
            borderColor: '#38bdf8',
            borderWidth: 2.5,
            spanGaps: true,
            yAxisID: 'yAim',
            tension: 0.2,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: { display: true, text: 'Panel 4: Dual-Brain Neural Telemetry (SLM Loss & ACT 80.8M Aim Error)', color: '#f8fafc', font: { size: 15 } },
          legend: { labels: { color: '#94a3b8' } }
        },
        scales: {
          yLoss: { type: 'linear', position: 'left', min: 0, grid: { color: '#334155' }, ticks: { color: '#f43f5e' } },
          yAim: { type: 'linear', position: 'right', min: 0, grid: { drawOnChartArea: false }, ticks: { color: '#38bdf8' } },
          x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
        }
      }
    });
  </script>
</body>
</html>`;
}

if (require.main === module) {
  generateTrainingVisuals();
}
