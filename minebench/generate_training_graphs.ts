/**
 * NeuroCraft Training Graph & Visual Dashboard Generator
 * =======================================================
 * Reads `training_state.json` and renders:
 * 1. `minebench/results/training_curves.svg` (High-res standalone vector chart)
 * 2. `minebench/results/training_dashboard.html` (Interactive web dashboard with Canvas graphs)
 * 3. `minebench/results/training_history.csv` (Tabular format for Excel / Pandas / Seaborn)
 */

import * as fs from 'fs';
import * as path from 'path';

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

export function generateTrainingVisuals(): void {
  const resultsDir = path.resolve(process.cwd(), 'minebench', 'results');
  const statePath = path.join(resultsDir, 'training_state.json');

  if (!fs.existsSync(statePath)) {
    console.error(`[GraphGen] Error: training_state.json not found at ${statePath}`);
    return;
  }

  const state: PersistentTrainingState = JSON.parse(fs.readFileSync(statePath, 'utf-8'));
  const history: IterationRecord[] = state.history || [];

  if (history.length === 0) {
    console.warn(`[GraphGen] Warning: No iteration history found in training_state.json`);
    return;
  }

  // 1. Export CSV
  const csvPath = path.join(resultsDir, 'training_history.csv');
  const csvRows = [
    'iteration,phase,timestamp,mean_score,pass_rate_pct,tasks_passed,total_tasks,slm_initial_loss,slm_final_loss,dragon_survival_score,milestone',
  ];
  for (const h of history) {
    csvRows.push(
      `${h.iteration},"${h.phase || ''}","${h.timestamp}",${h.meanScore},${h.passRate},${h.tasksPassed},${h.totalTasks},${h.slmInitialLoss ?? ''},${h.slmFinalLoss ?? ''},${h.dragonSurvivalScore ?? ''},"${h.milestone || ''}"`
    );
  }
  fs.writeFileSync(csvPath, csvRows.join('\n'), 'utf-8');
  console.log(`[GraphGen] Saved CSV: ${csvPath}`);

  // 2. Render SVG Vector Graphic
  const svgPath = path.join(resultsDir, 'training_curves.svg');
  const svgContent = renderSvgChart(history, state);
  fs.writeFileSync(svgPath, svgContent, 'utf-8');
  console.log(`[GraphGen] Saved SVG Chart: ${svgPath}`);

  // 3. Render Interactive HTML Dashboard
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
    } catch {}
  }
}

function renderSvgChart(history: IterationRecord[], state: PersistentTrainingState): string {
  const width = 1100;
  const height = 700;
  const margin = { top: 70, right: 80, bottom: 60, left: 70 };

  const plotW = width - margin.left - margin.right;
  const plotH1 = 240; // Top chart (Scores & Pass Rate)
  const gap = 60;
  const plotH2 = 200; // Bottom chart (SLM Loss)

  const topY0 = margin.top;
  const botY0 = margin.top + plotH1 + gap;

  const minIter = Math.min(0, ...history.map((h) => h.iteration));
  const maxIter = Math.max(10, ...history.map((h) => h.iteration));

  const getX = (iter: number) => margin.left + ((iter - minIter) / (maxIter - minIter)) * plotW;
  const getYScore = (score: number) => topY0 + plotH1 - (Math.max(0, Math.min(100, score)) / 100) * plotH1;

  // Max loss for scaling
  const validLosses = history.map((h) => h.slmFinalLoss).filter((l): l is number => typeof l === 'number');
  const maxLoss = validLosses.length > 0 ? Math.max(0.35, ...validLosses) : 0.4;
  const getYLoss = (loss: number) => botY0 + plotH2 - (Math.max(0, Math.min(maxLoss, loss)) / maxLoss) * plotH2;

  // Generate Score line path
  const scorePath = history
    .map((h, i) => `${i === 0 ? 'M' : 'L'} ${getX(h.iteration).toFixed(1)} ${getYScore(h.meanScore).toFixed(1)}`)
    .join(' ');

  // Generate Pass Rate line path
  const passPath = history
    .map((h, i) => `${i === 0 ? 'M' : 'L'} ${getX(h.iteration).toFixed(1)} ${getYScore(h.passRate).toFixed(1)}`)
    .join(' ');

  // Generate Loss line path
  const lossPoints = history.filter((h) => typeof h.slmFinalLoss === 'number');
  const lossPath = lossPoints
    .map((h, i) => `${i === 0 ? 'M' : 'L'} ${getX(h.iteration).toFixed(1)} ${getYLoss(h.slmFinalLoss!).toFixed(1)}`)
    .join(' ');

  // Milestone pins
  const milestonePins = history
    .filter((h) => h.milestone)
    .map((h) => {
      const cx = getX(h.iteration);
      const cy = getYScore(h.meanScore);
      return `
        <circle cx="${cx}" cy="${cy}" r="6" fill="#f59e0b" stroke="#ffffff" stroke-width="2"/>
        <text x="${cx}" y="${cy - 12}" fill="#f59e0b" font-size="11" font-weight="bold" text-anchor="middle">${h.milestone}</text>
      `;
    })
    .join('\n');

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" style="background-color: #0f172a; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  <defs>
    <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#3b82f6" stop-opacity="0.4"/>
      <stop offset="100%" stop-color="#3b82f6" stop-opacity="0.0"/>
    </linearGradient>
    <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#ef4444" stop-opacity="0.3"/>
      <stop offset="100%" stop-color="#ef4444" stop-opacity="0.0"/>
    </linearGradient>
  </defs>

  <!-- Title & Subtitle -->
  <text x="${margin.left}" y="35" fill="#f8fafc" font-size="20" font-weight="bold">NeuroCraft Autonomous Training & Convergence Trajectory</text>
  <text x="${margin.left}" y="55" fill="#94a3b8" font-size="12">Dual-Tier Progress across ${history.length} Training Epochs & Live Mojang 1.20.4 Simulations | Best Score: ${state.bestMeanScore}/100</text>

  <!-- Legend -->
  <g transform="translate(${width - margin.right - 380}, 25)">
    <line x1="0" y1="10" x2="25" y2="10" stroke="#3b82f6" stroke-width="3"/>
    <text x="32" y="14" fill="#93c5fd" font-size="12">Mean Score (0-100)</text>

    <line x1="160" y1="10" x2="185" y2="10" stroke="#10b981" stroke-width="3" stroke-dasharray="4,3"/>
    <text x="192" y="14" fill="#6ee7b7" font-size="12">Pass Rate (%)</text>

    <line x1="280" y1="10" x2="305" y2="10" stroke="#f43f5e" stroke-width="3"/>
    <text x="312" y="14" fill="#fda4af" font-size="12">SLM Loss</text>
  </g>

  <!-- CHART 1: Scores & Pass Rate -->
  <rect x="${margin.left}" y="${topY0}" width="${plotW}" height="${plotH1}" fill="#1e293b" rx="6"/>
  <!-- Gridlines -->
  ${[0, 25, 50, 75, 100].map((val) => {
    const y = getYScore(val);
    return `
      <line x1="${margin.left}" y1="${y}" x2="${margin.left + plotW}" y2="${y}" stroke="#334155" stroke-width="1" stroke-dasharray="${val === 0 ? '0' : '2,2'}"/>
      <text x="${margin.left - 10}" y="${y + 4}" fill="#64748b" font-size="11" text-anchor="end">${val}</text>
    `;
  }).join('')}

  <!-- Paths -->
  <path d="${passPath}" fill="none" stroke="#10b981" stroke-width="2.5" stroke-dasharray="5,4"/>
  <path d="${scorePath}" fill="none" stroke="#3b82f6" stroke-width="3"/>

  <!-- Data Points for Scores -->
  ${history.map((h) => `<circle cx="${getX(h.iteration)}" cy="${getYScore(h.meanScore)}" r="3.5" fill="#3b82f6"/>`).join('')}

  <!-- Milestone Pins -->
  ${milestonePins}

  <!-- CHART 2: SLM Training Loss -->
  <rect x="${margin.left}" y="${botY0}" width="${plotW}" height="${plotH2}" fill="#1e293b" rx="6"/>
  <text x="${margin.left + 15}" y="${botY0 + 25}" fill="#f8fafc" font-size="13" font-weight="bold">System 2 Qwen2.5-Coder SLM Cross-Entropy Loss</text>

  <!-- Gridlines for Loss -->
  ${[0, 0.1, 0.2, 0.3].map((val) => {
    const y = getYLoss(val);
    return `
      <line x1="${margin.left}" y1="${y}" x2="${margin.left + plotW}" y2="${y}" stroke="#334155" stroke-width="1" stroke-dasharray="${val === 0 ? '0' : '2,2'}"/>
      <text x="${margin.left - 10}" y="${y + 4}" fill="#64748b" font-size="11" text-anchor="end">${val.toFixed(2)}</text>
    `;
  }).join('')}

  <!-- Loss Path -->
  <path d="${lossPath}" fill="none" stroke="#f43f5e" stroke-width="3"/>
  ${lossPoints.map((h) => `<circle cx="${getX(h.iteration)}" cy="${getYLoss(h.slmFinalLoss!)}" r="3" fill="#f43f5e"/>`).join('')}

  <!-- X Axis Labels -->
  ${history.map((h, i) => {
    if (i % Math.ceil(history.length / 15) === 0 || i === history.length - 1) {
      return `<text x="${getX(h.iteration)}" y="${botY0 + plotH2 + 25}" fill="#94a3b8" font-size="11" text-anchor="middle">Iter ${h.iteration}</text>`;
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

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NeuroCraft AI Agent — Autonomous Training & Convergence Dashboard</title>
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
    .chart-container { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; margin-bottom: 2rem; height: 380px; }
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
    <h1>🧠 NeuroCraft Autonomous Training & Convergence Dashboard <span class="badge">Live RTX 5060 + Mojang 1.20.4</span></h1>
    <p class="subtitle">Continuously tracked performance curves across System 1 (80.8M Action Chunking Transformer) & System 2 (Qwen2.5-Coder 1.5B QLoRA).</p>
  </header>

  <div class="grid-stats">
    <div class="card-stat">
      <div class="stat-title">Current Iteration</div>
      <div class="stat-val">#${state.currentIteration}</div>
      <div class="stat-sub">Session Count: ${state.sessionCount}</div>
    </div>
    <div class="card-stat">
      <div class="stat-title">All-Time Best Score</div>
      <div class="stat-val highlight">${bestScore} / 100</div>
      <div class="stat-sub">Best Pass Rate: ${bestPass}% (Best Score Iter #${bestScoreIter})</div>
    </div>
    <div class="card-stat">
      <div class="stat-title">Vanilla Dragon Survival</div>
      <div class="stat-val" style="color: #f59e0b;">${bestDragonScore.toFixed(0)}%</div>
      <div class="stat-sub">${bestDragonMilestones} / 8 Milestones Passed</div>
    </div>
    <div class="card-stat">
      <div class="stat-title">SLM Loss Low</div>
      <div class="stat-val" style="color: #60a5fa;">${minLoss.toFixed(4)}</div>
      <div class="stat-sub">Down from ${initialLoss.toFixed(4)} initial loss</div>
    </div>
    <div class="card-stat">
      <div class="stat-title">Action Head Specs</div>
      <div class="stat-val">80.8M</div>
      <div class="stat-sub">8-Tick (400ms) Chunk Trajectory</div>
    </div>
    <div class="card-stat">
      <div class="stat-title">GPU VRAM Resident</div>
      <div class="stat-val" style="color: #a78bfa;">3.6 GB</div>
      <div class="stat-sub">4.2 GB Headroom on RTX 5060</div>
    </div>
  </div>

  <div class="chart-container">
    <canvas id="scoreChart"></canvas>
  </div>

  <div class="chart-container">
    <canvas id="lossChart"></canvas>
  </div>

  <h2 style="margin-bottom: 0.5rem; font-size: 1.2rem;">Training Iteration Logs</h2>
  <table>
    <thead>
      <tr>
        <th>Iter</th>
        <th>Phase / Note</th>
        <th>Mean Score</th>
        <th>Pass Rate</th>
        <th>Tasks Passed</th>
        <th>SLM Final Loss</th>
        <th>Dragon Survival</th>
        <th>Milestone / Breakthrough</th>
      </tr>
    </thead>
    <tbody>
      ${history.slice().reverse().map((h) => `
        <tr>
          <td><strong>#${h.iteration}</strong></td>
          <td>${h.phase || 'Continuous Loop'}</td>
          <td><strong>${h.meanScore}</strong> / 100</td>
          <td><span class="tag ${h.passRate >= 40 ? 'tag-pass' : ''}">${h.passRate}%</span></td>
          <td>${h.tasksPassed} / ${h.totalTasks}</td>
          <td style="color: #f43f5e; font-family: monospace;">${h.slmFinalLoss !== undefined ? h.slmFinalLoss.toFixed(4) : '-'}</td>
          <td>${h.dragonSurvivalScore !== undefined ? `<span class="tag tag-amber">${h.dragonSurvivalScore.toFixed(0)}% (${h.dragonMilestones || 0}/8)</span>` : '-'}</td>
          <td>${h.milestone ? `<span class="tag tag-amber">⭐ ${h.milestone}</span>` : '-'}</td>
        </tr>
      `).join('')}
    </tbody>
  </table>

  <script>
    const historyData = ${JSON.stringify(history)};
    const labels = historyData.map(h => 'Iter ' + h.iteration);
    const scores = historyData.map(h => h.meanScore);
    const passRates = historyData.map(h => h.passRate);
    const slmLosses = historyData.map(h => h.slmFinalLoss !== undefined ? h.slmFinalLoss : null);
    const dragonScores = historyData.map(h => typeof h.dragonSurvivalScore === 'number' ? h.dragonSurvivalScore : null);

    // Chart 1: Scores & Pass Rate
    new Chart(document.getElementById('scoreChart'), {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Benchmark Mean Score (0-100)',
            data: scores,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 3,
            fill: true,
            tension: 0.2,
          },
          {
            label: 'Pass Rate (%)',
            data: passRates,
            borderColor: '#10b981',
            borderWidth: 2,
            borderDash: [5, 5],
            fill: false,
            tension: 0.2,
          },
          {
            label: 'Vanilla Dragon Survival (%)',
            data: dragonScores,
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.2)',
            borderWidth: 2.5,
            pointRadius: 6,
            pointBackgroundColor: '#f59e0b',
            pointBorderColor: '#ffffff',
            pointBorderWidth: 2,
            spanGaps: true,
            tension: 0.2,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: { display: true, text: 'Benchmark Score, Pass Rate & Dragon Survival Trajectory', color: '#f8fafc', font: { size: 16 } },
          legend: { labels: { color: '#94a3b8' } }
        },
        scales: {
          y: { min: 0, max: 100, grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
          x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
        }
      }
    });

    // Chart 2: Loss
    new Chart(document.getElementById('lossChart'), {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'System 2 SLM Cross-Entropy Loss',
            data: slmLosses,
            borderColor: '#f43f5e',
            backgroundColor: 'rgba(244, 63, 94, 0.1)',
            borderWidth: 3,
            fill: true,
            tension: 0.2,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: { display: true, text: 'System 2 SLM Convergence (QLoRA Fine-Tuning Loss)', color: '#f8fafc', font: { size: 16 } },
          legend: { labels: { color: '#94a3b8' } }
        },
        scales: {
          y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
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
