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
  heldOutMeanScore?: number;
  heldOutPassRate?: number;
  heldOutPassed?: number;
  heldOutTotal?: number;
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
  'Tactical Combat': '#c92a2a',
  'Crafting & Tech': '#d9480f',
  'Agility & Parkour': '#15803d',
  'Building & 3D': '#5b21b6',
  'Survival & Affordances': '#1d4ed8',
  'Puzzles & Tooling': '#0f766e',
  'Multiplayer': '#9d174d',
};

const DOMAIN_CODES: Record<string, { code: string; short: string }> = {
  'Tactical Combat': { code: 'CH-01', short: 'COMBAT' },
  'Crafting & Tech': { code: 'CH-02', short: 'CRAFT' },
  'Agility & Parkour': { code: 'CH-03', short: 'AGILITY' },
  'Building & 3D': { code: 'CH-04', short: 'BUILD' },
  'Survival & Affordances': { code: 'CH-05', short: 'SURVIVAL' },
  'Puzzles & Tooling': { code: 'CH-06', short: 'PUZZLE' },
  'Multiplayer': { code: 'CH-07', short: 'MP' },
};

function escapeXml(value: unknown): string {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function escapeHtml(value: unknown): string {
  return escapeXml(value).replace(/'/g, '&#39;');
}

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
    'iteration,phase,timestamp,mean_score,pass_rate_pct,tasks_passed,total_tasks,held_out_score,held_out_pass_pct,slm_initial_loss,slm_final_loss,act_aim_err_deg,survival_tier,survival_advancements,dragon_survival_score,milestone',
  ];
  for (const h of history) {
    csvRows.push(
      `${h.iteration},"${h.phase || ''}","${h.timestamp}",${h.meanScore},${h.passRate},${h.tasksPassed},${h.totalTasks},${h.heldOutMeanScore ?? ''},${h.heldOutPassRate ?? ''},${h.slmInitialLoss ?? ''},${h.slmFinalLoss ?? ''},${h.actAimErrorDeg ?? ''},"${h.survivalCampaign?.techTierReached ?? ''}",${h.survivalCampaign?.advancementsUnlocked?.length ?? ''},${h.dragonSurvivalScore ?? ''},"${h.milestone || ''}"`
    );
  }
  fs.writeFileSync(csvPath, csvRows.join('\n'), 'utf-8');
  console.log(`[GraphGen] Saved CSV: ${csvPath}`);

  // 2. Render SVG Vector Graphic (Mechanical Concrete Telemetry Sheet)
  const svgPath = path.join(resultsDir, 'training_curves.svg');
  const svgContent = renderSvgChart(history, state);
  fs.writeFileSync(svgPath, svgContent, 'utf-8');
  console.log(`[GraphGen] Saved SVG Chart: ${svgPath}`);

  // 3. Render Interactive Concrete & Mechanical HTML Dashboard
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

function buildSvgPolyline(points: Array<{ x: number; y: number }>): string {
  if (points.length === 0) return '';
  return points.map((pt, i) => `${i === 0 ? 'M' : 'L'} ${pt.x.toFixed(1)} ${pt.y.toFixed(1)}`).join(' ');
}

function renderSvgChart(history: IterationRecord[], state: PersistentTrainingState): string {
  const width = 1360;
  const height = 1280;
  const padX = 48;
  const contentW = width - padX * 2;

  const yHeader = 18;
  const hHeader = 64;

  // Panel boxes
  const y1 = 102;
  const h1 = 232;
  const y2 = 356;
  const h2 = 248;
  const y3 = 626;
  const h3 = 200;
  const y4 = 848;
  const h4 = 368;

  const plotLeft = padX + 54;
  const plotRight = padX + contentW - 22;
  const plotW = plotRight - plotLeft;

  const minIter = history.length > 0 ? Math.min(1, ...history.map((h) => h.iteration)) : 0;
  const maxIter = history.length > 0 ? Math.max(10, ...history.map((h) => h.iteration)) : 10;
  const getX = (iter: number) => plotLeft + ((iter - minIter) / Math.max(1, maxIter - minIter)) * plotW;

  const hasPhaseSplit = history.some((h) => h.iteration < 41) && history.some((h) => h.iteration >= 41);
  const phaseBX = getX(41);

  const renderPctGrid = (topY: number, innerH: number) => {
    const mapY = (v: number) => topY + innerH - (Math.max(0, Math.min(100, v)) / 100) * innerH;
    const lines = [0, 25, 50, 75, 100]
      .map((v) => {
        const yy = mapY(v);
        return `<line x1="${plotLeft}" y1="${yy.toFixed(1)}" x2="${plotRight}" y2="${yy.toFixed(1)}" stroke="#c7c3b7" stroke-width="1" ${v > 0 && v < 100 ? 'stroke-dasharray="3,3"' : ''}/><text x="${plotLeft - 8}" y="${(yy + 3.5).toFixed(1)}" fill="#52524b" font-size="10" text-anchor="end">${v}</text>`;
      })
      .join('');
    const iterTicks = history
      .filter((_, i) => i % Math.max(1, Math.ceil(history.length / 12)) === 0 || i === history.length - 1)
      .map((h) => `<text x="${getX(h.iteration).toFixed(1)}" y="${(topY + innerH + 16).toFixed(1)}" fill="#52524b" font-size="10" text-anchor="middle">I${h.iteration}</text>`)
      .join('');
    const phaseMark = hasPhaseSplit
      ? `<line x1="${phaseBX.toFixed(1)}" y1="${topY}" x2="${phaseBX.toFixed(1)}" y2="${topY + innerH}" stroke="#c92a2a" stroke-width="1.4" stroke-dasharray="4,3"/><rect x="${(phaseBX + 4).toFixed(1)}" y="${topY + 3}" width="60" height="14" fill="#c92a2a"/><text x="${(phaseBX + 34).toFixed(1)}" y="${topY + 13}" fill="#efede6" font-size="8.5" font-weight="bold" text-anchor="middle">PHASE B</text>`
      : '';
    return { mapY, grid: lines + iterTicks + phaseMark };
  };

  // Panel 1: Original Standardized Exam curves
  const p1Top = y1 + 52;
  const p1InnerH = h1 - 78;
  const p1Grid = renderPctGrid(p1Top, p1InnerH);
  const scorePath = buildSvgPolyline(history.map((h) => ({ x: getX(h.iteration), y: p1Grid.mapY(h.meanScore) })));
  const passPath = buildSvgPolyline(history.map((h) => ({ x: getX(h.iteration), y: p1Grid.mapY(h.passRate) })));
  const heldOutPath = buildSvgPolyline(
    history.filter((h) => typeof h.heldOutMeanScore === 'number').map((h) => ({ x: getX(h.iteration), y: p1Grid.mapY(h.heldOutMeanScore!) }))
  );

  // Panel 2: All 7 Category Pass Rate curves
  const p2Top = y2 + 56;
  const p2InnerH = h2 - 82;
  const p2Grid = renderPctGrid(p2Top, p2InnerH);
  const categoryPathsSvg = DOMAIN_BUCKETS.map((dom) => {
    const pts = history
      .filter((h) => typeof h.domainBreakdown?.[dom]?.passRate === 'number')
      .map((h) => ({ x: getX(h.iteration), y: p2Grid.mapY(h.domainBreakdown![dom].passRate) }));
    const d = buildSvgPolyline(pts);
    const color = DOMAIN_COLORS[dom] || '#1b1c19';
    return d ? `<path d="${d}" fill="none" stroke="${color}" stroke-width="2.1"/>` : '';
  }).join('');

  const p2LegendSvg = DOMAIN_BUCKETS.map((dom, idx) => {
    const lx = padX + 16 + idx * 178;
    const ly = y2 + 37;
    const color = DOMAIN_COLORS[dom] || '#1b1c19';
    const meta = DOMAIN_CODES[dom];
    const latestRec = [...history].reverse().find((h) => h.domainBreakdown?.[dom]);
    const latestPct = latestRec?.domainBreakdown?.[dom]?.passRate;
    return `<rect x="${lx}" y="${ly - 8}" width="10" height="10" fill="${color}"/><text x="${lx + 15}" y="${ly}" fill="#1b1c19" font-size="10" font-weight="bold">${meta.code} ${meta.short} ${latestPct !== undefined ? latestPct.toFixed(0) + '%' : '--'}</text>`;
  }).join('');

  // Panel 3: 7-Channel Small Multiples Strip
  const chGap = 10;
  const chW = (contentW - chGap * (DOMAIN_BUCKETS.length - 1)) / DOMAIN_BUCKETS.length;
  const channelsSvg = DOMAIN_BUCKETS.map((dom, idx) => {
    const cx = padX + idx * (chW + chGap);
    const color = DOMAIN_COLORS[dom] || '#1b1c19';
    const meta = DOMAIN_CODES[dom];
    const series = history
      .filter((h) => h.domainBreakdown?.[dom])
      .map((h) => ({ iter: h.iteration, pass: h.domainBreakdown![dom].passRate, score: h.domainBreakdown![dom].meanScore }));
    const latestVal = series.length > 0 ? series[series.length - 1].pass : 0;
    const peakVal = series.length > 0 ? Math.max(...series.map((s) => s.pass)) : 0;
    const subLeft = cx + 10;
    const subW = chW - 20;
    const subTop = y3 + 58;
    const subH = h3 - 78;
    const subX = (iter: number) => subLeft + ((iter - minIter) / Math.max(1, maxIter - minIter)) * subW;
    const subY = (v: number) => subTop + subH - (Math.max(0, Math.min(100, v)) / 100) * subH;
    const passD = buildSvgPolyline(series.map((s) => ({ x: subX(s.iter), y: subY(s.pass) })));
    const scoreD = buildSvgPolyline(series.map((s) => ({ x: subX(s.iter), y: subY(s.score) })));
    const phaseMini = hasPhaseSplit
      ? `<line x1="${subX(41).toFixed(1)}" y1="${subTop}" x2="${subX(41).toFixed(1)}" y2="${subTop + subH}" stroke="#c92a2a" stroke-width="1" stroke-dasharray="2,2"/>`
      : '';
    return `
      <rect x="${(cx + 3).toFixed(1)}" y="${y3 + 3}" width="${chW.toFixed(1)}" height="${h3}" fill="#1b1c19"/>
      <rect x="${cx.toFixed(1)}" y="${y3}" width="${chW.toFixed(1)}" height="${h3}" fill="#efede6" stroke="#1b1c19" stroke-width="1.5"/>
      <rect x="${cx.toFixed(1)}" y="${y3}" width="${chW.toFixed(1)}" height="24" fill="#1b1c19"/>
      <rect x="${cx.toFixed(1)}" y="${y3}" width="6" height="24" fill="${color}"/>
      <text x="${(cx + 12).toFixed(1)}" y="${y3 + 16}" fill="#efede6" font-size="10.5" font-weight="bold">${meta.code} // ${meta.short}</text>
      <text x="${(cx + 10).toFixed(1)}" y="${y3 + 42}" fill="#1b1c19" font-size="11" font-weight="bold">NOW ${latestVal.toFixed(0)}%</text>
      <text x="${(cx + chW - 10).toFixed(1)}" y="${y3 + 42}" fill="#52524b" font-size="10" text-anchor="end">PK ${peakVal.toFixed(0)}%</text>
      <rect x="${subLeft.toFixed(1)}" y="${subTop}" width="${subW.toFixed(1)}" height="${subH}" fill="#e6e3da" stroke="#c7c3b7" stroke-width="1"/>
      <line x1="${subLeft.toFixed(1)}" y1="${subY(50).toFixed(1)}" x2="${(subLeft + subW).toFixed(1)}" y2="${subY(50).toFixed(1)}" stroke="#c7c3b7" stroke-width="1" stroke-dasharray="2,2"/>
      ${phaseMini}
      ${scoreD ? `<path d="${scoreD}" fill="none" stroke="#1b1c19" stroke-opacity="0.45" stroke-width="1.4" stroke-dasharray="3,2"/>` : ''}
      ${passD ? `<path d="${passD}" fill="none" stroke="${color}" stroke-width="2.2"/>` : ''}
      <text x="${subLeft.toFixed(1)}" y="${y3 + h3 - 6}" fill="#52524b" font-size="9">I${minIter}</text>
      <text x="${(subLeft + subW).toFixed(1)}" y="${y3 + h3 - 6}" fill="#52524b" font-size="9" text-anchor="end">I${maxIter}</text>
    `;
  }).join('');

  // Panel 4 Split Row: Left = Latest Bar Snapshot, Right = Dual-Brain Convergence (SLM Loss + ACT Aim Error)
  const splitGap = 18;
  const halfW = (contentW - splitGap) / 2;
  const leftX = padX;
  const rightX = padX + halfW + splitGap;

  const latestWithDomains =
    history.length > 0 ? [...history].reverse().find((h) => h.domainBreakdown) || history[history.length - 1] : undefined;
  const bd = latestWithDomains?.domainBreakdown || {};
  const barAreaLeft = leftX + 46;
  const barAreaW = halfW - 66;
  const barAreaTop = y4 + 54;
  const barAreaH = h4 - 94;
  const slotW = barAreaW / DOMAIN_BUCKETS.length;
  const barsSvg = DOMAIN_BUCKETS.map((dom, idx) => {
    const snap = bd[dom];
    const passVal = snap ? snap.passRate : 0;
    const scoreVal = snap ? snap.meanScore : 0;
    const gx = barAreaLeft + idx * slotW + 6;
    const subBarW = Math.max(8, Math.floor((slotW - 14) / 2));
    const hPass = (Math.max(0, Math.min(100, passVal)) / 100) * barAreaH;
    const hScore = (Math.max(0, Math.min(100, scoreVal)) / 100) * barAreaH;
    const yPass = barAreaTop + barAreaH - hPass;
    const yScore = barAreaTop + barAreaH - hScore;
    const color = DOMAIN_COLORS[dom] || '#1b1c19';
    const meta = DOMAIN_CODES[dom];
    return `
      <rect x="${gx.toFixed(1)}" y="${yScore.toFixed(1)}" width="${subBarW}" height="${hScore.toFixed(1)}" fill="${color}" fill-opacity="0.32" stroke="#1b1c19" stroke-width="0.8"/>
      <rect x="${(gx + subBarW + 2).toFixed(1)}" y="${yPass.toFixed(1)}" width="${subBarW}" height="${hPass.toFixed(1)}" fill="${color}" stroke="#1b1c19" stroke-width="0.8"/>
      <text x="${(gx + subBarW).toFixed(1)}" y="${(Math.min(yScore, yPass) - 5).toFixed(1)}" fill="#1b1c19" font-size="9.5" font-weight="bold" text-anchor="middle">${passVal.toFixed(0)}%</text>
      <text x="${(gx + subBarW).toFixed(1)}" y="${barAreaTop + barAreaH + 15}" fill="#3a3c36" font-size="9.5" font-weight="bold" text-anchor="middle">${meta.short}</text>
    `;
  }).join('');

  // Right panel: SLM Loss (left) + ACT Aim Error (right)
  const nPlotLeft = rightX + 54;
  const nPlotRight = rightX + halfW - 50;
  const nPlotW = nPlotRight - nPlotLeft;
  const nPlotTop = y4 + 54;
  const nPlotH = h4 - 94;
  const getNX = (iter: number) => nPlotLeft + ((iter - minIter) / Math.max(1, maxIter - minIter)) * nPlotW;
  const lossPoints = history.filter((h) => typeof h.slmFinalLoss === 'number');
  const maxLoss = lossPoints.length > 0 ? Math.max(1.0, ...lossPoints.map((h) => h.slmFinalLoss!)) : 2.5;
  const getNYLoss = (loss: number) => nPlotTop + nPlotH - (Math.max(0, Math.min(maxLoss, loss)) / maxLoss) * nPlotH;
  const slmLossD = buildSvgPolyline(lossPoints.map((h) => ({ x: getNX(h.iteration), y: getNYLoss(h.slmFinalLoss!) })));

  const aimPoints = history.filter((h) => typeof h.actAimErrorDeg === 'number');
  const maxAim = aimPoints.length > 0 ? Math.max(10, Math.ceil(Math.max(...aimPoints.map((h) => h.actAimErrorDeg!)) / 2) * 2) : 12;
  const getNYAim = (deg: number) => nPlotTop + nPlotH - (Math.max(0, Math.min(maxAim, deg)) / maxAim) * nPlotH;
  const actAimD = buildSvgPolyline(aimPoints.map((h) => ({ x: getNX(h.iteration), y: getNYAim(h.actAimErrorDeg!) })));

  const latestRecord = history[history.length - 1];
  const latestTier = escapeXml(latestRecord?.survivalCampaign?.techTierReached || 'Ready (Loop 0)');
  const latestAimStr = typeof latestRecord?.actAimErrorDeg === 'number' ? `${latestRecord.actAimErrorDeg.toFixed(2)}°` : 'N/A';

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" style="background-color: #dad7ce; font-family: 'Cascadia Mono', 'JetBrains Mono', Consolas, monospace;">
  <defs>
    <pattern id="draftGrid" width="40" height="40" patternUnits="userSpaceOnUse">
      <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#c8c4b8" stroke-width="0.8"/>
    </pattern>
  </defs>
  <rect width="${width}" height="${height}" fill="#dad7ce"/>
  <rect width="${width}" height="${height}" fill="url(#draftGrid)"/>

  <!-- Header Block -->
  <rect x="${padX + 4}" y="${yHeader + 4}" width="${contentW}" height="${hHeader}" fill="#1b1c19"/>
  <rect x="${padX}" y="${yHeader}" width="${contentW}" height="${hHeader}" fill="#efede6" stroke="#1b1c19" stroke-width="2"/>
  <rect x="${padX}" y="${yHeader}" width="10" height="${hHeader}" fill="#d9480f"/>
  <text x="${padX + 24}" y="${yHeader + 27}" fill="#1b1c19" font-size="19" font-weight="bold" letter-spacing="0.8">SPEC // NEUROCRAFT MECHANICAL TELEMETRY SHEET</text>
  <text x="${padX + 24}" y="${yHeader + 48}" fill="#494942" font-size="11.5">ITERS: ${history.length}  |  EXAM PEAK: ${state.bestMeanScore}/100 (${state.bestPassRate}%)  |  LATEST PASS: ${state.latestPassRate}%  |  ACT AIM ERR: ${latestAimStr}  |  SURVIVAL: ${latestTier}</text>

  <!-- Panel 1: Original Standardized Exam Over Time -->
  <rect x="${padX + 4}" y="${y1 + 4}" width="${contentW}" height="${h1}" fill="#1b1c19"/>
  <rect x="${padX}" y="${y1}" width="${contentW}" height="${h1}" fill="#efede6" stroke="#1b1c19" stroke-width="1.8"/>
  <rect x="${padX}" y="${y1}" width="${contentW}" height="28" fill="#1b1c19"/>
  <text x="${padX + 14}" y="${y1 + 18}" fill="#efede6" font-size="12" font-weight="bold">MOD-01 // STANDARDIZED MINEBENCH EXAM OVER TIME (MEAN SCORE, PASS RATE %, 20% HELD-OUT)</text>
  <rect x="${padX + 14}" y="${y1 + 35}" width="14" height="3" fill="#1b1c19"/><text x="${padX + 34}" y="${y1 + 40}" fill="#1b1c19" font-size="10" font-weight="bold">MEAN SCORE (0-100)</text>
  <line x1="${padX + 180}" y1="${y1 + 36.5}" x2="${padX + 196}" y2="${y1 + 36.5}" stroke="#15803d" stroke-width="2.5" stroke-dasharray="5,3"/><text x="${padX + 202}" y="${y1 + 40}" fill="#15803d" font-size="10" font-weight="bold">PASS RATE (%)</text>
  <line x1="${padX + 318}" y1="${y1 + 36.5}" x2="${padX + 334}" y2="${y1 + 36.5}" stroke="#d9480f" stroke-width="2.2" stroke-dasharray="2,3"/><text x="${padX + 340}" y="${y1 + 40}" fill="#d9480f" font-size="10" font-weight="bold">20% HELD-OUT EVAL</text>
  ${p1Grid.grid}
  ${passPath ? `<path d="${passPath}" fill="none" stroke="#15803d" stroke-width="2.2" stroke-dasharray="6,4"/>` : ''}
  ${heldOutPath ? `<path d="${heldOutPath}" fill="none" stroke="#d9480f" stroke-width="2.2" stroke-dasharray="2,3"/>` : ''}
  ${scorePath ? `<path d="${scorePath}" fill="none" stroke="#1b1c19" stroke-width="2.8"/>` : `<text x="${plotLeft + plotW / 2}" y="${p1Top + p1InnerH / 2}" fill="#52524b" font-size="13" text-anchor="middle">LOOP 0 INITIALIZED — AWAITING ITERATION #1</text>`}

  <!-- Panel 2: All-Category Training Curves -->
  <rect x="${padX + 4}" y="${y2 + 4}" width="${contentW}" height="${h2}" fill="#1b1c19"/>
  <rect x="${padX}" y="${y2}" width="${contentW}" height="${h2}" fill="#efede6" stroke="#1b1c19" stroke-width="1.8"/>
  <rect x="${padX}" y="${y2}" width="${contentW}" height="28" fill="#1b1c19"/>
  <text x="${padX + 14}" y="${y2 + 18}" fill="#efede6" font-size="12" font-weight="bold">MOD-02 // COMPOSITE PER-CATEGORY TRAINING CURVES (PASS RATE % ACROSS ALL ITERATIONS)</text>
  ${p2LegendSvg}
  ${p2Grid.grid}
  ${categoryPathsSvg}

  <!-- Panel 3: 7-Channel Per-Category Mini-Traces -->
  ${channelsSvg}

  <!-- Panel 4A: Latest Domain Snapshot Bars -->
  <rect x="${leftX + 4}" y="${y4 + 4}" width="${halfW}" height="${h4}" fill="#1b1c19"/>
  <rect x="${leftX}" y="${y4}" width="${halfW}" height="${h4}" fill="#efede6" stroke="#1b1c19" stroke-width="1.8"/>
  <rect x="${leftX}" y="${y4}" width="${halfW}" height="28" fill="#1b1c19"/>
  <text x="${leftX + 14}" y="${y4 + 18}" fill="#efede6" font-size="12" font-weight="bold">MOD-04A // PER-DOMAIN SNAPSHOT (ITER #${latestWithDomains?.iteration ?? state.currentIteration}: DIM=SCORE, SOLID=PASS%)</text>
  ${[0, 25, 50, 75, 100].map((v) => {
    const yy = barAreaTop + barAreaH - (v / 100) * barAreaH;
    return `<line x1="${barAreaLeft}" y1="${yy.toFixed(1)}" x2="${barAreaLeft + barAreaW}" y2="${yy.toFixed(1)}" stroke="#c7c3b7" stroke-width="1" stroke-dasharray="2,2"/><text x="${barAreaLeft - 8}" y="${(yy + 3.5).toFixed(1)}" fill="#52524b" font-size="10" text-anchor="end">${v}</text>`;
  }).join('')}
  ${barsSvg}

  <!-- Panel 4B: Dual-Brain Loss & Aim Error -->
  <rect x="${rightX + 4}" y="${y4 + 4}" width="${halfW}" height="${h4}" fill="#1b1c19"/>
  <rect x="${rightX}" y="${y4}" width="${halfW}" height="${h4}" fill="#efede6" stroke="#1b1c19" stroke-width="1.8"/>
  <rect x="${rightX}" y="${y4}" width="${halfW}" height="28" fill="#1b1c19"/>
  <text x="${rightX + 14}" y="${y4 + 18}" fill="#efede6" font-size="12" font-weight="bold">MOD-04B // DUAL-BRAIN CONVERGENCE (SLM LOSS + ACT AIM ERR)</text>
  <rect x="${rightX + 14}" y="${y4 + 35}" width="12" height="3" fill="#c92a2a"/><text x="${rightX + 32}" y="${y4 + 40}" fill="#c92a2a" font-size="10" font-weight="bold">SLM LOSS (LEFT)</text>
  <rect x="${rightX + 150}" y="${y4 + 35}" width="12" height="3" fill="#1d4ed8"/><text x="${rightX + 168}" y="${y4 + 40}" fill="#1d4ed8" font-size="10" font-weight="bold">ACT AIM ERR DEG (RIGHT)</text>
  ${[0, 0.25, 0.5, 0.75, 1].map((frac) => {
    const yy = nPlotTop + nPlotH - frac * nPlotH;
    return `<line x1="${nPlotLeft}" y1="${yy.toFixed(1)}" x2="${nPlotRight}" y2="${yy.toFixed(1)}" stroke="#c7c3b7" stroke-width="1" stroke-dasharray="2,2"/><text x="${nPlotLeft - 8}" y="${(yy + 3.5).toFixed(1)}" fill="#c92a2a" font-size="9.5" text-anchor="end">${(frac * maxLoss).toFixed(2)}</text><text x="${nPlotRight + 8}" y="${(yy + 3.5).toFixed(1)}" fill="#1d4ed8" font-size="9.5" text-anchor="start">${(frac * maxAim).toFixed(0)}°</text>`;
  }).join('')}
  ${hasPhaseSplit ? `<line x1="${getNX(41).toFixed(1)}" y1="${nPlotTop}" x2="${getNX(41).toFixed(1)}" y2="${nPlotTop + nPlotH}" stroke="#c92a2a" stroke-width="1.2" stroke-dasharray="4,3"/>` : ''}
  ${slmLossD ? `<path d="${slmLossD}" fill="none" stroke="#c92a2a" stroke-width="2.2"/>` : ''}
  ${actAimD ? `<path d="${actAimD}" fill="none" stroke="#1d4ed8" stroke-width="2.4"/>` : ''}
  ${history
    .filter((_, i) => i % Math.max(1, Math.ceil(history.length / 8)) === 0 || i === history.length - 1)
    .map((h) => `<text x="${getNX(h.iteration).toFixed(1)}" y="${nPlotTop + nPlotH + 16}" fill="#52524b" font-size="9.5" text-anchor="middle">I${h.iteration}</text>`)
    .join('')}
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

  // Sample up to 10 evenly spaced time increments t (plus latest) for the grouped increment bar snapshot
  const domainHistory = history.filter((h) => h.domainBreakdown);
  const sampledDomainIters: IterationRecord[] = [];
  if (domainHistory.length <= 10) {
    sampledDomainIters.push(...domainHistory);
  } else {
    const step = (domainHistory.length - 1) / 9;
    for (let i = 0; i < 9; i++) sampledDomainIters.push(domainHistory[Math.round(i * step)]);
    sampledDomainIters.push(domainHistory[domainHistory.length - 1]);
  }

  // Build HTML cards for the 7 per-category training curve channels
  const channelCardsHtml = DOMAIN_BUCKETS.map((dom, idx) => {
    const color = DOMAIN_COLORS[dom] || '#1b1c19';
    const meta = DOMAIN_CODES[dom];
    const points = history
      .filter((h) => h.domainBreakdown?.[dom])
      .map((h) => ({ iter: h.iteration, snap: h.domainBreakdown![dom] }));
    const latestSnap = points.length > 0 ? points[points.length - 1].snap : undefined;
    const latestPass = latestSnap ? latestSnap.passRate : 0;
    const latestScore = latestSnap ? latestSnap.meanScore : 0;
    let peakPass = 0;
    let peakIter = 0;
    for (const p of points) {
      if (p.snap.passRate >= peakPass) {
        peakPass = p.snap.passRate;
        peakIter = p.iter;
      }
    }
    const last5 = points.slice(-5);
    const avg5 = last5.length > 0 ? last5.reduce((s, p) => s + p.snap.passRate, 0) / last5.length : 0;
    const prev5 = points.slice(-10, -5);
    const delta5 =
      prev5.length > 0 ? avg5 - prev5.reduce((s, p) => s + p.snap.passRate, 0) / prev5.length : 0;
    const deltaStr = `${delta5 >= 0 ? '+' : ''}${delta5.toFixed(1)}%`;
    const deltaCls = delta5 > 0.5 ? 'delta-up' : delta5 < -0.5 ? 'delta-down' : 'delta-flat';
    const countLabel = latestSnap ? `${latestSnap.passed}/${latestSnap.total} TASKS` : '0/0 TASKS';

    return `
      <div class="slab channel-slab">
        <div class="slab-head">
          <div class="head-left">
            <span class="color-lug" style="background:${color};"></span>
            <span class="mod-code">${meta.code}</span>
            <span class="mod-title">${escapeHtml(dom.toUpperCase())}</span>
          </div>
          <span class="head-meta">${countLabel}</span>
        </div>
        <div class="channel-readout">
          <div><span class="rk">PASS NOW</span><strong style="color:${color};">${latestPass.toFixed(1)}%</strong></div>
          <div><span class="rk">SCORE NOW</span><strong>${latestScore.toFixed(1)}</strong></div>
          <div><span class="rk">PEAK</span><strong>${peakPass.toFixed(1)}% <small>@I${peakIter}</small></strong></div>
          <div><span class="rk">5-IT AVG / &Delta;</span><strong class="${deltaCls}">${avg5.toFixed(1)}% (${deltaStr})</strong></div>
        </div>
        <div class="channel-canvas-wrap">
          <canvas id="catChart_${idx}"></canvas>
        </div>
      </div>`;
  }).join('');

  const tableRowsHtml =
    history.length === 0
      ? `<tr><td colspan="14" style="text-align:center;color:#52524b;padding:1.5rem;">LOOP 0 INITIALIZED — AWAITING ITERATION #1 TELEMETRY</td></tr>`
      : history
          .slice()
          .reverse()
          .map((h) => {
            const bd = h.domainBreakdown || {};
            const cellPct = (dom: string) => {
              const v = bd[dom]?.passRate;
              return v !== undefined ? `${v.toFixed(0)}%` : '-';
            };
            const phaseBanner =
              h.iteration === 41
                ? `<tr class="phase-divider-row"><td colspan="14">&#9658; PHASE B ENTRY // 211.4M MULTI-MODAL ACTION HEAD TOOK LIVE 20HZ MOTOR CONTROL (ITER #41+)</td></tr>`
                : '';
            return `${phaseBanner}
        <tr>
          <td class="mono bold">#${h.iteration}</td>
          <td class="mono bold">${h.meanScore}</td>
          <td class="mono"><span class="badge-cell ${h.passRate >= 50 ? 'badge-ok' : h.passRate >= 35 ? 'badge-warn' : 'badge-low'}">${h.passRate}%</span> <small>(${h.tasksPassed}/${h.totalTasks})</small></td>
          <td class="mono">${h.heldOutPassRate !== undefined ? `${h.heldOutPassRate}%` : '-'}</td>
          <td class="mono">${cellPct('Tactical Combat')}</td>
          <td class="mono">${cellPct('Crafting & Tech')}</td>
          <td class="mono">${cellPct('Agility & Parkour')}</td>
          <td class="mono">${cellPct('Building & 3D')}</td>
          <td class="mono">${cellPct('Survival & Affordances')}</td>
          <td class="mono">${cellPct('Puzzles & Tooling')}</td>
          <td class="mono">${cellPct('Multiplayer')}</td>
          <td class="mono ink-red">${h.slmFinalLoss !== undefined ? h.slmFinalLoss.toFixed(4) : '-'}</td>
          <td class="mono ink-blue">${h.actAimErrorDeg !== undefined ? h.actAimErrorDeg.toFixed(2) + '°' : '-'}</td>
          <td class="mono">${h.survivalCampaign ? `<span class="badge-cell badge-tier">${escapeHtml(h.survivalCampaign.techTierReached)} (${h.survivalCampaign.advancementsUnlocked.length})</span>` : escapeHtml(h.milestone || '-')}</td>
        </tr>`;
          })
          .join('');

  const safeHistoryJson = JSON.stringify(history).replace(/<\//g, '<\\/');
  const safeSampledJson = JSON.stringify(sampledDomainIters).replace(/<\//g, '<\\/');
  const safeBucketsJson = JSON.stringify(DOMAIN_BUCKETS).replace(/<\//g, '<\\/');
  const safeColorsJson = JSON.stringify(DOMAIN_COLORS).replace(/<\//g, '<\\/');
  const safeCodesJson = JSON.stringify(DOMAIN_CODES).replace(/<\//g, '<\\/');

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NEUROCRAFT // Telemetry Rig NC-V2</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --concrete: #dad7ce;
      --slab: #efede6;
      --recess: #e4e1d8;
      --grid-minor: rgba(27, 28, 25, 0.055);
      --grid-major: rgba(27, 28, 25, 0.12);
      --ink: #1b1c19;
      --ink-muted: #52524b;
      --steel: #252622;
      --signal-orange: #d9480f;
      --signal-red: #c92a2a;
      --signal-green: #15803d;
      --signal-blue: #1d4ed8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; border-radius: 0 !important; }
    body {
      background-color: var(--concrete);
      background-image:
        linear-gradient(to right, var(--grid-minor) 1px, transparent 1px),
        linear-gradient(to bottom, var(--grid-minor) 1px, transparent 1px),
        linear-gradient(to right, var(--grid-major) 1px, transparent 1px),
        linear-gradient(to bottom, var(--grid-major) 1px, transparent 1px);
      background-size: 24px 24px, 24px 24px, 120px 120px, 120px 120px;
      color: var(--ink);
      font-family: 'Cascadia Mono', 'JetBrains Mono', ui-monospace, 'SFMono-Regular', Consolas, monospace;
      padding: 20px;
      line-height: 1.35;
      font-variant-numeric: tabular-nums;
      position: relative;
    }
    .noise-overlay {
      position: fixed;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      opacity: 0.04;
      z-index: 0;
    }
    .rig-shell { position: relative; z-index: 1; max-width: 1720px; margin: 0 auto; }

    /* Title Plate */
    .title-plate {
      background: var(--slab);
      border: 2px solid var(--ink);
      box-shadow: 4px 4px 0 var(--ink);
      margin-bottom: 18px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: stretch;
      border-left: 10px solid var(--signal-orange);
    }
    .title-main { padding: 14px 18px; flex: 1 1 560px; }
    .title-kicker {
      font-size: 11px;
      letter-spacing: 0.12em;
      color: var(--ink-muted);
      text-transform: uppercase;
      margin-bottom: 4px;
    }
    .title-main h1 {
      font-size: 20px;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .title-sub {
      margin-top: 6px;
      font-size: 12px;
      color: var(--ink-muted);
    }
    .rig-controls {
      background: var(--recess);
      border-left: 2px solid var(--ink);
      padding: 12px 16px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 8px;
      min-width: 280px;
    }
    .ctrl-label { font-size: 10px; letter-spacing: 0.1em; color: var(--ink-muted); text-transform: uppercase; }
    .btn-bank { display: flex; gap: 4px; flex-wrap: wrap; }
    .mech-btn {
      appearance: none;
      border: 1.5px solid var(--ink);
      background: var(--slab);
      color: var(--ink);
      font-family: inherit;
      font-size: 11px;
      font-weight: 700;
      padding: 5px 10px;
      cursor: pointer;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      box-shadow: 2px 2px 0 var(--ink);
      transition: transform 0.05s ease, box-shadow 0.05s ease;
    }
    .mech-btn:active { transform: translate(1px, 1px); box-shadow: 1px 1px 0 var(--ink); }
    .mech-btn.active {
      background: var(--ink);
      color: var(--slab);
      border-bottom: 3px solid var(--signal-orange);
    }

    /* Section Headers */
    .sec-bar {
      background: var(--ink);
      color: var(--slab);
      padding: 7px 14px;
      margin: 20px 0 12px 0;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-left: 6px solid var(--signal-orange);
    }
    .sec-bar span.sec-note { color: #b8b5aa; font-size: 10.5px; font-weight: 500; }

    /* KPI Gauge Strip */
    .kpi-strip {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .gauge {
      background: var(--slab);
      border: 1.5px solid var(--ink);
      box-shadow: 3px 3px 0 var(--ink);
      padding: 12px 14px;
      border-top: 4px solid var(--ink);
    }
    .gauge.accent-green { border-top-color: var(--signal-green); }
    .gauge.accent-orange { border-top-color: var(--signal-orange); }
    .gauge.accent-blue { border-top-color: var(--signal-blue); }
    .gauge-k { font-size: 10.5px; color: var(--ink-muted); letter-spacing: 0.08em; text-transform: uppercase; }
    .gauge-v { font-size: 24px; font-weight: 800; margin: 4px 0 2px 0; }
    .gauge-s { font-size: 11px; color: var(--ink-muted); }

    /* Slabs / Chart Modules */
    .slab {
      background: var(--slab);
      border: 1.5px solid var(--ink);
      box-shadow: 3px 3px 0 var(--ink);
      display: flex;
      flex-direction: column;
    }
    .slab-head {
      background: var(--steel);
      color: var(--slab);
      padding: 7px 12px;
      font-size: 11.5px;
      font-weight: 700;
      letter-spacing: 0.07em;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      border-bottom: 1.5px solid var(--ink);
    }
    .head-left { display: flex; align-items: center; gap: 8px; min-width: 0; }
    .color-lug { width: 10px; height: 14px; display: inline-block; flex-shrink: 0; }
    .mod-code { color: #d9480f; font-weight: 800; }
    .mod-title { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .head-meta { font-size: 10.5px; color: #c8c4b8; white-space: nowrap; }
    .slab-toolbar {
      background: var(--recess);
      border-bottom: 1px solid #c2beb2;
      padding: 6px 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
      font-size: 10.5px;
      color: var(--ink-muted);
    }
    .slab-body { padding: 12px; height: 340px; position: relative; }
    .slab-body.tall { height: 370px; }

    .deck-2 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(540px, 1fr));
      gap: 16px;
      margin-bottom: 16px;
    }

    /* 7-Channel Per-Category Bank */
    .channel-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }
    .channel-readout {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      background: var(--recess);
      border-bottom: 1px solid #c2beb2;
      padding: 6px 10px;
      gap: 6px;
    }
    .channel-readout > div { display: flex; flex-direction: column; }
    .channel-readout .rk { font-size: 9px; color: var(--ink-muted); letter-spacing: 0.06em; }
    .channel-readout strong { font-size: 12px; }
    .channel-readout small { font-weight: 500; color: var(--ink-muted); font-size: 10px; }
    .delta-up { color: var(--signal-green); }
    .delta-down { color: var(--signal-red); }
    .delta-flat { color: var(--ink); }
    .channel-canvas-wrap { padding: 10px; height: 195px; position: relative; }

    /* Register Table */
    .table-wrap {
      background: var(--slab);
      border: 1.5px solid var(--ink);
      box-shadow: 3px 3px 0 var(--ink);
      overflow-x: auto;
      max-height: 520px;
      overflow-y: auto;
    }
    table { width: 100%; border-collapse: collapse; font-size: 11.5px; }
    th, td { padding: 6px 8px; text-align: left; border-bottom: 1px solid #cbc7bc; border-right: 1px solid #dcd8cd; white-space: nowrap; }
    th {
      background: var(--ink);
      color: var(--slab);
      position: sticky;
      top: 0;
      z-index: 2;
      font-size: 10.5px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    tbody tr:nth-child(even) { background: #e8e5dc; }
    tbody tr:hover { background: #dedacf; }
    tr.phase-divider-row td {
      background: #c92a2a;
      color: #efede6;
      font-weight: 800;
      letter-spacing: 0.07em;
      padding: 6px 10px;
    }
    .mono { font-variant-numeric: tabular-nums; }
    .bold { font-weight: 800; }
    .ink-red { color: var(--signal-red); font-weight: 700; }
    .ink-blue { color: var(--signal-blue); font-weight: 700; }
    .badge-cell { display: inline-block; padding: 1px 5px; font-weight: 700; border: 1px solid var(--ink); font-size: 10.5px; }
    .badge-ok { background: #dcfce7; color: #14532d; }
    .badge-warn { background: #ffedd5; color: #9a3412; }
    .badge-low { background: #fee2e2; color: #7f1d1d; }
    .badge-tier { background: var(--recess); color: var(--ink); }

    @media (max-width: 680px) {
      .deck-2 { grid-template-columns: 1fr; }
      .channel-grid { grid-template-columns: 1fr; }
      .channel-readout { grid-template-columns: repeat(2, 1fr); }
    }
  </style>
</head>
<body>
  <svg class="noise-overlay" aria-hidden="true">
    <filter id="concreteNoise">
      <feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="3" stitchTiles="stitch"/>
    </filter>
    <rect width="100%" height="100%" filter="url(#concreteNoise)"/>
  </svg>

  <div class="rig-shell">
    <header class="title-plate">
      <div class="title-main">
        <div class="title-kicker">NC-V2 // CAST CONCRETE TELEMETRY INSTRUMENT</div>
        <h1>NEUROCRAFT DUAL-BRAIN RL BENCHMARK RIG</h1>
        <p class="title-sub">HW: RTX-5060 (6.0GB CEILING) // ENV: MOJANG 1.20.4 REAL PHYSICS // S2: QWEN2.5-1.5B + S1: 211.4M MULTI-MODAL ACT</p>
      </div>
      <div class="rig-controls">
        <span class="ctrl-label">CURVE FILTER // GLOBAL LINE SMOOTHING</span>
        <div class="btn-bank">
          <button class="mech-btn active" id="btnSmoothRaw" type="button" onclick="setSmoothingMode(false)">RAW</button>
          <button class="mech-btn" id="btnSmoothSma" type="button" onclick="setSmoothingMode(true)">SMA-3</button>
        </div>
        <span class="ctrl-label" style="margin-top:2px;">PHASE MARKER: ITER 41 = LIVE NEURAL MOTOR HANDOFF</span>
      </div>
    </header>

    <section class="kpi-strip">
      <div class="gauge">
        <div class="gauge-k">01 // ITERATIONS LOGGED</div>
        <div class="gauge-v">#${state.currentIteration}</div>
        <div class="gauge-s">${state.totalEpisodesPlayed} EPISODES // SESSION #${state.sessionCount}</div>
      </div>
      <div class="gauge accent-green">
        <div class="gauge-k">02 // EXAM ALL-TIME PEAK</div>
        <div class="gauge-v">${bestScore} / 100</div>
        <div class="gauge-s">PASS PEAK: ${bestPass}% (ITER #${bestScoreIter})</div>
      </div>
      <div class="gauge accent-orange">
        <div class="gauge-k">03 // LATEST ITERATION</div>
        <div class="gauge-v">${latestRecord ? `${latestRecord.passRate}%` : '--'}</div>
        <div class="gauge-s">${latestRecord ? `MEAN ${latestRecord.meanScore}/100 // HELD ${latestRecord.heldOutPassRate ?? '--'}%` : 'AWAITING ITER #1'}</div>
      </div>
      <div class="gauge">
        <div class="gauge-k">04 // OPEN-WORLD &amp; DRAGON</div>
        <div class="gauge-v">${escapeHtml((latestSurvival?.techTierReached || 'READY').toUpperCase())}</div>
        <div class="gauge-s">${latestSurvival ? `${latestSurvival.advancementsUnlocked.length} ADV` : '0 ADV'} // DRAGON BEST: ${bestDragonScore.toFixed(0)}% (${bestDragonMilestones}/8)</div>
      </div>
      <div class="gauge accent-blue">
        <div class="gauge-k">05 // DUAL-BRAIN CONVERGENCE</div>
        <div class="gauge-v">${typeof latestRecord?.actAimErrorDeg === 'number' ? latestRecord.actAimErrorDeg.toFixed(2) + '°' : '--'}</div>
        <div class="gauge-s">ACT AIM ERR // SLM MIN LOSS: ${minLoss.toFixed(4)} (INIT ${initialLoss.toFixed(4)})</div>
      </div>
    </section>

    <div class="sec-bar">
      <span>SEC-01 // PRIMARY EXAM &amp; COMPOSITE CATEGORY OVERVIEW</span>
      <span class="sec-note">ORIGINAL PANEL 1 + ALL 7 CATEGORIES OVER TIME (t)</span>
    </div>

    <div class="deck-2">
      <!-- Original Panel 1 Preserved -->
      <div class="slab">
        <div class="slab-head">
          <div class="head-left">
            <span class="color-lug" style="background:#1b1c19;"></span>
            <span class="mod-code">MOD-01</span>
            <span class="mod-title">STANDARDIZED MINEBENCH EXAM OVER TIME (t)</span>
          </div>
          <span class="head-meta">84 TASKS + 20% HELD-OUT</span>
        </div>
        <div class="slab-toolbar">
          <span>MEAN SCORE (SOLID) / PASS RATE % (DASHED) / HELD-OUT (DOTTED)</span>
          <span>SCALE: 0–100</span>
        </div>
        <div class="slab-body tall">
          <canvas id="scoreChart"></canvas>
        </div>
      </div>

      <!-- Composite All-Category Training Curves -->
      <div class="slab">
        <div class="slab-head">
          <div class="head-left">
            <span class="color-lug" style="background:#d9480f;"></span>
            <span class="mod-code">MOD-02</span>
            <span class="mod-title">COMPOSITE CATEGORY TRAINING CURVES</span>
          </div>
          <span class="head-meta">7 DOMAINS // FULL HISTORY</span>
        </div>
        <div class="slab-toolbar">
          <div class="btn-bank">
            <button class="mech-btn active" id="btnCatPass" type="button" onclick="setCategoryMetric('passRate')">PASS %</button>
            <button class="mech-btn" id="btnCatMean" type="button" onclick="setCategoryMetric('meanScore')">MEAN SCORE</button>
          </div>
          <div class="btn-bank">
            <button class="mech-btn" type="button" onclick="setAllCategoriesVisible(true)">ALL ON</button>
            <button class="mech-btn" type="button" onclick="setAllCategoriesVisible(false)">CLEAR</button>
          </div>
        </div>
        <div class="slab-body tall">
          <canvas id="categoryCompositeChart"></canvas>
        </div>
      </div>
    </div>

    <div class="sec-bar">
      <span>SEC-02 // PER-CATEGORY TRAINING CURVES (CH-01 .. CH-07)</span>
      <span class="sec-note">SOLID = PASS RATE (%) // DASHED BLACK = MEAN SCORE (0–100)</span>
    </div>

    <div class="channel-grid">
      ${channelCardsHtml}
    </div>

    <div class="sec-bar">
      <span>SEC-03 // SURVIVAL CAMPAIGN, NEURAL CONVERGENCE &amp; INCREMENT BARS</span>
      <span class="sec-note">SURVIVAL TRACK + SYSTEM 1/2 TELEMETRY + SAMPLED BAR SNAPSHOT</span>
    </div>

    <div class="deck-2">
      <div class="slab">
        <div class="slab-head">
          <div class="head-left">
            <span class="color-lug" style="background:#15803d;"></span>
            <span class="mod-code">MOD-03</span>
            <span class="mod-title">OPEN-WORLD SURVIVAL &amp; CAMPAIGN PROGRESS</span>
          </div>
          <span class="head-meta">PERSISTENT WORLD</span>
        </div>
        <div class="slab-body">
          <canvas id="survivalChart"></canvas>
        </div>
      </div>

      <div class="slab">
        <div class="slab-head">
          <div class="head-left">
            <span class="color-lug" style="background:#1d4ed8;"></span>
            <span class="mod-code">MOD-04</span>
            <span class="mod-title">DUAL-BRAIN NEURAL CONVERGENCE</span>
          </div>
          <span class="head-meta">SLM LOSS (L) / ACT AIM ERR (R)</span>
        </div>
        <div class="slab-body">
          <canvas id="neuralChart"></canvas>
        </div>
      </div>
    </div>

    <div class="slab" style="margin-bottom: 18px;">
      <div class="slab-head">
        <div class="head-left">
          <span class="color-lug" style="background:#5b21b6;"></span>
          <span class="mod-code">MOD-05</span>
          <span class="mod-title">PER-DOMAIN PASS RATE (%) GROUPED BARS ACROSS INCREMENTS (t)</span>
        </div>
        <span class="head-meta">10 EVENLY SAMPLED CHECKPOINTS</span>
      </div>
      <div class="slab-body">
        <canvas id="domainBarChart"></canvas>
      </div>
    </div>

    <div class="sec-bar">
      <span>SEC-04 // ITERATION TELEMETRY REGISTER</span>
      <span class="sec-note">REVERSE CHRONOLOGICAL LOG // PASS RATES BY CATEGORY</span>
    </div>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>ITER</th>
            <th>MEAN</th>
            <th>PASS RATE</th>
            <th>HELD</th>
            <th>CMB%</th>
            <th>CRF%</th>
            <th>AGL%</th>
            <th>BLD%</th>
            <th>SRV%</th>
            <th>PZL%</th>
            <th>MP%</th>
            <th>SLM LOSS</th>
            <th>ACT AIM</th>
            <th>SURVIVAL</th>
          </tr>
        </thead>
        <tbody>
          ${tableRowsHtml}
        </tbody>
      </table>
    </div>
  </div>

  <script>
    const historyData = ${safeHistoryJson};
    const sampledDomains = ${safeSampledJson};
    const domainBuckets = ${safeBucketsJson};
    const domainColors = ${safeColorsJson};
    const domainCodes = ${safeCodesJson};

    Chart.defaults.font.family = "'Cascadia Mono', 'JetBrains Mono', ui-monospace, Consolas, monospace";
    Chart.defaults.font.size = 11;
    Chart.defaults.color = '#3a3c36';

    const hasPrePhaseB = historyData.some(h => h.iteration < 41);

    const phaseDividerPlugin = {
      id: 'phaseDivider',
      afterDraw(chart) {
        if (!hasPrePhaseB) return;
        const labels = chart.data && chart.data.labels;
        if (!labels) return;
        const idx = labels.indexOf('Iter 41');
        if (idx < 0) return;
        const xScale = chart.scales && chart.scales.x;
        const area = chart.chartArea;
        if (!xScale || !area) return;
        const xPos = xScale.getPixelForValue(idx);
        if (!Number.isFinite(xPos)) return;
        const ctx = chart.ctx;
        ctx.save();
        ctx.strokeStyle = '#c92a2a';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(xPos, area.top);
        ctx.lineTo(xPos, area.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#c92a2a';
        ctx.fillRect(xPos + 3, area.top + 3, 56, 14);
        ctx.fillStyle = '#efede6';
        ctx.font = 'bold 9px monospace';
        ctx.fillText('PHASE B', xPos + 7, area.top + 13);
        ctx.restore();
      }
    };
    Chart.register(phaseDividerPlugin);

    const labels = historyData.map(h => 'Iter ' + h.iteration);
    let useSma3 = false;
    let activeCatMetric = 'passRate';

    function smoothSeries(arr) {
      if (!useSma3) return arr;
      return arr.map((v, i) => {
        if (v === null || v === undefined) return null;
        const win = [];
        for (let j = Math.max(0, i - 2); j <= i; j++) {
          if (arr[j] !== null && arr[j] !== undefined && Number.isFinite(arr[j])) win.push(arr[j]);
        }
        return win.length > 0 ? Number((win.reduce((a, b) => a + b, 0) / win.length).toFixed(2)) : v;
      });
    }

    // Raw series builders
    const rawExamMean = historyData.map(h => h.meanScore);
    const rawExamPass = historyData.map(h => h.passRate);
    const rawExamHeld = historyData.map(h => h.heldOutMeanScore ?? null);
    const rawSlmLoss = historyData.map(h => h.slmFinalLoss ?? null);
    const rawActAim = historyData.map(h => h.actAimErrorDeg ?? null);

    // 1. MOD-01: Original Standardized Exam Over Time
    const scoreChart = new Chart(document.getElementById('scoreChart'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Exam Mean Score (0-100)',
            data: smoothSeries(rawExamMean),
            borderColor: '#1b1c19',
            backgroundColor: 'rgba(27, 28, 25, 0.08)',
            borderWidth: 2.6,
            pointRadius: 2,
            pointHoverRadius: 4,
            fill: true,
            tension: 0.15,
          },
          {
            label: 'Exam Pass Rate (%)',
            data: smoothSeries(rawExamPass),
            borderColor: '#15803d',
            borderWidth: 2.1,
            borderDash: [6, 4],
            pointRadius: 1.5,
            tension: 0.15,
          },
          {
            label: '20% Held-Out Eval Score',
            data: smoothSeries(rawExamHeld),
            borderColor: '#d9480f',
            borderWidth: 2.1,
            borderDash: [2, 3],
            pointRadius: 1.5,
            spanGaps: true,
            tension: 0.15,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { labels: { color: '#1b1c19', boxWidth: 14 } }
        },
        scales: {
          y: { min: 0, max: 100, grid: { color: '#cfccc0' }, ticks: { color: '#3a3c36' } },
          x: { grid: { color: '#e1ded3' }, ticks: { color: '#3a3c36', maxTicksLimit: 12 } }
        }
      }
    });

    // 2. MOD-02: Composite Category Training Curves
    function getCategorySeries(dom, metricKey) {
      return historyData.map(h => h.domainBreakdown?.[dom]?.[metricKey] ?? null);
    }

    const categoryCompositeChart = new Chart(document.getElementById('categoryCompositeChart'), {
      type: 'line',
      data: {
        labels,
        datasets: domainBuckets.map(dom => ({
          label: domainCodes[dom].code + ' ' + dom,
          data: smoothSeries(getCategorySeries(dom, activeCatMetric)),
          borderColor: domainColors[dom] || '#1b1c19',
          backgroundColor: domainColors[dom] || '#1b1c19',
          borderWidth: 2,
          pointRadius: 1.5,
          pointHoverRadius: 4,
          spanGaps: true,
          tension: 0.15,
        }))
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { labels: { color: '#1b1c19', boxWidth: 11, font: { size: 10 } } }
        },
        scales: {
          y: { min: 0, max: 100, grid: { color: '#cfccc0' }, ticks: { color: '#3a3c36' } },
          x: { grid: { color: '#e1ded3' }, ticks: { color: '#3a3c36', maxTicksLimit: 12 } }
        }
      }
    });

    // 3. SEC-02: 7 Per-Category Small-Multiples Line Charts
    const channelCharts = domainBuckets.map((dom, idx) => {
      const color = domainColors[dom] || '#1b1c19';
      const el = document.getElementById('catChart_' + idx);
      return new Chart(el, {
        type: 'line',
        data: {
          labels,
          datasets: [
            {
              label: 'Pass %',
              data: smoothSeries(getCategorySeries(dom, 'passRate')),
              borderColor: color,
              backgroundColor: color + '1a',
              borderWidth: 2.2,
              pointRadius: 0,
              pointHoverRadius: 3,
              fill: true,
              spanGaps: true,
              tension: 0.15,
            },
            {
              label: 'Mean Score',
              data: smoothSeries(getCategorySeries(dom, 'meanScore')),
              borderColor: 'rgba(27, 28, 25, 0.55)',
              borderWidth: 1.4,
              borderDash: [3, 3],
              pointRadius: 0,
              pointHoverRadius: 3,
              spanGaps: true,
              tension: 0.15,
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          plugins: {
            legend: { display: false }
          },
          scales: {
            y: { min: 0, max: 100, grid: { color: '#cfccc0' }, ticks: { stepSize: 50, color: '#52524b', font: { size: 9.5 } } },
            x: { grid: { display: false }, ticks: { color: '#52524b', maxTicksLimit: 6, font: { size: 9.5 } } }
          }
        }
      });
    });

    // 4. MOD-03: Open-World Survival & Campaign
    const tierScale = { Raw: 10, Wood: 25, Stone: 45, Iron: 65, Diamond: 80, Nether: 92, End: 100 };
    new Chart(document.getElementById('survivalChart'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Tech Tier Depth (0-100)',
            data: historyData.map(h => h.survivalCampaign ? (tierScale[h.survivalCampaign.techTierReached] ?? h.survivalCampaign.score) : (h.dragonSurvivalScore ?? null)),
            backgroundColor: 'rgba(21, 128, 61, 0.65)',
            borderColor: '#15803d',
            borderWidth: 1,
          },
          {
            label: 'Advancements (Count x 10)',
            data: historyData.map(h => h.survivalCampaign ? (h.survivalCampaign.advancementsUnlocked.length * 10) : null),
            type: 'line',
            borderColor: '#d9480f',
            backgroundColor: '#d9480f',
            borderWidth: 2.4,
            pointRadius: 2,
            spanGaps: true,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#1b1c19', boxWidth: 12 } }
        },
        scales: {
          y: { min: 0, max: 100, grid: { color: '#cfccc0' }, ticks: { color: '#3a3c36' } },
          x: { grid: { color: '#e1ded3' }, ticks: { color: '#3a3c36', maxTicksLimit: 12 } }
        }
      }
    });

    // 5. MOD-04: Dual-Brain Neural Convergence
    const neuralChart = new Chart(document.getElementById('neuralChart'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'System 2 SLM Loss',
            data: smoothSeries(rawSlmLoss),
            borderColor: '#c92a2a',
            backgroundColor: 'rgba(201, 42, 42, 0.08)',
            borderWidth: 2.2,
            pointRadius: 1.5,
            fill: true,
            yAxisID: 'yLoss',
            spanGaps: true,
            tension: 0.15,
          },
          {
            label: 'System 1 211.4M ACT Aim Err (deg)',
            data: smoothSeries(rawActAim),
            borderColor: '#1d4ed8',
            borderWidth: 2.4,
            pointRadius: 1.5,
            spanGaps: true,
            yAxisID: 'yAim',
            tension: 0.15,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { labels: { color: '#1b1c19', boxWidth: 12 } }
        },
        scales: {
          yLoss: { type: 'linear', position: 'left', min: 0, grid: { color: '#cfccc0' }, ticks: { color: '#c92a2a' } },
          yAim: { type: 'linear', position: 'right', min: 0, grid: { drawOnChartArea: false }, ticks: { color: '#1d4ed8' } },
          x: { grid: { color: '#e1ded3' }, ticks: { color: '#3a3c36', maxTicksLimit: 12 } }
        }
      }
    });

    // 6. MOD-05: Per-Domain Pass Rate (%) Grouped Bars across Increments (t)
    new Chart(document.getElementById('domainBarChart'), {
      type: 'bar',
      data: {
        labels: sampledDomains.map(h => 'Iter ' + h.iteration),
        datasets: domainBuckets.map(dom => ({
          label: domainCodes[dom].code + ' ' + dom,
          data: sampledDomains.map(h => h.domainBreakdown?.[dom]?.passRate ?? 0),
          backgroundColor: domainColors[dom] || '#52524b',
          borderColor: '#1b1c19',
          borderWidth: 1,
        }))
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#1b1c19', boxWidth: 11 } }
        },
        scales: {
          y: { min: 0, max: 100, grid: { color: '#cfccc0' }, ticks: { color: '#3a3c36' } },
          x: { grid: { color: '#e1ded3' }, ticks: { color: '#3a3c36' } }
        }
      }
    });

    // Interactive Controls
    function refreshLineCharts() {
      scoreChart.data.datasets[0].data = smoothSeries(rawExamMean);
      scoreChart.data.datasets[1].data = smoothSeries(rawExamPass);
      scoreChart.data.datasets[2].data = smoothSeries(rawExamHeld);
      scoreChart.update('none');

      domainBuckets.forEach((dom, i) => {
        categoryCompositeChart.data.datasets[i].data = smoothSeries(getCategorySeries(dom, activeCatMetric));
        channelCharts[i].data.datasets[0].data = smoothSeries(getCategorySeries(dom, 'passRate'));
        channelCharts[i].data.datasets[1].data = smoothSeries(getCategorySeries(dom, 'meanScore'));
        channelCharts[i].update('none');
      });
      categoryCompositeChart.update('none');

      neuralChart.data.datasets[0].data = smoothSeries(rawSlmLoss);
      neuralChart.data.datasets[1].data = smoothSeries(rawActAim);
      neuralChart.update('none');
    }

    window.setSmoothingMode = function(enableSma) {
      useSma3 = Boolean(enableSma);
      document.getElementById('btnSmoothRaw').classList.toggle('active', !useSma3);
      document.getElementById('btnSmoothSma').classList.toggle('active', useSma3);
      refreshLineCharts();
    };

    window.setCategoryMetric = function(metricKey) {
      activeCatMetric = metricKey === 'meanScore' ? 'meanScore' : 'passRate';
      document.getElementById('btnCatPass').classList.toggle('active', activeCatMetric === 'passRate');
      document.getElementById('btnCatMean').classList.toggle('active', activeCatMetric === 'meanScore');
      refreshLineCharts();
    };

    window.setAllCategoriesVisible = function(visible) {
      domainBuckets.forEach((_, i) => {
        categoryCompositeChart.setDatasetVisibility(i, Boolean(visible));
      });
      categoryCompositeChart.update('none');
    };
  </script>
</body>
</html>`;
}

if (require.main === module) {
  generateTrainingVisuals();
}
