const fs = require('fs');

const lines = fs.readFileSync('minebench/results/rl_replay_buffer.jsonl', 'utf8')
  .trim()
  .split('\n')
  .filter(Boolean)
  .map(l => JSON.parse(l));

const failed = lines.filter(l => !l.passed);

failed.forEach((ep, i) => {
  console.log(`\n--- [${i+1}] ${ep.taskId} (Score: ${ep.score}) ---`);
  console.log(`Prompt: "${ep.prompt}"`);
  if (ep.executedTools) {
    ep.executedTools.forEach((t, idx) => {
      console.log(`  Tool ${idx+1}: ${t.name}(${JSON.stringify(t.arguments || t.args)}) -> success: ${t.success}`);
    });
  }
  if (ep.steps) {
    ep.steps.forEach((s, idx) => {
      console.log(`  Step ${idx+1}: ${JSON.stringify(s).substring(0, 200)}`);
    });
  }
});
