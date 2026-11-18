import test from 'node:test';
import assert from 'node:assert/strict';

// Dynamically require compiled train_loop module to avoid rootDir TS6059 cross-boundary errors
const {
  parseConcurrencyArg,
  getWorkerSpatialOffset,
  getWorkerBotName,
  shiftCommandCoordinates,
  createShardedTaskSpec,
  ConcurrentTaskQueue,
  runTaskQueuePool,
  AsyncMutex,
} = require('../../minebench/dist/minebench/train_loop');

test('Multi-Bot Spatial Arena Sharding & Concurrency Suite', async (t) => {
  await t.test('1. Concurrency CLI argument parsing, aliases, clamping, and env fallback', () => {
    // Default value when no args
    assert.equal(parseConcurrencyArg([]), 3, 'Default concurrency should be 3');

    // --concurrency flag
    assert.equal(parseConcurrencyArg(['--concurrency', '2']), 2);
    assert.equal(parseConcurrencyArg(['--concurrency', '4']), 4);
    assert.equal(parseConcurrencyArg(['--concurrency=5']), 5);

    // --bots alias
    assert.equal(parseConcurrencyArg(['--bots', '1']), 1);
    assert.equal(parseConcurrencyArg(['--bots=4']), 4);

    // Clamping [1, 5]
    assert.equal(parseConcurrencyArg(['--concurrency', '0']), 1, 'Should clamp minimum to 1');
    assert.equal(parseConcurrencyArg(['--concurrency', '-5']), 1, 'Should clamp negative to 1');
    assert.equal(parseConcurrencyArg(['--concurrency', '6']), 5, 'Should clamp maximum to 5');
    assert.equal(parseConcurrencyArg(['--concurrency', '100']), 5, 'Should clamp extreme to 5');

    // Case-insensitivity and decimal clamping
    assert.equal(parseConcurrencyArg(['--Concurrency', '4']), 4, 'Should be case-insensitive');
    assert.equal(parseConcurrencyArg(['--BOTS', '2']), 2, 'Should be case-insensitive');
    assert.equal(parseConcurrencyArg(['--concurrency', '2.8']), 2, 'Should truncate decimal');

    // Trailing flag or followed by non-numeric flag
    assert.equal(parseConcurrencyArg(['--concurrency']), 3, 'Trailing flag should fall back to default');
    assert.equal(parseConcurrencyArg(['--concurrency', '--eval-only']), 3, 'Flag followed by flag should fall back to default');

    // Environment variable fallback
    assert.equal(parseConcurrencyArg([], '4'), 4, 'Should read from env if CLI arg omitted');
    assert.equal(parseConcurrencyArg([], '  4  '), 4, 'Should handle whitespace in env');
    assert.equal(parseConcurrencyArg(['--concurrency', '2'], '4'), 2, 'CLI arg should take precedence over env');
    assert.equal(parseConcurrencyArg([], '10'), 5, 'Env var should also be clamped to 5');
    assert.equal(parseConcurrencyArg([], 'invalid'), 3, 'Invalid env var should fallback to 3');

    // Invalid CLI args fallback
    assert.equal(parseConcurrencyArg(['--concurrency', 'abc']), 3, 'Invalid CLI arg should fallback to default 3');
  });

  await t.test('2. Spatial offset calculation (ΔX = i * 4000)', () => {
    assert.equal(getWorkerSpatialOffset(0), 0);
    assert.equal(getWorkerSpatialOffset(1), 4000);
    assert.equal(getWorkerSpatialOffset(2), 8000);
    assert.equal(getWorkerSpatialOffset(3), 12000);
    assert.equal(getWorkerSpatialOffset(4), 16000);
  });

  await t.test('3. Worker bot username resolution', () => {
    // When concurrency <= 1, backward-compatible single bot name
    assert.equal(getWorkerBotName(0, 1), 'NeuroAgent');
    assert.equal(getWorkerBotName(0, 0), 'NeuroAgent');
    assert.equal(getWorkerBotName(0, -1), 'NeuroAgent');

    // When concurrency > 1, 1-indexed worker bot names
    assert.equal(getWorkerBotName(0, 3), 'NeuroAgent_1');
    assert.equal(getWorkerBotName(1, 3), 'NeuroAgent_2');
    assert.equal(getWorkerBotName(2, 3), 'NeuroAgent_3');
    assert.equal(getWorkerBotName(4, 5), 'NeuroAgent_5');
  });

  await t.test('4. Command coordinate shifting for /fill, /setblock, /tp, /summon, /clone', () => {
    const deltaX = 4000;

    // Delta = 0 returns command unchanged
    const cmd1 = 'fill 100 80 100 128 88 106 minecraft:air';
    assert.equal(shiftCommandCoordinates(cmd1, 0), cmd1);

    // /fill
    assert.equal(
      shiftCommandCoordinates('fill 100 80 100 128 88 106 minecraft:air', deltaX),
      'fill 4100 80 100 4128 88 106 minecraft:air'
    );
    assert.equal(
      shiftCommandCoordinates('/fill -10 60 -10 10 70 10 minecraft:stone', deltaX),
      '/fill 3990 60 -10 4010 70 10 minecraft:stone'
    );

    // /setblock
    assert.equal(
      shiftCommandCoordinates('setblock 104 79 100 minecraft:stone_bricks', deltaX),
      'setblock 4104 79 100 minecraft:stone_bricks'
    );

    // /tp with target
    assert.equal(
      shiftCommandCoordinates('tp NeuroAgent 100 80 100 -90 0', deltaX),
      'tp NeuroAgent 4100 80 100 -90 0'
    );

    // /tp without target
    assert.equal(
      shiftCommandCoordinates('tp 100 80 100 0 0', deltaX),
      'tp 4100 80 100 0 0'
    );

    // /teleport alias (with and without target)
    assert.equal(
      shiftCommandCoordinates('teleport NeuroAgent 100 80 100 -90 0', deltaX),
      'teleport NeuroAgent 4100 80 100 -90 0'
    );
    assert.equal(
      shiftCommandCoordinates('/teleport 100 80 100', deltaX),
      '/teleport 4100 80 100'
    );

    // /spawnpoint (with and without target)
    assert.equal(
      shiftCommandCoordinates('spawnpoint NeuroAgent 100 80 100', deltaX),
      'spawnpoint NeuroAgent 4100 80 100'
    );
    assert.equal(
      shiftCommandCoordinates('/spawnpoint 100 80 100', deltaX),
      '/spawnpoint 4100 80 100'
    );

    // /setworldspawn
    assert.equal(
      shiftCommandCoordinates('setworldspawn 100 80 100', deltaX),
      'setworldspawn 4100 80 100'
    );

    // /summon
    assert.equal(
      shiftCommandCoordinates('summon creeper 105 80 100 {Fuse:30}', deltaX),
      'summon creeper 4105 80 100 {Fuse:30}'
    );

    // /clone
    assert.equal(
      shiftCommandCoordinates('clone 10 20 30 15 25 35 100 200 300', deltaX),
      'clone 4010 20 30 4015 25 35 4100 200 300'
    );

    // Selector coordinates x=...
    assert.equal(
      shiftCommandCoordinates('kill @e[type=creeper,x=100,y=80,z=100,distance=..64]', deltaX),
      'kill @e[type=creeper,x=4100,y=80,z=100,distance=..64]'
    );

    // Embedded selector in /tp command
    assert.equal(
      shiftCommandCoordinates('tp @e[type=zombie,x=100,y=64,z=100] 200 64 200', deltaX),
      'tp @e[type=zombie,x=4100,y=64,z=100] 4200 64 200'
    );
  });

  await t.test('5. Task specification spatial sharding and bot renaming', () => {
    const baseTask = {
      id: 'test_parkour',
      title: 'Parkour Test Course',
      domain: 'AGILITY_PARKOUR',
      tier: 1,
      maxDurationSeconds: 30,
      maxSteps: 3,
      prompt: 'Reach the goal block!',
      origin: { x: 100, y: 80, z: 100 },
      initialLoadout: [],
      buildArenaCommands: (orig: { x: number; y: number; z: number }) => [
        `fill ${orig.x - 2} ${orig.y - 1} ${orig.z - 2} ${orig.x + 2} ${orig.y - 1} ${orig.z + 2} minecraft:stone`,
        `tp NeuroAgent ${orig.x} ${orig.y} ${orig.z} 0 0`,
        `tp @p ${orig.x} ${orig.y} ${orig.z}`,
      ],
    };

    // Concurrency = 1 (worker 0) preserves original task identically
    const singleTask = createShardedTaskSpec(baseTask, 0, 1);
    assert.equal(singleTask.origin.x, 100);
    assert.equal(singleTask, baseTask, 'Single bot execution must return base task directly');

    // Worker 1 of 3 (offset +4000, bot username NeuroAgent_2)
    const worker1Task = createShardedTaskSpec(baseTask, 1, 3);
    assert.equal(worker1Task.origin.x, 4100, 'Origin X must be shifted by +4000');
    assert.equal(worker1Task.origin.y, 80);
    assert.equal(worker1Task.origin.z, 100);

    const cmds = worker1Task.buildArenaCommands(worker1Task.origin);
    assert.equal(cmds[0], 'fill 4098 79 98 4102 79 102 minecraft:stone');
    assert.equal(cmds[1], 'tp NeuroAgent_2 4100 80 100 0 0', 'NeuroAgent must be replaced with NeuroAgent_2');
    assert.equal(cmds[2], 'tp NeuroAgent_2 4100 80 100', '@p must be replaced with NeuroAgent_2');

    // Calling buildArenaCommands with NO arguments defaults cleanly to sharded origin without throwing
    const cmdsNoArgs = worker1Task.buildArenaCommands();
    assert.equal(cmdsNoArgs[0], 'fill 4098 79 98 4102 79 102 minecraft:stone');
    assert.equal(cmdsNoArgs[1], 'tp NeuroAgent_2 4100 80 100 0 0');

    // Calling buildArenaCommands with unshifted base origin shifts commands by deltaX
    const cmdsBaseArg = worker1Task.buildArenaCommands(baseTask.origin);
    assert.equal(cmdsBaseArg[0], 'fill 4098 79 98 4102 79 102 minecraft:stone');

    // Sharding a task with static/hardcoded commands shifts coordinates by deltaX
    const staticTask = {
      id: 'static_task',
      title: 'Static Course',
      domain: 'AGILITY_PARKOUR',
      tier: 1,
      maxDurationSeconds: 30,
      maxSteps: 3,
      prompt: 'Jump!',
      origin: { x: 100, y: 80, z: 100 },
      initialLoadout: [],
      buildArenaCommands: () => [
        'fill 98 79 98 102 79 102 minecraft:stone',
        'tp NeuroAgent 100 80 100 0 0',
      ],
    };
    const shardedStatic = createShardedTaskSpec(staticTask, 1, 3);
    const staticCmds = shardedStatic.buildArenaCommands(shardedStatic.origin);
    assert.equal(staticCmds[0], 'fill 4098 79 98 4102 79 102 minecraft:stone', 'Static fill command must be shifted by +4000');
    assert.equal(staticCmds[1], 'tp NeuroAgent_2 4100 80 100 0 0', 'Static tp command must be shifted by +4000');

    // Worker 2 of 3 (offset +8000, bot username NeuroAgent_3)
    const worker2Task = createShardedTaskSpec(baseTask, 2, 3);
    assert.equal(worker2Task.origin.x, 8100, 'Origin X must be shifted by +8000');
    const cmds2 = worker2Task.buildArenaCommands(worker2Task.origin);
    assert.equal(cmds2[1], 'tp NeuroAgent_3 8100 80 100 0 0');
  });

  await t.test('6. Concurrent Task Queue & Worker Pool execution distribution', async () => {
    const tasks = Array.from({ length: 15 }, (_, i) => ({ id: `task_${i}` }));
    const processedTasks: Array<{ taskId: string; workerId: number; startTime: number; endTime: number }> = [];

    let currentActiveWorkers = 0;
    let maxSimultaneousWorkers = 0;

    const concurrency = 3;
    await runTaskQueuePool(tasks, concurrency, async (task: any, index: number, workerId: number) => {
      currentActiveWorkers++;
      maxSimultaneousWorkers = Math.max(maxSimultaneousWorkers, currentActiveWorkers);

      const t0 = Date.now();
      // Simulate variable task duration (10-30ms)
      await new Promise((r) => setTimeout(r, 10 + (index % 3) * 10));
      const t1 = Date.now();

      processedTasks.push({ taskId: task.id, workerId, startTime: t0, endTime: t1 });
      currentActiveWorkers--;
      return { success: true };
    });

    // 1. All tasks must be executed
    assert.equal(processedTasks.length, 15, 'All 15 tasks must be completed');

    // 2. Each task must be executed exactly once
    const taskIds = processedTasks.map((p) => p.taskId).sort();
    const expectedIds = tasks.map((t) => t.id).sort();
    assert.deepEqual(taskIds, expectedIds, 'Every task must be executed without omission or duplication');

    // 3. Concurrency limit enforced
    assert.ok(maxSimultaneousWorkers <= concurrency, `Active workers (${maxSimultaneousWorkers}) must not exceed limit (${concurrency})`);
    assert.ok(maxSimultaneousWorkers >= 2, 'Should achieve concurrent execution with multiple workers');

    // 4. All workers participated
    const uniqueWorkers = new Set(processedTasks.map((p) => p.workerId));
    assert.equal(uniqueWorkers.size, concurrency, 'All configured workers must participate in task execution');

    // 5. Empty queue handling
    const emptyResults = await runTaskQueuePool([], 3, async () => 'not_called');
    assert.deepEqual(emptyResults, []);
  });

  await t.test('7. AsyncMutex exclusivity under concurrent contention', async () => {
    const mutex = new AsyncMutex();
    let counter = 0;
    const contentionCount = 20;

    const promises = Array.from({ length: contentionCount }, async () => {
      await mutex.runExclusive(async () => {
        const local = counter;
        await new Promise((r) => setTimeout(r, 2));
        counter = local + 1;
      });
    });

    await Promise.all(promises);
    assert.equal(counter, contentionCount, 'AsyncMutex must prevent race conditions across parallel callers');
  });

  await t.test('8. Atomic episode transition buffer updates under multi-worker concurrency', async () => {
    const stateWriteMutex = new AsyncMutex();
    const sharedTransitions: Array<{ workerId: number; stepIdx: number; reward: number }> = [];
    const sharedScores: number[] = [];

    // Simulate 3 workers each completing 4 episodes with multiple step transitions
    const workersCount = 3;
    const episodesPerWorker = 4;

    const workerTasks = Array.from({ length: workersCount }, async (_, workerId) => {
      for (let ep = 0; ep < episodesPerWorker; ep++) {
        // Collect local episode transitions during the episode
        const episodeTransitions = [
          { workerId, stepIdx: 0, reward: 10 + ep },
          { workerId, stepIdx: 1, reward: 20 + ep },
        ];
        const episodeScore = 50 + workerId * 10 + ep;

        // Atomically flush at episode completion
        await stateWriteMutex.runExclusive(async () => {
          for (const tr of episodeTransitions) {
            sharedTransitions.push(tr);
          }
          sharedScores.push(episodeScore);
        });
      }
    });

    await Promise.all(workerTasks);

    assert.equal(
      sharedScores.length,
      workersCount * episodesPerWorker,
      'All episodes must be recorded atomically'
    );
    assert.equal(
      sharedTransitions.length,
      workersCount * episodesPerWorker * 2,
      'All transitions must be recorded without data loss'
    );
  });
});
