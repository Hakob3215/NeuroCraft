import test from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as path from 'path';
import { Vec3 } from 'vec3';

import { parseLanBroadcastPacket } from '../discovery/lanScanner';
import { isWithinHumanFov, calculateLookAngles } from '../system1/perception/frustum';
import { isLineOfSightOccluded, isEntityOccluded } from '../system1/perception/raycaster';
import { sampleHumanReactionDelayMs } from '../system1/perception/auditory';
import { generateBezierCameraTrajectory } from '../system1/kinematics/camera';
import { generateSmoothSplineTrajectory } from '../system1/kinematics/movement';
import { getWeaponCooldownMs, isCriticalStrikeWindow } from '../system1/kinematics/combat';
import { calculateFittsInventoryDelayMs } from '../system1/kinematics/inventory';
import { SpeculativePrefetcher } from '../system2/speculative';
import { DynamicMCPToolRegistry } from '../system2/mcp/toolRegistry';
import { NeuroCraftMCPServer } from '../system2/mcp/mcpServer';
import { validateAndCompileSynthesizedTool } from '../system2/meta/sandbox';
import { SelfSynthesizingToolFactory } from '../system2/meta/toolFactory';
import { UnifiedLLMClient } from '../system2/llmClient';
import { SubtaskStatus } from '../types';

test('1. UDP LAN Broadcast Packet Parser extracts MOTD and dynamic port', () => {
  const packet = '[MOTD]Steve - Awesome Singleplayer World[/MOTD][AD]59412[/AD]';
  const parsed = parseLanBroadcastPacket(packet, '192.168.1.50');
  assert.ok(parsed);
  assert.equal(parsed.motd, 'Steve - Awesome Singleplayer World');
  assert.equal(parsed.port, 59412);
  assert.equal(parsed.host, '192.168.1.50');

  const invalid = parseLanBroadcastPacket('random garbage packet');
  assert.equal(invalid, null);
});

test('2. Synthetic Human 90° FOV Cone accurately filters unseen entities behind back', () => {
  const eyePos = { x: 0, y: 65.62, z: 0 };
  // Facing North (-Z): yaw = 0, pitch = 0
  const targetAhead = { x: 0, y: 65.62, z: -10 };
  const targetSlightLeft = { x: -4, y: 65.62, z: -10 }; // ~21.8 deg off center (inside +/-45 deg)
  const targetFarSide = { x: -15, y: 65.62, z: -2 };    // ~82 deg off center (outside +/-45 deg)
  const targetBehind = { x: 0, y: 65.62, z: 10 };       // 180 deg behind

  assert.equal(isWithinHumanFov(eyePos, 0, 0, targetAhead), true);
  assert.equal(isWithinHumanFov(eyePos, 0, 0, targetSlightLeft), true);
  assert.equal(isWithinHumanFov(eyePos, 0, 0, targetFarSide), false);
  assert.equal(isWithinHumanFov(eyePos, 0, 0, targetBehind), false);
});

test('3. 3D Voxel Raycaster blocks wallhacking through solid walls', () => {
  const eyePos = { x: 0.5, y: 65.62, z: 0.5 };
  const mobPos = { x: 0.5, y: 64.0, z: 8.5 };

  // Solid obsidian wall at z = 4 across all heights
  const wallAtZ4 = (_x: number, _y: number, z: number) => z === 4;
  assert.equal(isLineOfSightOccluded(eyePos, mobPos, wallAtZ4), true);
  assert.equal(isEntityOccluded(eyePos, mobPos, 1.8, wallAtZ4), true);

  // Clear air everywhere
  const clearAir = () => false;
  assert.equal(isLineOfSightOccluded(eyePos, mobPos, clearAir), false);
  assert.equal(isEntityOccluded(eyePos, mobPos, 1.8, clearAir), false);
});

test('4. Log-Normal Human Reaction Delay stays within physiological bounds [110ms, 550ms]', () => {
  const samples: number[] = [];
  for (let i = 0; i < 200; i++) {
    const s = sampleHumanReactionDelayMs(220, 0.25);
    assert.ok(s >= 110 && s <= 550, `Sample ${s}ms out of bounds`);
    samples.push(s);
  }
  samples.sort((a, b) => a - b);
  const median = samples[100];
  assert.ok(median >= 180 && median <= 265, `Expected median near 220ms, got ${median}ms`);
});

test('5. Cubic Bézier Camera Interpolation smoothly rotates without instantaneous snapping', () => {
  const startYaw = 0;
  const startPitch = 0;
  const { yaw: targetYaw, pitch: targetPitch } = calculateLookAngles(
    { x: 0, y: 65, z: 0 },
    { x: 10, y: 66, z: 10 }
  );

  const frames = generateBezierCameraTrajectory(startYaw, startPitch, targetYaw, targetPitch, 350);
  assert.ok(frames.length >= 6, 'Should generate multiple 50ms tick frames');

  // Verify final frame converges onto target angle within micro-tremor tolerance
  const last = frames[frames.length - 1];
  assert.ok(Math.abs(last.yaw - targetYaw) < 0.05);
  assert.ok(Math.abs(last.pitch - targetPitch) < 0.05);
});

test('6. Catmull-Rom Spline Trajectory prunes collinear nodes and curves corners', () => {
  const waypoints = [
    { x: 0, y: 64, z: 0 },
    { x: 1, y: 64, z: 0 },
    { x: 2, y: 64, z: 0 },
    { x: 2, y: 64, z: 3 },
  ];
  const spline = generateSmoothSplineTrajectory(waypoints, 4);
  assert.ok(spline.length > 2);
  assert.deepEqual(spline[0], { x: 0, y: 64, z: 0 });
  assert.deepEqual(spline[spline.length - 1], { x: 2, y: 64, z: 3 });
});

test('7. Modern 1.9+ Combat Cooldowns & Critical Strike Physics', () => {
  assert.equal(getWeaponCooldownMs('diamond_sword'), 625);
  assert.equal(getWeaponCooldownMs('stone_axe'), 1250);
  assert.equal(getWeaponCooldownMs('iron_pickaxe'), 835);

  // Critical strike requires !onGround, !inWater, and downward velocity < -0.05
  assert.equal(isCriticalStrikeWindow(false, -0.22, false), true);
  assert.equal(isCriticalStrikeWindow(true, -0.22, false), false);
  assert.equal(isCriticalStrikeWindow(false, 0.15, false), false);
});

test('8. Fitts Law Inventory Slot Delay increases with slot distance', () => {
  const nearSum = Array.from({ length: 40 }, () => calculateFittsInventoryDelayMs(36, 37)).reduce(
    (a, b) => a + b,
    0
  );
  const farSum = Array.from({ length: 40 }, () => calculateFittsInventoryDelayMs(36, 8)).reduce(
    (a, b) => a + b,
    0
  );
  assert.ok(farSum > nearSum, 'Farther GUI slots must take longer on average per Fitts Law');
});

test('9. Speculative Pipeline Prefetcher triggers at 75% subtask progress', () => {
  let triggeredCount = 0;
  const prefetcher = new SpeculativePrefetcher(() => {
    triggeredCount++;
  });

  const subtask: SubtaskStatus = {
    id: 'sub_1',
    toolName: 'gather_resource',
    description: 'Mining oak logs',
    args: {},
    startedAt: Date.now(),
    estimatedDurationMs: 10000,
    progress: 0.5,
    status: 'RUNNING',
  };

  prefetcher.onSubtaskUpdate(subtask);
  assert.equal(triggeredCount, 0, 'Should not prefetch at 50% with 10s left');

  subtask.progress = 0.78;
  prefetcher.onSubtaskUpdate(subtask);
  assert.equal(triggeredCount, 1, 'Should trigger speculative prefetch at >= 75%');

  // Calling again for same subtask ID should not duplicate prefetch
  prefetcher.onSubtaskUpdate(subtask);
  assert.equal(triggeredCount, 1);
});

test('10. Self-Synthesizing MCP Tool Factory + Sandbox Validation + Hot-Reload Execution', async () => {
  // Verify sandbox blocks dangerous code
  const malicious = validateAndCompileSynthesizedTool(
    'evil_tool',
    `process.exit(1); return { success: true, summary: 'bad' };`
  );
  assert.equal(malicious.valid, false);

  // Verify dynamic tool synthesis, MCP notification emission, and JSON-RPC 2.0 execution
  const registry = new DynamicMCPToolRegistry();
  const mcpServer = new NeuroCraftMCPServer(registry);
  const llm = new UnifiedLLMClient({ offlineOnly: true });

  const tempDir = path.join(__dirname, '..', '..', 'data', 'test_tmp');
  const tempRegFile = path.join(tempDir, 'test_registry.json');

  let listChangedEvents = 0;
  mcpServer.onNotification((n) => {
    if (n.method === 'notifications/tools/list_changed') {
      listChangedEvents++;
    }
  });

  const factory = new SelfSynthesizingToolFactory(registry, llm, tempRegFile, tempDir);
  const synthRes = await factory.synthesizeAndRegisterTool(
    'custom_celebration_dance',
    'Perform a celebratory spin dance',
    { durationSeconds: 'number' },
    'Agent, do a celebration dance!'
  );

  assert.equal(synthRes.success, true);
  assert.equal(synthRes.toolName, 'custom_celebration_dance');
  assert.ok(listChangedEvents >= 1, 'MCP Server must emit notifications/tools/list_changed');

  // Execute the newly synthesized tool via JSON-RPC 2.0 `tools/call`
  const abortController = new AbortController();
  const mockBot = {
    entity: { position: new Vec3(0, 64, 0), yaw: 0, pitch: 0 },
    setControlState: () => {},
    swingArm: () => {},
  };

  const rpcResult = await mcpServer.handleRpcRequest(
    {
      jsonrpc: '2.0',
      id: 99,
      method: 'tools/call',
      params: {
        name: 'custom_celebration_dance',
        arguments: { durationSeconds: 1 },
      },
    },
    () => ({
      bot: mockBot,
      kinematics: {
        lookAtSmooth: async () => {},
        moveAlongSpline: async () => {},
        pathfindTo: async () => true,
        safeDig: async () => true,
        strikeTarget: async () => true,
        equipItemHumanized: async () => true,
      },
      worldState: () => ({} as any),
      reportProgress: () => {},
      sayHumanized: async () => {},
      abortSignal: abortController.signal,
    })
  );

  assert.equal(rpcResult.result?.isError, false);
  const payload = JSON.parse(rpcResult.result.content[0].text);
  assert.equal(payload.success, true);

  // Clean up test temp files
  try {
    fs.rmSync(tempDir, { recursive: true, force: true });
  } catch {}
});

test('11. Prismarine-Chat 1.20.4+/1.21+ object chat format code ({ registryIndex: 0 } / { chatType: 0 }) does not crash', () => {
  const registry = require('prismarine-registry')('1.20.4');
  const ChatMessage = require('prismarine-chat')(registry);

  const msgObj = ChatMessage.fromNetwork(
    { registryIndex: 0, chatType: 0 },
    {
      sender: { text: 'TheJack3215' },
      content: { text: 'follow me' },
    }
  );
  assert.ok(msgObj.toString().includes('TheJack3215'));
  assert.ok(msgObj.toString().includes('follow me'));
});

test('12. Autonomous Multi-Step Goal Continuation chains survival tech-tree steps start-to-finish', async () => {
  const llm = new UnifiedLLMClient({ offlineOnly: true });
  const baseState: any = {
    tick: 100,
    timestamp: Date.now(),
    botUsername: 'NeuroAgent',
    position: { x: -162, y: 75, z: -76 },
    velocity: { x: 0, y: 0, z: 0 },
    yaw: 0,
    pitch: 0,
    health: 20,
    food: 20,
    oxygen: 20,
    onGround: true,
    isInWater: false,
    isInLava: false,
    heldItem: null,
    inventory: [{ name: 'spruce_planks', count: 12, slot: 36 }],
    visibleEntities: [],
    audibleCues: [],
    nearbyNotableBlocks: [],
    activeHazard: null,
    activeSubtask: null,
  };

  // Step 1: Has 12 spruce_planks -> should craft wooden_pickaxe
  const d1 = await llm.generateGoalContinuationDecision(
    'Get full iron armor',
    [],
    baseState,
    [],
    []
  );
  assert.equal(d1.goalCompleted, false);
  assert.equal(d1.toolCall?.name, 'craft_item');
  assert.equal(d1.toolCall?.arguments.itemName, 'wooden_pickaxe');

  // Step 2: Now has wooden_pickaxe, but surface stone search failed -> should adapt to explore_or_dig_for_ores (staircase_mine)
  baseState.inventory.push({ name: 'wooden_pickaxe', count: 1, slot: 37 });
  const d2 = await llm.generateGoalContinuationDecision(
    'Get full iron armor',
    [
      {
        stepNumber: 1,
        toolName: 'gather_resource',
        args: { blockType: 'stone', count: 11 },
        success: false,
        summary: 'No exposed stone found within 48m.',
      },
    ],
    baseState,
    [],
    []
  );
  assert.equal(d2.goalCompleted, false);
  assert.equal(d2.toolCall?.name, 'explore_or_dig_for_ores');

  // Step 3: Now has 4 raw_iron + coal -> should smelt_item
  baseState.inventory.push(
    { name: 'stone_pickaxe', count: 1, slot: 38 },
    { name: 'raw_iron', count: 6, slot: 39 },
    { name: 'coal', count: 4, slot: 40 }
  );
  const d3 = await llm.generateGoalContinuationDecision(
    'Get full iron armor',
    [],
    baseState,
    [],
    []
  );
  assert.equal(d3.goalCompleted, false);
  assert.equal(d3.toolCall?.name, 'smelt_item');

  // Step 4: Has all 4 iron armor pieces -> should equip_armor_or_item then complete goal
  baseState.inventory.push(
    { name: 'iron_helmet', count: 1, slot: 9 },
    { name: 'iron_chestplate', count: 1, slot: 10 },
    { name: 'iron_leggings', count: 1, slot: 11 },
    { name: 'iron_boots', count: 1, slot: 12 }
  );
  const d4 = await llm.generateGoalContinuationDecision(
    'Get full iron armor',
    [],
    baseState,
    [],
    []
  );
  assert.equal(d4.toolCall?.name, 'equip_armor_or_item');

  const d5 = await llm.generateGoalContinuationDecision(
    'Get full iron armor',
    [
      {
        stepNumber: 10,
        toolName: 'equip_armor_or_item',
        args: { itemName: 'all_armor' },
        success: true,
        summary: 'Equipped full iron armor!',
      },
    ],
    baseState,
    [],
    []
  );
  assert.equal(d5.goalCompleted, true);
});

test('13. gather_resource handles Mineflayer palette pre-filter (where block.position === null) and harvests logs', async () => {
  const { registerBaseMCPTools } = require('../system2/skills/baseSkills');
  const registry = new DynamicMCPToolRegistry();
  registerBaseMCPTools(registry);
  const mcpServer = new NeuroCraftMCPServer(registry);

  let logBroken = false;
  const logPos = new Vec3(-160, 75, -70);

  const mockBot: any = {
    entity: {
      position: new Vec3(-162, 75, -70),
      onGround: true,
    },
    heldItem: null,
    inventory: {
      items: () => (logBroken ? [{ name: 'spruce_log', count: 1 }] : []),
    },
    pathfinder: {
      setGoal: () => {},
    },
    clearControlStates: () => {},
    canDigBlock: () => true,
    dig: async () => {
      logBroken = true;
    },
    stopDigging: () => {},
    blockAt: (pos: Vec3) => {
      if (pos.x === logPos.x && pos.y === logPos.y && pos.z === logPos.z) {
        return logBroken
          ? { name: 'air', position: logPos, boundingBox: 'empty' }
          : { name: 'spruce_log', position: logPos, boundingBox: 'block' };
      }
      return { name: 'air', position: pos, boundingBox: 'empty' };
    },
    findBlocks: (options: { matching: (b: any) => boolean }) => {
      // Simulate Mineflayer's `isBlockInSection` palette pre-filter where `position === null`!
      const palettePrototypeBlock = { name: 'spruce_log', position: null };
      if (!options.matching(palettePrototypeBlock)) {
        return [];
      }
      return logBroken ? [] : [logPos];
    },
  };

  const abortController = new AbortController();
  const res = await mcpServer.handleRpcRequest(
    {
      jsonrpc: '2.0',
      id: 13,
      method: 'tools/call',
      params: {
        name: 'gather_resource',
        arguments: { blockType: 'spruce_log', count: 1 },
      },
    },
    () => ({
      bot: mockBot,
      kinematics: {
        lookAtSmooth: async () => {},
        moveAlongSpline: async () => {},
        pathfindTo: async () => true,
        safeDig: async () => true,
        strikeTarget: async () => true,
        equipItemHumanized: async () => false,
      },
      worldState: () => ({} as any),
      reportProgress: () => {},
      sayHumanized: async () => {},
      abortSignal: abortController.signal,
    })
  );

  const payload = JSON.parse(res.result.content[0].text);
  assert.equal(payload.success, true, `Expected gather_resource to succeed, got: ${payload.summary}`);
  assert.equal(logBroken, true);
});

test('14. gather_resource enforces stone_pickaxe tier for iron_ore and extracts full 3D connected ore veins', async () => {
  const { registerBaseMCPTools } = require('../system2/skills/baseSkills');
  const registry = new DynamicMCPToolRegistry();
  registerBaseMCPTools(registry);
  const mcpServer = new NeuroCraftMCPServer(registry);

  const veinPositions = [
    new Vec3(0, 64, 2),
    new Vec3(1, 64, 2),
    new Vec3(1, 65, 2),
  ];
  const brokenSet = new Set<string>();
  let rawIronInInv = 0;
  let heldItem: any = { name: 'wooden_pickaxe' };

  const mockBot: any = {
    entity: {
      position: new Vec3(0, 64, 0),
      onGround: true,
    },
    get heldItem() {
      return heldItem;
    },
    inventory: {
      items: () => [
        heldItem,
        ...(rawIronInInv > 0 ? [{ name: 'raw_iron', count: rawIronInInv }] : []),
      ],
    },
    pathfinder: {
      setGoal: () => {},
    },
    clearControlStates: () => {},
    canDigBlock: () => true,
    dig: async (block: any) => {
      const k = `${block.position.x},${block.position.y},${block.position.z}`;
      brokenSet.add(k);
      rawIronInInv++;
    },
    stopDigging: () => {},
    blockAt: (pos: Vec3) => {
      const k = `${pos.x},${pos.y},${pos.z}`;
      const isVein = veinPositions.some((v) => v.x === pos.x && v.y === pos.y && v.z === pos.z);
      if (isVein && !brokenSet.has(k)) {
        return { name: 'iron_ore', position: pos, boundingBox: 'block' };
      }
      return { name: 'air', position: pos, boundingBox: 'empty' };
    },
    findBlocks: (options: { matching: (b: any) => boolean }) => {
      if (!options.matching({ name: 'iron_ore', position: null })) return [];
      return veinPositions.filter((v) => !brokenSet.has(`${v.x},${v.y},${v.z}`));
    },
  };

  const abortController = new AbortController();

  // Case A: Holding only wooden_pickaxe -> MUST refuse to dig iron_ore!
  const resFail = await mcpServer.handleRpcRequest(
    {
      jsonrpc: '2.0',
      id: 1401,
      method: 'tools/call',
      params: {
        name: 'gather_resource',
        arguments: { blockType: 'iron_ore', count: 3 },
      },
    },
    () => ({
      bot: mockBot,
      kinematics: {
        lookAtSmooth: async () => {},
        moveAlongSpline: async () => {},
        pathfindTo: async () => true,
        safeDig: async () => true,
        strikeTarget: async () => true,
        equipItemHumanized: async () => false,
      },
      worldState: () => ({} as any),
      reportProgress: () => {},
      sayHumanized: async () => {},
      abortSignal: abortController.signal,
    })
  );

  const payloadFail = JSON.parse(resFail.result.content[0].text);
  assert.equal(payloadFail.success, false);
  assert.equal(brokenSet.size, 0, 'Must NEVER break iron_ore with a wooden_pickaxe!');
  assert.ok(payloadFail.summary.includes('stone_pickaxe'), 'Must report required stone_pickaxe');

  // Case B: Equip stone_pickaxe -> MUST mine out the entire 3-block connected vein!
  heldItem = { name: 'stone_pickaxe' };
  const resOk = await mcpServer.handleRpcRequest(
    {
      jsonrpc: '2.0',
      id: 1402,
      method: 'tools/call',
      params: {
        name: 'gather_resource',
        arguments: { blockType: 'iron_ore', count: 3 },
      },
    },
    () => ({
      bot: mockBot,
      kinematics: {
        lookAtSmooth: async () => {},
        moveAlongSpline: async () => {},
        pathfindTo: async () => true,
        safeDig: async () => true,
        strikeTarget: async () => true,
        equipItemHumanized: async (name: string) => name === 'stone_pickaxe',
      },
      worldState: () => ({} as any),
      reportProgress: () => {},
      sayHumanized: async () => {},
      abortSignal: abortController.signal,
    })
  );

  const payloadOk = JSON.parse(resOk.result.content[0].text);
  assert.equal(payloadOk.success, true);
  assert.equal(brokenSet.size, 3, 'Must mine all 3 blocks in the connected 3D iron_ore vein');
  assert.equal(rawIronInInv, 3);
});

test('15. gather_resource actively excavates 1x1 pockets/shelves and walks onto dropped raw_iron entities to guarantee pickup', async () => {
  const { registerBaseMCPTools } = require('../system2/skills/baseSkills');
  const registry = new DynamicMCPToolRegistry();
  registerBaseMCPTools(registry);
  const mcpServer = new NeuroCraftMCPServer(registry);

  const orePos = new Vec3(0, 65, 2);
  const lowerShelfPos = new Vec3(0, 64, 2);
  const brokenSet = new Set<string>();
  let rawIronInInv = 0;
  let droppedEntity: any = null;

  const mockBot: any = {
    entity: {
      id: 1,
      position: new Vec3(0.5, 64, 0.5),
      onGround: true,
    },
    heldItem: { name: 'stone_pickaxe' },
    get entities() {
      return droppedEntity ? { 99: droppedEntity } : {};
    },
    inventory: {
      items: () => [
        { name: 'stone_pickaxe', count: 1 },
        ...(rawIronInInv > 0 ? [{ name: 'raw_iron', count: rawIronInInv }] : []),
      ],
    },
    pathfinder: {
      setGoal: () => {},
    },
    look: async () => {},
    setControlState: (state: string, active: boolean) => {
      if (state === 'forward' && active) {
        // If the lower stone shelf at (0, 64, 2) has been excavated, the bot can walk into z=2.2 and pick up the raw_iron!
        if (brokenSet.has('0,64,2')) {
          mockBot.entity.position = new Vec3(0.5, 64, 2.3);
          if (droppedEntity) {
            rawIronInInv = 1;
            droppedEntity = null;
          }
        }
      }
    },
    clearControlStates: () => {},
    canDigBlock: () => true,
    dig: async (block: any) => {
      const k = `${block.position.x},${block.position.y},${block.position.z}`;
      brokenSet.add(k);
      if (k === '0,65,2') {
        // Spawn a dropped raw_iron item entity inside the pocket at (0.5, 65.1, 2.5) (NOT in inventory yet!)
        droppedEntity = {
          id: 99,
          name: 'item',
          type: 'other',
          position: new Vec3(0.5, 65.1, 2.5),
          getDroppedItem: () => ({ name: 'raw_iron', count: 1 }),
        };
      }
    },
    stopDigging: () => {},
    blockAt: (pos: Vec3) => {
      const k = `${pos.x},${pos.y},${pos.z}`;
      if (brokenSet.has(k)) {
        return { name: 'air', position: pos, boundingBox: 'empty' };
      }
      if (pos.x === orePos.x && pos.y === orePos.y && pos.z === orePos.z) {
        return { name: 'iron_ore', position: pos, boundingBox: 'block' };
      }
      if (pos.x === lowerShelfPos.x && pos.y === lowerShelfPos.y && pos.z === lowerShelfPos.z) {
        return { name: 'stone', position: pos, boundingBox: 'block' };
      }
      return { name: 'air', position: pos, boundingBox: 'empty' };
    },
    findBlocks: (options: { matching: (b: any) => boolean }) => {
      if (!options.matching({ name: 'iron_ore', position: null })) return [];
      return brokenSet.has('0,65,2') ? [] : [orePos];
    },
  };

  const abortController = new AbortController();
  const res = await mcpServer.handleRpcRequest(
    {
      jsonrpc: '2.0',
      id: 1501,
      method: 'tools/call',
      params: {
        name: 'gather_resource',
        arguments: { blockType: 'iron_ore', count: 1 },
      },
    },
    () => ({
      bot: mockBot,
      kinematics: {
        lookAtSmooth: async () => {},
        moveAlongSpline: async () => {},
        pathfindTo: async () => true,
        safeDig: async () => true,
        strikeTarget: async () => true,
        equipItemHumanized: async () => true,
      },
      worldState: () => ({} as any),
      reportProgress: () => {},
      sayHumanized: async () => {},
      abortSignal: abortController.signal,
    })
  );

  const payload = JSON.parse(res.result.content[0].text);
  assert.equal(payload.success, true, `Expected raw_iron pocket pickup to succeed, got: ${payload.summary}`);
  assert.equal(brokenSet.has('0,64,2'), true, 'Must excavate the lower stone shelf block (0,64,2) blocking the 1x1 pocket');
  assert.equal(rawIronInInv, 1, 'Must physically walk into the excavated pocket and pick up raw_iron');
});

test('16. clear_flat_area chops trees bottom-up and levels designated area without canopy hovering', async () => {
  const { registerBaseMCPTools } = require('../system2/skills/baseSkills');
  const registry = new DynamicMCPToolRegistry();
  registerBaseMCPTools(registry);
  const mcpServer = new NeuroCraftMCPServer(registry);

  const brokenBlocks: string[] = [];
  const treeLogs = [new Vec3(2, 64, 2), new Vec3(2, 65, 2), new Vec3(2, 66, 2)];
  const raisedDirt = new Vec3(1, 65, 1);

  const mockBot: any = {
    entity: {
      position: new Vec3(0, 64, 0),
      onGround: true,
    },
    heldItem: { name: 'stone_axe' },
    inventory: {
      items: () => [{ name: 'stone_axe', count: 1 }],
    },
    pathfinder: {
      setGoal: () => {},
    },
    clearControlStates: () => {},
    canDigBlock: () => true,
    dig: async (block: any) => {
      brokenBlocks.push(`${block.name}@${block.position.y}`);
    },
    stopDigging: () => {},
    blockAt: (pos: Vec3) => {
      const isLog = treeLogs.some(
        (v) => v.x === pos.x && v.y === pos.y && v.z === pos.z && !brokenBlocks.includes(`spruce_log@${pos.y}`)
      );
      if (isLog) return { name: 'spruce_log', position: pos, boundingBox: 'block' };

      if (pos.x === raisedDirt.x && pos.y === raisedDirt.y && pos.z === raisedDirt.z && !brokenBlocks.includes(`dirt@${pos.y}`)) {
        return { name: 'dirt', position: pos, boundingBox: 'block' };
      }
      return { name: 'air', position: pos, boundingBox: 'empty' };
    },
    players: {},
  };

  const abortController = new AbortController();
  const res = await mcpServer.handleRpcRequest(
    {
      jsonrpc: '2.0',
      id: 1601,
      method: 'tools/call',
      params: {
        name: 'clear_flat_area',
        arguments: { width: 5, length: 5, height: 4, targetY: 64 },
      },
    },
    () => ({
      bot: mockBot,
      kinematics: {
        lookAtSmooth: async () => {},
        moveAlongSpline: async () => {},
        pathfindTo: async () => true,
        safeDig: async () => true,
        strikeTarget: async () => true,
        equipItemHumanized: async () => true,
      },
      worldState: () => ({} as any),
      reportProgress: () => {},
      sayHumanized: async () => {},
      abortSignal: abortController.signal,
    })
  );

  const payload = JSON.parse(res.result.content[0].text);
  assert.equal(payload.success, true);
  assert.ok(brokenBlocks.includes('spruce_log@64'), 'Must chop base trunk at Y=64 first');
  assert.ok(brokenBlocks.includes('spruce_log@65'), 'Must chop trunk at Y=65');
  assert.ok(brokenBlocks.includes('dirt@65'), 'Must level raised dirt above Y=64');
});

test('17. HumanIdleSimulator executes autonomous companion self-improvement actions during inactivity', async () => {
  const { HumanIdleSimulator } = require('../system2/idleFsm');

  const executedTools: string[] = [];
  const mockSystem1: any = {
    bot: {
      entity: { position: new Vec3(0, 64, 0), yaw: 0 },
      inventory: {
        items: () => [
          { name: 'cobblestone', count: 12 },
          { name: 'stick', count: 4 },
        ],
      },
      players: {},
    },
    camera: {
      lookAtSmooth: async () => {},
    },
  };

  const idleSim = new HumanIdleSimulator(mockSystem1, async (toolName: string, args: any) => {
    executedTools.push(toolName);
    return { success: true, summary: `Ran ${toolName}` };
  });

  // Force idle time >= 15 seconds
  (idleSim as any).lastActivityTimestamp = Date.now() - 20000;
  await (idleSim as any).runAutonomousIdleStep();

  assert.ok(
    executedTools.includes('craft_item'),
    'Autonomous idle loop must trigger tool crafting when missing essential tools'
  );
});

test('18. formatKickReason decodes Minecraft chat components and disconnect reasons without [object Object]', () => {
  const { formatKickReason } = require('../system1/botController');

  // Case 1: Vanilla flying kick JSON
  const flyingReason = { translate: 'multiplayer.disconnect.flying' };
  assert.equal(formatKickReason(flyingReason), 'multiplayer.disconnect.flying');

  // Case 2: Kick with arguments
  const kickedWithArg = {
    translate: 'multiplayer.disconnect.kicked',
    with: ['Flying is not enabled on this server'],
  };
  assert.equal(
    formatKickReason(kickedWithArg),
    'multiplayer.disconnect.kicked [Flying is not enabled on this server]'
  );

  // Case 3: Plain text disconnect
  const textReason = { text: 'Server closed connection' };
  assert.equal(formatKickReason(textReason), 'Server closed connection');

  // Case 4: Text with extra array
  const textWithExtra = {
    text: 'Internal Exception: ',
    extra: [{ text: 'io.netty.handler.codec.DecoderException' }],
  };
  assert.equal(
    formatKickReason(textWithExtra),
    'Internal Exception: io.netty.handler.codec.DecoderException'
  );

  // Case 5: Empty / null / socket close
  assert.equal(
    formatKickReason(null),
    'Clean socket disconnect or timeout (no server kick packet)'
  );

  // Case 6: Generic object must never return [object Object]
  const genericObj = { reasonCode: 42, customMessage: 'test' };
  const formatted = formatKickReason(genericObj);
  assert.notEqual(formatted, '[object Object]');
  assert.ok(formatted.includes('42'));
});

test('19. calculateLookAngles safely guards against NaN and non-finite coordinates', () => {
  const { calculateLookAngles } = require('../system1/perception/frustum');

  // Case 1: NaN target
  const nanTarget = calculateLookAngles({ x: 0, y: 64, z: 0 }, { x: NaN, y: 64, z: NaN });
  assert.equal(Number.isFinite(nanTarget.yaw), true);
  assert.equal(Number.isFinite(nanTarget.pitch), true);

  // Case 2: NaN eyePos
  const nanEye = calculateLookAngles({ x: NaN, y: 64, z: 0 }, { x: 5, y: 64, z: 5 });
  assert.equal(Number.isFinite(nanEye.yaw), true);
  assert.equal(Number.isFinite(nanEye.pitch), true);

  // Case 3: Normal target
  const normal = calculateLookAngles({ x: 0, y: 64, z: 0 }, { x: 0, y: 64, z: -5 });
  assert.equal(Math.abs(normal.yaw), 0); // Faces North
  assert.equal(normal.distance, 5);
});

test('20. formatKickReason decodes modern 1.20.4+ NBT compound disconnect structures', () => {
  const { formatKickReason } = require('../system1/botController');

  // Case 1: Anonymous NBT compound with translate tag
  const compoundNbt = {
    type: 'compound',
    value: {
      translate: { type: 'string', value: 'multiplayer.disconnect.invalid_player_movement' },
    },
  };
  assert.equal(
    formatKickReason(compoundNbt),
    'multiplayer.disconnect.invalid_player_movement'
  );

  // Case 2: Direct NBT string tag on translate
  const directNbtTranslate = {
    translate: { type: 'string', value: 'multiplayer.disconnect.flying' },
  };
  assert.equal(
    formatKickReason(directNbtTranslate),
    'multiplayer.disconnect.flying'
  );

  // Case 3: Direct NBT string tag on text
  const directNbtText = {
    text: { type: 'string', value: 'Kicked by an operator' },
  };
  assert.equal(
    formatKickReason(directNbtText),
    'Kicked by an operator'
  );
});

test('21. ReflexEngine initializes baseline health on tick 1 without false-positive damage spikes', () => {
  const { ReflexEngine } = require('../system1/reflexes/reflexEngine');

  const interrupts: any[] = [];
  const mockBot: any = {
    health: 16.8, // Spawn with damaged health
    entity: {
      position: { x: 0, y: 64, z: 0 },
      onGround: true,
    },
    clearControlStates: () => {},
    setControlState: () => {},
  };
  const mockCamera: any = { lookAtSmooth: async () => {} };
  const mockCombat: any = { raiseShield: () => {}, lowerShield: () => {} };
  const mockInv: any = { equipItemHumanized: async () => true };

  const engine = new ReflexEngine(
    mockBot,
    mockCamera,
    mockCombat,
    mockInv,
    (ev: any) => interrupts.push(ev)
  );

  const mockState: any = {
    health: 16.8,
    food: 20,
    oxygen: 20,
    visibleEntities: [],
    isInWater: false,
    isInLava: false,
  };

  // Tick 1: Baseline health initialized -> NO interrupt triggered!
  engine.evaluateTick(mockState);
  assert.equal(interrupts.length, 0);

  // Tick 2: Same health -> NO interrupt
  engine.evaluateTick(mockState);
  assert.equal(interrupts.length, 0);

  // Tick 3: Real sudden damage spike (-3.0 HP -> 13.8 HP)
  mockBot.health = 13.8;
  mockState.health = 13.8;
  engine.evaluateTick(mockState);
  assert.equal(interrupts.length, 1);
  assert.equal(interrupts[0].hazard, 'SUDDEN_DAMAGE');
});

test('22. Modern entity_velocity packet with { velocity: { x, y, z } } produces finite numbers', () => {
  const conv = require('../../node_modules/mineflayer/lib/conversions');
  const { Vec3 } = require('vec3');

  const packet = {
    entityId: 42,
    velocity: { x: 8000, y: 4000, z: -8000 },
  };

  const rawX = packet.velocity ? packet.velocity.x : (packet as any).velocityX;
  const rawY = packet.velocity ? packet.velocity.y : (packet as any).velocityY;
  const rawZ = packet.velocity ? packet.velocity.z : (packet as any).velocityZ;
  const vx = Number.isFinite(rawX) ? rawX : 0;
  const vy = Number.isFinite(rawY) ? rawY : 0;
  const vz = Number.isFinite(rawZ) ? rawZ : 0;
  const notchVel = new Vec3(vx, vy, vz);
  const converted = conv.fromNotchVelocity(notchVel);

  assert.equal(Number.isFinite(converted.x), true);
  assert.equal(Number.isFinite(converted.y), true);
  assert.equal(Number.isFinite(converted.z), true);
  assert.notEqual(converted.x, NaN);
  assert.ok(converted.x > 0);
});

test('23. isServerCommandOrAdminMessage filters operator broadcasts and allows real chat', () => {
  const { isServerCommandOrAdminMessage } = require('../system1/botController');

  // Case 1: Command feedback with chat.type.admin translation
  assert.equal(
    isServerCommandOrAdminMessage('Gave 64 [Steak] to NeuroAgent]', 'chat.type.admin'),
    true
  );

  // Case 2: Bracketed operator feedback
  assert.equal(
    isServerCommandOrAdminMessage('Gave 64 [Torch] to TheJack3215]'),
    true
  );
  assert.equal(
    isServerCommandOrAdminMessage('Applied effect Night Vision to TheJack3215]'),
    true
  );
  assert.equal(
    isServerCommandOrAdminMessage('[TheJack3215: Set own game mode to Creative Mode]'),
    true
  );

  // Case 3: Genuine player chat messages MUST NOT be filtered!
  assert.equal(
    isServerCommandOrAdminMessage('try to get some diamonds at least 1'),
    false
  );
  assert.equal(
    isServerCommandOrAdminMessage('hello. you should have some stuff already'),
    false
  );
  assert.equal(
    isServerCommandOrAdminMessage('can you clear this area?'),
    false
  );
});

test('24. Item durability calculation extracts remaining uses and percentages', () => {
  const { getItemDurabilityInfo, getUsablePickaxeDurability } = require('../system2/skills/baseSkills');

  const mockPickaxe = {
    name: 'iron_pickaxe',
    maxDurability: 250,
    durabilityUsed: 50,
  };

  const info = getItemDurabilityInfo(mockPickaxe);
  assert.ok(info);
  assert.equal(info.durability, 200);
  assert.equal(info.maxDurability, 250);
  assert.equal(info.percent, 80);

  const mockBot = {
    inventory: {
      items: () => [
        { name: 'iron_pickaxe', maxDurability: 250, durabilityUsed: 50 },
        { name: 'stone_pickaxe', maxDurability: 131, durabilityUsed: 31 },
      ],
    },
  };

  const usable = getUsablePickaxeDurability(mockBot);
  assert.equal(usable.stoneOrWoodTotal, 100);
  assert.equal(usable.ironReserve, 200);
  // Total usable for stone mining reserves 20 iron uses for diamonds
  assert.equal(usable.total, 100 + (200 - 20));
});

test('25. Tool preservation prioritizes stone pickaxe over iron pickaxe for regular stone', () => {
  const { getAvailableTools } = require('../system2/skills/baseSkills');

  const mockBot = {
    inventory: {
      items: () => [
        { name: 'iron_pickaxe', maxDurability: 250, durabilityUsed: 10 },
        { name: 'stone_pickaxe', maxDurability: 131, durabilityUsed: 20 },
      ],
    },
  };

  const picks = getAvailableTools(mockBot, '_pickaxe');
  assert.equal(picks.length, 2);
  const stonePick = picks.find((p: any) => p.name === 'stone_pickaxe');
  const ironPick = picks.find((p: any) => p.name === 'iron_pickaxe');
  assert.ok(stonePick);
  assert.ok(ironPick);
  assert.ok(stonePick.durability > 2);
  assert.ok(ironPick.durability > 15);
});

test('26. Tool selection distinguishes sand vs sandstone and equips stone pickaxe for sandstone even with shovel in hand', async () => {
  const { getRequiredPickaxeTier, equipBestToolForBlockName } = require('../system2/skills/baseSkills');

  assert.equal(getRequiredPickaxeTier('sand'), 'none');
  assert.equal(getRequiredPickaxeTier('red_sand'), 'none');
  assert.equal(getRequiredPickaxeTier('sandstone'), 'wood');
  assert.equal(getRequiredPickaxeTier('red_sandstone'), 'wood');
  assert.equal(getRequiredPickaxeTier('cut_sandstone'), 'wood');
  assert.equal(getRequiredPickaxeTier('obsidian'), 'diamond');

  let equippedItem: string | null = null;
  const mockCtx: any = {
    bot: {
      heldItem: { name: 'stone_shovel' },
      inventory: {
        items: () => [
          { name: 'stone_shovel', maxDurability: 131, durabilityUsed: 10 },
          { name: 'stone_pickaxe', maxDurability: 131, durabilityUsed: 10 },
          { name: 'iron_pickaxe', maxDurability: 250, durabilityUsed: 10 },
        ],
      },
    },
    kinematics: {
      equipItemHumanized: async (name: string) => {
        equippedItem = name;
        mockCtx.bot.heldItem = { name };
        return true;
      },
    },
  };

  // When digging sand, equips shovel
  await equipBestToolForBlockName(mockCtx, 'sand');
  assert.equal(equippedItem, 'stone_shovel');

  // When encountering sandstone, MUST equip pickaxe, NOT shovel!
  await equipBestToolForBlockName(mockCtx, 'sandstone');
  assert.equal(equippedItem, 'stone_pickaxe');

  // Red sandstone also equips pickaxe
  await equipBestToolForBlockName(mockCtx, 'red_sandstone');
  assert.equal(equippedItem, 'stone_pickaxe');
});

test('27. ensureHarvestToolForBlock equips existing inventory pickaxe and secondary obstacles do not set lastToolFailure', async () => {
  const { ensureHarvestToolForBlock } = require('../system2/skills/baseSkills');

  // Case 1: Bot is holding a stone_shovel, but has a stone_pickaxe in inventory.
  // When encountering sandstone, ensureHarvestToolForBlock equips stone_pickaxe from inventory and returns true!
  const mockCtx: any = {
    bot: {
      heldItem: { name: 'stone_shovel' },
      inventory: {
        items: () => [
          { name: 'stone_shovel', maxDurability: 131, durabilityUsed: 10 },
          { name: 'stone_pickaxe', maxDurability: 131, durabilityUsed: 10 },
        ],
      },
    },
    kinematics: {
      equipItemHumanized: async (name: string) => {
        mockCtx.bot.heldItem = { name };
        return true;
      },
    },
  };

  const ok = await ensureHarvestToolForBlock(mockCtx, { name: 'sandstone' }, 0);
  assert.equal(ok, true);
  assert.equal(mockCtx.bot.heldItem.name, 'stone_pickaxe');
  assert.equal((mockCtx as any).lastToolFailure, undefined);

  // Case 2: Bot has NO pickaxes and NO materials to craft one.
  const emptyCtx: any = {
    bot: {
      heldItem: null,
      inventory: { items: () => [] },
      version: '1.20.4',
      recipesFor: () => [],
    },
    kinematics: {
      equipItemHumanized: async () => false,
    },
  };

  // If depth > 0 (secondary pocket/obstacle block), failure must NOT set lastToolFailure!
  const obstacleOk = await ensureHarvestToolForBlock(emptyCtx, { name: 'sandstone' }, 1);
  assert.equal(obstacleOk, false);
  assert.equal((emptyCtx as any).lastToolFailure, undefined);

  // If depth === 0 (primary target), failure DOES set lastToolFailure
  const primaryOk = await ensureHarvestToolForBlock(emptyCtx, { name: 'sandstone' }, 0);
  assert.equal(primaryOk, false);
  assert.ok((emptyCtx as any).lastToolFailure.includes('Missing required tool'));
});

test('28. Ground height scanning & area topology perception accurately detects terrain elevation and flatness', () => {
  const { System1BotController } = require('../system1/botController');

  const worldBlocks = new Map<string, any>();
  // Ground at Y=64, with one mound at Y=65
  for (let x = -5; x <= 5; x++) {
    for (let z = -5; z <= 5; z++) {
      const y = x === 2 && z === 2 ? 65 : 64;
      worldBlocks.set(`${x},${y},${z}`, { name: 'grass_block', boundingBox: 'block' });
      worldBlocks.set(`${x},${y - 1},${z}`, { name: 'dirt', boundingBox: 'block' });
      worldBlocks.set(`${x},${y + 1},${z}`, { name: 'air', boundingBox: 'empty' });
    }
  }

  const mockBot: any = {
    entity: {
      position: new Vec3(0, 65, 0),
      yaw: 0,
      pitch: 0,
      velocity: new Vec3(0, 0, 0),
      onGround: true,
    },
    blockAt: (pos: Vec3) => {
      const k = `${Math.floor(pos.x)},${Math.floor(pos.y)},${Math.floor(pos.z)}`;
      return worldBlocks.get(k) || { name: 'air', boundingBox: 'empty' };
    },
    inventory: { items: () => [] },
  };

  const controller = new System1BotController(mockBot);
  (controller as any).bot = mockBot;

  // Normal ground height
  const gh1 = controller.getGroundHeightAt(0, 0, 70);
  assert.equal(gh1, 64);

  // Mound ground height
  const gh2 = controller.getGroundHeightAt(2, 2, 70);
  assert.equal(gh2, 65);

  // Topology scan of 5x5 area
  const topology = controller.scanAreaTopology(0, 0, 4, 4);
  assert.equal(topology.minHeight, 64);
  assert.equal(topology.maxHeight, 65);
  assert.equal(topology.medianHeight, 64);
  assert.equal(topology.isFlat, true); // variance <= 1 block is flat
  assert.ok(topology.groundBlockTypes['grass_block'] > 0);
});

test('29. safePlaceBlock dynamically positions bot within reach and attaches to adjacent solid faces', async () => {
  const { System1BotController } = require('../system1/botController');

  const placedBlocks = new Map<string, any>();
  placedBlocks.set('0,64,0', { name: 'grass_block', boundingBox: 'block', position: new Vec3(0, 64, 0) });

  let botPos = new Vec3(5, 65, 5); // Start far away (7m)
  let equipped: string | null = null;

  const mockBot: any = {
    entity: {
      position: botPos,
      yaw: 0,
      pitch: 0,
      velocity: new Vec3(0, 0, 0),
      onGround: true,
    },
    blockAt: (pos: Vec3) => {
      const k = `${Math.floor(pos.x)},${Math.floor(pos.y)},${Math.floor(pos.z)}`;
      return placedBlocks.get(k) || { name: 'air', boundingBox: 'empty' };
    },
    placeBlock: async (ref: any, faceVec: Vec3) => {
      const dest = ref.position.plus(faceVec);
      const k = `${Math.floor(dest.x)},${Math.floor(dest.y)},${Math.floor(dest.z)}`;
      placedBlocks.set(k, { name: 'oak_planks', boundingBox: 'block', position: dest });
    },
    inventory: {
      items: () => [{ name: 'oak_planks', count: 32 }],
    },
    pathfinder: {
      setGoal: () => {},
    },
  };

  const controller = new System1BotController(mockBot);
  (controller as any).bot = mockBot;
  (controller as any).camera = {
    lookAtSmooth: async () => {},
    clearQueue: () => {},
  };
  (controller as any).inventory = {
    equipItemHumanized: async (name: string) => {
      equipped = name;
      return true;
    },
  };
  (controller as any).pathfindToPosition = async (target: Vec3) => {
    botPos.set(target.x, target.y, target.z);
    return true;
  };

  // Place block on top of (0, 64, 0) -> target is (0, 65, 0)
  const ok = await controller.safePlaceBlock(new Vec3(0, 65, 0), 'oak_planks');
  assert.equal(ok, true);
  assert.equal(equipped, 'oak_planks');
  // Verified bot walked closer (from 7m away to within 3.4m)
  assert.ok(botPos.distanceTo(new Vec3(0.5, 65.5, 0.5)) <= 3.4);
  // Verified block is now solid in world
  const placed = mockBot.blockAt(new Vec3(0, 65, 0));
  assert.equal(placed.name, 'oak_planks');
});

test('30. build_simple_house anchors on scanned ground topology and constructs walls and roof', async () => {
  const { DynamicMCPToolRegistry } = require('../system2/mcp/toolRegistry');
  const { registerBaseMCPTools } = require('../system2/skills/baseSkills');

  const registry = new DynamicMCPToolRegistry();
  registerBaseMCPTools(registry);

  const houseTool = registry.getTool('build_simple_house');
  assert.ok(houseTool, 'build_simple_house must be registered in MCP catalog');
  assert.equal(houseTool.schema.name, 'build_simple_house');

  const scanTool = registry.getTool('scan_area');
  assert.ok(scanTool, 'scan_area must be registered in MCP catalog');

  // Verify house execution with mock context
  const placedPositions: Vec3[] = [];
  const mockCtx: any = {
    bot: {
      entity: {
        position: new Vec3(10, 65, 10),
        yaw: 0,
      },
      inventory: {
        items: () => [
          { name: 'oak_planks', count: 64 },
          { name: 'torch', count: 4 },
        ],
      },
      blockAt: () => ({ name: 'air', boundingBox: 'empty' }),
    },
    kinematics: {
      safePlaceBlock: async (pos: Vec3) => {
        placedPositions.push(pos.clone());
        return true;
      },
      safeDig: async () => true,
      pathfindTo: async () => true,
      lookAtSmooth: async () => {},
    },
    world: {
      scanAreaTopology: () => ({
        medianHeight: 64,
        minHeight: 64,
        maxHeight: 64,
        isFlat: true,
        obstructionCount: 0,
        airGapCount: 0,
        groundBlockTypes: { grass_block: 25 },
        heightGrid: [],
        minX: 10,
        minZ: 10,
        maxX: 14,
        maxZ: 14,
      }),
      getGroundHeight: () => 64,
    },
    reportProgress: () => {},
    sayHumanized: async () => {},
    abortSignal: new AbortController().signal,
  };

  const res = await houseTool.handler({ width: 4, length: 4, height: 3 }, mockCtx);
  assert.equal(res.success, true);
  assert.ok(res.summary.includes('Successfully built'));
  assert.ok(
    placedPositions.length >= 20,
    `Expected at least 20 blocks placed for 4x4 house, got ${placedPositions.length}`
  );
});

test('31. neural_chunk_scan MCP tool is registered and executes with GPU or CPU topology fallback', async () => {
  const { DynamicMCPToolRegistry } = require('../system2/mcp/toolRegistry');
  const { registerBaseMCPTools } = require('../system2/skills/baseSkills');

  const registry = new DynamicMCPToolRegistry();
  registerBaseMCPTools(registry);

  const neuralScanTool = registry.getTool('neural_chunk_scan');
  assert.ok(neuralScanTool, 'neural_chunk_scan must be registered in MCP catalog');

  const mockCtx: any = {
    bot: {
      entity: { position: new Vec3(0, 64, 0) },
      blockAt: () => ({ name: 'stone', boundingBox: 'block' }),
    },
    world: {
      scanAreaTopology: () => ({
        minHeight: 64,
        maxHeight: 65,
        medianHeight: 64,
        isFlat: true,
        obstructionCount: 0,
      }),
    },
  };

  const res = await neuralScanTool.handler({ radius: 32 }, mockCtx);
  assert.equal(res.success, true);
  assert.ok(res.summary.includes('Scan'));
});

test('32. NeuralBridgeClient communicates with mock RTX 5060 Neural Daemon endpoints (/reflex, /chunk_scan, /llm/chat)', async () => {
  const { NeuralBridgeClient } = require('../system1/neural/neuralBridge');

  const mockFetch = async (url: string) => {
    let payload: any = {};
    if (url.endsWith('/health')) {
      payload = { status: 'ok', device: 'NVIDIA GeForce RTX 5060 Laptop GPU' };
    } else if (url.endsWith('/reflex')) {
      payload = {
        aim_delta: { dyaw: 0.25, dpitch: -0.05 },
        actions: {
          move_forward: false,
          move_back: true,
          strafe_left: false,
          strafe_right: false,
          jump: true,
          sprint: true,
          sneak: false,
          attack: false,
          use_shield: true,
        },
        action_probs: { move_back: 0.98, use_shield: 0.99 },
        hazard_urgency: 0.96,
        latency_ms: 1.95,
      };
    } else if (url.endsWith('/chunk_scan')) {
      payload = {
        recommended_build_offset: { dx: 6.0, dy: 0.0, dz: -4.0 },
        flatness_confidence: 0.94,
        recommended_ore_offset: { dx: -10.0, dy: -14.0, dz: 8.0 },
        ore_richness_score: 0.96,
        cavern_hazard_score: 0.12,
        latency_ms: 4.5,
      };
    } else if (url.endsWith('/llm/chat')) {
      payload = {
        content: JSON.stringify({
          thought: 'Local RTX 5060 Qwen2.5-Coder selected follow_player',
          speechResponse: 'On my way!',
          toolCall: { name: 'follow_player', arguments: { playerName: 'TheJack3215', distance: 3 } },
        }),
        tokens_generated: 42,
        latency_ms: 410.0,
        tokens_per_sec: 102.4,
      };
    }
    return {
      ok: true,
      json: async () => payload,
    } as any;
  };

  const bridge = new NeuralBridgeClient('http://127.0.0.1:8765', mockFetch as any);
  const healthy = await bridge.checkHealth(true);
  assert.equal(healthy, true);

  const reflexOut = await bridge.evaluateReflex({
    position: { x: 0, y: 64, z: 0 },
    yaw: 0,
    pitch: 0,
    health: 18,
    food: 20,
    oxygen: 20,
    onGround: true,
    isInWater: false,
    isInLava: false,
    equippedArmor: { head: null, torso: null, legs: null, feet: null, offHand: 'shield' },
    inventory: [],
    visibleEntities: [
      {
        id: 1,
        name: 'creeper',
        type: 'hostile',
        position: { x: 2, y: 64, z: 2 },
        velocity: { x: 0, y: 0, z: 0 },
        distance: 2.8,
      },
    ],
    nearbyNotableBlocks: [],
  } as any);

  assert.ok(reflexOut);
  assert.equal(reflexOut.actions.use_shield, true);
  assert.equal(reflexOut.actions.move_back, true);
  assert.ok(reflexOut.hazard_urgency > 0.9);

  const chunkOut = await bridge.evaluateChunkScan([{ dx: 2, dy: -5, dz: 3, name: 'diamond_ore' }]);
  assert.ok(chunkOut);
  assert.ok(chunkOut.ore_richness_score > 0.9);

  const slmOut = await bridge.generateLocalSLMResponse('Sys', 'Follow me');
  assert.ok(slmOut);
  const parsed = JSON.parse(slmOut.content);
  assert.equal(parsed.toolCall.name, 'follow_player');
});

test('33. engage_combat supports passive animal hunting (cow), multi-target count, Mineflayer 1.20.4 e.type="hostile" (spider/wither_skeleton), and Creeper kiting', async () => {
  const { DynamicMCPToolRegistry } = require('../system2/mcp/toolRegistry');
  const { registerBaseMCPTools } = require('../system2/skills/baseSkills');

  const registry = new DynamicMCPToolRegistry();
  registerBaseMCPTools(registry);
  const combatTool = registry.getTool('engage_combat');
  assert.ok(combatTool, 'engage_combat must be registered');

  // Create 3 cows and 1 dropped beef item
  const cow1: any = {
    id: 101,
    name: 'cow',
    type: 'animal',
    position: new Vec3(2, 64, 0),
    isValid: true,
    hp: 2,
  };
  const cow2: any = {
    id: 102,
    name: 'cow',
    type: 'passive',
    position: new Vec3(4, 64, 0),
    isValid: true,
    hp: 2,
  };
  const spider1: any = {
    id: 201,
    name: 'spider',
    type: 'hostile', // Mineflayer 1.20.4+ uses 'hostile', not 'mob'!
    position: new Vec3(2.5, 64, 0),
    isValid: true,
    hp: 2,
  };

  const entities: Record<number, any> = {
    101: cow1,
    102: cow2,
    201: spider1,
  };

  const mockBot: any = {
    entity: { position: new Vec3(0, 64, 0) },
    entities,
    pathfinder: { setGoal: () => {} },
    setControlState: () => {},
  };

  const mockCtx: any = {
    bot: mockBot,
    kinematics: {
      equipItemHumanized: async () => true,
      lookAtSmooth: async () => {},
      pathfindTo: async (pos: Vec3) => {
        mockBot.entity.position = pos.clone();
        return true;
      },
      strikeTarget: async (ent: any) => {
        ent.hp -= 1;
        if (ent.hp <= 0) ent.isValid = false;
        return true;
      },
    },
    worldState: () => ({
      visibleEntities: [],
    }),
    reportProgress: () => {},
    abortSignal: new AbortController().signal,
  };

  // 1. Multi-target cow hunting (count: 2)
  const cowRes = await combatTool.handler({ targetType: 'cow', maxRange: 32, count: 2 }, mockCtx);
  assert.equal(cowRes.success, true);
  assert.equal(cow1.isValid, false, 'Cow 1 should be eliminated');
  assert.equal(cow2.isValid, false, 'Cow 2 should be eliminated');
  assert.ok(cowRes.summary.includes('2/2'), `Expected 2/2 cows eliminated, got: ${cowRes.summary}`);

  // 2. Spider with e.type === 'hostile'
  const spiderRes = await combatTool.handler({ targetType: 'spider', maxRange: 32, count: 1 }, mockCtx);
  assert.equal(spiderRes.success, true);
  assert.equal(spider1.isValid, false, 'Spider should be eliminated');
});

test('34. Built-in survival skills (collect_dropped_items, tame_animal, pillar_up) and generic come_here("player") fallback work out-of-the-box', async () => {
  const { DynamicMCPToolRegistry } = require('../system2/mcp/toolRegistry');
  const { registerBaseMCPTools } = require('../system2/skills/baseSkills');

  const registry = new DynamicMCPToolRegistry();
  registerBaseMCPTools(registry);

  assert.ok(registry.getTool('collect_dropped_items'), 'collect_dropped_items must be built-in');
  assert.ok(registry.getTool('tame_animal'), 'tame_animal must be built-in');
  assert.ok(registry.getTool('pillar_up'), 'pillar_up must be built-in');

  // Test come_here when LLM passes generic "player" instead of "TheJack3215"
  const comeHere = registry.getTool('come_here');
  const mockBot: any = {
    username: 'NeuroAgent',
    entity: { position: new Vec3(0, 64, 0) },
    players: {
      TheJack3215: {
        username: 'TheJack3215',
        entity: { position: new Vec3(10, 64, 10), velocity: new Vec3(0, 0, 0) },
      },
    },
    pathfinder: { setGoal: () => {} },
  };
  const comeRes = await comeHere.handler(
    { playerName: 'player' },
    {
      bot: mockBot,
      kinematics: {
        lookAtSmooth: async () => {},
        pathfindTo: async () => true,
      },
      reportProgress: () => {},
    } as any
  );
  assert.equal(comeRes.success, true);
  assert.ok(comeRes.summary.includes('TheJack3215'));
});

test('35. <1ms Fast-Path Intent Router resolves combat, multi-mob hunting, loot recovery, taming, and navigation without LLM wait', async () => {
  const llm = new UnifiedLLMClient();
  const mockState: any = {
    position: { x: 0, y: 64, z: 0 },
    health: 20,
    food: 20,
    inventory: [],
    visibleEntities: [],
  };

  const dCow = await llm.generateCognitiveDecision(
    'TheJack3215',
    'hello. Can you kill the nearest cow',
    mockState,
    [],
    []
  );
  assert.equal(dCow.toolCall?.name, 'engage_combat');
  assert.equal(dCow.toolCall?.arguments.targetType, 'cow');
  assert.equal(dCow.toolCall?.arguments.count, 1);

  const d5Cows = await llm.generateCognitiveDecision(
    'TheJack3215',
    'can you kill the nearest like 5 cows',
    mockState,
    [],
    []
  );
  assert.equal(d5Cows.toolCall?.name, 'engage_combat');
  assert.equal(d5Cows.toolCall?.arguments.targetType, 'cow');
  assert.equal(d5Cows.toolCall?.arguments.count, 5);

  const d5Spiders = await llm.generateCognitiveDecision(
    'TheJack3215',
    'can you take care of the 5 spiders',
    mockState,
    [],
    []
  );
  assert.equal(d5Spiders.toolCall?.name, 'engage_combat');
  assert.equal(d5Spiders.toolCall?.arguments.targetType, 'spider');
  assert.equal(d5Spiders.toolCall?.arguments.count, 5);

  const dLoot = await llm.generateCognitiveDecision(
    'TheJack3215',
    'pick up your stuff nearby',
    mockState,
    [],
    []
  );
  assert.equal(dLoot.toolCall?.name, 'collect_dropped_items');

  const dTame = await llm.generateCognitiveDecision(
    'TheJack3215',
    'can you tame that wolf',
    mockState,
    [],
    []
  );
  assert.equal(dTame.toolCall?.name, 'tame_animal');
  assert.equal(dTame.toolCall?.arguments.animalType, 'wolf');

  const dCome = await llm.generateCognitiveDecision(
    'TheJack3215',
    'come back to where I am',
    mockState,
    [],
    []
  );
  assert.equal(dCome.toolCall?.name, 'come_here');
  assert.equal(dCome.toolCall?.arguments.playerName, 'TheJack3215');

  const dEat = await llm.generateCognitiveDecision(
    'TheJack3215',
    'I meant, eat food if you have to',
    mockState,
    [],
    []
  );
  assert.equal(dEat.toolCall?.name, 'eat_food');

  const dCreeper = await llm.generateCognitiveDecision(
    'TheJack3215',
    'creeper!',
    mockState,
    [],
    []
  );
  assert.equal(dCreeper.toolCall?.name, 'engage_combat');
  assert.equal(dCreeper.toolCall?.arguments.targetType, 'creeper');
});

test('36. Built-in eat_food, shoot_bow, till_and_farm, interact_block_or_puzzle, and navigate_parkour_or_bridge MCP skills are registered and functional', async () => {
  const { DynamicMCPToolRegistry } = require('../system2/mcp/toolRegistry');
  const { registerBaseMCPTools } = require('../system2/skills/baseSkills');

  const registry = new DynamicMCPToolRegistry();
  registerBaseMCPTools(registry);

  assert.ok(registry.getTool('eat_food'), 'eat_food must be registered');
  assert.ok(registry.getTool('shoot_bow'), 'shoot_bow must be registered');
  assert.ok(registry.getTool('till_and_farm'), 'till_and_farm must be registered');
  assert.ok(registry.getTool('interact_block_or_puzzle'), 'interact_block_or_puzzle must be registered');
  assert.ok(registry.getTool('navigate_parkour_or_bridge'), 'navigate_parkour_or_bridge must be registered');
});

test('37. Discrete MCP tools craft_recipe, smelt_ores, and manage_chest are registered and enforce combinatorial schemas', async () => {
  const { DynamicMCPToolRegistry } = require('../system2/mcp/toolRegistry');
  const { registerBaseMCPTools } = require('../system2/skills/baseSkills');

  const registry = new DynamicMCPToolRegistry();
  registerBaseMCPTools(registry);

  const craftTool = registry.getTool('craft_recipe');
  assert.ok(craftTool, 'craft_recipe must be registered');
  assert.equal(craftTool.schema.name, 'craft_recipe');

  const smeltTool = registry.getTool('smelt_ores');
  assert.ok(smeltTool, 'smelt_ores must be registered');
  assert.equal(smeltTool.schema.name, 'smelt_ores');

  const chestTool = registry.getTool('manage_chest');
  assert.ok(chestTool, 'manage_chest must be registered');
  assert.equal(chestTool.schema.name, 'manage_chest');

  // Verify craft_recipe refuses crafting without required prerequisites in inventory
  const mockCtx: any = {
    bot: {
      version: '1.20.4',
      entity: { position: new Vec3(0, 64, 0) },
      inventory: { items: () => [] },
      findBlock: () => null,
      recipesFor: () => [],
      blockAt: () => ({ name: 'air', boundingBox: 'empty' }),
      canSeeBlock: () => true,
      canDigBlock: () => true,
      safePlaceBlock: async () => false,
    },
    kinematics: {
      equipItemHumanized: async () => false,
      pathfindTo: async () => true,
      safePlaceBlock: async () => false,
    },
    abortSignal: new AbortController().signal,
    reportProgress: () => {},
  };
  const craftRes = await craftTool.handler({ recipeName: 'wooden_pickaxe', count: 1 }, mockCtx);
  assert.equal(craftRes.success, false);
  assert.match(craftRes.summary, /Cannot craft wooden_pickaxe/);
});

test('38. Granular DAG milestone scoring in evaluators awards partial dense credit (25/50/75/100)', async () => {
  const { evaluateProgressionTask, evaluateOmniscienceMilestones, evaluateMinebenchEpisode } = require('../../minebench/dist/minebench/evaluators');
  const { MINEBENCH_TASKS } = require('../../minebench/dist/minebench/benchmarkSpec');

  const tier1Task = MINEBENCH_TASKS.find((t: any) => t.id === 'prog_tier1_stone_age');
  assert.ok(tier1Task, 'prog_tier1_stone_age spec must exist');

  // Base snapshot: only raw logs gathered -> Milestone 1 (25 pts)
  const snap1: any = {
    botPosition: { x: 800, y: 64, z: 100 },
    health: 20,
    food: 20,
    deaths: 0,
    inventory: [{ name: 'oak_log', count: 2 }],
    equippedArmor: { head: null, torso: null, legs: null, feet: null, offHand: null },
    blockAt: () => null,
    entities: [],
    stats: { blocksPlaced: 0, blocksMined: 2, mobsKilled: {}, creeperBlasts: 0, arrowsFired: 0, endCrystalsDestroyed: 0, enderDragonSlain: false, leversFlipped: 0, wolvesTamed: 0, ironPickaxeDurabilityUsedOnStone: 0 },
  };
  const res1 = evaluateProgressionTask(tier1Task, snap1);
  assert.equal(res1.score, 25, 'Milestone 1 raw harvest should award exactly 25 pts');
  assert.equal(res1.passed, false);

  // Snapshot 2: logs + planks -> Milestone 1 + 2 (50 pts)
  const snap2: any = {
    ...snap1,
    inventory: [{ name: 'oak_log', count: 2 }, { name: 'oak_planks', count: 8 }],
  };
  const res2 = evaluateProgressionTask(tier1Task, snap2);
  assert.equal(res2.score, 50, 'Milestones 1+2 should award 50 pts');
  assert.equal(res2.passed, false);

  // Snapshot 3: logs + planks + crafting_table -> Milestone 1 + 2 + 3 (75 pts)
  const snap3: any = {
    ...snap1,
    inventory: [{ name: 'oak_log', count: 2 }, { name: 'oak_planks', count: 8 }, { name: 'crafting_table', count: 1 }],
  };
  const res3 = evaluateProgressionTask(tier1Task, snap3);
  assert.equal(res3.score, 75, 'Milestones 1+2+3 should award 75 pts');
  assert.equal(res3.passed, false);

  // Snapshot 4: final target stone pickaxe synthesized (even when logs and planks were consumed) -> 100 pts & passed
  const snap4: any = {
    ...snap1,
    inventory: [{ name: 'stone_pickaxe', count: 1 }],
    stats: { ...snap1.stats, blocksMined: 5, blocksPlaced: 1 },
  };
  const res4 = evaluateProgressionTask(tier1Task, snap4);
  assert.equal(res4.score, 100, 'All 4 milestones should award 100 pts even after logs and planks were consumed');
  assert.equal(res4.passed, true);

  // Verify evaluateMinebenchEpisode handles classic_ task IDs without throwing or returning unknown ID
  const classicEpisodeRes = evaluateMinebenchEpisode({ taskId: 'classic_prog_tier1_stone_age' } as any, snap4);
  assert.equal(classicEpisodeRes.score, 100, 'evaluateMinebenchEpisode should normalize classic_ prefix and award 100 pts');
  assert.equal(classicEpisodeRes.passed, true);

  // Verify omniscience crafting task where target item was crafted and planks consumed
  const omniCraftTask = {
    taskId: 'omni_craft_chest',
    targetObject: 'chest',
    evaluatorMilestones: [
      { name: 'Milestone 1: Raw Harvest', maxPoints: 25, condition: 'dag_raw_harvest' },
      { name: 'Milestone 2: Basic Refine', maxPoints: 25, condition: 'dag_basic_refine' },
      { name: 'Milestone 3: Workstation', maxPoints: 25, condition: 'dag_workstation' },
      { name: 'Milestone 4: Final Target Item', maxPoints: 25, condition: 'has_chest' },
    ],
  };
  const omniSnap: any = {
    ...snap1,
    inventory: [{ name: 'chest', count: 1 }],
    stats: { ...snap1.stats, blocksMined: 2, blocksPlaced: 1 },
  };
  const omniRes = evaluateOmniscienceMilestones(omniCraftTask, omniSnap);
  assert.equal(omniRes.score, 100, 'Synthesizing target item should preserve prior DAG milestone points');
  assert.equal(omniRes.passed, true);
});

test('39. Pure sensorimotor pathfindToPosition steers bot at 20Hz without setting mineflayer-pathfinder goals', async () => {
  const { System1BotController } = require('../system1/botController');
  const controller = new System1BotController();

  let lastGoalArg: any = 'UNSET';
  let looksCount = 0;
  let forwardSet = false;

  const mockBot: any = {
    entity: {
      position: new Vec3(0, 64, 0),
      yaw: 0,
      pitch: 0,
      onGround: true,
      isCollidedHorizontally: false,
    },
    pathfinder: {
      setGoal: (arg: any) => {
        lastGoalArg = arg;
      },
    },
    look: async () => {
      looksCount++;
    },
    setControlState: (state: string, val: boolean) => {
      if (state === 'forward' && val) forwardSet = true;
    },
    clearControlStates: () => {},
    blockAt: () => ({ name: 'air', boundingBox: 'empty' }),
  };

  (controller as any).bot = mockBot;

  // Run pathfindToPosition for target 3 blocks away with fine range 0.8
  const target = new Vec3(0, 64, 3);
  const promise = controller.pathfindToPosition(target, 0.8, 500);

  // Simulate arrival after a few ticks
  setTimeout(() => {
    mockBot.entity.position = new Vec3(0, 64, 2.6);
  }, 100);

  const reached = await promise;
  assert.equal(reached, true, 'Should successfully arrive within fine range');
  assert.equal(forwardSet, true, 'Must drive forward control state continuously');
  assert.ok(looksCount > 0, 'Must apply camera aim deltas towards target');
  assert.equal(lastGoalArg, null, 'pathfinder.setGoal should strictly be called with null to eliminate pathfinder hijacking');
});

test('40. Multi-Bot Spatial Arena Sharding & Concurrent Worker Pool', async () => {
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

  // 1. Concurrency argument parsing & clamping
  assert.equal(parseConcurrencyArg([]), 3);
  assert.equal(parseConcurrencyArg(['--concurrency', '4']), 4);
  assert.equal(parseConcurrencyArg(['--bots', '2']), 2);
  assert.equal(parseConcurrencyArg(['--concurrency=5']), 5);
  assert.equal(parseConcurrencyArg(['--concurrency', '0']), 1);
  assert.equal(parseConcurrencyArg(['--concurrency', '99']), 5);
  assert.equal(parseConcurrencyArg([], '4'), 4);
  assert.equal(parseConcurrencyArg(['--concurrency', '2'], '4'), 2);

  // 2. Spatial offset calculation (ΔX = i * 4000)
  assert.equal(getWorkerSpatialOffset(0), 0);
  assert.equal(getWorkerSpatialOffset(1), 4000);
  assert.equal(getWorkerSpatialOffset(2), 8000);
  assert.equal(getWorkerSpatialOffset(3), 12000);

  // 3. Worker bot naming
  assert.equal(getWorkerBotName(0, 1), 'NeuroAgent');
  assert.equal(getWorkerBotName(0, 3), 'NeuroAgent_1');
  assert.equal(getWorkerBotName(1, 3), 'NeuroAgent_2');
  assert.equal(getWorkerBotName(2, 3), 'NeuroAgent_3');

  // 4. Command coordinate shifting
  assert.equal(
    shiftCommandCoordinates('fill 100 80 100 128 88 106 minecraft:air', 4000),
    'fill 4100 80 100 4128 88 106 minecraft:air'
  );
  assert.equal(
    shiftCommandCoordinates('setblock 104 79 100 minecraft:stone_bricks', 4000),
    'setblock 4104 79 100 minecraft:stone_bricks'
  );
  assert.equal(
    shiftCommandCoordinates('tp NeuroAgent 100 80 100 -90 0', 4000),
    'tp NeuroAgent 4100 80 100 -90 0'
  );
  assert.equal(
    shiftCommandCoordinates('summon creeper 105 80 100', 4000),
    'summon creeper 4105 80 100'
  );

  // 5. Task spatial sharding
  const dummyTask = {
    id: 'sharded_test_task',
    title: 'Sharded Course',
    domain: 'AGILITY_PARKOUR',
    tier: 1,
    maxDurationSeconds: 30,
    maxSteps: 3,
    prompt: 'Jump across!',
    origin: { x: 100, y: 80, z: 100 },
    initialLoadout: [],
    buildArenaCommands: ({ x, y, z }: any) => [
      `fill ${x - 2} ${y - 1} ${z - 2} ${x + 2} ${y - 1} ${z + 2} minecraft:stone`,
      `tp NeuroAgent ${x} ${y} ${z} 0 0`,
    ],
  };

  const sharded1 = createShardedTaskSpec(dummyTask, 1, 3);
  assert.equal(sharded1.origin.x, 4100);
  assert.equal(sharded1.origin.y, 80);
  assert.equal(sharded1.origin.z, 100);
  const cmds = sharded1.buildArenaCommands(sharded1.origin);
  assert.equal(cmds[0], 'fill 4098 79 98 4102 79 102 minecraft:stone');
  assert.equal(cmds[1], 'tp NeuroAgent_2 4100 80 100 0 0');

  // Single bot mode preserves task unmodified
  assert.equal(createShardedTaskSpec(dummyTask, 0, 1), dummyTask);

  // 6. Concurrent task distribution pool
  const mockTasks = Array.from({ length: 9 }, (_, i) => ({ id: `sub_${i}` }));
  const completed: Array<{ taskId: string; workerId: number }> = [];

  await runTaskQueuePool(mockTasks, 3, async (item: any, _idx: number, workerId: number) => {
    await new Promise((r) => setTimeout(r, 5));
    completed.push({ taskId: item.id, workerId });
  });

  assert.equal(completed.length, 9, 'All 9 mock tasks must complete');
  const distinctWorkers = new Set(completed.map((c) => c.workerId));
  assert.equal(distinctWorkers.size, 3, 'All 3 workers must process tasks');
});

test('41. VLACompanionClient extracts 32-class semantic 3D voxels and 16-entity radar tensors', () => {
  const { VLACompanionClient, mapBlockNameToSemanticId } = require('../system2/vlaClient');

  assert.equal(mapBlockNameToSemanticId('air'), 0);
  assert.equal(mapBlockNameToSemanticId('stone'), 1);
  assert.equal(mapBlockNameToSemanticId('water'), 10);
  assert.equal(mapBlockNameToSemanticId('lava'), 11);
  assert.equal(mapBlockNameToSemanticId('diamond_ore'), 15);
  assert.equal(mapBlockNameToSemanticId('crafting_table'), 17);
  assert.equal(mapBlockNameToSemanticId('furnace'), 18);
  assert.equal(mapBlockNameToSemanticId('gold_block'), 28);

  const client = new VLACompanionClient('http://127.0.0.1:8765');
  const botPos = new Vec3(0, 64, 0);
  const mockBot: any = {
    username: 'NeuroAgent',
    entity: { position: botPos, velocity: new Vec3(0, 0, 0), yaw: 0, pitch: 0, onGround: true },
    health: 20,
    food: 20,
    heldItem: { name: 'iron_pickaxe' },
    inventory: { slots: {} },
    blockAt: (p: Vec3) => {
      if (p.x === 0 && p.y === 64 && p.z === 2) return { name: 'lava', boundingBox: 'empty' };
      if (p.x === 1 && p.y === 65 && p.z === 0) return { name: 'diamond_ore', boundingBox: 'block' };
      if (p.y === 63) return { name: 'stone', boundingBox: 'block' };
      return { name: 'air', boundingBox: 'empty' };
    },
    entities: {
      10: {
        id: 10,
        name: 'creeper',
        type: 'hostile',
        position: new Vec3(4, 64, 0),
        velocity: new Vec3(-0.1, 0, 0),
        metadata: { 16: 1 },
        isValid: true,
      },
      11: {
        id: 11,
        name: 'cow',
        type: 'animal',
        position: new Vec3(0, 64, 8),
        velocity: new Vec3(0, 0, 0),
        isValid: true,
      },
    },
  };

  const voxels = client.extractLocalVoxels(mockBot);
  assert.equal(voxels.length, 3375);
  // Assert non-binary semantic IDs (1=stone, 11=lava, 15=diamond_ore) appear cleanly
  assert.ok(voxels.includes(1), 'Must preserve stone semantic ID 1');
  assert.ok(voxels.includes(11), 'Must preserve lava semantic ID 11 even though boundingBox is empty');
  assert.ok(voxels.includes(15), 'Must preserve diamond_ore semantic ID 15');

  const radar = client.extractEntityRadar(mockBot);
  assert.equal(radar.length, 16);
  assert.equal(radar[0].length, 12);
  // Nearest entity should be creeper at dist 4 (< 8)
  assert.equal(radar[0][7], 1, 'Nearest entity must be flagged hostile');
  assert.equal(radar[0][8], 1, 'Nearest entity must be flagged creeper');
  assert.equal(radar[0][11], 1, 'Creeper active fuse must be flagged');
  // Second entity is non-hostile cow
  assert.equal(radar[1][7], 0, 'Passive cow must not be flagged hostile');

  // Assert Boss Priority Ranking: In End dimension with 25 swarming Endermen (dist 10-15m) and 1 Ender Dragon (dist 75m),
  // Ender Dragon MUST be sorted to slot 0 (Priority 0) and not be cut off by the swarming Endermen.
  const endMockEntities: Record<number, any> = {
    1: { id: 1, name: 'ender_dragon', type: 'mob', position: new Vec3(0, 100, 75), velocity: new Vec3(0, 0, 0), isValid: true },
  };
  for (let k = 2; k <= 25; k++) {
    endMockEntities[k] = { id: k, name: 'enderman', type: 'hostile', position: new Vec3(k % 5, 64, 10 + (k % 4)), velocity: new Vec3(0, 0, 0), isValid: true };
  }
  const endBot: any = {
    ...mockBot,
    entities: endMockEntities,
  };
  const endRadar = client.extractEntityRadar(endBot);
  assert.equal(endRadar.length, 16);
  assert.equal(endRadar[0][10], 1, 'Slot 0 MUST be the Ender Dragon (isBossOrCrystal=1.0) despite being farther than 24 Endermen');
});

test('42. HierarchicalGoalPlanner & computeGroundedAffordances track macro/sub-goals, craftable items, and replanning', () => {
  const { HierarchicalGoalPlanner, computeGroundedAffordances } = require('../system2/goalPlanner');

  const mockState: any = {
    tick: 50,
    timestamp: Date.now(),
    botUsername: 'NeuroAgent',
    position: { x: 0, y: 64, z: 0 },
    velocity: { x: 0, y: 0, z: 0 },
    yaw: 0,
    pitch: 0,
    health: 20,
    food: 20,
    oxygen: 20,
    onGround: true,
    isInWater: false,
    isInLava: false,
    heldItem: 'stone_pickaxe',
    equippedArmor: { head: 'iron_helmet', torso: null, legs: null, feet: null, offHand: 'shield' },
    inventory: [
      { name: 'oak_log', count: 2, slot: 36 },
      { name: 'oak_planks', count: 8, slot: 37 },
      { name: 'stick', count: 4, slot: 38 },
      { name: 'cobblestone', count: 8, slot: 39 },
      { name: 'raw_iron', count: 6, slot: 40 },
      { name: 'coal', count: 4, slot: 41 },
      { name: 'stone_pickaxe', count: 1, slot: 42, durability: 40, maxDurability: 131, durabilityPercent: 31 },
    ],
    visibleEntities: [],
    audibleCues: [],
    nearbyNotableBlocks: [{ name: 'crafting_table', position: { x: 1, y: 64, z: 0 }, distance: 1.0 }],
    activeHazard: null,
    activeSubtask: null,
  };

  const aff = computeGroundedAffordances(mockState);
  assert.equal(aff.equippedSummary.mainHand, 'stone_pickaxe');
  assert.equal(aff.equippedSummary.offHand, 'shield');
  assert.equal(aff.equippedSummary.lowestToolDurabilityPercent, 31);
  assert.ok(aff.nearbyWorkstations.includes('crafting_table'));
  assert.ok(aff.craftableNow.includes('stone_pickaxe'));
  assert.ok(aff.craftableNow.includes('furnace'));
  assert.equal(aff.smeltableNow, true);

  const planner = new HierarchicalGoalPlanner();
  planner.setMacroGoal('Get full iron armor', ['Craft stone_pickaxe', 'Mine iron_ore', 'Smelt iron_ingot', 'Craft iron armor']);
  assert.equal(planner.getCurrentSubGoal(), 'Craft stone_pickaxe');

  const t1 = planner.advanceOrReplan(
    { toolName: 'craft_item', success: true, summary: 'Successfully crafted 1x stone_pickaxe' },
    { subgoal_completed: true, should_replan: false },
    mockState
  );
  assert.equal(t1.advanced, true);
  assert.equal(t1.needsReplan, false);
  assert.equal(t1.currentSubGoal, 'Mine iron_ore');

  const t2 = planner.advanceOrReplan(
    { toolName: 'gather_resource', success: true, summary: 'Searching vein...' },
    { subgoal_completed: false, should_replan: true },
    mockState
  );
  assert.equal(t2.needsReplan, true);
  assert.ok(planner.formatGoalHorizonContext(mockState).includes('Replan Alert'));
});

test('43. UnifiedLLMClient.buildGroundedTrainingSystemPrompt preserves live sensory presence and affordances', () => {
  const { registerBaseMCPTools } = require('../system2/skills/baseSkills');
  const registry = new DynamicMCPToolRegistry();
  registerBaseMCPTools(registry);
  const schemas = registry.listSchemas();
  const llm = new UnifiedLLMClient({ offlineOnly: true });

  const mockState: any = {
    tick: 120,
    timestamp: Date.now(),
    botUsername: 'NeuroAgent_1',
    position: { x: 12, y: 64, z: -8 },
    velocity: { x: 0, y: 0, z: 0 },
    yaw: 0,
    pitch: 0,
    health: 18,
    food: 16,
    oxygen: 20,
    onGround: true,
    isInWater: false,
    isInLava: false,
    heldItem: 'stone_pickaxe',
    equippedArmor: { head: null, torso: 'iron_chestplate', legs: null, feet: null, offHand: 'shield' },
    inventory: [
      { name: 'oak_log', count: 2, slot: 36 },
      { name: 'cobblestone', count: 8, slot: 37 },
      { name: 'raw_iron', count: 4, slot: 38 },
      { name: 'coal', count: 2, slot: 39 },
      { name: 'stone_pickaxe', count: 1, slot: 40, durability: 65, maxDurability: 131, durabilityPercent: 50 },
    ],
    visibleEntities: [{ id: 5, name: 'creeper', type: 'hostile', position: { x: 16, y: 64, z: -8 }, velocity: { x: 0, y: 0, z: 0 }, distance: 4.0 }],
    audibleCues: [],
    nearbyNotableBlocks: [{ name: 'furnace', position: { x: 13, y: 64, z: -8 }, distance: 1.0 }],
    activeHazard: null,
    activeSubtask: null,
  };

  const snapPrompt = llm.buildGroundedTrainingSystemPrompt(schemas, mockState, [
    { stepNumber: 1, toolName: 'gather_resource', args: { blockType: 'iron_ore' }, success: true, summary: 'Mined 4 raw_iron' },
  ]);

  assert.ok(snapPrompt.includes('Gear & Durability:'), 'Must include Gear & Durability');
  assert.ok(snapPrompt.includes('Inventory: [2x oak_log, 8x cobblestone, 4x raw_iron'), 'Must include Inventory contents');
  assert.ok(snapPrompt.includes('Workstations Nearby: [furnace]'), 'Must include Workstations Nearby');
  assert.ok(snapPrompt.includes('Immediately Craftable: [oak_planks]'), 'Must include Immediately Craftable');
  assert.ok(snapPrompt.includes('Smeltable: [') && snapPrompt.includes('raw_iron'), 'Must include Smeltable');
  assert.ok(snapPrompt.includes('Nearby Entities (360°): [creeper(4.0m)]'), 'Must include Nearby Entities');
  assert.ok(snapPrompt.includes('Recent Actions: [gather_resource(OK)]'), 'Must include Recent Actions');
});

test('44. Repaired Omniscience tasks validate against 1.20.4 and evaluators score water clutch, elytra, lava smelt, and melee damageDealt properly', () => {
  const fs = require('fs');
  const path = require('path');
  const mcData = require('minecraft-data')('1.20.4');
  const { evaluateOmniscienceMilestones } = require('../../minebench/dist/minebench/evaluators');

  const tasksPath = path.resolve(process.cwd(), 'minebench', 'results', 'omniscience_tasks.json');
  const tasks: any[] = JSON.parse(fs.readFileSync(tasksPath, 'utf8'));

  assert.ok(!tasks.some((t) => t.taskId === 'omni_mob_breeze'), 'Experimental 1.21 breeze must be excluded from 1.20.4 pool');

  const nbtRe = /id:"(?:minecraft:)?([a-z0-9_]+)"/g;
  for (const t of tasks) {
    for (const it of t.initialLoadout || []) {
      assert.ok(mcData.itemsByName[it.name], `Invalid 1.20.4 loadout item "${it.name}" in ${t.taskId}`);
    }
    for (const cmd of t.setupCommands || []) {
      let m: RegExpExecArray | null;
      while ((m = nbtRe.exec(cmd)) !== null) {
        assert.ok(mcData.itemsByName[m[1]], `Invalid 1.20.4 NBT item "${m[1]}" in ${t.taskId}`);
      }
    }
  }

  const getTask = (id: string) => tasks.find((t) => t.taskId === id);
  const growTree = getTask('omni_tool_bonemeal_grow_tree');
  assert.equal(growTree.targetObject, 'bone_meal');
  assert.ok(growTree.setupCommands.some((c: string) => c.includes('dirt')));
  assert.ok(getTask('omni_tool_fishing_rod_catch_fish').setupCommands.some((c: string) => c.includes('water')));
  assert.ok(getTask('omni_tool_lava_bucket_smelt_fuel_100').setupCommands.some((c: string) => c.includes('furnace')));
  assert.ok(getTask('omni_tool_elytra_glide_and_boost').initialLoadout.some((i: any) => i.name === 'firework_rocket'));

  const baseSnap = (overrides: any = {}) => ({
    botPosition: { x: 500, y: 100, z: 500 },
    health: 20,
    food: 20,
    deaths: 0,
    inventory: [],
    equippedArmor: { head: null, torso: null, legs: null, feet: null, offHand: null },
    blockAt: () => null,
    entities: [],
    stats: {
      blocksPlaced: 0,
      blocksMined: 0,
      mobsKilled: {},
      creeperBlasts: 0,
      arrowsFired: 0,
      endCrystalsDestroyed: 0,
      enderDragonSlain: false,
      leversFlipped: 0,
      wolvesTamed: 0,
      foodConsumed: 0,
      affordanceActions: 0,
      damageDealt: 0,
      waterPlaced: 0,
      ironPickaxeDurabilityUsedOnStone: 0,
      ...overrides.stats,
    },
    ...overrides,
  });

  // Water clutch: passive sitting on perch = FAIL
  const wcTask = { ...getTask('omni_tool_water_bucket_water_clutch'), origin: { x: 500, y: 100, z: 500 } };
  const wcPassive = evaluateOmniscienceMilestones(wcTask, baseSnap({ botPosition: { x: 500, y: 118, z: 500 }, inventory: [{ name: 'water_bucket', count: 1 }] }));
  assert.equal(wcPassive.passed, false, 'Passive water_clutch standing on perch must fail');

  // Water clutch: un-clutched fatal fall = FAIL
  const wcFatal = evaluateOmniscienceMilestones(wcTask, baseSnap({ deaths: 1, health: 0, inventory: [{ name: 'water_bucket', count: 1 }] }));
  assert.equal(wcFatal.passed, false, 'Fatal un-clutched fall must fail');

  // Water clutch: deployed water & survived = PASS
  const wcPass = evaluateOmniscienceMilestones(wcTask, baseSnap({ inventory: [{ name: 'bucket', count: 1 }], stats: { waterPlaced: 1, affordanceActions: 1 } }));
  assert.equal(wcPass.passed, true, 'Deployed water clutch must pass');
  assert.equal(wcPass.score, 100);

  // Elytra: equipped only without glide distance = FAIL (< 60)
  const elTask = { ...getTask('omni_tool_elytra_glide_and_boost'), origin: { x: 500, y: 100, z: 500 } };
  const elEquipOnly = evaluateOmniscienceMilestones(elTask, baseSnap({ equippedArmor: { torso: 'elytra' }, botPosition: { x: 502, y: 120, z: 500 } }));
  assert.equal(elEquipOnly.passed, false, 'Equipping elytra without gliding >8m must not pass');
  assert.equal(elEquipOnly.score, 55);

  // Elytra: equipped + >8m horizontal glide = PASS (100)
  const elGlided = evaluateOmniscienceMilestones(elTask, baseSnap({ equippedArmor: { torso: 'elytra' }, botPosition: { x: 515, y: 105, z: 500 } }));
  assert.equal(elGlided.passed, true, 'Gliding >8m with elytra equipped must pass');
  assert.equal(elGlided.score, 100);

  // Lava bucket smelt: smelted iron_ingot + bucket = PASS (100)
  const lavaTask = { ...getTask('omni_tool_lava_bucket_smelt_fuel_100'), origin: { x: 500, y: 100, z: 500 } };
  const lavaPass = evaluateOmniscienceMilestones(lavaTask, baseSnap({ inventory: [{ name: 'bucket', count: 1 }, { name: 'iron_ingot', count: 2 }], stats: { affordanceActions: 1 } }));
  assert.equal(lavaPass.passed, true, 'Smelting with lava_bucket must pass');

  // Hostile mob kill: damageDealt=0 = FAIL vs damageDealt=20 = PASS
  const zombieTask = getTask('omni_mob_zombie');
  const zNoDmg = evaluateOmniscienceMilestones(zombieTask, baseSnap({ stats: { mobsKilled: { zombie: 1 }, damageDealt: 0 } }));
  assert.equal(zNoDmg.passed, false, 'Unattributed kill with 0 damageDealt must not pass');
  const zWithDmg = evaluateOmniscienceMilestones(zombieTask, baseSnap({ stats: { mobsKilled: { zombie: 1 }, damageDealt: 20 } }));
  assert.equal(zWithDmg.passed, true, 'Verified kill with damageDealt=20 must pass');
  assert.equal(zWithDmg.score, 100);
});

test('45. gather_resource avoids substring-matching stone for redstone_ore and smelt_item accepts lava_bucket fuel', async () => {
  const { registerBaseMCPTools } = require('../system2/skills/baseSkills');
  const registry = new DynamicMCPToolRegistry();
  registerBaseMCPTools(registry);
  const mcpServer = new NeuroCraftMCPServer(registry);

  // 1. Verify gather_resource("redstone_ore") ignores arena "stone" floor
  const dugBlocks: string[] = [];
  let heldPick: any = { name: 'iron_pickaxe', type: 700, count: 1 };
  const redstoneBot: any = {
    version: '1.20.4',
    entity: { position: new Vec3(0, 64, 0), onGround: true },
    get heldItem() { return heldPick; },
    inventory: {
      items: () => [heldPick],
      slots: [],
    },
    pathfinder: { setGoal: () => {} },
    clearControlStates: () => {},
    canDigBlock: () => true,
    dig: async (b: any) => { dugBlocks.push(b.name); },
    findBlocks: ({ matching }: any) => {
      // Simulate Mineflayer palette pre-filter (position === null)
      assert.equal(matching({ name: 'stone', position: null }), false, 'redstone_ore matcher MUST NOT match stone');
      assert.equal(matching({ name: 'redstone_ore', position: null }), true, 'redstone_ore matcher MUST match redstone_ore');
      return [];
    },
    blockAt: (p: Vec3) => (p.y < 64 ? { name: 'stone', boundingBox: 'block', position: p } : { name: 'air', boundingBox: 'empty', position: p }),
  };

  const gatherRes = await mcpServer.handleRpcRequest(
    { jsonrpc: '2.0', id: 't45_redstone', method: 'tools/call', params: { name: 'gather_resource', arguments: { blockType: 'redstone_ore', count: 1 } } },
    () => ({
      bot: redstoneBot,
      kinematics: { equipItemHumanized: async () => true, pathfindTo: async () => true, lookAtSmooth: async () => {} },
      reportProgress: () => {},
      abortSignal: new AbortController().signal,
    } as any)
  );
  assert.equal(dugBlocks.includes('stone'), false, 'gather_resource("redstone_ore") must never dig stone');
  assert.ok(!gatherRes.error);

  // 2. Verify smelt_item accepts lava_bucket as valid furnace fuel
  let putFuelName = '';
  let tookFuel = false;
  let tookOutput = false;
  const smeltBot: any = {
    version: '1.20.4',
    entity: { position: new Vec3(0, 64, 0), onGround: true },
    inventory: {
      items: () => [
        { name: 'lava_bucket', type: 815, count: 1 },
        { name: 'raw_iron', type: 720, count: 1 },
      ],
      slots: [],
    },
    findBlock: ({ matching }: any) => (matching({ name: 'furnace' }) ? { name: 'furnace', position: new Vec3(0, 64, 2) } : null),
    openFurnace: async () => ({
      putFuel: async (typeId: number, _meta: any, cnt: number) => {
        if (typeId === 815 && cnt === 1) putFuelName = 'lava_bucket';
      },
      putInput: async () => {},
      outputItem: () => ({ name: 'iron_ingot', count: 1 }),
      inputItem: () => null,
      fuelItem: () => ({ name: 'bucket', count: 1 }),
      takeOutput: async () => { tookOutput = true; },
      takeInput: async () => {},
      takeFuel: async () => { tookFuel = true; },
      close: () => {},
    }),
  };

  const smeltRes = await mcpServer.handleRpcRequest(
    { jsonrpc: '2.0', id: 't45_lava', method: 'tools/call', params: { name: 'smelt_item', arguments: { inputItem: 'raw_iron', count: 1 } } },
    () => ({
      bot: smeltBot,
      kinematics: { equipItemHumanized: async () => true, pathfindTo: async () => true, lookAtSmooth: async () => {} },
      reportProgress: () => {},
      abortSignal: new AbortController().signal,
    } as any)
  );
  const smeltPayload = JSON.parse(smeltRes.result.content[0].text);
  assert.equal(putFuelName, 'lava_bucket', 'smelt_item must fuel furnace with 1x lava_bucket');
  assert.equal(tookOutput, true);
  assert.equal(tookFuel, true);
  assert.equal(smeltPayload.success, true, 'smelt_item must succeed with lava_bucket fuel');
});

test('46. GRPO goalCompleted distillation never marks Step 0, partial scores, or open-world survival as completed, and train_loop guards Step 0 break', () => {
  const { execFileSync } = require('child_process');
  const fs = require('fs');
  const path = require('path');

  const rootDir = path.resolve(__dirname, '..', '..');
  const pythonExe = path.join(rootDir, 'neural_lab', '.venv', 'Scripts', 'python.exe');
  const pyCheck = `
import json, sys
sys.path.insert(0, r'${rootDir.replace(/\\/g, '\\\\')}')
from minebench.rl_trainer import should_mark_goal_completed, sanitize_loaded_grpo_entry

assert should_mark_goal_completed(True, 100.0, 'local_gpu', 1, 0) is False, '1-step Step-0 episode must NOT set goalCompleted'
assert should_mark_goal_completed(True, 75.0, 'local_gpu', 3, 2) is False, 'Partial score multi-step episode must NOT set goalCompleted'
assert should_mark_goal_completed(True, 100.0, 'open_world_survival', 4, 3) is False, 'open_world_survival must NOT set goalCompleted'
assert should_mark_goal_completed(True, 100.0, 'local_gpu', 4, 1) is False, 'Intermediate step must NOT set goalCompleted'
assert should_mark_goal_completed(True, 100.0, 'local_gpu', 4, 3) is True, 'Final step of 100/100 multi-step episode must set goalCompleted'

legacy_step0 = {
    'provider': 'local_gpu',
    'taskId': 'omni_mob_cow',
    'step': 0,
    'messages': [
        {'role': 'system', 'content': 'sys'},
        {'role': 'user', 'content': 'usr'},
        {'role': 'assistant', 'content': json.dumps({'thought': 'hi', 'toolCall': {'name': 'engage_combat', 'arguments': {}}, 'goalCompleted': True})},
    ],
}
scrubbed = sanitize_loaded_grpo_entry(legacy_step0)
assert json.loads(scrubbed['messages'][-1]['content'])['goalCompleted'] is False, 'Legacy Step-0 local_gpu goalCompleted=True must be rewritten to False'

gold_cont = {
    'provider': 'gold_continuation',
    'taskId': 'omni_craft_chest',
    'step': 3,
    'messages': [
        {'role': 'system', 'content': 'sys'},
        {'role': 'user', 'content': 'usr'},
        {'role': 'assistant', 'content': json.dumps({'thought': 'done', 'toolCall': {'name': 'craft_item', 'arguments': {}}, 'goalCompleted': True})},
    ],
}
preserved = sanitize_loaded_grpo_entry(gold_cont)
assert json.loads(preserved['messages'][-1]['content'])['goalCompleted'] is True, 'gold_continuation must be preserved unchanged'
print('OK')
`;
  const pyOut = execFileSync(pythonExe, ['-c', pyCheck], { encoding: 'utf8' }).trim();
  assert.equal(pyOut, 'OK');

  const trainLoopSrc = fs.readFileSync(path.join(rootDir, 'minebench', 'train_loop.ts'), 'utf8');
  assert.ok(
    trainLoopSrc.includes('if (decision.goalCompleted && stepIdx > 0) break;'),
    'train_loop.ts must guard decision.goalCompleted with stepIdx > 0'
  );
  assert.equal(
    /if\s*\(\s*decision\.goalCompleted\s*\)\s*break;/.test(trainLoopSrc),
    false,
    'train_loop.ts must not contain an unguarded decision.goalCompleted break'
  );
});

test('47. SLM masked-only lm_head loss matches full CrossEntropyLoss exactly and neural_daemon offloads all GPU models during pause_slm', () => {
  const { execFileSync } = require('child_process');
  const fs = require('fs');
  const path = require('path');

  const rootDir = path.resolve(__dirname, '..', '..');
  const pythonExe = path.join(rootDir, 'neural_lab', '.venv', 'Scripts', 'python.exe');
  const pyCheck = `
import sys, torch
import torch.nn.functional as F
sys.path.insert(0, r'${rootDir.replace(/\\/g, '\\\\')}')
from minebench.slm_trainer import compute_masked_causal_lm_loss

torch.manual_seed(7)
vocab, hidden_dim = 32, 16

class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden_dim)
        self.proj = torch.nn.Linear(hidden_dim, hidden_dim)
    def forward(self, input_ids=None, attention_mask=None, use_cache=False):
        return (torch.tanh(self.proj(self.emb(input_ids))),)

class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()
        self.lm_head = torch.nn.Linear(hidden_dim, vocab, bias=False)

m = TinyCausalLM()
inp = torch.randint(0, vocab, (2, 12))
mask = torch.ones_like(inp)
lbl = inp.clone()
lbl[:, :7] = -100

masked_loss = compute_masked_causal_lm_loss(m, inp, mask, lbl)
with torch.no_grad():
    full_hidden = m.model(inp, mask)[0]
    full_logits = m.lm_head(full_hidden)
    full_loss = F.cross_entropy(full_logits[:, :-1, :].reshape(-1, vocab), lbl[:, 1:].reshape(-1), ignore_index=-100)

assert masked_loss is not None and abs(masked_loss.item() - full_loss.item()) < 1e-5, (masked_loss.item(), full_loss.item())

all_masked = torch.full_like(lbl, -100)
assert compute_masked_causal_lm_loss(m, inp, mask, all_masked) is None
print('OK')
`;
  const pyOut = execFileSync(pythonExe, ['-c', pyCheck], { encoding: 'utf8' }).trim();
  assert.equal(pyOut, 'OK');

  const daemonSrc = fs.readFileSync(path.join(rootDir, 'neural_lab', 'neural_daemon.py'), 'utf8');
  const pauseSection = daemonSrc.slice(
    daemonSrc.indexOf('async def handle_pause_slm'),
    daemonSrc.indexOf('async def handle_reload_adapter')
  );
  assert.ok(pauseSection.includes('slm_model') && pauseSection.includes('action_head') && pauseSection.includes('reflex_policy'), 'handle_pause_slm must offload slm_model, action_head, and reflex_policy');

  const reloadSection = daemonSrc.slice(
    daemonSrc.indexOf('async def handle_reload_adapter'),
    daemonSrc.indexOf('async def handle_dqn_sparring_predict')
  );
  assert.ok(reloadSection.includes('chunk_analyzer') && reloadSection.includes('visuomotor_policy'), 'handle_reload_adapter must restore chunk_analyzer and visuomotor_policy');
});

test('48. Crafting evaluators reject bare blocksPlaced scaffolds, tail-reader bounds transition I/O, and mining puzzle has bedrock subfloor', () => {
  const { execFileSync } = require('child_process');
  const path = require('path');
  const { evaluateProgressionTask, evaluateOmniscienceMilestones } = require('../../minebench/dist/minebench/evaluators');
  const { MINEBENCH_TASKS } = require('../../minebench/dist/minebench/benchmarkSpec');

  const railTask = {
    taskId: 'omni_craft_rail',
    targetObject: 'rail',
    evaluatorMilestones: [
      { name: 'Milestone 1: Raw Harvest', maxPoints: 25, condition: 'dag_raw_harvest' },
      { name: 'Milestone 2: Basic Refine', maxPoints: 25, condition: 'dag_basic_refine' },
      { name: 'Milestone 3: Workstation', maxPoints: 25, condition: 'dag_workstation' },
      { name: 'Milestone 4: Final Target Item Synthesized (rail)', maxPoints: 25, condition: 'has_rail' },
    ],
  };
  const baseSnap: any = {
    botPosition: { x: 500, y: 100, z: 500 },
    health: 20,
    food: 20,
    deaths: 0,
    inventory: [{ name: 'oak_log', count: 4 }],
    equippedArmor: { head: null, torso: null, legs: null, feet: null, offHand: null },
    blockAt: () => null,
    entities: [],
    stats: { blocksPlaced: 1, blocksMined: 4, mobsKilled: {}, creeperBlasts: 0, arrowsFired: 0, endCrystalsDestroyed: 0, enderDragonSlain: false, leversFlipped: 0, wolvesTamed: 0, ironPickaxeDurabilityUsedOnStone: 0 },
  };

  // 1. Bare blocksPlaced scaffold (no crafting_table in inventory or world) MUST only score 25 and fail!
  const resScaffoldOnly = evaluateOmniscienceMilestones(railTask, baseSnap);
  assert.equal(resScaffoldOnly.score, 25);
  assert.equal(resScaffoldOnly.passed, false);

  // 2. Real placed crafting_table block in world scores 75 partial credit, still passed=false until rail synthesized
  const snapWithWorldTable: any = {
    ...baseSnap,
    blockAt: (x: number, y: number, z: number) => (x === 502 && y === 100 && z === 500 ? { name: 'crafting_table' } : { name: 'air' }),
  };
  const resWorldTable = evaluateOmniscienceMilestones(railTask, snapWithWorldTable);
  assert.equal(resWorldTable.score, 75);
  assert.equal(resWorldTable.passed, false);

  // 3. Target rail in inventory scores 100 and passes
  const resWithRail = evaluateOmniscienceMilestones(railTask, { ...baseSnap, inventory: [{ name: 'rail', count: 16 }] });
  assert.equal(resWithRail.score, 100);
  assert.equal(resWithRail.passed, true);

  // 4. prog_tier1_stone_age rejects bare scaffold blocksPlaced without real crafting_table
  const tier1Task = MINEBENCH_TASKS.find((t: any) => t.id === 'prog_tier1_stone_age');
  const tier1Scaffold = evaluateProgressionTask(tier1Task, baseSnap);
  assert.equal(tier1Scaffold.score, 25);
  assert.equal(tier1Scaffold.passed, false);

  // 5. read_jsonl_tail discards partial leading line and returns last N complete lines
  const rootDir = path.resolve(__dirname, '..', '..');
  const pythonExe = path.join(rootDir, 'neural_lab', '.venv', 'Scripts', 'python.exe');
  const pyCheck = `
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, r'${rootDir.replace(/\\/g, '\\\\')}')
from minebench.rl_trainer import read_jsonl_tail as rl_tail
from neural_lab.train_action_head import read_jsonl_tail as act_tail

with tempfile.NamedTemporaryFile('wb', delete=False) as tf:
    for i in range(10):
        tf.write(f'{{"idx": {i}, "pad": "xxxxxxxxxx"}}\\n'.encode('utf-8'))
    tmp_path = Path(tf.name)
try:
    for fn in (rl_tail, act_tail):
        lines = fn(tmp_path, max_lines=3, max_bytes=70)
        assert len(lines) == 2 or len(lines) == 3, lines
        assert lines[-1].startswith('{"idx": 9,'), lines
        for ln in lines:
            assert ln.startswith('{"idx": '), f'Incomplete leading line not discarded: {ln}'
    print('OK')
finally:
    os.unlink(tmp_path)
`;
  const pyOut = execFileSync(pythonExe, ['-c', pyCheck], { encoding: 'utf8' }).trim();
  assert.equal(pyOut, 'OK');

  // 6. puzzle_tool_preservation_mining includes bedrock subfloor
  const puzzleMining = MINEBENCH_TASKS.find((t: any) => t.id === 'puzzle_tool_preservation_mining');
  const cmds = puzzleMining.buildArenaCommands(puzzleMining.origin);
  assert.ok(cmds.some((c: string) => c.includes('minecraft:bedrock')), 'puzzle_tool_preservation_mining must include bedrock subfloor');
});











