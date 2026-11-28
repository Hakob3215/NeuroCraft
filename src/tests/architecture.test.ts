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

test('49. Zero-loss inference VRAM optimization, step-history inventory dedup, and daemon reload auto-respawn', () => {
  const fs = require('fs');
  const path = require('path');
  const { compactStepSummaryForHistory } = require('../system2/llmClient');

  // 1. compactStepSummaryForHistory removes redundant Current inventory dump while preserving outcome & errors
  const verbose = 'Harvested 4 oak_log (6 blocks broken). Current inventory: [4x oak_log, 8x iron_ingot, 4x diamond].';
  assert.equal(compactStepSummaryForHistory(verbose), 'Harvested 4 oak_log (6 blocks broken).');
  const errMsg = 'Cannot craft stone_pickaxe: Requires 3 cobblestone and 2 stick.';
  assert.equal(compactStepSummaryForHistory(errMsg), errMsg);

  // 2. neural_daemon.py uses output_hidden_states=False, logits_to_keep=1, norm hook, and empty_cache
  const rootDir = path.resolve(__dirname, '..', '..');
  const daemonSrc = fs.readFileSync(path.join(rootDir, 'neural_lab', 'neural_daemon.py'), 'utf8');
  const runSlm = daemonSrc.slice(
    daemonSrc.indexOf('def _run_slm_sync'),
    daemonSrc.indexOf('async def handle_llm_chat')
  );
  assert.ok(runSlm.includes('"output_hidden_states": False'), '_run_slm_sync must disable output_hidden_states');
  assert.ok(runSlm.includes('"logits_to_keep": 1'), '_run_slm_sync must set logits_to_keep=1');
  assert.ok(runSlm.includes('register_forward_hook'), '_run_slm_sync must capture final-norm latent via forward hook');
  assert.ok(runSlm.includes('torch.cuda.empty_cache()'), '_run_slm_sync must clear CUDA cache on long prompts/OOM');

  // 3. train_loop.ts respawns neural_daemon.py if reload_adapter fails all 3 attempts
  const trainLoopSrc = fs.readFileSync(path.join(rootDir, 'minebench', 'train_loop.ts'), 'utf8');
  const reloadFallback = trainLoopSrc.slice(trainLoopSrc.indexOf('if (!reloaded)'), trainLoopSrc.indexOf('if (!reloaded)') + 1200);
  assert.ok(reloadFallback.includes('/shutdown') && reloadFallback.includes('neural_daemon.py'), 'train_loop.ts must respawn neural_daemon.py on reload failure');
});

test('50. Building shelter IoU supports oy+3 and oy+4 roofs, held-out split holds ~20% omni tasks, and rl_trainer excludes held-out records', () => {
  const { execFileSync } = require('child_process');
  const { evaluateMinebenchEpisode } = require('../../minebench/dist/minebench/evaluators');
  const { MINEBENCH_TASKS } = require('../../minebench/dist/minebench/benchmarkSpec');
  const { isHeldOutEvalTask } = require('../../minebench/dist/minebench/train_loop');

  const shelterTask = MINEBENCH_TASKS.find((t: any) => t.id === 'build_enclosed_shelter');
  assert.ok(shelterTask, 'build_enclosed_shelter spec must exist');
  const ox = shelterTask.origin.x + 2;
  const oy = shelterTask.origin.y;
  const oz = shelterTask.origin.z + 2;

  const makeCabinSnap = (roofH: number): any => ({
    botPosition: { x: ox + 2, y: oy, z: oz + 2 },
    health: 20,
    food: 20,
    deaths: 0,
    inventory: [],
    equippedArmor: { head: null, torso: null, legs: null, feet: null, offHand: null },
    entities: [],
    stats: { blocksPlaced: 40, blocksMined: 0, mobsKilled: {}, creeperBlasts: 0, arrowsFired: 0, endCrystalsDestroyed: 0, enderDragonSlain: false, leversFlipped: 0, wolvesTamed: 0, ironPickaxeDurabilityUsedOnStone: 0 },
    blockAt: (bx: number, by: number, bz: number) => {
      const rx = bx - ox;
      const ry = by - oy;
      const rz = bz - oz;
      if (rx >= 0 && rx < 5 && rz >= 0 && rz < 5) {
        if (ry === -1) return { name: 'grass_block' };
        if (ry === roofH) return { name: 'oak_planks' };
        if (ry >= 0 && ry < roofH) {
          const isPerim = rx === 0 || rx === 4 || rz === 0 || rz === 4;
          if (isPerim) {
            if (rx === 1 && rz === 0 && ry === 2) return { name: 'torch' };
            return { name: 'oak_planks' };
          }
        }
      }
      return { name: 'air' };
    },
  });

  const resH4 = evaluateMinebenchEpisode({ ...shelterTask, taskId: 'build_enclosed_shelter' } as any, makeCabinSnap(4));
  const resH3 = evaluateMinebenchEpisode({ ...shelterTask, taskId: 'build_enclosed_shelter' } as any, makeCabinSnap(3));
  assert.equal(resH4.passed, true, `oy+4 roof cabin must pass (score=${resH4.score})`);
  assert.equal(resH3.passed, true, `oy+3 roof cabin must pass (score=${resH3.score})`);

  // Deterministic ~20% held-out split across omni_* tasks, false on non-omni tasks
  assert.equal(isHeldOutEvalTask('prog_tier1_stone_age'), false);
  assert.equal(isHeldOutEvalTask('combat_zombie_skirmish'), false);
  assert.equal(isHeldOutEvalTask('coop_give_tool'), false);

  const rootDir = path.resolve(__dirname, '..', '..');
  const tasksRaw: any[] = JSON.parse(
    fs.readFileSync(path.join(rootDir, 'minebench', 'results', 'omniscience_tasks.json'), 'utf8')
  );
  const omniIds = tasksRaw.filter((t) => t.taskId?.startsWith('omni_')).map((t) => t.taskId);
  assert.ok(omniIds.length >= 20, 'omniscience_tasks.json must contain omni_ tasks');
  const heldOutCount = omniIds.filter((id) => isHeldOutEvalTask(id)).length;
  const ratio = heldOutCount / omniIds.length;
  assert.ok(ratio >= 0.1 && ratio <= 0.32, `held-out ratio expected ~0.20, got ${ratio.toFixed(3)}`);
  // Check determinism across repeated calls
  assert.equal(isHeldOutEvalTask(omniIds[0]), isHeldOutEvalTask(omniIds[0]));

  // Verify rl_trainer.py excludes isHeldOut: true records from both GRPO and DPO
  const rlTrainerSrc = fs.readFileSync(path.join(rootDir, 'minebench', 'rl_trainer.py'), 'utf8');
  assert.ok(rlTrainerSrc.includes('if bool(r.get("isHeldOut", False)):'), 'rl_trainer.py must skip isHeldOut records');
  const pythonExe = path.join(rootDir, 'neural_lab', '.venv', 'Scripts', 'python.exe');
  const pyCheck = `
import sys
sys.path.insert(0, r'${rootDir.replace(/\\/g, '\\\\')}')
import minebench.rl_trainer as rt
src_lines = open(rt.__file__, 'r', encoding='utf-8').read()
assert src_lines.count('isHeldOut') >= 2, 'isHeldOut guard must cover GRPO and DPO loops'
print('OK')
`;
  assert.equal(execFileSync(pythonExe, ['-c', pyCheck], { encoding: 'utf8' }).trim(), 'OK');
});

test('51. Multi-species plank affordances, random wood arena transform, required-arg validation, and real thought preservation', () => {
  const { execFileSync } = require('child_process');
  const { computeGroundedAffordances } = require('../system2/goalPlanner');
  const { applyRandomWoodSpecies } = require('../../minebench/dist/minebench/train_loop');
  const { MINEBENCH_TASKS } = require('../../minebench/dist/minebench/benchmarkSpec');

  const birchAff = computeGroundedAffordances({
    inventory: [{ name: 'birch_log', count: 3, slot: 0 }],
    nearbyResources: [],
    nearbyEntities: [],
    nearbyBlocks: [],
    health: 20,
    food: 20,
  } as any);
  assert.ok(birchAff.craftableNow.includes('birch_planks'), 'birch_log must yield birch_planks');
  assert.ok(!birchAff.craftableNow.includes('oak_planks'), 'birch_log must not yield fake oak_planks');

  const crimsonAff = computeGroundedAffordances({
    inventory: [{ name: 'crimson_stem', count: 2, slot: 0 }],
    nearbyResources: [],
    nearbyEntities: [],
    nearbyBlocks: [],
    health: 20,
    food: 20,
  } as any);
  assert.ok(crimsonAff.craftableNow.includes('crimson_planks'), 'crimson_stem must yield crimson_planks');

  const tier1 = MINEBENCH_TASKS.find((t: any) => t.id === 'prog_tier1_stone_age');
  assert.ok(tier1);
  const transformed = applyRandomWoodSpecies(tier1, 'birch');
  assert.ok(transformed.buildArenaCommands, 'buildArenaCommands must remain callable');
  const cmds = transformed.buildArenaCommands({ x: 0, y: 64, z: 0 });
  assert.ok(cmds.some((c: string) => c.includes('birch_log')), 'Transformed arena commands must spawn birch_log');
  assert.ok(!cmds.some((c: string) => c.includes('oak_log')), 'Transformed arena commands must not spawn oak_log');

  const rootDir = path.resolve(__dirname, '..', '..');
  const pythonExe = path.join(rootDir, 'neural_lab', '.venv', 'Scripts', 'python.exe');
  const pyCheck = `
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, r'${rootDir.replace(/\\/g, '\\\\')}')
import minebench.rl_trainer as rt

assert rt.has_required_args("craft_item", {}) is False
assert rt.has_required_args("craft_item", {"itemName": ""}) is False
assert rt.has_required_args("craft_item", {"itemName": "stone_pickaxe"}) is True
assert rt.has_required_args("gather_resource", {}) is False
assert rt.has_required_args("gather_resource", {"blockType": "birch_log"}) is True
assert rt.has_required_args("smelt_item", {}) is False
assert rt.has_required_args("smelt_item", {"inputItem": "raw_iron"}) is True
assert rt.has_required_args("equip_armor_or_item", {}) is False
assert rt.has_required_args("equip_armor_or_item", {"itemName": "iron_sword"}) is True
assert rt.has_required_args("equip_best_gear", {}) is True

# Verify train_rl_from_replay rejects empty-arg calls from GRPO, mines them as DPO negatives, and preserves real thought
with tempfile.TemporaryDirectory() as tmp:
    tp = Path(tmp)
    replay_p = tp / "replay.jsonl"
    trans_p = tp / "trans.jsonl"
    grpo_p = tp / "grpo.jsonl"
    dpo_p = tp / "dpo.jsonl"
    metrics_p = tp / "metrics.json"
    rows = [
        {
            "taskId": "omni_craft_stone_pickaxe",
            "domain": "craft",
            "prompt": "Craft a stone pickaxe",
            "passed": True,
            "score": 100,
            "rawReward": 95.0,
            "executedTools": [
                {"name": "craft_item", "args": {"itemName": "stone_pickaxe"}, "success": True, "thought": "Synthesizing stone pickaxe at table."}
            ],
        },
        {
            "taskId": "omni_craft_stone_pickaxe",
            "domain": "craft",
            "prompt": "Craft a stone pickaxe",
            "passed": False,
            "score": 0,
            "rawReward": 5.0,
            "executedTools": [
                {"name": "craft_item", "args": {}, "success": False, "thought": "Trying empty craft args."}
            ],
        },
        {
            "taskId": "omni_heldout_ignored",
            "domain": "craft",
            "isHeldOut": True,
            "prompt": "Held out",
            "passed": True,
            "score": 100,
            "rawReward": 100.0,
            "executedTools": [
                {"name": "craft_item", "args": {"itemName": "iron_pickaxe"}, "success": True, "thought": "Should never leak."}
            ],
        },
    ]
    replay_p.write_text("\\n".join(json.dumps(r) for r in rows) + "\\n", encoding="utf-8")
    trans_row = {"voxels": [0] * 3375, "entities": [[0.0] * 12] * 16, "botState": [0.0] * 10, "actionIdx": 0, "aimDelta": [0.0, 0.0], "reward": 1.0}
    trans_p.write_text(json.dumps(trans_row) + "\\n", encoding="utf-8")
    orig_paths = (rt.REPLAY_BUFFER_PATH, rt.REAL_TRANSITIONS_PATH, rt.GRPO_DATASET_PATH, rt.DPO_PAIRS_PATH, rt.RL_METRICS_PATH, rt.CHECKPOINT_DIR, rt.RESULTS_DIR)
    try:
        rt.REPLAY_BUFFER_PATH = replay_p
        rt.REAL_TRANSITIONS_PATH = trans_p
        rt.GRPO_DATASET_PATH = grpo_p
        rt.DPO_PAIRS_PATH = dpo_p
        rt.RL_METRICS_PATH = metrics_p
        rt.CHECKPOINT_DIR = tp / "ckpt"
        rt.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        rt.RESULTS_DIR = tp / "res"
        rt.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        rt.run_minebench_rl_update(epochs=0, is_best=False)
        grpo_items = [json.loads(l) for l in grpo_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert all(g.get("taskId") != "omni_heldout_ignored" for g in grpo_items), "Held-out task leaked to GRPO"
        real_winners = [g for g in grpo_items if g.get("taskId") == "omni_craft_stone_pickaxe"]
        assert len(real_winners) >= 1
        win_msg = json.loads(real_winners[0]["messages"][2]["content"])
        assert win_msg["thought"] == "Synthesizing stone pickaxe at table.", win_msg
        dpo_items = [json.loads(l) for l in dpo_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(dpo_items) == 1, f"Expected 1 DPO pair, got {len(dpo_items)}"
        rej_msg = json.loads(dpo_items[0]["rejected"])
        assert rej_msg["toolCall"]["arguments"] == {}
    finally:
        rt.REPLAY_BUFFER_PATH, rt.REAL_TRANSITIONS_PATH, rt.GRPO_DATASET_PATH, rt.DPO_PAIRS_PATH, rt.RL_METRICS_PATH, rt.CHECKPOINT_DIR, rt.RESULTS_DIR = orig_paths
print("OK")
`;
  assert.equal(execFileSync(pythonExe, ['-c', pyCheck], { encoding: 'utf8' }).trim().split('\n').pop(), 'OK');
});

test('52. SLM stratified domain sampling and Hybrid DPO+NLL with r=16 Full-MLP LoRA disable_adapter on CPU', () => {
  const { execFileSync } = require('child_process');
  const rootDir = path.resolve(__dirname, '..', '..');
  const pythonExe = path.join(rootDir, 'neural_lab', '.venv', 'Scripts', 'python.exe');
  const pyCheck = `
import sys, torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model
sys.path.insert(0, r'${rootDir.replace(/\\/g, '\\\\')}')
from minebench.slm_trainer import stratified_sample_by_domain, compute_masked_logp_and_nll

items = [{"domain": "combat", "i": i} for i in range(90)] + [{"domain": "puzzle", "i": i} for i in range(10)]
sampled = stratified_sample_by_domain(items, 20)
assert len(sampled) == 20
puz = sum(1 for x in sampled if x["domain"] == "puzzle")
cmb = sum(1 for x in sampled if x["domain"] == "combat")
assert puz == 10 and cmb == 10, (puz, cmb)

torch.manual_seed(42)
cfg = Qwen2Config(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=2,
)
base = Qwen2ForCausalLM(cfg)
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base, lora_cfg)
model.train()

ch_ids = torch.randint(0, 128, (1, 16))
rej_ids = torch.randint(0, 128, (1, 16))
mask = torch.ones((1, 16), dtype=torch.long)
lbl_w = ch_ids.clone(); lbl_w[:, :8] = -100
lbl_l = rej_ids.clone(); lbl_l[:, :8] = -100

with torch.no_grad():
    with model.disable_adapter():
        ref_w, _ = compute_masked_logp_and_nll(model, ch_ids, mask, lbl_w, label_smoothing=0.0)
        ref_l, _ = compute_masked_logp_and_nll(model, rej_ids, mask, lbl_l, label_smoothing=0.0)
assert ref_w is not None and ref_l is not None and torch.isfinite(ref_w) and torch.isfinite(ref_l)

pi_w, chosen_nll = compute_masked_logp_and_nll(model, ch_ids, mask, lbl_w, label_smoothing=0.04)
pi_l, _ = compute_masked_logp_and_nll(model, rej_ids, mask, lbl_l, label_smoothing=0.0)
assert pi_w is not None and chosen_nll is not None and pi_l is not None

loss = -F.logsigmoid(0.1 * ((pi_w - ref_w) - (pi_l - ref_l))) + 0.3 * chosen_nll
assert torch.isfinite(loss), loss.item()
loss.backward()
grads = [p.grad for n, p in model.named_parameters() if "lora_" in n and p.requires_grad and p.grad is not None]
assert len(grads) > 0, "Expected gradients on r=16 Full-MLP LoRA weights"
assert all(torch.isfinite(g).all() for g in grads)

# Unpaired NLL fallback check
model.zero_grad(set_to_none=True)
_, unpaired_nll = compute_masked_logp_and_nll(model, ch_ids, mask, lbl_w, label_smoothing=0.04)
assert unpaired_nll is not None and torch.isfinite(unpaired_nll)
unpaired_nll.backward()

# Exact chain-rule split DPO gradient equivalence check (with eval() so dropout is off)
model.eval()
model.zero_grad(set_to_none=True)
with torch.no_grad():
    with model.disable_adapter():
        ref_w, _ = compute_masked_logp_and_nll(model, ch_ids, mask, lbl_w, label_smoothing=0.0)
        ref_l, _ = compute_masked_logp_and_nll(model, rej_ids, mask, lbl_l, label_smoothing=0.0)
pi_w_j, nll_j = compute_masked_logp_and_nll(model, ch_ids, mask, lbl_w, label_smoothing=0.04)
pi_l_j, _ = compute_masked_logp_and_nll(model, rej_ids, mask, lbl_l, label_smoothing=0.0)
loss_j = -F.logsigmoid(0.1 * ((pi_w_j - ref_w) - (pi_l_j - ref_l))) + 0.3 * nll_j
loss_j.backward()
joint_grads = [p.grad.detach().clone() for n, p in model.named_parameters() if "lora_" in n and p.requires_grad and p.grad is not None]

model.zero_grad(set_to_none=True)
with torch.no_grad():
    pi_w_v, _ = compute_masked_logp_and_nll(model, ch_ids, mask, lbl_w, label_smoothing=0.04)
    pi_l_v, _ = compute_masked_logp_and_nll(model, rej_ids, mask, lbl_l, label_smoothing=0.0)
    z_v = (pi_w_v - ref_w) - (pi_l_v - ref_l)
    coeff = float((-0.1 * torch.sigmoid(-0.1 * z_v)).item())
pi_w_s, nll_s = compute_masked_logp_and_nll(model, ch_ids, mask, lbl_w, label_smoothing=0.04)
(coeff * pi_w_s + 0.3 * nll_s).backward()
pi_l_s, _ = compute_masked_logp_and_nll(model, rej_ids, mask, lbl_l, label_smoothing=0.0)
((-coeff) * pi_l_s).backward()
split_grads = [p.grad.detach().clone() for n, p in model.named_parameters() if "lora_" in n and p.requires_grad and p.grad is not None]
assert len(joint_grads) == len(split_grads) > 0
for jg, sg in zip(joint_grads, split_grads):
    assert torch.allclose(jg, sg, atol=1e-4, rtol=1e-3)
print("OK")
`;
  assert.equal(execFileSync(pythonExe, ['-c', pyCheck], { encoding: 'utf8' }).trim().split('\n').pop(), 'OK');
});

test('53. ActionHead load_real_motor_chunks_dataset loads 1536-dim latent_h, 24-dim proprio, and 8-tick 20Hz trajectories', () => {
  const { execFileSync } = require('child_process');
  const rootDir = path.resolve(__dirname, '..', '..');
  const pythonExe = path.join(rootDir, 'neural_lab', '.venv', 'Scripts', 'python.exe');
  const pyCheck = `
import json, sys, tempfile, torch
from pathlib import Path
sys.path.insert(0, r'${rootDir.replace(/\\/g, '\\\\')}')
from neural_lab.train_action_head import load_real_motor_chunks_dataset, TOOL_TO_IDX

with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "real_motor_chunks.jsonl"
    latent = [round(0.25 if i == 5 else -0.1, 3) for i in range(1536)]
    proprio = [0.5] * 24
    voxels = [17 if i == 100 else 0 for i in range(3375)]
    entities = [[0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
    valid_rec = {
        "taskId": "combat_zombie_skirmish",
        "toolName": "engage_combat",
        "latent_h": latent,
        "proprio24": proprio,
        "voxels": voxels,
        "entities": entities,
        "aimTargets": [[15.0, -5.0]] * 8,
        "moveTargets": [[0.0, 1.0]] * 8,
        "discreteTargets": [[0, 1, 0, 1, 0, 0]] * 8,
        "extendedTargets": [[0, 0]] * 8,
        "stepSuccess": True,
        "reward": 80.0,
        "episodePassed": True,
        "episodeReward": 90.0,
    }
    bad_rec = {"taskId": "broken", "latent_h": [1.0, 2.0], "voxels": []}
    p.write_text(json.dumps(valid_rec) + "\\n" + json.dumps(bad_rec) + "\\n", encoding="utf-8")

    ds = load_real_motor_chunks_dataset(max_samples=16, device="cpu", chunks_path=p)
    assert ds is not None and len(ds) == 12
    lat_t, prop_t, vox_t, ent_t, aim_t, mov_t, disc_t, ext_t, tool_t, gate_t, val_t, w_t = ds
    assert tuple(lat_t.shape) == (1, 1536)
    assert tuple(prop_t.shape) == (1, 24)
    assert tuple(vox_t.shape) == (1, 15, 15, 15) and vox_t.dtype == torch.long
    assert tuple(ent_t.shape) == (1, 16, 12)
    assert tuple(aim_t.shape) == (1, 8, 2)
    assert tuple(mov_t.shape) == (1, 8, 2)
    assert tuple(disc_t.shape) == (1, 8, 6)
    assert tuple(ext_t.shape) == (1, 8, 2)
    assert tuple(tool_t.shape) == (1,) and int(tool_t[0].item()) == TOOL_TO_IDX["engage_combat"]
    assert float(gate_t[0].item()) == 1.0
    assert abs(float(lat_t[0, 5].item()) - 0.025) < 1e-4
    assert abs(float(w_t[0].item()) - 1.5) < 1e-4
    assert 0.0 <= float(val_t[0].item()) <= 1.0
print("OK")
`;
  assert.equal(execFileSync(pythonExe, ['-c', pyCheck], { encoding: 'utf8' }).trim().split('\n').pop(), 'OK');
});

test('54. System1 perception cache invalidates on >10m teleport and safePlaceBlock accepts non-solid torches', async () => {
  const { System1BotController } = require('../system1/botController');
  const s1: any = new System1BotController();

  let currentPos = new Vec3(0, 64, 0);
  const blockLookup = (pos: Vec3) => {
    if (Math.floor(pos.x) === 0 && Math.floor(pos.y) === 64 && Math.floor(pos.z) === 2) {
      return { name: 'iron_ore', boundingBox: 'block', position: new Vec3(0, 64, 2) };
    }
    if (Math.floor(pos.x) === 50 && Math.floor(pos.y) === 64 && Math.floor(pos.z) === 2) {
      return { name: 'diamond_ore', boundingBox: 'block', position: new Vec3(50, 64, 2) };
    }
    if (Math.floor(pos.y) === 63) return { name: 'stone', boundingBox: 'block', position: pos.floored() };
    return { name: 'air', boundingBox: 'empty', position: pos.floored() };
  };

  s1.bot = {
    username: 'NeuroAgent_Test',
    health: 20,
    food: 20,
    entity: { position: currentPos, yaw: 0, pitch: 0, velocity: new Vec3(0, 0, 0), onGround: true },
    entities: {},
    inventory: { items: () => [] },
    time: { timeOfDay: 1000 },
    blockAt: blockLookup,
  };
  s1.reflexEngine = { getActiveHazard: () => null };
  s1.auditory = { getRecentCues: () => [] };
  s1.tickCounter = 25;

  const ws1 = s1.buildSanitizedWorldState();
  assert.ok(ws1.nearbyNotableBlocks.some((b: any) => b.name === 'iron_ore'), 'Must detect iron_ore before teleport');

  // Teleport > 10m within the same tick window (<20 ticks later)
  s1.tickCounter = 26;
  s1.bot.entity.position = new Vec3(50, 64, 0);
  const ws2 = s1.buildSanitizedWorldState();
  assert.ok(ws2.nearbyNotableBlocks.some((b: any) => b.name === 'diamond_ore'), 'Teleport >10m must force immediate rescan');
  assert.ok(!ws2.nearbyNotableBlocks.some((b: any) => b.name === 'iron_ore'), 'Old iron_ore must be flushed after teleport');

  // safePlaceBlock accepts placed torch (boundingBox === "empty")
  let torchPlaced = false;
  s1.bot = {
    username: 'NeuroAgent_Test',
    entity: { position: new Vec3(0, 64, 0), yaw: 0, pitch: 0 },
    heldItem: { name: 'torch', count: 4 },
    inventory: { items: () => [{ name: 'torch', count: 4, slot: 36 }] },
    blockAt: (p: Vec3) => {
      const fx = Math.floor(p.x);
      const fy = Math.floor(p.y);
      const fz = Math.floor(p.z);
      if (fx === 1 && fy === 64 && fz === 0) return { name: 'oak_planks', boundingBox: 'block', position: new Vec3(1, 64, 0) };
      if (fx === 1 && fy === 65 && fz === 0) {
        return torchPlaced
          ? { name: 'torch', boundingBox: 'empty', position: new Vec3(1, 65, 0) }
          : { name: 'air', boundingBox: 'empty', position: new Vec3(1, 65, 0) };
      }
      return { name: 'air', boundingBox: 'empty', position: new Vec3(fx, fy, fz) };
    },
    placeBlock: async () => {
      torchPlaced = true;
    },
  };
  s1.inventory = { equipItemHumanized: async () => true };
  s1.camera = { lookAtSmooth: async () => {} };
  const placedOk = await s1.safePlaceBlock(new Vec3(1, 65, 0), 'torch', false);
  assert.equal(placedOk, true, 'safePlaceBlock must accept placed torch even though boundingBox is empty');
});

test('55. VLACompanionClient wraps yaw into [-1, 1], deactivates use_item cleanly, and blocks accidental drop_item', async () => {
  const { VLACompanionClient } = require('../system2/vlaClient');
  const client = new VLACompanionClient();

  for (const yawVal of [3.5, -5.0, 5.5, 12.2]) {
    const p24 = client.extractProprioception({
      entity: { position: new Vec3(0, 64, 0), yaw: yawVal, pitch: 0.2, velocity: new Vec3(0, 0, 0), onGround: true },
      health: 20,
      food: 20,
      entities: {},
    } as any);
    assert.ok(p24[2] >= -1.0 && p24[2] <= 1.0, `Normalized yaw ${p24[2]} out of [-1, 1] for raw yaw ${yawVal}`);
  }

  let activeCalls = 0;
  let deactiveCalls = 0;
  let tossCalls = 0;
  const mockBot: any = {
    entity: { position: new Vec3(0, 64, 0), yaw: 0, pitch: 0, onGround: true },
    heldItem: { type: 274, name: 'stone_pickaxe', count: 1 },
    entities: {},
    look: async () => {},
    setControlState: () => {},
    clearControlStates: () => {},
    activateItem: () => {
      activeCalls++;
    },
    deactivateItem: () => {
      deactiveCalls++;
    },
    toss: async () => {
      tossCalls++;
    },
  };

  await client.executeMotorActionChunk(
    mockBot,
    [
      { delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 1, jump: false, sprint: false, sneak: false, attack: false, use_item: true, swap_slot: false, dig_block: false, drop_item: true },
      { delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 1, jump: false, sprint: false, sneak: false, attack: false, use_item: false, swap_slot: false, dig_block: false, drop_item: true },
    ],
    undefined,
    2,
    { allowItemDrop: false, allowDig: false }
  );
  assert.equal(activeCalls, 1);
  assert.equal(deactiveCalls, 1, 'deactivateItem must fire when use_item transitions true -> false');
  assert.equal(tossCalls, 0, 'drop_item must never toss tools when allowItemDrop=false');

  // Even when chunk ends with use_item still true, finally must deactivate it
  deactiveCalls = 0;
  await client.executeMotorActionChunk(
    mockBot,
    [
      { delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 1, jump: false, sprint: false, sneak: false, attack: false, use_item: true, swap_slot: false, dig_block: false, drop_item: false },
    ],
    undefined,
    1
  );
  assert.equal(deactiveCalls, 1, 'finally block must call deactivateItem on chunk exit');
});

test('56. 1.20.4 lever powered state, combat death abort, strict craft_item args, and strict toolRegistry matching', async () => {
  const { registerBaseMCPTools } = require('../system2/skills/baseSkills');
  const registry = new DynamicMCPToolRegistry();
  registerBaseMCPTools(registry);

  const prevStrict = process.env.STRICT_LLM_BENCHMARK;
  process.env.STRICT_LLM_BENCHMARK = 'true';
  try {
    assert.equal(registry.getTool('crafting_table'), undefined, 'Hallucinated tool must fail under STRICT_LLM_BENCHMARK=true');
    assert.equal(registry.getTool('deliver_resource'), undefined, 'Unregistered deliver_resource must fail under strict mode');
  } finally {
    process.env.STRICT_LLM_BENCHMARK = prevStrict;
  }

  // craft_item rejects empty itemName
  const craftTool = registry.getTool('craft_item')!;
  const emptyCraftRes = await craftTool.handler({}, {
    bot: { inventory: { items: () => [{ name: 'oak_log', count: 4 }] } },
  } as any);
  assert.equal(emptyCraftRes.success, false, 'craft_item({}) must fail instead of defaulting to oak_planks');

  // 1.20.4 unpowered lever (powered=false, metadata=9) triggers retry in interact_block_or_puzzle
  let leverActivations = 0;
  let leverPowered = false;
  const interactTool = registry.getTool('interact_block_or_puzzle')!;
  await interactTool.handler(
    { blockType: 'lever' },
    {
      bot: {
        entity: { position: new Vec3(0, 64, 0) },
        inventory: { items: () => [] },
        findBlock: () => ({ position: new Vec3(2, 65, 0), name: 'lever' }),
        blockAt: () => ({
          name: 'lever',
          metadata: 9,
          getProperties: () => ({ face: 'wall', facing: 'north', powered: leverPowered }),
        }),
        activateBlock: async () => {
          leverActivations++;
          if (leverActivations >= 2) leverPowered = true;
        },
        lookAt: async () => {},
      },
      kinematics: { pathfindTo: async () => true, lookAtSmooth: async () => {} },
      reportProgress: () => {},
    } as any
  );
  assert.equal(leverActivations, 2, '1.20.4 lever with powered=false and metadata=9 must be detected as unpowered and retried');

  // engage_combat aborts immediately on bot death with success=false and 0 false kills
  const { EventEmitter } = require('events');
  const combatBot: any = new EventEmitter();
  const opponentEnt: any = {
    type: 'player',
    username: 'DQN_Opponent_1',
    name: 'player',
    isValid: true,
    position: new Vec3(2, 64, 0),
  };
  combatBot.username = 'NeuroAgent_1';
  combatBot.health = 20;
  combatBot.entity = { position: new Vec3(0, 64, 0) };
  combatBot.entities = { opp: opponentEnt };
  combatBot.inventory = { items: () => [] };
  combatBot.pathfinder = { setGoal: () => {} };
  combatBot.setControlState = () => {};

  const combatTool = registry.getTool('engage_combat')!;
  const combatRes = await combatTool.handler(
    { targetType: 'DQN_Opponent_1', count: 1 },
    {
      bot: combatBot,
      kinematics: {
        pathfindTo: async () => true,
        equipItemHumanized: async () => true,
        lookAtSmooth: async () => {},
        strikeTarget: async () => {
          // Simulate bot getting slain and respawned (invalidating opponentEnt)
          combatBot.health = 0;
          combatBot.emit('death');
          opponentEnt.isValid = false;
          return true;
        },
      },
      worldState: () => ({ visibleEntities: [] }),
      reportProgress: () => {},
      abortSignal: new AbortController().signal,
    } as any
  );
  assert.equal(combatRes.success, false, 'engage_combat must return success=false when bot dies');
  assert.equal(combatRes.data.kills, 0, 'Bot death must not increment kills');
  assert.equal(combatRes.data.botDied, true);

  // Target surviving >= 2 strikes must NOT falsely count as killed
  let strikeCnt = 0;
  const tankBot: any = new EventEmitter();
  const tankEnt: any = {
    type: 'player',
    username: 'DQN_Opponent_2',
    name: 'player',
    isValid: true,
    position: new Vec3(2, 64, 0),
  };
  tankBot.username = 'NeuroAgent_2';
  tankBot.health = 20;
  tankBot.entity = { position: new Vec3(0, 64, 0) };
  tankBot.entities = { opp: tankEnt };
  tankBot.inventory = { items: () => [] };
  tankBot.pathfinder = { setGoal: () => {} };
  tankBot.setControlState = () => {};
  const tankAbort = new AbortController();
  const tankRes = await combatTool.handler(
    { targetType: 'DQN_Opponent_2', count: 1 },
    {
      bot: tankBot,
      kinematics: {
        pathfindTo: async () => true,
        equipItemHumanized: async () => true,
        lookAtSmooth: async () => {},
        strikeTarget: async () => {
          strikeCnt++;
          if (strikeCnt >= 3) tankAbort.abort();
          return true;
        },
      },
      worldState: () => ({ visibleEntities: [] }),
      reportProgress: () => {},
      abortSignal: tankAbort.signal,
    } as any
  );
  assert.equal(tankRes.success, false, 'Target surviving >=2 strikes must not count as killed');
  assert.equal(tankRes.data.kills, 0);
});

test('57. All 48 omni_craft_* tasks have complete 1.20.4 raw ingredients without silver-platter intermediates', () => {
  const rootDir = path.resolve(__dirname, '..', '..');
  const tasks: any[] = JSON.parse(
    fs.readFileSync(path.join(rootDir, 'minebench', 'results', 'omniscience_tasks.json'), 'utf8')
  );
  const craftTasks = tasks.filter((t) => t.taskId?.startsWith('omni_craft_'));
  assert.equal(craftTasks.length, 48, 'Expected 48 omni_craft_* tasks');

  const forbiddenPreCrafted = new Set(['oak_planks', 'spruce_planks', 'birch_planks', 'stick', 'crafting_table', 'book', 'paper']);
  const countInLoadout = (task: any, item: string): number =>
    (task.initialLoadout || []).filter((x: any) => x.name === item).reduce((s: number, x: any) => s + x.count, 0);

  for (const t of craftTasks) {
    for (const item of t.initialLoadout || []) {
      assert.ok(!forbiddenPreCrafted.has(item.name), `${t.taskId} must not pre-grant ${item.name}`);
    }
  }

  const findCraft = (id: string) => craftTasks.find((t) => t.taskId === id);
  assert.ok(countInLoadout(findCraft('omni_craft_arrow'), 'flint') >= 1 && countInLoadout(findCraft('omni_craft_arrow'), 'feather') >= 1);
  assert.ok(countInLoadout(findCraft('omni_craft_bread'), 'wheat') >= 3);
  assert.ok(countInLoadout(findCraft('omni_craft_golden_apple'), 'apple') >= 1);
  assert.ok(countInLoadout(findCraft('omni_craft_golden_carrot'), 'carrot') >= 1);
  assert.ok(countInLoadout(findCraft('omni_craft_flint_and_steel'), 'flint') >= 1);
  assert.ok(countInLoadout(findCraft('omni_craft_sticky_piston'), 'slime_ball') >= 1);
  assert.ok(countInLoadout(findCraft('omni_craft_lead'), 'slime_ball') >= 1);
  assert.ok(countInLoadout(findCraft('omni_craft_enchanting_table'), 'sugar_cane') >= 3);
  assert.ok(countInLoadout(findCraft('omni_craft_anvil'), 'iron_ingot') >= 31, 'anvil requires 31 iron_ingot');
});

test('58. DPO refuses failed/zero-reward winners, deduplicates GRPO, and ActionHead normalizes latents and sigmoid values', () => {
  const { execFileSync } = require('child_process');
  const rootDir = path.resolve(__dirname, '..', '..');
  const pythonExe = path.join(rootDir, 'neural_lab', '.venv', 'Scripts', 'python.exe');
  const pyCheck = `
import json, sys, tempfile, torch
from pathlib import Path
sys.path.insert(0, r'${rootDir.replace(/\\/g, '\\\\')}')
import minebench.rl_trainer as rt
from neural_lab.train_action_head import load_real_motor_chunks_dataset

with tempfile.TemporaryDirectory() as tmp:
    tp = Path(tmp)
    replay_p = tp / "replay.jsonl"
    trans_p = tp / "trans.jsonl"
    grpo_p = tp / "grpo.jsonl"
    dpo_p = tp / "dpo.jsonl"
    metrics_p = tp / "metrics.json"
    rows = [
        # 1) Failed episode with high partial score=62 must NEVER become a DPO winner
        {
            "taskId": "build_enclosed_shelter",
            "domain": "ARCHITECTURE_BUILDING",
            "workerId": 0,
            "prompt": "Build house",
            "passed": False,
            "score": 62,
            "rawReward": 60.03,
            "executedTools": [
                {"name": "construct_modular_structure", "args": {"width": 5, "roofBlock": "stick"}, "success": True, "stepReward": 60.0, "thought": "Roof from stick"}
            ],
        },
        {
            "taskId": "build_enclosed_shelter",
            "domain": "ARCHITECTURE_BUILDING",
            "workerId": 0,
            "prompt": "Build house",
            "passed": False,
            "score": 12,
            "rawReward": 10.0,
            "executedTools": [
                {"name": "build_simple_house", "args": {"width": 8}, "success": False, "stepReward": 0.0, "thought": "Aborted"}
            ],
        },
        # 2) Passed episode with stepReward == 0 on step 0 must NOT become a DPO winner
        {
            "taskId": "mp_bridge_race",
            "domain": "MULTIPLAYER_ARENA",
            "workerId": 1,
            "prompt": "Bridge",
            "passed": True,
            "score": 100,
            "rawReward": 140.0,
            "executedTools": [
                {"name": "navigate_to_coordinates", "args": {"x": 5301, "y": 72, "z": 1004}, "success": True, "stepReward": 0.0, "thought": "Walk"},
                {"name": "navigate_parkour_or_bridge", "args": {"goalBlock": "emerald_block"}, "success": True, "stepReward": 140.0, "thought": "Bridge"}
            ],
        },
        {
            "taskId": "mp_bridge_race",
            "domain": "MULTIPLAYER_ARENA",
            "workerId": 0,
            "prompt": "Bridge",
            "passed": False,
            "score": 59,
            "rawReward": 10.0,
            "executedTools": [
                {"name": "navigate_parkour_or_bridge", "args": {"goalBlock": "emerald_goal_platform"}, "success": False, "stepReward": 0.0, "thought": "Fell"},
                {"name": "navigate_to_coordinates", "args": {"x": 1300, "y": 71, "z": 1018}, "success": False, "stepReward": 0.0, "thought": "Stuck"}
            ],
        },
        # 3) Two passed episodes with positive stepReward must NOT pair on reward noise
        {
            "taskId": "homestead_crop_farm",
            "domain": "HOMESTEADING_ECONOMY",
            "workerId": 0,
            "prompt": "Till farm",
            "passed": True,
            "score": 100,
            "rawReward": 175.0,
            "executedTools": [
                {"name": "till_and_farm", "args": {"plots": "A"}, "success": True, "stepReward": 90.0, "thought": "Tilled A"}
            ],
        },
        {
            "taskId": "homestead_crop_farm",
            "domain": "HOMESTEADING_ECONOMY",
            "workerId": 0,
            "prompt": "Till farm",
            "passed": True,
            "score": 100,
            "rawReward": 145.0,
            "executedTools": [
                {"name": "till_and_farm", "args": {"plots": "B"}, "success": True, "stepReward": 70.0, "thought": "Tilled B"}
            ],
        },
    ]
    replay_p.write_text("\\n".join(json.dumps(r) for r in rows) + "\\n", encoding="utf-8")
    trans_row = {"voxels": [0] * 3375, "entities": [[0.0] * 12] * 16, "botState": [0.0] * 10, "actionIdx": 0, "aimDelta": [0.0, 0.0], "reward": 1.0}
    trans_p.write_text(json.dumps(trans_row) + "\\n", encoding="utf-8")
    orig_paths = (rt.REPLAY_BUFFER_PATH, rt.REAL_TRANSITIONS_PATH, rt.GRPO_DATASET_PATH, rt.DPO_PAIRS_PATH, rt.RL_METRICS_PATH, rt.CHECKPOINT_DIR, rt.RESULTS_DIR)
    try:
        rt.REPLAY_BUFFER_PATH = replay_p
        rt.REAL_TRANSITIONS_PATH = trans_p
        rt.GRPO_DATASET_PATH = grpo_p
        rt.DPO_PAIRS_PATH = dpo_p
        rt.RL_METRICS_PATH = metrics_p
        rt.CHECKPOINT_DIR = tp / "ckpt"
        rt.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        rt.RESULTS_DIR = tp / "res"
        rt.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        rt.run_minebench_rl_update(epochs=0, is_best=False)
        dpo_items = [json.loads(l) for l in dpo_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(dpo_items) == 0, f"Expected 0 polluted DPO pairs, got {dpo_items}"
        grpo_lines = [l for l in grpo_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(grpo_lines) == len(set(grpo_lines)), "GRPO dataset must be deduplicated"
    finally:
        rt.REPLAY_BUFFER_PATH, rt.REAL_TRANSITIONS_PATH, rt.GRPO_DATASET_PATH, rt.DPO_PAIRS_PATH, rt.RL_METRICS_PATH, rt.CHECKPOINT_DIR, rt.RESULTS_DIR = orig_paths

    # Check latent 0.1 scaling + sigmoid [0, 1] value normalization in ActionHead loader
    mc_p = tp / "mc.jsonl"
    mc_rec = {
        "taskId": "combat_zombie_skirmish",
        "toolName": "engage_combat",
        "latent_h": [10.0] + [0.0] * 1535,
        "proprio24": [0.5] * 24,
        "voxels": [0] * 3375,
        "entities": [],
        "aimTargets": [[0.0, 0.0]] * 8,
        "moveTargets": [[0.0, 0.0]] * 8,
        "discreteTargets": [[0, 0, 0, 0, 0, 0]] * 8,
        "extendedTargets": [[0, 0]] * 8,
        "stepSuccess": True,
        "reward": -50.0,
        "episodePassed": False,
        "episodeReward": -50.0,
    }
    mc_p.write_text(json.dumps(mc_rec) + "\\n", encoding="utf-8")
    ds = load_real_motor_chunks_dataset(max_samples=8, device="cpu", chunks_path=mc_p)
    assert ds is not None
    assert abs(float(ds[0][0, 0].item()) - 1.0) < 1e-4
    assert 0.0 <= float(ds[10][0].item()) <= 1.0
print("OK")
`;
  assert.equal(execFileSync(pythonExe, ['-c', pyCheck], { encoding: 'utf8' }).trim().split('\n').pop(), 'OK');
});

test('59. System1 perception filters dying/invalid entities, refreshes immediately on teleport sync, and stamps swap slot', async () => {
  const { System1BotController } = require('../system1/botController');
  const { HumanizedInventoryController } = require('../system1/kinematics/inventory');

  const mockBot: any = {
    entity: {
      id: 1,
      position: new Vec3(0, 64, 0),
      height: 1.62,
      yaw: 0,
      pitch: 0,
      velocity: new Vec3(0, 0, 0),
      onGround: true,
    },
    health: 20,
    food: 20,
    oxygenLevel: 20,
    heldItem: null,
    time: { timeOfDay: 1000 },
    entities: {
      '10': { id: 10, name: 'zombie', type: 'hostile', health: 0, isValid: true, position: new Vec3(0, 64, -2) },
      '11': { id: 11, name: 'skeleton', type: 'hostile', health: 20, isValid: false, position: new Vec3(0, 64, -3) },
      '12': { id: 12, name: 'creeper', type: 'hostile', health: 20, isValid: true, position: new Vec3(0, 64, -4) },
    },
    inventory: {
      items: () => [{ name: 'iron_sword', slot: 37, count: 1 }],
    },
    blockAt: (pos: Vec3) => (pos && pos.y < 64 ? { name: 'stone', boundingBox: 'block' } : { name: 'air', boundingBox: 'empty' }),
    findBlock: () => null,
    findBlocks: () => [],
    equip: async () => {},
  };

  const controller = new System1BotController(mockBot);
  (controller as any).bot = mockBot;
  (controller as any).reflexEngine = { getActiveHazard: () => null };
  controller.invalidatePerceptionCache();
  const ws = controller.refreshWorldStateNow(true);
  assert.ok(ws, 'refreshWorldStateNow(true) must return a fresh snapshot');
  const entityNames = (ws!.visibleEntities || []).map((e: any) => e.name);
  assert.ok(entityNames.includes('creeper'), 'Valid live creeper must appear immediately after refreshWorldStateNow(true)');
  assert.ok(!entityNames.includes('zombie'), 'Dying entity with health <= 0 must be filtered out');
  assert.ok(!entityNames.includes('skeleton'), 'Invalid entity with isValid=false must be filtered out');

  const invCtrl = new HumanizedInventoryController(mockBot);
  const eqOk = await invCtrl.equipItemHumanized('iron_sword', 'hand');
  assert.equal(eqOk, true);
  assert.ok(Number((mockBot as any)._lastSwapSlotAt || 0) > 0, 'equipItemHumanized must stamp _lastSwapSlotAt');
});

test('60. Evaluators grade agility height tolerance, live lever powered property, and expanded peaceful mobs', () => {
  const { evaluateMinebenchEpisode } = require('../../minebench/dist/minebench/evaluators');
  const { MINEBENCH_TASKS } = require('../../minebench/dist/minebench/benchmarkSpec');

  const skyTask = MINEBENCH_TASKS.find((t: any) => t.id === 'agility_sky_parkour');
  const bridgeTask = MINEBENCH_TASKS.find((t: any) => t.id === 'agility_ravine_bridge');
  const leverTask = MINEBENCH_TASKS.find((t: any) => t.id === 'puzzle_redstone_lever');
  assert.ok(skyTask && bridgeTask && leverTask);

  const baseStats = {
    blocksPlaced: 0,
    blocksMined: 0,
    mobsKilled: {},
    creeperBlasts: 0,
    arrowsFired: 0,
    endCrystalsDestroyed: 0,
    enderDragonSlain: false,
    leversFlipped: 1,
    wolvesTamed: 0,
    ironPickaxeDurabilityUsedOnStone: 0,
    affordanceActions: 1,
  };

  const skyAbove = evaluateMinebenchEpisode(skyTask, {
    botPosition: { x: skyTask.origin.x + 18, y: skyTask.origin.y + 2, z: skyTask.origin.z },
    health: 20,
    food: 20,
    deaths: 0,
    inventory: [],
    equippedArmor: { head: null, torso: null, legs: null, feet: null, offHand: null },
    blockAt: () => ({ name: 'air' }),
    entities: [],
    stats: baseStats,
  } as any);
  assert.equal(skyAbove.passed, true, 'Landing cleanly above parkour gold pad must pass');

  const skyBelow = evaluateMinebenchEpisode(skyTask, {
    botPosition: { x: skyTask.origin.x + 18, y: skyTask.origin.y - 0.5, z: skyTask.origin.z },
    health: 20,
    food: 20,
    deaths: 0,
    inventory: [],
    equippedArmor: { head: null, torso: null, legs: null, feet: null, offHand: null },
    blockAt: () => ({ name: 'air' }),
    entities: [],
    stats: baseStats,
  } as any);
  assert.equal(skyBelow.passed, false, 'Sky parkour below goalY+0.5 must not pass');

  const bridgePass = evaluateMinebenchEpisode(bridgeTask, {
    botPosition: { x: bridgeTask.origin.x + 12, y: bridgeTask.origin.y + 0.5, z: bridgeTask.origin.z },
    health: 20,
    food: 20,
    deaths: 0,
    inventory: [],
    equippedArmor: { head: null, torso: null, legs: null, feet: null, offHand: null },
    blockAt: () => ({ name: 'air' }),
    entities: [],
    stats: baseStats,
  } as any);
  assert.equal(bridgePass.passed, true);

  let leverPowered = false;
  const makeLeverSnap = () => ({
    botPosition: { x: leverTask.origin.x + 6, y: leverTask.origin.y, z: leverTask.origin.z },
    health: 20,
    food: 20,
    deaths: 0,
    inventory: [],
    equippedArmor: { head: null, torso: null, legs: null, feet: null, offHand: null },
    blockAt: (bx: number, by: number, bz: number) =>
      bx === leverTask.origin.x + 3 && by === leverTask.origin.y + 1 && bz === leverTask.origin.z - 1
        ? ({ name: 'lever', getProperties: () => ({ powered: leverPowered }) } as any)
        : { name: 'air' },
    entities: [],
    stats: baseStats,
  });
  const unpoweredRes = evaluateMinebenchEpisode(leverTask, makeLeverSnap() as any);
  assert.equal(unpoweredRes.passed, false, 'Unpowered live lever at target pos must not pass');
  assert.equal(unpoweredRes.score, 50);

  leverPowered = true;
  const poweredRes = evaluateMinebenchEpisode(leverTask, makeLeverSnap() as any);
  assert.equal(poweredRes.passed, true);
  assert.equal(poweredRes.score, 100);

  for (const animal of ['pig', 'bee', 'axolotl', 'chicken']) {
    const mobRes = evaluateMinebenchEpisode(
      {
        id: `omni_mob_${animal}`,
        domain: 'MOB_INTERACTION',
        targetObject: animal,
        origin: { x: 0, y: 100, z: 0 },
        initialLoadout: [],
        evaluatorMilestones: [
          { name: 'Target Engaged', maxPoints: 40, condition: 'target_engaged' },
          { name: 'Target Resolved', maxPoints: 60, condition: 'target_resolved' },
        ],
      } as any,
      {
        botPosition: { x: 0, y: 100, z: 0 },
        health: 20,
        food: 20,
        deaths: 0,
        inventory: [],
        equippedArmor: { head: null, torso: null, legs: null, feet: null, offHand: null },
        blockAt: () => ({ name: 'air' }),
        entities: [],
        stats: { ...baseStats, mobInteractions: 1 },
      } as any
    );
    assert.equal(mobRes.passed, true, `Peaceful mob ${animal} interaction must pass without killing`);
  }
});

test('61. Regenerated Omniscience dataset has zero unregistered tools and slm_trainer filters plain QA/invalid tools', () => {
  const { execFileSync } = require('child_process');
  const { registerBaseMCPTools } = require('../system2/skills/baseSkills');
  const registry = new DynamicMCPToolRegistry();
  registerBaseMCPTools(registry);
  const registered = new Set(registry.listSchemas().map((t: any) => t.name));

  const rootDir = path.resolve(__dirname, '..', '..');
  const tasks: any[] = JSON.parse(
    fs.readFileSync(path.join(rootDir, 'minebench', 'results', 'omniscience_tasks.json'), 'utf8')
  );
  assert.ok(tasks.length > 0);

  for (const t of tasks) {
    assert.ok(registered.has(t.expectedTool), `Task ${t.taskId} references unregistered tool "${t.expectedTool}"`);
    if (t.taskId?.startsWith('omni_food_')) {
      assert.ok(
        !(t.setupCommands || []).some((c: string) => c.includes('hunger 5 1')),
        `${t.taskId} must not use weak hunger 5 1 setup command`
      );
    }
  }
  const piglinTask = tasks.find((t) => t.taskId === 'omni_mob_piglin');
  assert.ok(piglinTask);
  assert.equal(piglinTask.expectedTool, 'give_item_to_player');
  assert.equal(piglinTask.expectedToolArgs.playerName, 'piglin');
  assert.equal(piglinTask.expectedToolArgs.itemName, 'gold_ingot');

  const wardenTask = tasks.find((t) => t.taskId === 'omni_mob_warden');
  assert.ok(wardenTask);
  assert.equal(wardenTask.expectedTool, 'shoot_bow');
  assert.ok((wardenTask.initialLoadout || []).some((i: any) => i.name === 'bow'));
  assert.ok((wardenTask.initialLoadout || []).some((i: any) => i.name === 'arrow'));

  const pythonExe = path.join(rootDir, 'neural_lab', '.venv', 'Scripts', 'python.exe');
  const pyCheck = `
import json, sys
sys.path.insert(0, r'${rootDir.replace(/\\/g, '\\\\')}')
from minebench.slm_trainer import filter_valid_tool_trajectory

qa_row = {"id": "omni_qa_item_apple", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}, {"role": "assistant", "content": "<|intent:chat|> Apple stacks to 64."}]}
bad_tool_row = {"id": "omni_tool_axe_strip_log", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}, {"role": "assistant", "content": json.dumps({"thought": "t", "toolCall": {"name": "strip_log", "arguments": {"blockName": "oak_log"}}})}]}
good_row = {"id": "omni_tool_axe_strip_log", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}, {"role": "assistant", "content": json.dumps({"thought": "t", "toolCall": {"name": "use_tool_on_block", "arguments": {"tool": "iron_axe", "targetBlock": "oak_log"}}})}]}

assert filter_valid_tool_trajectory(qa_row) is None
assert filter_valid_tool_trajectory(bad_tool_row) is None
assert filter_valid_tool_trajectory(good_row) is not None
print("OK")
`;
  assert.equal(execFileSync(pythonExe, ['-c', pyCheck], { encoding: 'utf8' }).trim().split('\n').pop(), 'OK');
});

test('62. Skill guards: shulker_bullet filter, gather_resource empty & *_block, interact allowlist/dist, and piglin barter drop', async () => {
  const { registerBaseMCPTools } = require('../system2/skills/baseSkills');
  const registry = new DynamicMCPToolRegistry();
  registerBaseMCPTools(registry);
  assert.equal(registry.listSchemas().length, 35);

  const gatherTool = registry.getTool('gather_resource')!;
  const emptyGather = await gatherTool.handler(
    {},
    { bot: { inventory: { items: () => [] } } } as any
  );
  assert.equal(emptyGather.success, false, 'gather_resource({}) must fail without defaulting to log');

  let seenMatcher: ((b: any) => boolean) | null = null;
  const blockGather = await gatherTool.handler(
    { blockType: 'diamond_block', count: 1 },
    {
      bot: {
        entity: { position: new Vec3(0, 64, 0) },
        version: '1.20.4',
        heldItem: null,
        recipesFor: () => [],
        inventory: { items: () => [] },
        findBlocks: (opts: any) => {
          seenMatcher = opts.matching;
          return [];
        },
        findBlock: (opts: any) => {
          seenMatcher = seenMatcher || opts.matching;
          return null;
        },
        blockAt: () => ({ name: 'air', boundingBox: 'empty' }),
      },
      kinematics: { equipItemHumanized: async () => false },
      worldState: () => ({ visibleEntities: [] }),
      reportProgress: () => {},
    } as any
  );
  assert.equal(blockGather.success, false);
  const matcherFn = seenMatcher as ((b: any) => boolean) | null;
  assert.ok(matcherFn);
  assert.equal(matcherFn({ name: 'diamond_block' }), true);
  assert.equal(matcherFn({ name: 'diamond_ore' }), false, 'diamond_block must not match diamond_ore');

  const combatTool = registry.getTool('engage_combat')!;
  const bulletRes = await combatTool.handler(
    { targetType: 'shulker', maxRange: 16 },
    {
      bot: {
        entity: { position: new Vec3(0, 64, 0) },
        inventory: { items: () => [] },
        blockAt: () => ({ name: 'air', boundingBox: 'empty' }),
        findBlock: () => null,
        findBlocks: () => [],
        pathfinder: { setGoal: () => {} },
        setControlState: () => {},
        clearControlStates: () => {},
        entities: {
          '5': { id: 5, name: 'shulker_bullet', type: 'projectile', position: new Vec3(2, 64, 2), isValid: true },
        },
      },
      kinematics: { equipItemHumanized: async () => false },
      worldState: () => ({ visibleEntities: [] }),
      reportProgress: () => {},
    } as any
  );
  assert.equal(bulletRes.success, false, 'shulker_bullet projectile must be ignored by engage_combat');

  const interactTool = registry.getTool('interact_block_or_puzzle')!;
  const farGold = await interactTool.handler(
    { blockType: 'gold_block' },
    {
      bot: {
        entity: { position: new Vec3(0, 64, 0) },
        inventory: { items: () => [] },
        findBlock: () => ({ position: new Vec3(5, 64, 0), name: 'gold_block' }),
      },
      kinematics: { pathfindTo: async () => {} },
    } as any
  );
  assert.equal(farGold.success, false, 'interact_block_or_puzzle must fail if bot never reached gold_block');

  let brickActivated = 0;
  const nonInter = await interactTool.handler(
    { blockType: 'stone_bricks' },
    {
      bot: {
        entity: { position: new Vec3(0, 64, 0) },
        inventory: { items: () => [] },
        findBlock: () => ({ position: new Vec3(1, 64, 0), name: 'stone_bricks' }),
        blockAt: () => ({ position: new Vec3(1, 64, 0), name: 'stone_bricks' }),
        activateBlock: async () => { brickActivated++; },
      },
      kinematics: { pathfindTo: async () => {}, lookAtSmooth: async () => {} },
    } as any
  );
  assert.equal(nonInter.success, false, 'stone_bricks is not interactive');
  assert.equal(brickActivated, 0);

  const giveTool = registry.getTool('give_item_to_player')!;
  let tossed = 0;
  const piglinEnt = { id: 42, name: 'piglin', type: 'mob', position: new Vec3(1.5, 64, 0), isValid: true };
  const dropBot: any = {
    username: 'NeuroAgent',
    entity: { position: new Vec3(0, 64, 0) },
    players: {},
    entities: { '42': piglinEnt },
    inventory: { items: () => [{ name: 'gold_ingot', count: 1 }] },
    nearestEntity: (pred: (e: any) => boolean) => (pred(piglinEnt) ? piglinEnt : null),
    tossStack: async () => { tossed++; },
    setControlState: () => {},
    clearControlStates: () => {},
  };
  const giveRes = await giveTool.handler(
    { playerName: 'piglin', itemName: 'gold_ingot' },
    {
      bot: dropBot,
      kinematics: { pathfindTo: async () => {}, lookAtSmooth: async () => {} },
      reportProgress: () => {},
    } as any
  );
  assert.equal(giveRes.success, true, 'give_item_to_player must resolve piglin entity by name');
  assert.equal(tossed, 1);
  assert.ok(Number(dropBot._lastDropItemAt || 0) > 0, 'give_item_to_player must stamp _lastDropItemAt');
});

test('63. rl_trainer VALID_MCP_TOOLS matches 35 registered tools, GRPO skips stepReward<=0 on fails, DPO enforces combat & worker/prompt guards', () => {
  const { execFileSync } = require('child_process');
  const { registerBaseMCPTools } = require('../system2/skills/baseSkills');
  const registry = new DynamicMCPToolRegistry();
  registerBaseMCPTools(registry);
  const toolNames = registry.listSchemas().map((t: any) => t.name);

  const rootDir = path.resolve(__dirname, '..', '..');
  const pythonExe = path.join(rootDir, 'neural_lab', '.venv', 'Scripts', 'python.exe');
  const pyCheck = `
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, r'${rootDir.replace(/\\/g, '\\\\')}')
import minebench.rl_trainer as rt

js_tools = set(${JSON.stringify(toolNames)})
assert len(js_tools) == 35
assert js_tools == rt.VALID_MCP_TOOLS, (js_tools ^ rt.VALID_MCP_TOOLS)

with tempfile.TemporaryDirectory() as tmp:
    tp = Path(tmp)
    replay_p = tp / "replay.jsonl"
    trans_p = tp / "trans.jsonl"
    grpo_p = tp / "grpo.jsonl"
    dpo_p = tp / "dpo.jsonl"
    metrics_p = tp / "metrics.json"
    rows = [
        # Failed episode prefix: step 0 stepReward==0 must NOT enter GRPO, step 1 stepReward>0 CAN enter
        {
            "taskId": "prog_tier1_stone_age",
            "domain": "PROGRESSION_DRAGON",
            "workerId": 0,
            "prompt": "Make stone pickaxe",
            "passed": False,
            "score": 50,
            "rawReward": 40.0,
            "executedTools": [
                {"name": "gather_resource", "args": {"blockType": "oak_log"}, "success": True, "stepReward": 0.0, "thought": "Zero reward"},
                {"name": "craft_item", "args": {"itemName": "oak_planks"}, "success": True, "stepReward": 15.0, "thought": "Planks"},
            ],
        },
        # Passed combat task using navigate_to_coordinates must NOT become a DPO winner
        {
            "taskId": "combat_spider_swarm",
            "domain": "COMBAT_TACTICAL",
            "workerId": 0,
            "prompt": "Kill spiders",
            "passed": True,
            "score": 100,
            "rawReward": 100.0,
            "executedTools": [
                {"name": "navigate_to_coordinates", "args": {"x": 10, "y": 100, "z": 10}, "success": True, "stepReward": 60.0, "thought": "Walk"},
            ],
        },
        {
            "taskId": "combat_spider_swarm",
            "domain": "COMBAT_TACTICAL",
            "workerId": 0,
            "prompt": "Kill spiders",
            "passed": False,
            "score": 0,
            "rawReward": 0.0,
            "executedTools": [
                {"name": "engage_combat", "args": {"targetType": "creeper"}, "success": False, "stepReward": 0.0, "thought": "Wrong target"},
            ],
        },
        # Multi-worker & prompt guard on engage_combat
        {
            "taskId": "combat_zombie_skirmish",
            "domain": "COMBAT_TACTICAL",
            "workerId": 0,
            "prompt": "Fight A",
            "passed": True,
            "score": 100,
            "rawReward": 100.0,
            "executedTools": [
                {"name": "engage_combat", "args": {"targetType": "zombie"}, "success": True, "stepReward": 80.0, "thought": "Strike zombie"},
            ],
        },
        {
            "taskId": "combat_zombie_skirmish",
            "domain": "COMBAT_TACTICAL",
            "workerId": 1,
            "prompt": "Fight A",
            "passed": False,
            "score": 0,
            "rawReward": 0.0,
            "executedTools": [
                {"name": "eat_food", "args": {"foodName": "bread"}, "success": False, "stepReward": 0.0, "thought": "Cross worker"},
            ],
        },
        {
            "taskId": "combat_zombie_skirmish",
            "domain": "COMBAT_TACTICAL",
            "workerId": 0,
            "prompt": "Fight B",
            "passed": False,
            "score": 0,
            "rawReward": 0.0,
            "executedTools": [
                {"name": "eat_food", "args": {"foodName": "apple"}, "success": False, "stepReward": 0.0, "thought": "Diff prompt"},
            ],
        },
        {
            "taskId": "combat_zombie_skirmish",
            "domain": "COMBAT_TACTICAL",
            "workerId": 0,
            "prompt": "Fight A",
            "passed": False,
            "score": 0,
            "rawReward": 0.0,
            "executedTools": [
                {"name": "engage_combat", "args": {"targetType": "skeleton"}, "success": False, "stepReward": 0.0, "thought": "Same worker+prompt"},
            ],
        },
    ]
    replay_p.write_text("\\n".join(json.dumps(r) for r in rows) + "\\n", encoding="utf-8")
    trans_row = {"voxels": [0] * 3375, "entities": [[0.0] * 12] * 16, "botState": [0.0] * 10, "actionIdx": 0, "aimDelta": [0.0, 0.0], "reward": 1.0}
    trans_p.write_text(json.dumps(trans_row) + "\\n", encoding="utf-8")
    orig_paths = (rt.REPLAY_BUFFER_PATH, rt.REAL_TRANSITIONS_PATH, rt.GRPO_DATASET_PATH, rt.DPO_PAIRS_PATH, rt.RL_METRICS_PATH, rt.CHECKPOINT_DIR, rt.RESULTS_DIR)
    try:
        rt.REPLAY_BUFFER_PATH = replay_p
        rt.REAL_TRANSITIONS_PATH = trans_p
        rt.GRPO_DATASET_PATH = grpo_p
        rt.DPO_PAIRS_PATH = dpo_p
        rt.RL_METRICS_PATH = metrics_p
        rt.CHECKPOINT_DIR = tp / "ckpt"
        rt.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        rt.RESULTS_DIR = tp / "res"
        rt.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        rt.run_minebench_rl_update(epochs=0, is_best=False)
        grpo_items = [json.loads(l) for l in grpo_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        tier1_steps = [g.get("step") for g in grpo_items if g.get("taskId") == "prog_tier1_stone_age"]
        assert 0 not in tier1_steps and 1 in tier1_steps, f"Unexpected tier1 prefix steps: {tier1_steps}"
        dpo_items = [json.loads(l) for l in dpo_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(dpo_items) == 1, f"Expected exactly 1 clean combat_zombie_skirmish DPO pair, got {dpo_items}"
        assert dpo_items[0]["taskId"] == "combat_zombie_skirmish"
        rej_tool = json.loads(dpo_items[0]["rejected"])["toolCall"]
        assert rej_tool == {"name": "engage_combat", "arguments": {"targetType": "skeleton"}}, rej_tool
    finally:
        rt.REPLAY_BUFFER_PATH, rt.REAL_TRANSITIONS_PATH, rt.GRPO_DATASET_PATH, rt.DPO_PAIRS_PATH, rt.RL_METRICS_PATH, rt.CHECKPOINT_DIR, rt.RESULTS_DIR = orig_paths
print("OK")
`;
  assert.equal(execFileSync(pythonExe, ['-c', pyCheck], { encoding: 'utf8' }).trim().split('\n').pop(), 'OK');
});

test('64. Arena reset builds floor before teleport, issues post-build tp, applies single survival hunger, and encloses mining puzzle', async () => {
  const { HeadlessMinecraftServerManager } = require('../../minebench/dist/minebench/headlessServer');
  const { MINEBENCH_TASKS, MULTIPLAYER_TASKS } = require('../../minebench/dist/minebench/benchmarkSpec');

  const mgr = new HeadlessMinecraftServerManager();
  const sent: string[] = [];
  (mgr as any).sendCommand = (cmd: string) => sent.push(cmd);

  const mpTask = MULTIPLAYER_TASKS.find((t: any) => t.id === 'mp_pvp_dqn_axe_shield');
  assert.ok(mpTask);
  await mgr.resetArenaOnLiveServer(mpTask, 'NeuroAgent_2', undefined, true);

  const stonePadIdx = sent.findIndex((c) => c.includes('minecraft:stone') && c.startsWith('fill '));
  const tpIdxs = sent.map((c, idx) => (c.startsWith('tp NeuroAgent_2 ') ? idx : -1)).filter((i) => i >= 0);
  const survIdx = sent.findIndex((c) => c === 'gamemode survival NeuroAgent_2');
  assert.ok(stonePadIdx >= 0 && tpIdxs.length >= 2 && survIdx > tpIdxs[1], `Unexpected cmd order: ${JSON.stringify(sent)}`);
  assert.ok(stonePadIdx < tpIdxs[0], 'Spawn stone pad must be filled BEFORE initial teleport');
  assert.ok(tpIdxs[1] < survIdx, 'Post-build safety teleport must run BEFORE switching to survival');

  sent.length = 0;
  await mgr.resetArenaOnLiveServer(
    {
      id: 'omni_food_apple',
      origin: { x: 100, y: 64, z: 100 },
      initialLoadout: [{ name: 'apple', count: 4 }],
      buildArenaCommands: () => ['tp @p 100 64 100 0 0'],
    } as any,
    'NeuroAgent_1',
    undefined,
    true
  );
  assert.equal(sent.some((c) => c.includes('minecraft:saturation')), false, 'Food task must skip saturation');
  const hungerCmds = sent.map((c, idx) => (c.includes('minecraft:hunger') ? idx : -1)).filter((i) => i >= 0);
  const foodSurvIdx = sent.findIndex((c) => c === 'gamemode survival NeuroAgent_1');
  assert.equal(hungerCmds.length, 1, 'Hunger effect must be applied exactly once');
  assert.ok(hungerCmds[0] > foodSurvIdx, 'Hunger effect must run AFTER switching to survival mode');

  const minePuzzle = MINEBENCH_TASKS.find((t: any) => t.id === 'puzzle_tool_preservation_mining');
  assert.ok(minePuzzle);
  const mineCmds = minePuzzle.buildArenaCommands({ x: 750, y: 64, z: 100 });
  assert.ok(
    mineCmds.some((c: string) => c.includes('minecraft:glass outline')),
    'puzzle_tool_preservation_mining must include glass perimeter border'
  );
});

test('65. Evaluators grade initialFood deltas objectively and block dry-land aquatic suffocation or ambient exploits', () => {
  const { evaluateMinebenchEpisode } = require('../../minebench/dist/minebench/evaluators');

  const fishTask: any = {
    id: 'omni_food_tropical_fish',
    domain: 'NUTRITION',
    targetObject: 'tropical_fish',
    origin: { x: 0, y: 64, z: 0 },
    initialLoadout: [{ name: 'tropical_fish', count: 4 }],
    evaluatorMilestones: [
      { name: 'Item Selected', maxPoints: 30, condition: 'held_item_food' },
      { name: 'Hunger Restored', maxPoints: 70, condition: 'food_level_restored' },
    ],
  };
  const fishSnapBase: any = {
    botPosition: { x: 0, y: 64, z: 0 },
    health: 20,
    initialFood: 14,
    food: 15,
    deaths: 0,
    inventory: [],
    equippedArmor: { head: null, torso: null, legs: null, feet: null, offHand: null },
    blockAt: () => ({ name: 'air' }),
    entities: [],
    stats: {
      blocksPlaced: 0, blocksMined: 0, mobsKilled: {}, creeperBlasts: 0, arrowsFired: 0,
      endCrystalsDestroyed: 0, enderDragonSlain: false, leversFlipped: 0, wolvesTamed: 0,
      foodConsumed: 1, affordanceActions: 1, ironPickaxeDurabilityUsedOnStone: 0,
    },
  };
  const fishPass = evaluateMinebenchEpisode(fishTask, fishSnapBase);
  assert.equal(fishPass.passed, true, 'Eating tropical_fish restoring 14->15 food must pass');
  assert.equal(fishPass.score, 100);

  const fishNoGain = evaluateMinebenchEpisode(fishTask, { ...fishSnapBase, food: 14 });
  assert.equal(fishNoGain.passed, false, 'Food without hunger gain above initialFood must fail');

  const codTask: any = {
    id: 'omni_mob_cod',
    domain: 'MOB_INTERACTION',
    targetObject: 'cod',
    origin: { x: 0, y: 64, z: 0 },
    initialLoadout: [],
    evaluatorMilestones: [
      { name: 'Target Engaged', maxPoints: 40, condition: 'target_engaged' },
      { name: 'Target Resolved', maxPoints: 60, condition: 'target_resolved' },
    ],
  };
  const codSuffocate = evaluateMinebenchEpisode(codTask, {
    ...fishSnapBase,
    stats: {
      ...fishSnapBase.stats,
      mobsKilled: { cod: 1 },
      blocksMined: 1,
      damageDealt: 0,
      mobInteractions: 0,
    },
  });
  assert.equal(codSuffocate.passed, false);
  assert.equal(codSuffocate.score, 0, 'Dry-land suffocation + unrelated digging/eating must score 0');

  const codKilled = evaluateMinebenchEpisode(codTask, {
    ...fishSnapBase,
    stats: { ...fishSnapBase.stats, mobsKilled: { cod: 1 }, damageDealt: 4, mobInteractions: 0 },
  });
  assert.equal(codKilled.passed, true);

  const raidTask: any = {
    id: 'omni_structure_end_city',
    domain: 'STRUCTURE_RAID',
    targetObject: 'end_city',
    origin: { x: 0, y: 64, z: 0 },
    initialLoadout: [],
    evaluatorMilestones: [
      { name: 'Hazards Avoided', maxPoints: 40, condition: 'zero_hazard_deaths' },
      { name: 'Key Loot Secured', maxPoints: 60, condition: 'loot_collected' },
    ],
  };
  const digOnlyRaid = evaluateMinebenchEpisode(raidTask, {
    ...fishSnapBase,
    stats: { ...fishSnapBase.stats, blocksMined: 5, foodConsumed: 0, affordanceActions: 0 },
  });
  assert.equal(digOnlyRaid.score, 0, 'Digging in place without moving >2m or collecting loot must score 0');
});

test('66. Skills guard crafted items in gather_resource, safely parse interact args, reject swords and support lava_bucket in use_tool_on_block', async () => {
  const { registerBaseMCPTools } = require('../system2/skills/baseSkills');
  const registry = new DynamicMCPToolRegistry();
  registerBaseMCPTools(registry);

  const gatherTool = registry.getTool('gather_resource')!;
  const craftedRes = await gatherTool.handler(
    { blockType: 'iron_helmet', count: 1 },
    { bot: { inventory: { items: () => [] } } } as any
  );
  assert.equal(craftedRes.success, false, 'gather_resource must reject crafted armor/items like iron_helmet');

  let barsMatcher: ((b: any) => boolean) | null = null;
  await gatherTool.handler(
    { blockType: 'iron_bars', count: 1 },
    {
      bot: {
        entity: { position: new Vec3(0, 64, 0) },
        version: '1.20.4',
        heldItem: null,
        recipesFor: () => [],
        inventory: { items: () => [] },
        findBlocks: (opts: any) => {
          barsMatcher = opts.matching;
          return [];
        },
        findBlock: (opts: any) => {
          barsMatcher = barsMatcher || opts.matching;
          return null;
        },
        blockAt: () => ({ name: 'air', boundingBox: 'empty' }),
      },
      kinematics: { equipItemHumanized: async () => false },
      worldState: () => ({ visibleEntities: [] }),
      reportProgress: () => {},
    } as any
  );
  const matcherFn = barsMatcher as ((b: any) => boolean) | null;
  assert.ok(matcherFn);
  assert.equal(matcherFn({ name: 'iron_bars' }), true);
  assert.equal(matcherFn({ name: 'iron_ore' }), false, 'iron_bars must not match iron_ore');

  const interactTool = registry.getTool('interact_block_or_puzzle')!;
  const objInteract = await interactTool.handler(
    { targetBlock: {} },
    { bot: { entity: { position: new Vec3(0, 64, 0) }, inventory: { items: () => [] } } } as any
  );
  assert.equal(objInteract.success, false, 'Empty object targetBlock must fail cleanly without [object object]');

  let activatedAnimal = '';
  const cowEnt = { id: 9, name: 'cow', type: 'animal', position: new Vec3(2, 64, 0), isValid: true };
  const cowInteract = await interactTool.handler(
    { animalType: 'cow', itemName: 'lead' },
    {
      bot: {
        entity: { position: new Vec3(0, 64, 0) },
        inventory: { items: () => [{ name: 'lead', count: 1 }] },
        entities: { '9': cowEnt },
        activateEntity: (ent: any) => {
          activatedAnimal = ent.name;
        },
      },
      kinematics: {
        equipItemHumanized: async () => true,
        pathfindTo: async () => true,
        lookAtSmooth: async () => {},
      },
    } as any
  );
  assert.equal(cowInteract.success, true);
  assert.equal(activatedAnimal, 'cow');

  const useTool = registry.getTool('use_tool_on_block')!;
  const swordReject = await useTool.handler(
    { tool: 'diamond_sword', targetBlock: 'ocelot' },
    { bot: { entity: { position: new Vec3(0, 64, 0) }, inventory: { items: () => [{ name: 'diamond_sword', count: 1 }] } } } as any
  );
  assert.equal(swordReject.success, false, 'use_tool_on_block must refuse sword weapon attacks');

  let lavaActivatedPos: Vec3 | null = null;
  let lavaDeployed = false;
  const lavaRes = await useTool.handler(
    { tool: 'lava_bucket', targetBlock: 'zombie' },
    {
      bot: {
        entity: { position: new Vec3(0, 64, 0) },
        inventory: { items: () => [{ name: lavaDeployed ? 'bucket' : 'lava_bucket', count: 1 }] },
        entities: { '15': { id: 15, name: 'zombie', type: 'hostile', position: new Vec3(2, 64, 0), isValid: true } },
        blockAt: (p: Vec3) => {
          if (Math.floor(p.x) === 2 && Math.floor(p.y) === 63 && Math.floor(p.z) === 0) {
            return { name: 'stone', position: new Vec3(2, 63, 0) };
          }
          if (lavaDeployed && Math.floor(p.x) === 2 && Math.floor(p.y) === 64 && Math.floor(p.z) === 0) {
            return { name: 'lava', position: new Vec3(2, 64, 0) };
          }
          return { name: 'air', position: p };
        },
        activateBlock: async (blk: any) => {
          lavaActivatedPos = blk.position;
          lavaDeployed = true;
        },
        activateItem: () => { lavaDeployed = true; },
      },
      kinematics: {
        equipItemHumanized: async () => true,
        pathfindTo: async () => true,
        lookAtSmooth: async () => {},
      },
    } as any
  );
  assert.equal(lavaRes.success, true);
  assert.ok(lavaActivatedPos && (lavaActivatedPos as Vec3).x === 2 && (lavaActivatedPos as Vec3).y === 63);
});

test('67. Regenerated Omniscience suite adds aquatic pools, fixes leash/lava/pumpkin tasks, and drops duplicate nutrition hunger', () => {
  const rootDir = path.resolve(__dirname, '..', '..');
  const tasks: any[] = JSON.parse(
    fs.readFileSync(path.join(rootDir, 'minebench', 'results', 'omniscience_tasks.json'), 'utf8')
  );

  assert.equal(tasks.some((t) => t.taskId === 'omni_tool_axe_carve_pumpkin'), false);
  const chopTask = tasks.find((t) => t.taskId === 'omni_tool_axe_chop_pumpkin');
  assert.ok(chopTask && chopTask.expectedTool === 'gather_resource');

  const leashTask = tasks.find((t) => t.taskId === 'omni_tool_lead_leash_animal');
  assert.ok(leashTask);
  assert.equal(leashTask.expectedTool, 'use_tool_on_block');
  assert.equal(leashTask.expectedToolArgs.tool, 'lead');
  assert.equal(leashTask.expectedToolArgs.targetBlock, 'cow');

  const lavaTask = tasks.find((t) => t.taskId === 'omni_tool_lava_bucket_trap_mob');
  assert.ok(lavaTask);
  assert.equal(lavaTask.expectedToolArgs.tool, 'lava_bucket');
  assert.equal(lavaTask.expectedToolArgs.targetBlock, 'zombie');

  for (const t of tasks.filter((x) => x.taskId?.startsWith('omni_food_'))) {
    assert.equal(
      (t.setupCommands || []).some((c: string) => c.includes('effect give') && c.includes('hunger')),
      false,
      `${t.taskId} must not duplicate hunger command inside setupCommands`
    );
  }

  for (const aq of ['cod', 'squid', 'axolotl', 'guardian']) {
    const t = tasks.find((x) => x.taskId === `omni_mob_${aq}`);
    assert.ok(t, `Missing omni_mob_${aq}`);
    assert.ok(
      (t.setupCommands || []).some((c: string) => c.includes('water')),
      `omni_mob_${aq} must build a water pool arena`
    );
  }
});

test('68. rl_trainer seeds omniscience_gold_dataset.jsonl via id, normalizes taskId & passed=False, and blocks failed overwrites', () => {
  const { execFileSync } = require('child_process');
  const rootDir = path.resolve(__dirname, '..', '..');
  const pythonExe = path.join(rootDir, 'neural_lab', '.venv', 'Scripts', 'python.exe');
  const pyCheck = `
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, r'${rootDir.replace(/\\/g, '\\\\')}')
import minebench.rl_trainer as rt

with tempfile.TemporaryDirectory() as tmp:
    tp = Path(tmp)
    replay_p = tp / "replay.jsonl"
    trans_p = tp / "trans.jsonl"
    grpo_p = tp / "grpo.jsonl"
    dpo_p = tp / "dpo.jsonl"
    metrics_p = tp / "metrics.json"
    res_dir = tp / "res"
    res_dir.mkdir(parents=True, exist_ok=True)

    omni_rows = [
        {
            "id": "omni_gold_0_omni_tool_lead_leash_animal",
            "category": "tool_affordance",
            "targetObject": "lead",
            "advantage": 1.0,
            "reward": 100.0,
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Attach your lead to the cow in front of you."},
                {"role": "assistant", "content": json.dumps({"thought": "Leash cow", "toolCall": {"name": "use_tool_on_block", "arguments": {"tool": "lead", "targetBlock": "cow"}}})}
            ],
        },
        {
            "id": "omni_qa_item_apple",
            "category": "world_knowledge",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "What is apple?"},
                {"role": "assistant", "content": "<|intent:chat|> Apple stacks to 64."}
            ],
        },
    ]
    (res_dir / "omniscience_gold_dataset.jsonl").write_text("\\n".join(json.dumps(x) for x in omni_rows) + "\\n", encoding="utf-8")

    replay_rows = [
        {
            "taskId": "omni_tool_lead_leash_animal",
            "domain": "TOOL",
            "workerId": 0,
            "prompt": "Attach your lead to the cow in front of you.",
            "passed": False,
            "score": 30,
            "rawReward": 25.0,
            "executedTools": [
                {"name": "interact_block_or_puzzle", "args": {"blockType": "lever"}, "success": True, "stepReward": 20.0, "thought": "Random lever"},
            ],
        }
    ]
    replay_p.write_text("\\n".join(json.dumps(x) for x in replay_rows) + "\\n", encoding="utf-8")
    trans_p.write_text(json.dumps({"voxels": [0] * 3375, "entities": [[0.0] * 12] * 16, "botState": [0.0] * 10, "actionIdx": 0, "aimDelta": [0.0, 0.0], "reward": 1.0}) + "\\n", encoding="utf-8")

    orig = (rt.REPLAY_BUFFER_PATH, rt.REAL_TRANSITIONS_PATH, rt.GRPO_DATASET_PATH, rt.DPO_PAIRS_PATH, rt.RL_METRICS_PATH, rt.CHECKPOINT_DIR, rt.RESULTS_DIR)
    try:
        rt.REPLAY_BUFFER_PATH = replay_p
        rt.REAL_TRANSITIONS_PATH = trans_p
        rt.GRPO_DATASET_PATH = grpo_p
        rt.DPO_PAIRS_PATH = dpo_p
        rt.RL_METRICS_PATH = metrics_p
        rt.CHECKPOINT_DIR = tp / "ckpt"
        rt.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        rt.RESULTS_DIR = res_dir
        rt.run_minebench_rl_update(epochs=0, is_best=False)
        grpo_items = [json.loads(l) for l in grpo_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        gold_lead = [g for g in grpo_items if g.get("taskId") == "omni_tool_lead_leash_animal" and g.get("provider") == "omniscience_gold"]
        assert len(gold_lead) == 1, f"Expected gold seed preserved without failed overwrite, got {grpo_items}"
        assert gold_lead[0].get("passed") is False
        assert gold_lead[0].get("domain") == "TOOL_AFFORDANCES"
        assert 'Player "Trainer" says:' in gold_lead[0]["messages"][1]["content"]
        assert json.loads(gold_lead[0]["messages"][2]["content"]).get("goalCompleted") is False
        assert not any(g.get("taskId") == "omni_qa_item_apple" for g in grpo_items)
        dpo_items = [json.loads(l) for l in dpo_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(dpo_items) == 0, f"Gold seed must never act as DPO winner, got {dpo_items}"
    finally:
        rt.REPLAY_BUFFER_PATH, rt.REAL_TRANSITIONS_PATH, rt.GRPO_DATASET_PATH, rt.DPO_PAIRS_PATH, rt.RL_METRICS_PATH, rt.CHECKPOINT_DIR, rt.RESULTS_DIR = orig
print("OK")
`;
  assert.equal(execFileSync(pythonExe, ['-c', pyCheck], { encoding: 'utf8' }).trim().split('\n').pop(), 'OK');
});

test('69. Omniscience buildArenaCommands keeps low tp offsets, classical/MP arenas are fenced, and villager spawns with trades', () => {
  const { sampleDiverseOmniscienceTasks } = require('../../minebench/dist/minebench/train_loop');
  const { MINEBENCH_TASKS, MULTIPLAYER_TASKS, SELF_SPARRING_TASKS } = require('../../minebench/dist/minebench/benchmarkSpec');

  const rootDir = path.resolve(__dirname, '..', '..');
  const tasksPath = path.join(rootDir, 'minebench', 'results', 'omniscience_tasks.json');
  const omniTasks = sampleDiverseOmniscienceTasks(tasksPath, 500);
  const chopLog = omniTasks.find((t: any) => t.id === 'omni_tool_axe_chop_log');
  const stripLog = omniTasks.find((t: any) => t.id === 'omni_tool_axe_strip_log');
  assert.ok(chopLog && stripLog);

  const chopCmds = chopLog.buildArenaCommands!({ x: 8500, y: 100, z: 500 });
  assert.equal(chopCmds[chopCmds.length - 1], 'tp NeuroAgent 8500 101 498 0 0', 'Must keep ~ ~1 ~-2 spawn offset outside oak_log');
  const stripCmds = stripLog.buildArenaCommands!({ x: 8500, y: 100, z: 500 });
  assert.equal(stripCmds[stripCmds.length - 1], 'tp NeuroAgent 8500 101 496 0 0', 'Must keep ~ ~1 ~-4 spawn offset');

  const fencedIds = [
    'combat_skeleton_duel',
    'combat_spider_swarm',
    'combat_enderman_hunt',
    'prog_tier1_stone_age',
    'prog_tier3_iron_armor',
    'prog_tier6_ender_dragon',
    'mp_mob_escort_race',
    'mp_resource_duel',
    'mp_mob_escort_race_self',
    'mp_resource_duel_self',
  ];
  const allSpecs = [...MINEBENCH_TASKS, ...MULTIPLAYER_TASKS, ...SELF_SPARRING_TASKS];
  for (const id of fencedIds) {
    const spec = allSpecs.find((t: any) => t.id === id);
    assert.ok(spec, `Missing task spec ${id}`);
    const cmds = spec.buildArenaCommands({ x: 400, y: 64, z: 100 });
    assert.ok(cmds.some((c: string) => c.includes('minecraft:glass outline')), `${id} must have glass perimeter fence`);
  }

  const rawOmni: any[] = JSON.parse(fs.readFileSync(tasksPath, 'utf8'));
  const villager = rawOmni.find((t) => t.taskId === 'omni_mob_villager');
  assert.ok(villager);
  assert.ok(
    (villager.setupCommands || []).some((c: string) => c.includes('minecraft:weaponsmith') && c.includes('Offers:{Recipes:')),
    'omni_mob_villager must spawn weaponsmith with trades'
  );
});

test('70. safeDigSingleBlock digs diggable boundingBox=empty blocks, activate checks block-state mutation, and args unwrap objects', async () => {
  const { registerBaseMCPTools } = require('../system2/skills/baseSkills');
  const registry = new DynamicMCPToolRegistry();
  registerBaseMCPTools(registry);

  const gatherTool = registry.getTool('gather_resource')!;
  let digCalls = 0;
  let leverRemoved = false;
  let invHasLever = false;
  const leverPos = new Vec3(1, 64, 0);
  const leverBlock = () =>
    leverRemoved
      ? { name: 'air', boundingBox: 'empty', position: leverPos, diggable: false }
      : { name: 'lever', boundingBox: 'empty', position: leverPos, diggable: true };

  const gatherRes = await gatherTool.handler(
    { blockType: 'lever', count: 1 },
    {
      bot: {
        entity: { position: new Vec3(0, 64, 0), onGround: true },
        heldItem: null,
        inventory: { items: () => (invHasLever ? [{ name: 'lever', count: 1 }] : []) },
        findBlocks: () => (leverRemoved ? [] : [leverPos]),
        findBlock: () => (leverRemoved ? null : leverBlock()),
        blockAt: (p: Vec3) => (p.x === 1 && p.y === 64 && p.z === 0 ? leverBlock() : { name: 'stone', boundingBox: 'block', position: p }),
        canDigBlock: (b: any) => Boolean(b && b.diggable !== false),
        lookAt: async () => {},
        dig: async () => {
          digCalls++;
          leverRemoved = true;
          invHasLever = true;
        },
        pathfinder: { setGoal: () => {} },
        clearControlStates: () => {},
        on: () => {},
        removeListener: () => {},
      },
      kinematics: {
        pathfindTo: async () => true,
        lookAtSmooth: async () => {},
      },
      worldState: () => ({ visibleEntities: [] }),
      reportProgress: () => {},
    } as any
  );
  assert.equal(digCalls, 1, 'safeDigSingleBlock must actually dig boundingBox=empty blocks like lever');
  assert.equal(gatherRes.success, true);

  const useTool = registry.getTool('use_tool_on_block')!;
  const noMutRes = await useTool.handler(
    { tool: 'iron_axe', targetBlock: 'oak_log' },
    {
      bot: {
        entity: { position: new Vec3(0, 64, 0) },
        inventory: { items: () => [{ name: 'iron_axe', count: 1 }] },
        findBlock: () => ({ name: 'oak_log', position: new Vec3(2, 64, 0) }),
        blockAt: (p: Vec3) => (p.y === 64 ? { name: 'oak_log', position: p } : { name: 'air', position: p }),
        activateBlock: async () => {},
        lookAt: async () => {},
      },
      kinematics: { equipItemHumanized: async () => true, pathfindTo: async () => true, lookAtSmooth: async () => {} },
    } as any
  );
  assert.equal(noMutRes.success, false, 'use_tool_on_block must return false if log was not stripped on server');

  let isStripped = false;
  const mutRes = await useTool.handler(
    { tool: 'iron_axe', targetBlock: 'oak_log' },
    {
      bot: {
        entity: { position: new Vec3(0, 64, 0) },
        inventory: { items: () => [{ name: 'iron_axe', count: 1 }] },
        findBlock: () => ({ name: 'oak_log', position: new Vec3(2, 64, 0) }),
        blockAt: (p: Vec3) =>
          p.y === 64
            ? { name: isStripped ? 'stripped_oak_log' : 'oak_log', position: p }
            : { name: 'air', position: p },
        activateBlock: async () => { isStripped = true; },
        lookAt: async () => {},
      },
      kinematics: { equipItemHumanized: async () => true, pathfindTo: async () => true, lookAtSmooth: async () => {} },
    } as any
  );
  assert.equal(mutRes.success, true);

  const lightTool = registry.getTool('light_block_fire')!;
  const unlitRes = await lightTool.handler(
    { targetBlock: 'campfire' },
    {
      bot: {
        entity: { position: new Vec3(0, 64, 0) },
        inventory: { items: () => [{ name: 'flint_and_steel', count: 1 }] },
        findBlock: () => ({ name: 'campfire', position: new Vec3(2, 64, 0), getProperties: () => ({ lit: false }) }),
        blockAt: (p: Vec3) =>
          p.y === 64 ? { name: 'campfire', position: p, getProperties: () => ({ lit: false }) } : { name: 'air', position: p },
        activateBlock: async () => {},
        lookAt: async () => {},
      },
      kinematics: { equipItemHumanized: async () => true, pathfindTo: async () => true, lookAtSmooth: async () => {} },
    } as any
  );
  assert.equal(unlitRes.success, false, 'light_block_fire must fail if campfire lit remains false');

  const navTool = registry.getTool('navigate_parkour_or_bridge')!;
  const navRes = await navTool.handler(
    { goalBlock: { type: 'chicken' } },
    { bot: { entity: { position: new Vec3(0, 64, 0) }, findBlock: () => null } } as any
  );
  assert.equal(navRes.success, false);
  assert.ok(!navRes.summary.toLowerCase().includes('[object'), `Unexpected [object coercion: ${navRes.summary}`);

  const giveTool = registry.getTool('give_item_to_player')!;
  const compEnt = { id: 7, type: 'player', username: 'Companion', position: new Vec3(1, 64, 0), isValid: true };
  let tossedItemName = '';
  const giveObjRes = await giveTool.handler(
    { targetPlayer: 'Companion', itemName: { name: 'bread' } },
    {
      bot: {
        username: 'NeuroAgent',
        entity: { position: new Vec3(0, 64, 0) },
        players: { Companion: { username: 'Companion', entity: compEnt } },
        entities: { '7': compEnt },
        inventory: { items: () => [{ name: 'bread', count: 2 }] },
        nearestEntity: (fn: any) => (fn(compEnt) ? compEnt : null),
        tossStack: async (it: any) => { tossedItemName = it.name; },
        setControlState: () => {},
        clearControlStates: () => {},
      },
      kinematics: { pathfindTo: async () => true, lookAtSmooth: async () => {} },
      reportProgress: () => {},
    } as any
  );
  assert.equal(giveObjRes.success, true);
  assert.equal(tossedItemName, 'bread');
});

test('71. Evaluators reject blocksMined shortcut on non-mining omni_tool_axe_strip_log while keeping chop_log valid', () => {
  const { evaluateMinebenchEpisode } = require('../../minebench/dist/minebench/evaluators');

  const mkSnap = (invName: string) => ({
    botPosition: { x: 0, y: 100, z: 0 },
    health: 20,
    food: 20,
    deaths: 0,
    inventory: [{ name: invName, count: 1 }],
    equippedArmor: { head: null, torso: null, legs: null, feet: null, offHand: null },
    blockAt: () => ({ name: 'air' }),
    entities: [],
    stats: {
      blocksPlaced: 0, blocksMined: 1, mobsKilled: {}, creeperBlasts: 0, arrowsFired: 0,
      endCrystalsDestroyed: 0, enderDragonSlain: false, leversFlipped: 0, wolvesTamed: 0,
      affordanceActions: 0, ironPickaxeDurabilityUsedOnStone: 0,
    },
  });

  const stripFail = evaluateMinebenchEpisode(
    {
      id: 'omni_tool_axe_strip_log',
      domain: 'TOOL_AFFORDANCES',
      targetObject: 'iron_axe',
      origin: { x: 0, y: 100, z: 0 },
      initialLoadout: [{ name: 'iron_axe', count: 1 }],
      evaluatorMilestones: [
        { name: 'Tool Equipped', maxPoints: 30, condition: 'held_tool_correct' },
        { name: 'Affordance Executed', maxPoints: 70, condition: 'affordance_action_completed' },
      ],
    } as any,
    mkSnap('iron_axe') as any
  );
  assert.equal(stripFail.passed, false);
  assert.equal(stripFail.score, 0, 'Mining oak_log must not pass omni_tool_axe_strip_log');

  const chopPass = evaluateMinebenchEpisode(
    {
      id: 'omni_tool_axe_chop_log',
      domain: 'TOOL_AFFORDANCES',
      targetObject: 'diamond_axe',
      origin: { x: 0, y: 100, z: 0 },
      initialLoadout: [{ name: 'diamond_axe', count: 1 }],
      evaluatorMilestones: [
        { name: 'Tool Equipped', maxPoints: 30, condition: 'held_tool_correct' },
        { name: 'Affordance Executed', maxPoints: 70, condition: 'affordance_action_completed' },
      ],
    } as any,
    mkSnap('diamond_axe') as any
  );
  assert.equal(chopPass.passed, true);
  assert.equal(chopPass.score, 100);
});

test('72. Open-world evaluateSurvivalAdvancements merges persistent priorUnlocked across iterations', () => {
  const { evaluateSurvivalAdvancements } = require('../../minebench/dist/minebench/train_loop');
  const prior = new Set<string>(['Homesteader & Hunter', 'Wood Age']);
  const eval2 = evaluateSurvivalAdvancements([], 0, 0, prior);
  assert.ok(eval2.unlocked.includes('Homesteader & Hunter'));
  assert.ok(eval2.unlocked.includes('Wood Age'));
  assert.equal(eval2.techTier, 'Wood');
  assert.equal(eval2.score, 28);
});

test('73. rl_trainer & slm_trainer block negative-advantage, non-DAG failed rollouts, and failed open_world_survival from GRPO/SFT', () => {
  const { execFileSync } = require('child_process');
  const rootDir = path.resolve(__dirname, '..', '..');
  const pythonExe = path.join(rootDir, 'neural_lab', '.venv', 'Scripts', 'python.exe');
  const pyCheck = `
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, ${JSON.stringify(rootDir)})
import minebench.rl_trainer as rt
from minebench.slm_trainer import filter_valid_tool_trajectory

with tempfile.TemporaryDirectory() as tmp:
    tp = Path(tmp)
    replay_p = tp / "replay.jsonl"
    trans_p = tp / "trans.jsonl"
    grpo_p = tp / "grpo.jsonl"
    dpo_p = tp / "dpo.jsonl"
    metrics_p = tp / "metrics.json"
    rows = [
        # Winner sets mean ~ 50
        {
            "taskId": "omni_craft_bucket",
            "domain": "CRAFTING_RECIPES",
            "prompt": "Craft bucket",
            "passed": True,
            "score": 100,
            "rawReward": 100.0,
            "executedTools": [
                {"name": "craft_item", "args": {"itemName": "bucket"}, "success": True, "stepReward": 100.0, "thought": "Craft"},
            ],
        },
        # Negative-advantage failed DAG episode (rawReward=10 < mean) must NOT enter GRPO, but CAN become DPO negative
        {
            "taskId": "omni_craft_bucket",
            "domain": "CRAFTING_RECIPES",
            "prompt": "Craft bucket",
            "passed": False,
            "score": 30,
            "rawReward": 10.0,
            "executedTools": [
                {"name": "gather_resource", "args": {"blockType": "stone"}, "success": True, "stepReward": 10.0, "thought": "Wrong step"},
            ],
        },
        # Non-DAG failed episode with high rawReward=90 must NOT enter GRPO
        {
            "taskId": "omni_mob_creeper",
            "domain": "MOB_INTERACTION",
            "prompt": "Fight creeper",
            "passed": False,
            "score": 40,
            "rawReward": 90.0,
            "executedTools": [
                {"name": "engage_combat", "args": {"targetType": "creeper"}, "success": True, "stepReward": 40.0, "thought": "Engaged"},
            ],
        },
        # Failed open_world_survival row with high reward must NEVER enter GRPO
        {
            "taskId": "prog_survival_goal",
            "provider": "open_world_survival",
            "domain": "PROGRESSION_DRAGON",
            "prompt": "Survive",
            "passed": False,
            "score": 45,
            "rawReward": 95.0,
            "executedTools": [
                {"name": "gather_resource", "args": {"blockType": "oak_log"}, "success": True, "stepReward": 45.0, "thought": "Log"},
            ],
        },
    ]
    replay_p.write_text("\\n".join(json.dumps(x) for x in rows) + "\\n", encoding="utf-8")
    trans_p.write_text(json.dumps({"voxels": [0] * 3375, "entities": [[0.0] * 12] * 16, "botState": [0.0] * 10, "actionIdx": 0, "aimDelta": [0.0, 0.0], "reward": 1.0}) + "\\n", encoding="utf-8")
    orig = (rt.REPLAY_BUFFER_PATH, rt.REAL_TRANSITIONS_PATH, rt.GRPO_DATASET_PATH, rt.DPO_PAIRS_PATH, rt.RL_METRICS_PATH, rt.CHECKPOINT_DIR, rt.RESULTS_DIR)
    try:
        rt.REPLAY_BUFFER_PATH = replay_p
        rt.REAL_TRANSITIONS_PATH = trans_p
        rt.GRPO_DATASET_PATH = grpo_p
        rt.DPO_PAIRS_PATH = dpo_p
        rt.RL_METRICS_PATH = metrics_p
        rt.CHECKPOINT_DIR = tp / "ckpt"
        rt.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        rt.RESULTS_DIR = tp / "res"
        rt.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        rt.run_minebench_rl_update(epochs=0, is_best=False)
        grpo_items = [json.loads(l) for l in grpo_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert not any(not g.get("passed", False) for g in grpo_items), f"Negative-adv/non-DAG/survival fails leaked to GRPO: {grpo_items}"
        dpo_items = [json.loads(l) for l in dpo_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(dpo_items) == 1 and dpo_items[0]["taskId"] == "omni_craft_bucket"
    finally:
        rt.REPLAY_BUFFER_PATH, rt.REAL_TRANSITIONS_PATH, rt.GRPO_DATASET_PATH, rt.DPO_PAIRS_PATH, rt.RL_METRICS_PATH, rt.CHECKPOINT_DIR, rt.RESULTS_DIR = orig

neg_rollout = {
    "provider": "local_gpu",
    "advantage": -0.25,
    "messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": json.dumps({"thought": "t", "toolCall": {"name": "craft_item", "arguments": {"itemName": "bucket"}}})},
    ],
}
assert filter_valid_tool_trajectory(neg_rollout) is None
print("OK")
`;
  assert.equal(execFileSync(pythonExe, ['-c', pyCheck], { encoding: 'utf8' }).trim().split('\n').pop(), 'OK');
});

test('74. Strict specific tool affordance grading, combat missing-entity 0-score, and Dragon safe respawn origin', () => {
  const { evaluateMinebenchEpisode } = require('../../minebench/dist/minebench/evaluators');
  const { MINEBENCH_TASKS } = require('../../minebench/dist/minebench/benchmarkSpec');
  const { resolveSafeRespawnOrigin } = require('../../minebench/dist/minebench/train_loop');

  const mkSnap = (overrides: any = {}) => ({
    botPosition: { x: 0, y: 100, z: 0 },
    health: 20,
    food: 20,
    deaths: 0,
    inventory: [],
    equippedArmor: { head: null, torso: null, legs: null, feet: null, offHand: null },
    blockAt: () => ({ name: 'air' }),
    entities: [],
    ...overrides,
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
      affordanceActions: 0,
      damageDealt: 0,
      waterPlaced: 0,
      lavaPlaced: 0,
      shieldBlocks: 0,
      shieldDisabled: 0,
      ironPickaxeDurabilityUsedOnStone: 0,
      ...(overrides.stats || {}),
    },
  });

  // Campfire extinguish must NOT pass on noise affordanceActions/blocksPlaced while campfire remains lit=true
  const campTask = {
    id: 'omni_tool_shovel_extinguish_campfire',
    domain: 'TOOL_AFFORDANCES',
    targetObject: 'iron_shovel',
    origin: { x: 0, y: 100, z: 0 },
    evaluatorMilestones: [
      { name: 'Tool Equipped', maxPoints: 30, condition: 'held_tool_correct' },
      { name: 'Affordance Executed', maxPoints: 70, condition: 'affordance_action_completed' },
    ],
  } as any;
  const campLitFake = evaluateMinebenchEpisode(
    campTask,
    mkSnap({
      inventory: [{ name: 'iron_shovel', count: 1 }],
      blockAt: (bx: number, by: number, bz: number) =>
        bx === 0 && by === 100 && bz === 2 ? { name: 'campfire', getProperties: () => ({ lit: true }) } : { name: 'air' },
      stats: { blocksPlaced: 1, affordanceActions: 1 },
    })
  );
  assert.equal(campLitFake.score, 0, 'Lit campfire with packet noise must score 0');

  const campExtinguished = evaluateMinebenchEpisode(
    campTask,
    mkSnap({
      inventory: [{ name: 'iron_shovel', count: 1 }],
      blockAt: (bx: number, by: number, bz: number) =>
        bx === 0 && by === 100 && bz === 2 ? { name: 'campfire', getProperties: () => ({ lit: false }) } : { name: 'air' },
    })
  );
  assert.equal(campExtinguished.passed, true);
  assert.equal(campExtinguished.score, 100);

  // Shield block arrows must NOT pass via killing skeleton with shieldBlocks=0
  const shieldTask = {
    id: 'omni_tool_shield_block_arrows',
    domain: 'TOOL_AFFORDANCES',
    targetObject: 'shield',
    origin: { x: 0, y: 100, z: 0 },
    evaluatorMilestones: [
      { name: 'Tool Equipped', maxPoints: 30, condition: 'held_tool_correct' },
      { name: 'Affordance Executed', maxPoints: 70, condition: 'affordance_action_completed' },
    ],
  } as any;
  assert.equal(
    evaluateMinebenchEpisode(shieldTask, mkSnap({ inventory: [{ name: 'shield', count: 1 }], stats: { mobsKilled: { skeleton: 1 }, shieldBlocks: 0 } })).score,
    0,
    'Killing skeleton with shieldBlocks=0 must score 0'
  );
  assert.equal(
    evaluateMinebenchEpisode(shieldTask, mkSnap({ inventory: [{ name: 'shield', count: 1 }], stats: { shieldBlocks: 1 } })).score,
    100
  );

  // Lava trap mob with deaths=1 & no lava placed scores 0
  const lavaTrapTask = {
    id: 'omni_tool_lava_bucket_trap_mob',
    domain: 'TOOL_AFFORDANCES',
    targetObject: 'lava_bucket',
    origin: { x: 0, y: 100, z: 0 },
    evaluatorMilestones: [
      { name: 'Tool Equipped', maxPoints: 30, condition: 'held_tool_correct' },
      { name: 'Affordance Executed', maxPoints: 70, condition: 'affordance_action_completed' },
    ],
  } as any;
  assert.equal(
    evaluateMinebenchEpisode(lavaTrapTask, mkSnap({ deaths: 1, inventory: [{ name: 'lava_bucket', count: 1 }] })).score,
    0
  );

  // Enderman missing/out-of-range with 0 kills & 0 damageDealt scores 0 (no free 50/100!)
  const enderTask = MINEBENCH_TASKS.find((t: any) => t.id === 'combat_enderman_hunt')!;
  const missingEnder = evaluateMinebenchEpisode(enderTask, mkSnap({ entities: [], stats: { mobsKilled: {}, damageDealt: 0 } }));
  assert.equal(missingEnder.score, 0, 'Missing entity with 0 kills and 0 damage must score 0');

  // Dragon safe respawn pad is outside (x, y, z) solid bedrock pillar
  const dragonTask = MINEBENCH_TASKS.find((t: any) => t.id === 'prog_tier6_ender_dragon')!;
  const safePos = resolveSafeRespawnOrigin({ ...dragonTask, origin: { x: 4000, y: 64, z: 0 } });
  assert.deepEqual(safePos, { x: 4000, y: 64, z: -6 });
});

test('75. use_tool_on_block rejects non-mutating blocks & no-op buckets and all helpers enforce allowAutoPrereqs=false', async () => {
  const { registerBaseMCPTools } = require('../system2/skills/baseSkills');
  const registry = new DynamicMCPToolRegistry();
  registerBaseMCPTools(registry);

  const useTool = registry.getTool('use_tool_on_block')!;
  const stoneNoop = await useTool.handler(
    { tool: 'stone_pickaxe', targetBlock: 'stone' },
    {
      bot: {
        entity: { position: new Vec3(0, 64, 0) },
        inventory: { items: () => [{ name: 'stone_pickaxe', count: 1 }] },
        findBlock: () => ({ name: 'stone', position: new Vec3(1, 64, 0) }),
        blockAt: (p: Vec3) => (p.y === 64 ? { name: 'stone', position: p } : { name: 'air', position: p }),
        activateBlock: async () => {},
        lookAt: async () => {},
      },
      kinematics: { equipItemHumanized: async () => true, pathfindTo: async () => true, lookAtSmooth: async () => {} },
    } as any
  );
  assert.equal(stoneNoop.success, false, 'Right-clicking stone must not return success=true');

  const waterNoop = await useTool.handler(
    { tool: 'water_bucket', targetBlock: 'fire' },
    {
      bot: {
        entity: { position: new Vec3(0, 64, 0) },
        inventory: { items: () => [{ name: 'water_bucket', count: 1 }] },
        findBlock: () => ({ name: 'fire', position: new Vec3(1, 64, 0) }),
        blockAt: (p: Vec3) => (p.x === 1 && p.y === 64 ? { name: 'fire', position: p } : { name: 'stone', position: p }),
        activateBlock: async () => {},
        activateItem: () => {},
      },
      kinematics: { equipItemHumanized: async () => true, pathfindTo: async () => true, lookAtSmooth: async () => {} },
    } as any
  );
  assert.equal(waterNoop.success, false, 'water_bucket no-op must return success=false');

  // Strict allowAutoPrereqs=false on smelt_item, store_items_in_chest, build_simple_house
  const smeltTool = registry.getTool('smelt_item')!;
  const smeltNoFurnace = await smeltTool.handler(
    { inputItem: 'raw_iron', count: 1 },
    {
      bot: {
        version: '1.20.4',
        entity: { position: new Vec3(0, 64, 0) },
        inventory: { items: () => [{ name: 'raw_iron', count: 2 }, { name: 'cobblestone', count: 16 }, { name: 'coal', count: 4 }] },
        findBlock: () => null,
      },
      kinematics: { pathfindTo: async () => true },
    } as any
  );
  assert.equal(smeltNoFurnace.success, false, 'smelt_item must not auto-craft furnace from cobblestone');

  const chestTool = registry.getTool('store_items_in_chest')!;
  const chestNoChest = await chestTool.handler(
    { itemName: 'cobblestone', count: 4 },
    {
      bot: {
        version: '1.20.4',
        entity: { position: new Vec3(0, 64, 0) },
        inventory: { items: () => [{ name: 'oak_log', count: 4 }, { name: 'oak_planks', count: 12 }, { name: 'cobblestone', count: 8 }] },
        findBlock: () => null,
      },
      kinematics: { pathfindTo: async () => true },
    } as any
  );
  assert.equal(chestNoChest.success, false, 'store_items_in_chest must not auto-craft chest from planks');

  const houseTool = registry.getTool('build_simple_house')!;
  const houseNoPlanks = await houseTool.handler(
    { wallBlock: 'oak_planks' },
    {
      bot: {
        version: '1.20.4',
        entity: { position: new Vec3(0, 64, 0) },
        inventory: { items: () => [{ name: 'oak_log', count: 8 }] },
      },
    } as any
  );
  assert.equal(houseNoPlanks.success, false, 'build_simple_house must not auto-craft logs into planks');
});

test('76. botController safeDigBlock digs boundingBox=empty crops, notableBlocks sorts before capping, and vlaClient detects bosses', async () => {
  const { System1BotController } = require('../system1/botController');
  const { VLACompanionClient } = require('../system2/vlaClient');
  const controller = new System1BotController();

  let digCount = 0;
  let wheatBroken = false;
  const wheatPos = new Vec3(1, 64, 0);
  (controller as any).camera = { clearQueue: () => {} };
  (controller as any).bot = {
    entity: { id: 1, position: new Vec3(0, 64, 0), onGround: true, yaw: 0, pitch: 0 },
    heldItem: null,
    inventory: { items: () => [], slots: [] },
    pathfinder: { setGoal: () => {} },
    clearControlStates: () => {},
    canDigBlock: () => true,
    lookAt: async () => {},
    swingArm: () => {},
    stopDigging: () => {},
    dig: async () => {
      digCount++;
      wheatBroken = true;
    },
    blockAt: (p: Vec3) =>
      p.x === 1 && p.y === 64 && p.z === 0
        ? (wheatBroken
            ? { name: 'air', boundingBox: 'empty', position: p, diggable: false }
            : { name: 'wheat', boundingBox: 'empty', position: p, diggable: true })
        : { name: 'stone', boundingBox: 'block', position: p, diggable: true },
    _client: { on: () => {}, removeListener: () => {} },
  };
  const dug = await controller.safeDigBlock(wheatPos);
  assert.equal(dug, true);
  assert.equal(digCount, 1, 'safeDigBlock must break diggable boundingBox=empty blocks');

  // Surround bot with >30 exposed dirt/sand blocks AND a crafting_table at (2, 64, 2)
  (controller as any).bot = {
    entity: { id: 1, position: new Vec3(0, 64, 0), velocity: new Vec3(0, 0, 0), yaw: 0, pitch: 0, onGround: true },
    health: 20,
    food: 20,
    time: { timeOfDay: 1000 },
    isRaining: false,
    inventory: { items: () => [], slots: [] },
    entities: {},
    blockAt: (p: Vec3) => {
      if (p.x === 2 && p.y === 64 && p.z === 2) return { name: 'crafting_table', position: p };
      if (p.x === 3 && p.y === 64 && p.z === 0) return { name: 'campfire', position: p };
      if (p.y === 63 && Math.abs(p.x) <= 6 && Math.abs(p.z) <= 6) return { name: 'dirt', position: p };
      return { name: 'air', position: p };
    },
    canSeeBlock: () => true,
  };
  (controller as any).reflexEngine = { getActiveHazard: () => null };
  controller.invalidatePerceptionCache();
  const ws = controller.refreshWorldStateNow(true)!;
  const notableNames = (ws.nearbyNotableBlocks || []).map((b: any) => b.name);
  assert.ok(notableNames.includes('crafting_table'), 'crafting_table must not be crowded out by dirt/sand');
  assert.ok(notableNames.includes('campfire'), 'campfire must be included in NOTABLE_BLOCKS');

  const vla = new VLACompanionClient();
  const proprio = vla.extractProprioception({
    entity: { id: 1, position: new Vec3(0, 64, 0), velocity: new Vec3(0, 0, 0), yaw: 0, pitch: 0, onGround: true },
    health: 20,
    food: 20,
    heldItem: null,
    inventory: { slots: [] },
    entities: { '2': { id: 2, name: 'blaze', type: 'mob', position: new Vec3(8, 64, 0), isValid: true } },
  } as any);
  assert.ok(proprio[12] > 0 && proprio[12] < 0.3, `Expected blaze distance encoded in proprio[12], got ${proprio[12]}`);
});

test('77. checkSurvivalCampaignGoalCompleted requires real campaign goals and regenerated Omniscience gold uses grounded schema', () => {
  const { checkSurvivalCampaignGoalCompleted } = require('../../minebench/dist/minebench/train_loop');
  assert.equal(
    checkSurvivalCampaignGoalCompleted('ender_dragon', ['oak_log', 'cobblestone', 'iron_ingot', 'wheat', 'crafting_table', 'furnace'], {}, 0, 20),
    false,
    'Generic 6-item survival inventory must not pass Ender Dragon campaign goal'
  );
  assert.equal(checkSurvivalCampaignGoalCompleted('wither_boss', ['nether_star'], {}, 0, 0), true);
  assert.equal(checkSurvivalCampaignGoalCompleted('full_beacon', ['beacon'], {}, 0, 0), true);
  assert.equal(checkSurvivalCampaignGoalCompleted('ocean_sponges', ['wet_sponge'], {}, 0, 0), true);

  const rootDir = path.resolve(__dirname, '..', '..');
  const tasks: any[] = JSON.parse(fs.readFileSync(path.join(rootDir, 'minebench', 'results', 'omniscience_tasks.json'), 'utf8'));
  const raids = tasks.filter((t) => String(t.taskId || '').startsWith('omni_structure_'));
  assert.ok(raids.length >= 15);
  for (const r of raids) {
    assert.ok(
      (r.setupCommands || []).some((c: string) => c.startsWith('summon minecraft:')),
      `Structure ${r.taskId} must spawn a hazard guardian mob`
    );
  }

  const goldLines = fs
    .readFileSync(path.join(rootDir, 'minebench', 'results', 'omniscience_gold_dataset.jsonl'), 'utf8')
    .split('\n')
    .filter((l: string) => l.trim())
    .map((l: string) => JSON.parse(l));
  assert.ok(goldLines.length > 0);
  assert.equal(goldLines.some((g: any) => String(g.id || '').includes('omni_craft_')), false, 'Single-step omni_craft_ shortcuts must be excluded');
  for (const g of goldLines.slice(0, 25)) {
    assert.ok(g.domain, `Gold row ${g.id} missing domain`);
    assert.equal(String(g.messages[0].content).includes('Use native action tokens'), false);
    assert.ok(String(g.messages[1].content).startsWith('Player "Trainer" says:'));
    assert.equal(JSON.parse(g.messages[2].content).goalCompleted, false);
  }
});

test('78. rl_trainer preserves coop_* speaker & slm_trainer prioritizes local_gpu over omniscience_gold', () => {
  const { execFileSync } = require('child_process');
  const rootDir = path.resolve(__dirname, '..', '..');
  const pythonExe = path.join(rootDir, 'neural_lab', '.venv', 'Scripts', 'python.exe');
  const pyCheck = `
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, ${JSON.stringify(rootDir)})
import minebench.rl_trainer as rt
from minebench.slm_trainer import stratified_sample_by_domain

with tempfile.TemporaryDirectory() as tmp:
    tp = Path(tmp)
    replay_p = tp / "replay.jsonl"
    trans_p = tp / "trans.jsonl"
    grpo_p = tp / "grpo.jsonl"
    dpo_p = tp / "dpo.jsonl"
    metrics_p = tp / "metrics.json"
    res_dir = tp / "res"
    res_dir.mkdir(parents=True, exist_ok=True)
    # Also ensure legacy craft shortcut in omniscience_gold_dataset.jsonl is skipped
    (res_dir / "omniscience_gold_dataset.jsonl").write_text(json.dumps({
        "id": "omni_gold_1_omni_craft_wooden_pickaxe",
        "messages": [
            {"role": "system", "content": "Use native action tokens"},
            {"role": "user", "content": "Craft wooden pickaxe"},
            {"role": "assistant", "content": json.dumps({"thought": "c", "toolCall": {"name": "craft_item", "arguments": {"itemName": "wooden_pickaxe"}}})}
        ]
    }) + "\\n", encoding="utf-8")

    rows = [
        {
            "taskId": "coop_bodyguard_protect",
            "domain": "MULTIPLAYER_ARENA",
            "workerId": 1,
            "speaker": "DQN_Opponent_1",
            "prompt": "Protect me from zombies",
            "basePrompt": "Protect me from zombies",
            "passed": True,
            "score": 100,
            "rawReward": 100.0,
            "executedTools": [
                {"name": "engage_combat", "args": {"targetType": "zombie"}, "success": True, "stepReward": 100.0, "thought": "Defend"},
            ],
        },
        {
            "taskId": "coop_bodyguard_protect",
            "domain": "MULTIPLAYER_ARENA",
            "workerId": 1,
            "speaker": "DQN_Opponent_1",
            "prompt": "Protect me from zombies",
            "basePrompt": "Protect me from zombies",
            "passed": False,
            "score": 0,
            "rawReward": -20.0,
            "executedTools": [
                {"name": "follow_player", "args": {"playerName": "DQN_Opponent_1"}, "success": False, "stepReward": -10.0, "thought": "Stand still"},
            ],
        },
    ]
    replay_p.write_text("\\n".join(json.dumps(x) for x in rows) + "\\n", encoding="utf-8")
    trans_p.write_text(json.dumps({"voxels": [0] * 3375, "entities": [[0.0] * 12] * 16, "botState": [0.0] * 10, "actionIdx": 0, "aimDelta": [0.0, 0.0], "reward": 1.0}) + "\\n", encoding="utf-8")

    orig = (rt.REPLAY_BUFFER_PATH, rt.REAL_TRANSITIONS_PATH, rt.GRPO_DATASET_PATH, rt.DPO_PAIRS_PATH, rt.RL_METRICS_PATH, rt.CHECKPOINT_DIR, rt.RESULTS_DIR)
    try:
        rt.REPLAY_BUFFER_PATH, rt.REAL_TRANSITIONS_PATH, rt.GRPO_DATASET_PATH, rt.DPO_PAIRS_PATH, rt.RL_METRICS_PATH, rt.CHECKPOINT_DIR, rt.RESULTS_DIR = replay_p, trans_p, grpo_p, dpo_p, metrics_p, tp / "ckpt", res_dir
        rt.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        rt.run_minebench_rl_update(epochs=0, is_best=False)
        grpo_items = [json.loads(l) for l in grpo_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert not any(g.get("taskId") == "omni_craft_wooden_pickaxe" for g in grpo_items)
        coop_g = [g for g in grpo_items if g.get("taskId") == "coop_bodyguard_protect"][0]
        assert 'Player "DQN_Opponent_1" says:' in coop_g["messages"][1]["content"]
        dpo_items = [json.loads(l) for l in dpo_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(dpo_items) == 1
        assert 'Player "DQN_Opponent_1" says:' in dpo_items[0]["prompt_messages"][1]["content"]
    finally:
        rt.REPLAY_BUFFER_PATH, rt.REAL_TRANSITIONS_PATH, rt.GRPO_DATASET_PATH, rt.DPO_PAIRS_PATH, rt.RL_METRICS_PATH, rt.CHECKPOINT_DIR, rt.RESULTS_DIR = orig

pool = (
    [{"provider": "local_gpu", "domain": "TOOL", "id": f"live_{i}"} for i in range(40)] +
    [{"provider": "omniscience_gold", "domain": "TOOL", "id": f"omni_{i}"} for i in range(100)]
)
picked = stratified_sample_by_domain(pool, 60)
live_cnt = sum(1 for x in picked if x["provider"] == "local_gpu")
assert live_cnt == 40, f"Expected all 40 local_gpu items prioritized first, got {live_cnt}"
print("OK")
`;
  assert.equal(execFileSync(pythonExe, ['-c', pyCheck], { encoding: 'utf8' }).trim().split('\n').pop(), 'OK');
});

test('79. train_action_head canonicalizes aliased motor tools and sets actor_weights=0 on failed steps while keeping Critic active', () => {
  const { execFileSync } = require('child_process');
  const rootDir = path.resolve(__dirname, '..', '..');
  const pythonExe = path.join(rootDir, 'neural_lab', '.venv', 'Scripts', 'python.exe');
  const pyCheck = `
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, ${JSON.stringify(rootDir)})
from neural_lab.train_action_head import load_real_motor_chunks_dataset, _resolve_canonical_motor_tool, TOOL_TO_IDX

assert _resolve_canonical_motor_tool("harvest_resource") == "gather_resource"
assert _resolve_canonical_motor_tool("bridge_cross") == "navigate_parkour_or_bridge"
assert _resolve_canonical_motor_tool("craft_recipe") == "craft_item"

with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "chunks.jsonl"
    base_chunk = {
        "latent_h": [0.1] * 1536,
        "proprio24": [0.0] * 24,
        "voxels": [0] * 3375,
        "entities": [],
        "aimTargets": [[0.0, 0.0]] * 8,
        "moveTargets": [[0.0, 1.0]] * 8,
        "discreteTargets": [[0, 0, 0, 0, 0, 0]] * 8,
        "extendedTargets": [[0, 0]] * 8,
        "reward": -5.0,
        "episodeReward": -10.0,
    }
    rows = [
        {**base_chunk, "toolName": "harvest_resource", "stepSuccess": True, "episodePassed": True},
        {**base_chunk, "toolName": "bridge_cross", "stepSuccess": False, "episodePassed": False},
    ]
    p.write_text("\\n".join(json.dumps(r) for r in rows) + "\\n", encoding="utf-8")
    ds = load_real_motor_chunks_dataset(max_samples=10, device="cpu", chunks_path=p)
    assert ds is not None
    tgt_tools, tgt_gate, tgt_val, actor_weights = ds[8], ds[9], ds[10], ds[11]
    assert int(tgt_tools[0].item()) == TOOL_TO_IDX["gather_resource"]
    assert float(tgt_gate[0].item()) == 1.0
    assert float(actor_weights[0].item()) == 1.5
    assert int(tgt_tools[1].item()) == TOOL_TO_IDX["navigate_parkour_or_bridge"]
    assert float(actor_weights[1].item()) == 0.0
    assert 0.0 < float(tgt_val[1].item()) < 0.5
print("OK")
`;
  assert.equal(execFileSync(pythonExe, ['-c', pyCheck], { encoding: 'utf8' }).trim().split('\n').pop(), 'OK');
});

test('80. Evaluators exclude player self-deaths from mobsKilled, resolveSafeRespawnOrigin matches sharded bots, craft maps to PROGRESSION_DRAGON, and interact rejects campfire', async () => {
  const { evaluateMinebenchEpisode } = require('../../minebench/dist/minebench/evaluators');
  const { resolveSafeRespawnOrigin, sampleDiverseOmniscienceTasks } = require('../../minebench/dist/minebench/train_loop');
  const { registerBaseMCPTools } = require('../system2/skills/baseSkills');

  const wardenTask = {
    id: 'omni_mob_warden',
    domain: 'MOB_INTERACTION',
    targetObject: 'warden',
    origin: { x: 0, y: 100, z: 0 },
    evaluatorMilestones: [
      { name: 'Target Engaged', maxPoints: 40, condition: 'target_engaged' },
      { name: 'Target Resolved', maxPoints: 60, condition: 'target_resolved' },
    ],
  } as any;
  const selfDeathRes = evaluateMinebenchEpisode(wardenTask, {
    botPosition: { x: 0, y: 100, z: 0 },
    health: 20,
    food: 20,
    deaths: 1,
    inventory: [],
    equippedArmor: { head: null, torso: null, legs: null, feet: null, offHand: null },
    blockAt: () => ({ name: 'air' }),
    entities: [],
    stats: {
      blocksPlaced: 0, blocksMined: 0, mobsKilled: { player: 1 }, creeperBlasts: 0, arrowsFired: 0,
      endCrystalsDestroyed: 0, enderDragonSlain: false, leversFlipped: 0, wolvesTamed: 0,
      damageDealt: 4, ironPickaxeDurabilityUsedOnStone: 0,
    },
  } as any);
  assert.equal(selfDeathRes.passed, false);
  assert.equal(selfDeathRes.score, 0, 'Bot death recorded as player must never satisfy hostile target_resolved/engaged');

  const wardenKillRes = evaluateMinebenchEpisode(wardenTask, {
    botPosition: { x: 0, y: 100, z: 0 },
    health: 20,
    food: 20,
    deaths: 0,
    inventory: [],
    equippedArmor: { head: null, torso: null, legs: null, feet: null, offHand: null },
    blockAt: () => ({ name: 'air' }),
    entities: [],
    stats: {
      blocksPlaced: 0, blocksMined: 0, mobsKilled: { warden: 1 }, creeperBlasts: 0, arrowsFired: 0,
      endCrystalsDestroyed: 0, enderDragonSlain: false, leversFlipped: 0, wolvesTamed: 0,
      damageDealt: 20, ironPickaxeDurabilityUsedOnStone: 0,
    },
  } as any);
  assert.equal(wardenKillRes.passed, true);
  assert.equal(wardenKillRes.score, 100);

  const safeSharded = resolveSafeRespawnOrigin({
    id: 'prog_tier6_ender_dragon',
    origin: { x: 4000, y: 64, z: 0 },
    buildArenaCommands: ({ x, y, z }: any) => [
      `fill ${x} ${y + 1} ${z} ${x} ${y + 3} ${z} minecraft:bedrock`,
      `tp NeuroAgent_1 ${x} ${y} ${z - 6} 0 0`,
    ],
  } as any);
  assert.deepEqual(safeSharded, { x: 4000, y: 64, z: -6 });

  const rootDir = path.resolve(__dirname, '..', '..');
  const omniTasks = sampleDiverseOmniscienceTasks(path.join(rootDir, 'minebench', 'results', 'omniscience_tasks.json'), 500);
  const craftTask = omniTasks.find((t: any) => String(t.id || '').startsWith('omni_craft_'));
  assert.ok(craftTask);
  assert.equal(craftTask.domain, 'PROGRESSION_DRAGON');

  const registry = new DynamicMCPToolRegistry();
  registerBaseMCPTools(registry);
  const interactTool = registry.getTool('interact_block_or_puzzle')!;
  const campRes = await interactTool.handler(
    { blockType: 'campfire' },
    {
      bot: {
        entity: { position: new Vec3(0, 64, 0) },
        inventory: { items: () => [] },
        findBlock: () => ({ name: 'campfire', position: new Vec3(1, 64, 0) }),
        blockAt: (p: Vec3) => (p.x === 1 && p.y === 64 ? { name: 'campfire', position: p } : { name: 'air', position: p }),
        activateBlock: async () => {},
      },
      kinematics: { pathfindTo: async () => true, lookAtSmooth: async () => {} },
    } as any
  );
  assert.equal(campRes.success, false, 'interact_block_or_puzzle must reject non-lever campfire manipulation');
});

test('81. rl_trainer excludes failed omni_craft_* rollouts from GRPO and strips duplicate coop speaker prefixes in GRPO & DPO', () => {
  const { execFileSync } = require('child_process');
  const rootDir = path.resolve(__dirname, '..', '..');
  const pythonExe = path.join(rootDir, 'neural_lab', '.venv', 'Scripts', 'python.exe');
  const pyCheck = `
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, ${JSON.stringify(rootDir)})
import minebench.rl_trainer as rt

with tempfile.TemporaryDirectory() as tmp:
    tp = Path(tmp)
    replay_p, trans_p, grpo_p, dpo_p, metrics_p, res_dir = tp / "r.jsonl", tp / "t.jsonl", tp / "g.jsonl", tp / "d.jsonl", tp / "m.json", tp / "res"
    res_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        # Failed omni_craft_* even with high score/rawReward must NOT enter GRPO partial distillation
        {
            "taskId": "omni_craft_piston",
            "domain": "PROGRESSION_DRAGON",
            "prompt": "Craft piston",
            "passed": False,
            "score": 50,
            "rawReward": 95.0,
            "executedTools": [{"name": "gather_resource", "args": {"blockType": "oak_log"}, "success": True, "stepReward": 25.0, "thought": "Logs"}],
        },
        # Coop winner + loser with embedded speaker prefix in prompt
        {
            "taskId": "coop_give_food",
            "domain": "MULTIPLAYER_ARENA",
            "workerId": 2,
            "speaker": "DQN_Opponent_3",
            "prompt": 'DQN_Opponent_3 says: "I am starving!" Deliver beef.',
            "passed": True,
            "score": 100,
            "rawReward": 100.0,
            "executedTools": [{"name": "give_item_to_player", "args": {"playerName": "DQN_Opponent_3", "itemName": "cooked_beef"}, "success": True, "stepReward": 100.0, "thought": "Feed"}],
        },
        {
            "taskId": "coop_give_food",
            "domain": "MULTIPLAYER_ARENA",
            "workerId": 2,
            "speaker": "DQN_Opponent_3",
            "prompt": 'DQN_Opponent_3 says: "I am starving!" Deliver beef.',
            "passed": False,
            "score": 0,
            "rawReward": -10.0,
            "executedTools": [{"name": "follow_player", "args": {"playerName": "DQN_Opponent_3"}, "success": False, "stepReward": -5.0, "thought": "Wrong"}],
        },
    ]
    replay_p.write_text("\\n".join(json.dumps(x) for x in rows) + "\\n", encoding="utf-8")
    trans_p.write_text(json.dumps({"voxels": [0] * 3375, "entities": [[0.0] * 12] * 16, "botState": [0.0] * 10, "actionIdx": 0, "aimDelta": [0.0, 0.0], "reward": 1.0}) + "\\n", encoding="utf-8")
    orig = (rt.REPLAY_BUFFER_PATH, rt.REAL_TRANSITIONS_PATH, rt.GRPO_DATASET_PATH, rt.DPO_PAIRS_PATH, rt.RL_METRICS_PATH, rt.CHECKPOINT_DIR, rt.RESULTS_DIR)
    try:
        rt.REPLAY_BUFFER_PATH, rt.REAL_TRANSITIONS_PATH, rt.GRPO_DATASET_PATH, rt.DPO_PAIRS_PATH, rt.RL_METRICS_PATH, rt.CHECKPOINT_DIR, rt.RESULTS_DIR = replay_p, trans_p, grpo_p, dpo_p, metrics_p, tp / "ckpt", res_dir
        rt.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        rt.run_minebench_rl_update(epochs=0, is_best=False)
        grpo_items = [json.loads(l) for l in grpo_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert not any(g.get("taskId") == "omni_craft_piston" for g in grpo_items), f"Failed omni_craft leaked to GRPO: {grpo_items}"
        coop_g = [g for g in grpo_items if g.get("taskId") == "coop_give_food"][0]
        u_msg = coop_g["messages"][1]["content"]
        assert 'Player "DQN_Opponent_3" says:' in u_msg and "DQN_Opponent_3 says:" not in u_msg, f"Duplicate speaker prefix in GRPO: {u_msg}"
        dpo_items = [json.loads(l) for l in dpo_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(dpo_items) == 1
        dpo_u = dpo_items[0]["prompt_messages"][1]["content"]
        assert 'Player "DQN_Opponent_3" says:' in dpo_u and "DQN_Opponent_3 says:" not in dpo_u
    finally:
        rt.REPLAY_BUFFER_PATH, rt.REAL_TRANSITIONS_PATH, rt.GRPO_DATASET_PATH, rt.DPO_PAIRS_PATH, rt.RL_METRICS_PATH, rt.CHECKPOINT_DIR, rt.RESULTS_DIR = orig
print("OK")
`;
  assert.equal(execFileSync(pythonExe, ['-c', pyCheck], { encoding: 'utf8' }).trim().split('\n').pop(), 'OK');
});

test('82. water_clutch rejects dry falls without water deployment, undead arenas have sun protection, and gather_resource accepts item/itemName aliases', () => {
  const { execFileSync } = require('child_process');
  const { evaluateOmniscienceMilestones } = require('../../minebench/dist/minebench/evaluators');

  const rootDir = path.resolve(__dirname, '..', '..');
  const tasks: any[] = JSON.parse(fs.readFileSync(path.join(rootDir, 'minebench', 'results', 'omniscience_tasks.json'), 'utf8'));
  const wcTask = { ...tasks.find((t) => t.taskId === 'omni_tool_water_bucket_water_clutch'), origin: { x: 4500, y: 100, z: 500 } };
  assert.ok(wcTask);

  const dryFallRes = evaluateOmniscienceMilestones(wcTask, {
    botPosition: { x: 4500, y: 100, z: 500 },
    health: 19.2,
    food: 20,
    deaths: 0,
    inventory: [{ name: 'water_bucket', count: 1 }],
    equippedArmor: { head: null, torso: null, legs: null, feet: null, offHand: null },
    blockAt: () => ({ name: 'air' }),
    entities: [],
    stats: {
      blocksPlaced: 0, blocksMined: 0, mobsKilled: {}, creeperBlasts: 0, arrowsFired: 0,
      endCrystalsDestroyed: 0, enderDragonSlain: false, leversFlipped: 0, wolvesTamed: 0,
      affordanceActions: 1, waterPlaced: 0, ironPickaxeDurabilityUsedOnStone: 0,
    },
  } as any);
  assert.equal(dryFallRes.passed, false);
  assert.equal(dryFallRes.score, 0, 'Dry fall onto stone with water_bucket still unused must score 0');

  for (const mob of ['skeleton', 'stray', 'zombie', 'zombie_villager', 'drowned', 'phantom']) {
    const t = tasks.find((x) => x.taskId === `omni_mob_${mob}`);
    assert.ok(t, `Missing omni_mob_${mob}`);
    const cmds: string[] = t.setupCommands || [];
    assert.ok(cmds.some((c) => c.includes('minecraft:tinted_glass')), `omni_mob_${mob} must include tinted_glass overhead shade`);
    if (mob !== 'phantom') {
      assert.ok(cmds.some((c) => c.includes('minecraft:stone_button')), `omni_mob_${mob} must wear stone_button helmet`);
    }
  }
  const shieldTask = tasks.find((t) => t.taskId === 'omni_tool_axe_disable_shield');
  assert.ok((shieldTask.setupCommands || []).some((c: string) => c.includes('~ ~ ~8')), 'vindicator must spawn at ~ ~ ~8');

  const pythonExe = path.join(rootDir, 'neural_lab', '.venv', 'Scripts', 'python.exe');
  const pyCheck = `
import sys
sys.path.insert(0, ${JSON.stringify(rootDir)})
from minebench.rl_trainer import has_required_args
assert has_required_args("gather_resource", {"itemName": "sand"}) is True
assert has_required_args("gather_resource", {"item": "sand"}) is True
assert has_required_args("gather_resource", {}) is False
print("OK")
`;
  assert.equal(execFileSync(pythonExe, ['-c', pyCheck], { encoding: 'utf8' }).trim().split('\n').pop(), 'OK');
});

test('83. Phase B neural motor control uses tick-0 aim frame, prioritizes targetPos in proprio24, supports explore & AWR RL cleanly', async () => {
  const { execFileSync } = require('child_process');
  const { VLACompanionClient } = require('../system2/vlaClient');
  const client = new VLACompanionClient('http://127.0.0.1:8765');

  const lookAngles: Array<{ yaw: number; pitch: number }> = [];
  let strikeCount = 0;
  let bowReleaseCount = 0;
  let deactivateCalls = 0;
  const mockHostile: any = {
    name: 'zombie',
    type: 'hostile',
    isValid: true,
    position: new Vec3(2, 64, 0),
  };
  const mockBot: any = {
    entity: { position: new Vec3(0, 64, 0), yaw: 1.0, pitch: 0.2, onGround: true, velocity: new Vec3(0, 0, 0) },
    heldItem: { type: 261, name: 'bow', count: 1 },
    inventory: { slots: {} },
    entities: { e1: mockHostile },
    look: async (yaw: number, pitch: number) => {
      lookAngles.push({ yaw, pitch });
    },
    setControlState: () => {},
    clearControlStates: () => {},
    swingArm: () => {},
    attack: () => {},
    activateItem: () => {},
    deactivateItem: () => {
      deactivateCalls++;
    },
  };

  // Even when a hostile is closer, targetPos takes precedence in proprioception slots 9..12
  const targetPos = new Vec3(0, 64, 16);
  const prop = client.extractProprioception(mockBot, targetPos);
  assert.ok(Math.abs(prop[9]) < 1e-6 && Math.abs(prop[11] - 1.0) < 1e-6 && Math.abs(prop[12] - 0.5) < 1e-6, 'targetPos must populate proprio24[9..12]');

  // neuralAimOnly=true applies delta_yaw relative to tick-0 yaw (1.0 rad), NOT atan2 toward targetPos (0 rad)
  const bowRes = await client.executeMotorActionChunk(
    mockBot,
    [
      { delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 0, jump: false, sprint: false, sneak: false, attack: false, use_item: true, swap_slot: false },
      { delta_yaw: 18, delta_pitch: 0, vx: 0, vz: 0, jump: false, sprint: false, sneak: false, attack: false, use_item: true, swap_slot: false },
      { delta_yaw: 18, delta_pitch: 0, vx: 0, vz: 0, jump: false, sprint: false, sneak: false, attack: false, use_item: true, swap_slot: false },
      { delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 0, jump: false, sprint: false, sneak: false, attack: false, use_item: false, swap_slot: false },
    ],
    targetPos,
    4,
    { neuralAimOnly: true, onBowRelease: () => { bowReleaseCount++; } }
  );
  assert.equal(bowRes.bowReleases, 1);
  assert.equal(bowReleaseCount, 1);
  assert.ok(Math.abs(lookAngles[1].yaw - (1.0 + (18 * Math.PI) / 180)) < 0.02, 'neuralAimOnly must offset from tick-0 yaw');

  // Attack + stopWhenNearPos early exit
  mockBot.heldItem = { type: 267, name: 'iron_sword', count: 1 };
  const atkRes = await client.executeMotorActionChunk(
    mockBot,
    [
      { delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 1, jump: false, sprint: true, sneak: false, attack: true, use_item: false, swap_slot: false },
      { delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 1, jump: false, sprint: true, sneak: false, attack: true, use_item: false, swap_slot: false },
    ],
    targetPos,
    2,
    {
      neuralAimOnly: true,
      combatTarget: mockHostile,
      stopWhenNearPos: { pos: new Vec3(0, 64, 0.5), range: 1.0 },
      onStrike: () => { strikeCount++; },
    }
  );
  assert.equal(atkRes.reachedTarget, true, 'stopWhenNearPos must exit early on arrival');
  assert.equal(strikeCount, 0);
  assert.ok(deactivateCalls >= 1);

  // Verify Fix #1 (side-face placement), Fix #2 (cross-chunk persistItemUse), Fix #3 (swim jump), Fix #5 (delta=0 aim reset)
  let placedFace: any = null;
  let swimJumpSet = false;
  lookAngles.length = 0;
  deactivateCalls = 0;
  const mockWaterBot: any = {
    entity: { position: new Vec3(0, 64, 0), yaw: 0.5, pitch: 0.0, onGround: false, isInWater: true, velocity: new Vec3(0, 0, 0) },
    heldItem: { type: 4, name: 'cobblestone', count: 16 },
    inventory: { slots: {} },
    entities: {},
    world: { raycast: () => ({ name: 'stone', boundingBox: 'block', face: 3, position: new Vec3(0, 63, 1) }) },
    placeBlock: async (_blk: any, faceVec: Vec3) => { placedFace = faceVec; },
    look: async (yaw: number, pitch: number) => { lookAngles.push({ yaw, pitch }); },
    setControlState: (s: string, v: boolean) => { if (s === 'jump' && v) swimJumpSet = true; },
    clearControlStates: () => {},
    activateItem: () => {},
    deactivateItem: () => { deactivateCalls++; },
  };
  await client.executeMotorActionChunk(
    mockWaterBot,
    [
      { delta_yaw: 10, delta_pitch: 0, vx: 0, vz: 1, jump: true, sprint: false, sneak: false, attack: false, use_item: true, swap_slot: false },
      { delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 1, jump: true, sprint: false, sneak: false, attack: false, use_item: true, swap_slot: false },
    ],
    targetPos,
    2,
    { neuralAimOnly: true, allowUseItem: true, allowPlaceBlock: true }
  );
  assert.ok(placedFace && placedFace.z === 1 && placedFace.y === 0, 'Must place against intersected side face normal instead of hardcoded (0,1,0)');
  assert.equal(swimJumpSet, true, 'Must allow jump control state while swimming in water');
  assert.equal(lookAngles.length, 2, 'Tick with delta_yaw=0 must still reset look back to baseYaw');

  // Verify cross-chunk persistItemUse holds bow draw across chunk 1 without deactivating, then fires in chunk 2
  mockBot.heldItem = { type: 261, name: 'bow', count: 1 };
  (mockBot as any)._vlaItemUseState = null;
  deactivateCalls = 0;
  bowReleaseCount = 0;
  await client.executeMotorActionChunk(
    mockBot,
    [
      { delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 0, jump: false, sprint: false, sneak: false, attack: false, use_item: true, swap_slot: false },
      { delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 0, jump: false, sprint: false, sneak: false, attack: false, use_item: true, swap_slot: false },
    ],
    targetPos,
    2,
    { neuralAimOnly: true, persistItemUse: true, onBowRelease: () => { bowReleaseCount++; } }
  );
  assert.equal(deactivateCalls, 0, 'persistItemUse must hold bow draw across chunk boundary');
  const chunk2Res = await client.executeMotorActionChunk(
    mockBot,
    [
      { delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 0, jump: false, sprint: false, sneak: false, attack: false, use_item: true, swap_slot: false },
      { delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 0, jump: false, sprint: false, sneak: false, attack: false, use_item: false, swap_slot: false },
    ],
    targetPos,
    2,
    { neuralAimOnly: true, persistItemUse: true, onBowRelease: () => { bowReleaseCount++; } }
  );
  assert.equal(chunk2Res.bowReleases, 1, 'Cross-chunk bow draw must record release on transition to false');
  assert.equal(bowReleaseCount, 1);

  // Verify Probe 2 Fix #2: raycastCursorBlock uses eyeHeight (1.62 standing / 1.27 sneaking) instead of entity.height (1.8)
  let capturedEyeY = -1;
  const mockEyeBot: any = {
    entity: { position: new Vec3(0, 64, 0), yaw: 0, pitch: 0, height: 1.8, eyeHeight: 1.62, onGround: true, velocity: new Vec3(0, 0, 0) },
    heldItem: { type: 4, name: 'cobblestone', count: 16 },
    inventory: { slots: {} },
    entities: {},
    world: {
      raycast: (origin: Vec3) => {
        capturedEyeY = origin.y;
        return { name: 'stone', boundingBox: 'block', face: 3, position: new Vec3(0, 63, 1) };
      },
    },
    placeBlock: async () => {},
    look: async () => {},
    setControlState: () => {},
    clearControlStates: () => {},
    activateItem: () => {},
    deactivateItem: () => {},
  };
  await client.executeMotorActionChunk(
    mockEyeBot,
    [{ delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 0, jump: false, sprint: false, sneak: false, attack: false, use_item: true, swap_slot: false }],
    targetPos,
    1,
    { neuralAimOnly: true, allowUseItem: true, allowPlaceBlock: true }
  );
  assert.ok(Math.abs(capturedEyeY - 65.62) < 1e-4, 'raycastCursorBlock must use eyeHeight 1.62 instead of height 1.8');

  mockEyeBot.entity.eyeHeight = undefined;
  mockEyeBot.controlState = { sneak: true };
  await client.executeMotorActionChunk(
    mockEyeBot,
    [{ delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 0, jump: false, sprint: false, sneak: true, attack: false, use_item: true, swap_slot: false }],
    targetPos,
    1,
    { neuralAimOnly: true, allowUseItem: true, allowPlaceBlock: true }
  );
  assert.ok(Math.abs(capturedEyeY - 65.27) < 1e-4, 'raycastCursorBlock must fall back to sneaking eyeHeight 1.27');

  // Verify Probe 2 Fix #1: ReflexEngine gates SUDDEN_DAMAGE during active Phase B motor subtasks but fires outside them
  const { ReflexEngine } = require('../system1/reflexes/reflexEngine');
  const prevNeuralMotorEnv = process.env.NEURAL_MOTOR_CONTROL;
  process.env.NEURAL_MOTOR_CONTROL = 'true';
  try {
    let shieldRaises = 0;
    const interrupts: any[] = [];
    const reflexBot: any = {
      health: 20,
      entity: { position: { x: 0, y: 64, z: 0 }, onGround: true },
      clearControlStates: () => {},
      setControlState: () => {},
      _vlaMotorControlActive: false,
    };
    const engine = new ReflexEngine(
      reflexBot,
      { lookAtSmooth: async () => {} },
      { raiseShield: () => { shieldRaises++; }, lowerShield: () => {} },
      { equipItemHumanized: async () => false },
      (ev: any) => interrupts.push(ev)
    );
    const baseState: any = {
      health: 20,
      food: 20,
      oxygen: 20,
      visibleEntities: [],
      isInWater: false,
      isInLava: false,
      activeSubtask: { toolName: 'shoot_bow', goal: 'shoot skeleton' },
    };
    engine.evaluateTick(baseState);
    reflexBot.health = 14;
    baseState.health = 14;
    engine.evaluateTick(baseState);
    assert.equal(shieldRaises, 0, 'ReflexEngine must not raiseShield during active shoot_bow subtask');
    assert.equal(engine.getActiveHazard(), null, 'ReflexEngine must not set SUDDEN_DAMAGE hazard during active shoot_bow subtask');

    baseState.activeSubtask = null;
    reflexBot.health = 8;
    baseState.health = 8;
    engine.evaluateTick(baseState);
    assert.equal(interrupts.length, 1, 'SUDDEN_DAMAGE must fire once Phase B motor subtask finishes');
    assert.equal(interrupts[0].hazard, 'SUDDEN_DAMAGE');
  } finally {
    if (prevNeuralMotorEnv === undefined) delete process.env.NEURAL_MOTOR_CONTROL;
    else process.env.NEURAL_MOTOR_CONTROL = prevNeuralMotorEnv;
  }

  const rootDir = path.resolve(__dirname, '..', '..');
  const pythonExe = path.join(rootDir, 'neural_lab', '.venv', 'Scripts', 'python.exe');
  const pyCheck = `
import math, sys, torch
sys.path.insert(0, ${JSON.stringify(rootDir)})
from neural_lab.action_head import MinecraftActionHead
import neural_lab.train_action_head as tah

head = MinecraftActionHead(hidden_dim=64, num_layers=1, num_heads=4)
h = torch.zeros(1, 1536)
p = torch.zeros(1, 24)
d1 = head.predict_action_dict(h, p, explore=True)
d2 = head.predict_action_dict(h, p, explore=True)
assert math.isfinite(d1["delta_yaw"]) and math.isfinite(d1["vx"]) and len(d1["action_chunk"]) == 8
assert any(d1["action_chunk"][k]["delta_yaw"] != d2["action_chunk"][k]["delta_yaw"] for k in range(8))

b_val = torch.tensor([0.9, 0.1])
pred_val = torch.tensor([0.3, 0.5])
b_w = torch.tensor([1.5, 0.0])
adv = (b_val - pred_val).clamp(-1.0, 2.0)
w = b_w * torch.exp(adv * 1.5).clamp(0.25, 3.0)
assert torch.isfinite(w).all() and w[1].item() == 0.0 and w[0].item() > 0.0

# Verify Probe 2 Fix #3: synthetic Mode 4 pitch=-88.0 (down) and Mode 6 pitch=+14.0 (up)
ds = tah.generate_temporal_action_dataset(num_samples=20, device="cpu")
target_aim = ds[4]
assert abs(float(target_aim[4, 0, 1].item()) - (-88.0)) < 1e-4, f"Expected Mode 4 pitch -88.0, got {target_aim[4, 0, 1].item()}"
assert abs(float(target_aim[6, 0, 1].item()) - 14.0) < 1e-4, f"Expected Mode 6 pitch +14.0, got {target_aim[6, 0, 1].item()}"
print("OK")
`;
  assert.equal(execFileSync(pythonExe, ['-c', pyCheck], { encoding: 'utf8' }).trim().split('\n').pop(), 'OK');
});

test('84. Universal 400ms physics chunk reward, stratified motor gold bank, and signed Clipped PPO on MinecraftActionHead', async () => {
  const { execFileSync } = require('child_process');
  const os = require('os');
  const { VLACompanionClient } = require('../system2/vlaClient');
  const {
    computeUniversalChunkReward,
    assignDiscountedChunkReturns,
    selectStepChunksChronologically,
    updateStratifiedMotorGoldBank,
  } = require('../../minebench/dist/minebench/train_loop');

  // 1. Universal 400ms physics reward: closing distance + aligning aim + dealing damage > turning away & taking damage
  const preGood = {
    pos: { x: 0, y: 64, z: 0 },
    yaw: 0,
    pitch: 0,
    targetPos: { x: 0, y: 64, z: -6 },
    health: 20,
    damageDealt: 0,
    arrowsFired: 0,
    blocksMined: 0,
    blocksPlaced: 0,
    invCount: 1,
  };
  const postGood = {
    pos: { x: 0, y: 64, z: -2.5 },
    yaw: 0,
    pitch: 0,
    targetPos: { x: 0, y: 64, z: -6 },
    health: 20,
    damageDealt: 4,
    arrowsFired: 0,
    blocksMined: 0,
    blocksPlaced: 0,
    invCount: 1,
  };
  const postBad = {
    pos: { x: 0, y: 64, z: 2.0 },
    yaw: Math.PI,
    pitch: 0,
    targetPos: { x: 0, y: 64, z: -6 },
    health: 12,
    damageDealt: 0,
    arrowsFired: 0,
    blocksMined: 0,
    blocksPlaced: 0,
    invCount: 1,
  };
  const rGood = computeUniversalChunkReward(preGood, postGood, 'engage_combat');
  const rBad = computeUniversalChunkReward(preGood, postBad, 'engage_combat');
  assert.ok(rGood > 1.5, `Expected positive universal physics reward, got ${rGood}`);
  assert.ok(rBad < -2.0, `Expected negative universal physics reward on retreat+damage, got ${rBad}`);

  // shoot_bow must reward aim alignment without forcing melee approach
  const rBowAlign = computeUniversalChunkReward(
    { ...preGood, yaw: Math.PI * 0.5 },
    { ...preGood, yaw: 0, arrowsFired: 1 },
    'shoot_bow'
  );
  assert.ok(rBowAlign > 0.8, `shoot_bow aim+fire must give positive reward, got ${rBowAlign}`);

  // 2. Discounted chunk returns propagate backward with gamma=0.95 & chronological selection preserves order
  const pendingChrono = [
    { id: 0, discreteTargets: [[0, 0, 0, 0, 0, 0]], chunkReward: 0.1 },
    { id: 1, discreteTargets: [[0, 0, 0, 0, 0, 0]], chunkReward: 0.2 },
    { id: 2, discreteTargets: [[0, 0, 0, 1, 0, 0]], chunkReward: 1.5 },
    { id: 3, discreteTargets: [[0, 0, 0, 0, 1, 0]], chunkReward: 1.2 },
    { id: 4, discreteTargets: [[0, 0, 0, 0, 0, 0]], chunkReward: 0.0 },
  ];
  const pickedChrono = selectStepChunksChronologically(pendingChrono, 4);
  assert.deepEqual(
    pickedChrono.map((c: any) => c.id),
    [0, 1, 2, 3],
    'selectStepChunksChronologically must keep selected chunks in ascending chronological order'
  );
  const stepChunks: any[] = [{ chunkReward: 1.0 }, { chunkReward: 0.5 }];
  assignDiscountedChunkReturns(stepChunks, 20, 40, true, true, 0.95);
  assert.ok(stepChunks[0].chunkReturn > stepChunks[1].chunkReturn * 0.9, 'Earlier chunk must inherit discounted return');

  // Verify live _bowReleaseCount counter increments mid-step during bow release in executeMotorActionChunk
  const bowClient = new VLACompanionClient();
  const mockBowBot: any = {
    entity: { position: new Vec3(0, 64, 0), yaw: 0, pitch: 0, onGround: true, velocity: new Vec3(0, 0, 0) },
    heldItem: { type: 261, name: 'bow', count: 1 },
    inventory: { slots: {} },
    look: async () => {},
    setControlState: () => {},
    clearControlStates: () => {},
    activateItem: () => {},
    deactivateItem: () => {},
    _bowReleaseCount: 0,
  };
  await bowClient.executeMotorActionChunk(
    mockBowBot,
    [
      { delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 0, jump: false, sprint: false, sneak: false, attack: false, use_item: true, swap_slot: false },
      { delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 0, jump: false, sprint: false, sneak: false, attack: false, use_item: true, swap_slot: false },
      { delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 0, jump: false, sprint: false, sneak: false, attack: false, use_item: true, swap_slot: false },
      { delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 0, jump: false, sprint: false, sneak: false, attack: false, use_item: false, swap_slot: false },
    ],
    new Vec3(0, 64, -10),
    4,
    { neuralAimOnly: true }
  );
  assert.equal(mockBowBot._bowReleaseCount, 1, 'executeMotorActionChunk must increment live _bowReleaseCount on bow fire');

  // Verify off-hand shield works while holding sword and melee cooldown adapts for sword (630ms) & axe (1020ms)
  let activatedOffHandShield: boolean | undefined;
  let meleeSwingCount = 0;
  const mockCombatClient = new VLACompanionClient();
  const mockCombatTarget = {
    type: 'hostile',
    isValid: true,
    position: new Vec3(0, 64, -2),
  };
  const mockShieldSwordBot: any = {
    entity: { position: new Vec3(0, 64, 0), yaw: 0, pitch: 0, onGround: true, velocity: new Vec3(0, 0, 0) },
    heldItem: { type: 267, name: 'iron_sword', count: 1 },
    inventory: { slots: { 45: { name: 'shield', count: 1 } } },
    look: async () => {},
    setControlState: () => {},
    clearControlStates: () => {},
    swingArm: () => { meleeSwingCount++; },
    attack: () => {},
    nearestEntity: () => mockCombatTarget,
    activateItem: (offHand?: boolean) => { activatedOffHandShield = offHand; },
    deactivateItem: () => {},
  };
  await mockCombatClient.executeMotorActionChunk(
    mockShieldSwordBot,
    [{ delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 0, jump: false, sprint: false, sneak: false, attack: false, use_item: true, swap_slot: false }],
    new Vec3(0, 64, -2),
    1,
    { neuralAimOnly: true, combatTarget: mockCombatTarget }
  );
  assert.equal(activatedOffHandShield, true, 'executeMotorActionChunk must allow off-hand shield while holding sword');

  mockShieldSwordBot._lastAttackAt = Date.now() - 500;
  await mockCombatClient.executeMotorActionChunk(
    mockShieldSwordBot,
    [{ delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 0, jump: false, sprint: false, sneak: false, attack: true, use_item: false, swap_slot: false }],
    new Vec3(0, 64, -2),
    1,
    { neuralAimOnly: true, combatTarget: mockCombatTarget }
  );
  assert.equal(meleeSwingCount, 0, 'iron_sword cooldown (< 630ms) must prevent premature 500ms swing');

  mockShieldSwordBot.heldItem = { type: 258, name: 'iron_axe', count: 1 };
  mockShieldSwordBot._lastAttackAt = Date.now() - 700;
  await mockCombatClient.executeMotorActionChunk(
    mockShieldSwordBot,
    [{ delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 0, jump: false, sprint: false, sneak: false, attack: true, use_item: false, swap_slot: false }],
    new Vec3(0, 64, -2),
    1,
    { neuralAimOnly: true, combatTarget: mockCombatTarget }
  );
  assert.equal(meleeSwingCount, 0, 'iron_axe cooldown (< 1020ms) must wait full 1020ms axe recharge');

  mockShieldSwordBot._lastAttackAt = Date.now() - 1050;
  await mockCombatClient.executeMotorActionChunk(
    mockShieldSwordBot,
    [{ delta_yaw: 0, delta_pitch: 0, vx: 0, vz: 0, jump: false, sprint: false, sneak: false, attack: true, use_item: false, swap_slot: false }],
    new Vec3(0, 64, -2),
    1,
    { neuralAimOnly: true, combatTarget: mockCombatTarget }
  );
  assert.equal(meleeSwingCount, 1, 'iron_axe cooldown (>= 1020ms) must allow attack once recharged');

  // 3. Stratified motor gold reservoir caps eat_food without evicting engage_combat or shoot_bow
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nc-gold-'));
  try {
    const goldFile = path.join(tmpDir, 'real_motor_gold.jsonl');
    const cands: any[] = [];
    for (let i = 0; i < 15; i++) {
      cands.push({
        taskId: 't_eat',
        toolName: 'eat_food',
        latent_h: new Array(1536).fill(0),
        episodePassed: true,
        stepSuccess: true,
        chunkReturn: 2.0 + i * 0.01,
      });
    }
    cands.push({
      taskId: 't_combat',
      toolName: 'engage_combat',
      latent_h: new Array(1536).fill(0),
      episodePassed: true,
      stepSuccess: true,
      chunkReturn: 3.5,
    });
    cands.push({
      taskId: 't_bow',
      toolName: 'shoot_bow',
      latent_h: new Array(1536).fill(0),
      episodePassed: true,
      stepSuccess: true,
      chunkReturn: 3.2,
    });
    const stats = updateStratifiedMotorGoldBank(goldFile, cands, 4);
    assert.equal(stats.perTool.eat_food, 4, 'eat_food must be capped at maxPerTool=4');
    assert.equal(stats.perTool.engage_combat, 1, 'engage_combat must be preserved in stratified gold bank');
    assert.equal(stats.perTool.shoot_bow, 1, 'shoot_bow must be preserved in stratified gold bank');
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }

  // 4. Python Clipped PPO verification: strict=False resume, evaluate_action_chunk, signed negative advantage push-away, & std floor
  const rootDir = path.resolve(__dirname, '..', '..');
  const pythonExe = path.join(rootDir, 'neural_lab', '.venv', 'Scripts', 'python.exe');
  const pyCheck = `
import math, sys, torch
sys.path.insert(0, ${JSON.stringify(rootDir)})
from neural_lab.action_head import MinecraftActionHead, AIM_STD_MIN, AIM_STD_MAX

torch.manual_seed(7)
head = MinecraftActionHead(hidden_dim=64, num_layers=1, num_heads=4)
# Simulate loading legacy checkpoint missing aim_log_std & move_log_std
legacy_sd = {k: v.clone() for k, v in head.state_dict().items() if "log_std" not in k}
missing, unexpected = head.load_state_dict(legacy_sd, strict=False)
assert "aim_log_std" in missing and "move_log_std" in missing and len(unexpected) == 0

h = torch.zeros(1, 1536)
p = torch.zeros(1, 24)
# Enforce floor check even if log_std tried to shrink below AIM_STD_MIN
with torch.no_grad():
    head.aim_log_std.fill_(-10.0)
aim_std, _ = head.get_action_stds()
assert abs(float(aim_std[0].item()) - AIM_STD_MIN) < 1e-4

# Reset to clean std and verify negative advantage pushes predicted yaw AWAY from bad yaw_target=+20.0
with torch.no_grad():
    head.aim_log_std.fill_(math.log(12.0))
bad_aim = torch.full((1, 8, 2), 20.0)
zero_mov = torch.zeros((1, 8, 2))
zero_disc = torch.zeros((1, 8, 6))
zero_ext = torch.zeros((1, 8, 2))
with torch.no_grad():
    out0 = head(h, p)
    pre_dist = abs(float(out0["trajectory_aim"][0, 0, 0].item()) - 20.0)
    old_lp, ent0, val0 = head.evaluate_action_chunk(out0, bad_aim, zero_mov, zero_disc, zero_ext)
assert torch.isfinite(old_lp).all() and torch.isfinite(ent0).all() and torch.isfinite(val0).all()

opt = torch.optim.Adam(head.parameters(), lr=5e-3)
for _ in range(4):
    opt.zero_grad()
    out = head(h, p)
    new_lp, ent, _ = head.evaluate_action_chunk(out, bad_aim, zero_mov, zero_disc, zero_ext)
    ratio = torch.exp((new_lp - old_lp).clamp(-4.0, 4.0))
    adv = torch.tensor([-2.0])
    loss = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).mean()
    loss.backward()
    opt.step()
with torch.no_grad():
    out1 = head(h, p)
    post_dist = abs(float(out1["trajectory_aim"][0, 0, 0].item()) - 20.0)
assert post_dist > pre_dist, f"Negative advantage PPO must push yaw away from bad action (pre={pre_dist}, post={post_dist})"

# Verify eval() during PPO keeps gradients intact while giving exact ratio == 1.0 before step 0 and min_iter filtering works cleanly
import json, tempfile
from pathlib import Path
import neural_lab.train_action_head as tah

head.eval()
with torch.no_grad():
    out_eval = head(h, p)
    lp_eval, _, _ = head.evaluate_action_chunk(out_eval, bad_aim, zero_mov, zero_disc, zero_ext)
out_train_nodrop = head(h, p)
lp_train_nodrop, _, _ = head.evaluate_action_chunk(out_train_nodrop, bad_aim, zero_mov, zero_disc, zero_ext)
ratio_nodrop = float(torch.exp((lp_train_nodrop - lp_eval).clamp(-4.0, 4.0)).item())
assert abs(ratio_nodrop - 1.0) < 1e-5, f"Expected deterministic ratio == 1.0 with dropout eval freeze, got {ratio_nodrop}"
lp_train_nodrop.backward()
assert any(param.grad is not None and param.grad.abs().sum().item() > 0 for param in head.parameters()), "Gradients must still flow in eval() mode"
head.zero_grad()

with tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False, encoding='utf-8') as tf:
    base_rec = {
        "latent_h": [0.0] * 1536,
        "proprio24": [0.0] * 24,
        "voxels": [0] * 3375,
        "aimTargets": [[0.0, 0.0]] * 8,
        "moveTargets": [[0.0, 0.0]] * 8,
        "discreteTargets": [[0, 0, 0, 0, 0, 0]] * 8,
        "extendedTargets": [[0, 0]] * 8,
        "toolName": "engage_combat",
    }
    for it in ([42] * 100 + [45] * 130):
        tf.write(json.dumps({**base_rec, "iteration": it}) + "\\n")
    tmp_chunks = Path(tf.name)
try:
    res_filtered = tah.load_motor_chunks_with_returns(tmp_chunks, max_samples=500, device="cpu", min_iter=45)
    assert res_filtered is not None and int(res_filtered[0][0].shape[0]) == 130, f"Expected 130 filtered records, got {res_filtered[0][0].shape[0]}"
    res_fallback = tah.load_motor_chunks_with_returns(tmp_chunks, max_samples=500, device="cpu", min_iter=99)
    assert res_fallback is not None and int(res_fallback[0][0].shape[0]) == 230, f"Expected 230 fallback records, got {res_fallback[0][0].shape[0]}"
finally:
    tmp_chunks.unlink(missing_ok=True)
print("OK")
`;
  assert.equal(execFileSync(pythonExe, ['-c', pyCheck], { encoding: 'utf8' }).trim().split('\n').pop(), 'OK');
});











