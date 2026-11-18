// Synthesized MCP Tool: custom_i_meant_eat
// Created: 2026-11-09T06:10:08.007Z
// Description: Checks the bot's current hunger level and consumes available edible food from inventory if hunger is below maximum.

const bot = ctx.bot;
const currentFood = bot.food ?? 20;

if (currentFood >= 20) {
  await ctx.sayHumanized("I'm completely full right now!");
  return { success: true, summary: "Hunger is already full at 20." };
}

const edibleItemNames = new Set([
  'cooked_beef', 'steak', 'cooked_porkchop', 'cooked_mutton', 'cooked_chicken',
  'cooked_salmon', 'cooked_cod', 'baked_potato', 'bread', 'golden_carrot',
  'apple', 'melon_slice', 'sweet_berries', 'glow_berries', 'carrot'
]);

const items = bot.inventory.items();
let foodItem = items.find(item => edibleItemNames.has(item.name));

if (!foodItem) {
  foodItem = items.find(item => {
    const registryItem = bot.registry.itemsByName[item.name];
    return registryItem && registryItem.foodPoints !== undefined && registryItem.foodPoints > 0 && !item.name.includes('rotten_flesh') && !item.name.includes('spider_eye') && !item.name.includes('poisonous_potato');
  });
}

if (!foodItem) {
  await ctx.sayHumanized("I'm hungry, but I don't have any safe food in my inventory!");
  return { success: false, summary: "No edible food found in inventory." };
}

await ctx.sayHumanized(`Eating some ${foodItem.name.replace(/_/g, ' ')}...`);

try {
  await ctx.kinematics.equipItemHumanized(foodItem.name, 'hand');
  await bot.consume();
  await ctx.sayHumanized("All done, that was delicious!");
  return { success: true, summary: `Ate ${foodItem.name}. New food level: ${bot.food}` };
} catch (err) {
  return { success: false, summary: `Failed to consume food: ${err.message}` };
}
