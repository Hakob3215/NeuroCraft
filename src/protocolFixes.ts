/**
 * Protocol schema extensions and corrections for Minecraft 1.20.4+ in Mineflayer / node-minecraft-protocol.
 *
 * In vanilla 1.20.4 (protocol 765), `packet_explosion` serializes sound events as an `ItemSoundEvent`
 * (string sound identifier like 'minecraft:entity.generic.explode' + optional f32 fixedRange),
 * rather than an `ItemSoundHolder` registry entry. Providing this exact protocol definition
 * enables node-minecraft-protocol to deserialize the entire packet without leaving unparsed bytes
 * or triggering Protodef chunk-size warning dumps.
 */
export const MINECRAFT_1_20_CUSTOM_PACKETS: Record<string, any> = {
  '1.20': {
    play: {
      toClient: {
        types: {
          packet_explosion: [
            'container',
            [
              { name: 'x', type: 'f64' },
              { name: 'y', type: 'f64' },
              { name: 'z', type: 'f64' },
              { name: 'radius', type: 'f32' },
              {
                name: 'affectedBlockOffsets',
                type: [
                  'array',
                  {
                    countType: 'varint',
                    type: [
                      'container',
                      [
                        { name: 'x', type: 'i8' },
                        { name: 'y', type: 'i8' },
                        { name: 'z', type: 'i8' },
                      ],
                    ],
                  },
                ],
              },
              { name: 'playerMotionX', type: 'f32' },
              { name: 'playerMotionY', type: 'f32' },
              { name: 'playerMotionZ', type: 'f32' },
              { name: 'block_interaction_type', type: 'varint' },
              { name: 'small_explosion_particle', type: 'Particle' },
              { name: 'large_explosion_particle', type: 'Particle' },
              { name: 'sound', type: 'ItemSoundEvent' },
            ],
          ],
        },
      },
    },
  },
};
