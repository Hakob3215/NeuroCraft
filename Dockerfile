FROM node:20-alpine

WORKDIR /app

COPY package.json tsconfig.json ./
RUN npm install --no-fund --no-audit

COPY src ./src
RUN npm run build

# Expose UDP 4445 for Minecraft LAN discovery
EXPOSE 4445/udp

CMD ["node", "dist/index.js"]
