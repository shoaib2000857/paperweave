# syntax=docker/dockerfile:1.7
FROM node:22-alpine

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY frontend ./

EXPOSE 3000

CMD ["npm", "run", "dev"]
