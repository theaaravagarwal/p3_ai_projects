#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  # Load local defaults so the wrapper can use values from the repo's .env file.
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

detect_host_ip() {
  if command -v ipconfig >/dev/null 2>&1; then
    ifconfig_ip="$(ipconfig getifaddr en0 2>/dev/null || true)"
    if [[ -n "$ifconfig_ip" ]]; then
      printf '%s' "$ifconfig_ip"
      return
    fi
  fi

  if command -v hostname >/dev/null 2>&1; then
    local ips
    ips="$(hostname -I 2>/dev/null || true)"
    if [[ -n "$ips" ]]; then
      printf '%s' "${ips%% *}"
      return
    fi
  fi

  printf 'localhost'
}

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  echo "Docker Compose is not installed. Install Docker Desktop or docker-compose first."
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  ./docker.sh up       Build and start the API container
  ./docker.sh down     Stop and remove the container
  ./docker.sh logs     Stream container logs
  ./docker.sh restart  Rebuild and restart the container
  ./docker.sh public-up    Build and run a public, bind-mount-free container
  ./docker.sh public-down  Stop the public container
  ./docker.sh public-logs  Stream public container logs

Examples:
  OPENAI_API_KEY=sk-... ./docker.sh up
  OPENAI_API_KEY=sk-... PUBLIC_API_KEY=change-me ./docker.sh public-up
  ./docker.sh logs
EOF
}

cmd="${1:-up}"

case "$cmd" in
  up)
    "${COMPOSE[@]}" up --build -d
    echo "Container started. Use ./docker.sh logs to follow output."
    ;;
  down)
    "${COMPOSE[@]}" down
    ;;
  logs)
    "${COMPOSE[@]}" logs -f
    ;;
  restart)
    "${COMPOSE[@]}" down
    "${COMPOSE[@]}" up --build -d
    echo "Container restarted. Use ./docker.sh logs to follow output."
    ;;
  public-up)
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      echo "OPENAI_API_KEY must be set for public-up."
      exit 1
    fi
    host_ip="$(detect_host_ip)"
    docker build -t assistant-public .
    docker rm -f assistant-public >/dev/null 2>&1 || true
    docker run -d \
      --name assistant-public \
      -p "${HOST_PORT:-8000}:8000" \
      -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
      -e OPENAI_MODEL="${OPENAI_MODEL:-gpt-5-nano}" \
      -e OPENAI_MAX_OUTPUT_TOKENS="${OPENAI_MAX_OUTPUT_TOKENS:-64}" \
      -e OPENAI_MAX_HISTORY_MESSAGES="${OPENAI_MAX_HISTORY_MESSAGES:-2}" \
      -e OPENAI_REASONING_EFFORT="${OPENAI_REASONING_EFFORT:-minimal}" \
      -e OPENAI_REQUEST_TIMEOUT_SECONDS="${OPENAI_REQUEST_TIMEOUT_SECONDS:-45}" \
      -e PUBLIC_API_KEY="${PUBLIC_API_KEY:-}" \
      -e API_RATE_LIMIT_PER_MINUTE="${API_RATE_LIMIT_PER_MINUTE:-30}" \
      -e API_RATE_LIMIT_BURST="${API_RATE_LIMIT_BURST:-10}" \
      -e API_MAX_CHAT_MESSAGE_CHARS="${API_MAX_CHAT_MESSAGE_CHARS:-4000}" \
      -e API_MAX_CONVERSATION_MESSAGES="${API_MAX_CONVERSATION_MESSAGES:-12}" \
      -e CORS_ORIGINS="${CORS_ORIGINS:-}" \
      -e TRUST_PROXY_HEADERS="${TRUST_PROXY_HEADERS:-false}" \
      -e PUBLIC_UI_ONLY=true \
      assistant-public
    echo "Public container started."
    echo "Access URL: http://${host_ip}:${HOST_PORT:-8000}"
    echo "Use ./docker.sh public-logs to follow output."
    echo "Send X-API-Key: ${PUBLIC_API_KEY:-<set PUBLIC_API_KEY>}"
    ;;
  public-down)
    docker rm -f assistant-public >/dev/null 2>&1 || true
    ;;
  public-logs)
    docker logs -f assistant-public
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $cmd"
    usage
    exit 1
    ;;
esac
