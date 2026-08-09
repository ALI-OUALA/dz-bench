FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY schemas ./schemas
RUN uv sync --frozen --no-dev --extra service
RUN useradd --create-home --uid 10001 dzbench && chown -R dzbench:dzbench /app
USER 10001
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
EXPOSE 8001
ENTRYPOINT ["dz-bench-api"]
