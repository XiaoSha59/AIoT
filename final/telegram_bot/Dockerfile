FROM python:3.11-slim-bookworm
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=120
COPY pyproject.toml ./
COPY edgeface ./edgeface
COPY telegram_bot ./telegram_bot
COPY scripts ./scripts
COPY configs ./configs
RUN pip install '.[bot]'
RUN useradd -m -u 10001 edgeface && mkdir -p reports/generated && chown -R edgeface:edgeface /app
USER edgeface
CMD ["python", "-m", "telegram_bot.bot"]
