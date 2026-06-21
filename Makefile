.PHONY: build serve

build:
	cd frontend && pnpm install --frozen-lockfile && pnpm build

serve:
	uv run uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
