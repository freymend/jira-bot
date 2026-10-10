# Google Meet to Jira task assistant

This service joins a Google Meet with a locally hosted Attendee bot. Attendee sends meeting audio to local MLX Whisper for transcription; a local Gemma model then identifies explicit action items, which are stored in SQLite. Jira issue creation is off by default; proposals remain in `dry_run` for review until `JIRA_DRY_RUN=false`.

## Pipeline

1. `POST /meetings` creates an Attendee bot and configures `bot.state_change` and `transcript.update` callbacks.
2. Attendee sends raw audio segments to the local MLX Whisper service through its `custom_async_v2` provider and emits signed transcript webhooks. Webhooks are verified and deduplicated; utterances are persisted as they arrive.
3. On `post_processing_completed`, the service fetches the canonical locally generated transcript from Attendee and runs chunked task extraction. Duplicate webhook deliveries are ignored.
4. High-confidence tasks with a verbatim transcript quote are saved. Jira issue creation is attempted only when dry-run is disabled and Jira credentials are configured.

The transcript and extracted-task records are stored locally in SQLite at `DATABASE_PATH`. The transcript is sent to the configured local OpenAI-compatible model endpoint. Jira receives accepted task fields, evidence quotes, and the meeting link when issue creation is enabled.

## Requirements and setup

- macOS 15 or later on Apple Silicon.
- Docker Desktop with Docker Compose v2.20.3 or later. Compose runs the API and Attendee development stack together; the API image uses Python 3.14 and locked dependencies.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for local development.
- vLLM-Metal installed in its managed native arm64 Python 3.12 environment.
- The MLX Whisper transcriber runs natively on Apple Silicon with Python 3.14 and `ffmpeg`.
- The Attendee source checkout at `../attendee`. Its development stack requires AWS recording-storage credentials.
- An Attendee API key and webhook secret created in the local Attendee instance.
- A long random `MLX_TRANSCRIBER_API_KEY`, shared by the API and the local transcriber.
- A public HTTPS URL for Attendee callbacks (for local development, use a tunnel such as ngrok).
- Jira Cloud API credentials only when enabling issue creation.

```sh
cp .env.example .env
```

Project dependencies are declared in `pyproject.toml`; `uv.lock` records the resolved versions. For local development, run `uv sync --no-active --group dev` to create the project `.venv` and install the local MLX Whisper service package into that same environment.

## Configure Attendee locally

Attendee's upstream development stack is included in this project's Compose configuration. The API, Attendee API, worker, scheduler, Postgres, and Redis start and stop together. The API reaches Attendee over the Compose network; Attendee's worker sends raw PCM audio to MLX Whisper on the Mac at `host.docker.internal:8101/transcribe`. Gemma uses port `8100`.

Clone Attendee next to this repository once, then generate its development environment file:

```sh
git clone https://github.com/attendee-labs/attendee.git ../attendee
docker compose build attendee-app-local
docker compose run --rm -T attendee-app-local python init_env.py > ../attendee/.env
```

Build the image in a separate step; redirecting the `run` output to a file while the image still has to be built makes Docker's build progress writer fail with `failed to get console: provided file is not a console`.

Fill `../attendee/.env` with the required AWS recording-storage bucket and credentials. The `scripts/dev.py` launcher starts the native model servers and this Compose stack together; start it after configuring the local Gemma environment below.

## Run the complete local stack

MLX Whisper and vLLM-Metal need the macOS Metal runtime. Docker Desktop runs containers inside Linux, and MLX's Linux backend is CPU-only; the model servers therefore run natively while the API and Attendee services run in Compose. The local Whisper package shares the project's Python 3.14 `.venv`; the vLLM-Metal installation keeps its own Python environment. The launcher brings up both native servers and the full Compose stack with one command.

Install `ffmpeg`, set `MLX_TRANSCRIBER_API_KEY` in the project `.env`, and install vLLM-Metal if it is not installed yet:

```sh
brew install ffmpeg
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

See the [vLLM-Metal installation guide](https://docs.vllm.ai/projects/vllm-metal/en/stable/installation/) for platform requirements and the [supported-model list](https://docs.vllm.ai/projects/vllm-metal/en/stable/supported_models/) for Gemma 4 support.

Set `APP_PUBLIC_URL` to the HTTPS tunnel URL and choose a long random `APP_API_TOKEN`. Docker Compose passes `.env` to the API container at runtime; it is excluded from the image build. The API's SQLite database is stored in a persistent Docker volume. The first Whisper transcription downloads the model selected by `MLX_WHISPER_MODEL`; it defaults to `mlx-community/whisper-turbo`.

Activate the vLLM-Metal environment and start everything from the repository root:

```sh
source ~/.venv-vllm-metal/bin/activate
uv run --no-active --group dev python scripts/dev.py up
```

Apply Attendee's migrations, then create its local account:

```sh
docker compose exec attendee-app-local python manage.py migrate
```

Create an account at `http://localhost:8000`. Attendee prints the confirmation link in its app logs. After confirming, create an API key and configure the webhook secret in Attendee settings; put those values in this project's `.env` as `ATTENDEE_API_KEY` and `ATTENDEE_WEBHOOK_SECRET`.

The API is available at `http://127.0.0.1:5005`. Point the public HTTPS tunnel at that address for Attendee callbacks. Use `uv run --no-active --group dev python scripts/dev.py down` to stop the Compose stack and the native servers started by the launcher. Its native server logs are in `.dev-run/`. Docker volumes remain between restarts; `docker compose down -v` also deletes the Jira database and Attendee's Postgres data.

## API

All management routes require `Authorization: Bearer $APP_API_TOKEN`. `/health` and the signed webhook endpoint are exempt.

Create a meeting bot:

```sh
curl -X POST http://127.0.0.1:5005/meetings \
  -H 'Authorization: Bearer YOUR_APP_API_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{"meeting_url":"https://meet.google.com/abc-defg-hij"}'
```

Inspect status and extracted tasks using the returned `bot_id`:

```sh
curl http://127.0.0.1:5005/meetings/BOT_ID \
  -H 'Authorization: Bearer YOUR_APP_API_TOKEN'
```

Other endpoints are `POST /meetings/{bot_id}/leave` and `POST /meetings/{bot_id}/reprocess`.

The meeting status endpoint includes the Attendee bot state, pipeline status, final transcript, and task proposals. A meeting in `dry_run` can be reprocessed after changing `JIRA_DRY_RUN` to `false` and restarting the API.

## Jira configuration

Keep `JIRA_DRY_RUN=true` while evaluating proposals. Once validated, configure the Jira site, email, API token, project key, and issue type, then set `JIRA_DRY_RUN=false` and restart the service. Only `Zachary` or `Zac` can be assigned, and only when that exact name is spoken in the task's evidence quote and identifies who accepted it. Speaker labels, pronouns, and inferred identity do not qualify. `JIRA_SPEAKER_ACCOUNT_IDS` maps either eligible name to a Jira account ID; other owners remain unassigned. `JIRA_PRIORITY_MAP` maps model priority labels to Jira priority names. Due dates are included only when the model extracts a specific ISO date.

An issue-creation request that fails is recorded as `uncertain` so a retry does not accidentally create a duplicate issue. Inspect the meeting record and Jira before manually retrying an uncertain proposal.

## Current scope

This first slice supports manually submitted Google Meet URLs. Calendar discovery, automatic meeting scheduling, a review UI, and transcript retention controls are follow-up work.
