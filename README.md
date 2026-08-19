# Subtitle Profanity Filter

Self-hosted service that adds a second, selectable "Clean" audio track to
movies/episodes by muting the audio under profane subtitle cues -- detection
is subtitle-based (matches an .srt against a configurable, severity-tagged
word list), never ASR/transcription. The original audio track's bytes are
never modified; a verified copy with the added track atomically replaces the
file on disk, and the original is always kept as a timestamped backup under
`/data/backups`.

## How it works

1. **Parse** the episode/movie's `.srt` (`app/subtitles/parser.py`) into timed cues, stripping HTML/ASS-style tags and normalizing line breaks.
2. **Match** each cue's text against the word list (`app/subtitles/matcher.py`) -- case/punctuation-insensitive, whole-word by default, severity-tagged (mild/moderate/strong).
3. **Build mute intervals** from matched cues (`app/audio/mute.py`) with small padding and merging of close cues.
4. **Remux** (`app/mux/remux.py`): ffmpeg stream-copies every existing stream (video, original audio, subtitles, chapters) untouched and appends one new AAC audio track with a `volume=0` filter gated to the mute intervals, tagged `title=Clean`. The result is verified (stream count, duration, and a real decode of the new track) in a temp file before anything touches the original. Only then is the original moved to `/data/backups/<relative path>.<timestamp>.orig` and the verified file atomically swapped into place.

Because this is stream-copy plus one small filtered audio track, most of a file's bytes are literally copied, not re-encoded -- fast, and no quality loss on video/original audio.

## Design decisions made with the user

- **Track delivery**: in-place remux with a verify-then-backup-then-swap safety net (not a sidecar file), because neither Plex nor Jellyfin exposes an external sidecar audio file as a selectable track in the player's audio menu -- only a real extra stream in the container shows up there.
- **Automation trigger**: both Sonarr/Radarr's import webhook (poll for a subtitle to appear, since Bazarr may not have fetched one yet) and Bazarr's subtitle-downloaded event are supported and independently toggleable in Settings, with a selectable priority used only to de-duplicate near-simultaneous events for the same title. Bazarr's trigger defaults to **off** since the user's Bazarr instance was down at build time -- flip it on in Settings once it's back.

## Local dev

```bash
pip install -r requirements.txt
SPF_DATA_DIR=./data SPF_MEDIA_ROOT=./data uvicorn app.main:app --reload
```

Run tests:

```bash
pytest tests/ -q
```

## Wiring into an existing Sonarr/Radarr/Bazarr/Plex/Jellyfin stack

1. **Compose**: merge [`docker-compose.snippet.yml`](docker-compose.snippet.yml) into your existing compose file. Use the *exact same* media volume mount(s) and host paths your Sonarr/Radarr/Bazarr/Plex/Jellyfin services already use -- this service resolves file paths returned by the Sonarr/Radarr APIs directly, so a path mismatch means it won't find files that exist.

2. **API keys**: set `SONARR_API_KEY` / `RADARR_API_KEY` / `BAZARR_API_KEY` env vars (Settings > General > Security in each app). Set `SPF_WEBHOOK_TOKEN` to a random string and append `?token=<that value>` to the webhook URLs below -- otherwise anyone on your network can trigger processing.

3. **Sonarr**: Settings > Connect > add Webhook. URL: `http://subtitle-profanity-filter:8000/webhooks/sonarr?token=<token>`. Trigger on: *On Import*, *On Upgrade*.

4. **Radarr**: same as Sonarr, URL: `http://subtitle-profanity-filter:8000/webhooks/radarr?token=<token>`.

5. **Bazarr** (once it's working again): Settings > Subtitles > Custom Post-Processing. Enable it, and set the command to a `curl` call using Bazarr's template variables, e.g.:
   ```
   curl -s -X POST "http://subtitle-profanity-filter:8000/webhooks/bazarr?token=<token>&video_path={{directory}}/{{episode}}&subtitle_path={{subtitles}}&language={{subtitles_language_code2}}"
   ```
   (Bazarr's exact variable names vary by version -- check Settings > Subtitles > Custom Post-Processing for the list available in your version and adjust.) Then flip "Bazarr subtitle-downloaded event" on in this app's Settings page.

6. **First run**: open `http://<erwin>:8420/library` and click "Sync from Sonarr/Radarr" to pull in the existing library (this only reads metadata/paths via the Sonarr/Radarr APIs -- it does not process anything). Then go to `/wordlist` and build out your word list before processing anything for real.

7. **Plex/Jellyfin**: nothing to configure -- once a title has been processed, refresh/rescan that item's metadata (or wait for the next library scan) and the "Clean" audio track will appear in the audio track picker like any other track.

## Settings reference

- **Max concurrent ffmpeg jobs** -- caps how many remux jobs run at once.
- **Restrict processing to off-hours** -- when on, queued jobs wait until the configured window (handles windows that cross midnight).
- **Automation triggers** -- independently enable/disable the Sonarr/Radarr and Bazarr triggers, and set which one takes priority for de-duplication.

## Notes / known trade-offs

- Backups accumulate under `/data/backups` and are never auto-deleted -- prune manually once you've confirmed a batch of clean tracks look right.
- `backup_root` (`/data/backups`) should ideally be on the same filesystem as your media mount; otherwise the "move original aside" step becomes a slower copy+delete instead of an instant rename.
- Subtitle matching mutes the whole subtitle cue's time range (padded slightly), not word-level timing -- there's no word-level timestamp data available from a plain .srt.
