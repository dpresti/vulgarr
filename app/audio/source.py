"""Which audio to listen to when locating or transcribing speech.

Both forced alignment and the speech check used to (or would) decode with no
-map at all, letting ffmpeg pick a stream -- on a reprocessed file that can be
our own previously-muted Clean track -- and downmixing every channel, so
explosions and score sit on top of the dialogue. Pinning the same source
stream the Clean track is built from, and taking just its center channel on a
surround track (where film mixes put dialogue), gives both models far cleaner
speech to work with.
"""

from dataclasses import dataclass

from app.mux.remux import plan_audio_streams

# 5.0/5.1/6.1/7.1 all carry FC; quad (4.0 back-pair) and stereo don't.
_MIN_CHANNELS_WITH_CENTER = 5


@dataclass(frozen=True)
class AudioSource:
    stream_index: int  # absolute ffprobe stream index
    channels: int

    def ffmpeg_args(self) -> list[str]:
        mono = ["-af", "pan=mono|c0=FC"] if self.channels >= _MIN_CHANNELS_WITH_CENTER else ["-ac", "1"]
        return ["-map", f"0:{self.stream_index}", *mono]


def audio_source_for(probe_data: dict, clean_track_title: str, known_clean_indices: list[int] | None) -> AudioSource:
    plan = plan_audio_streams(probe_data, clean_track_title, 1, known_clean_indices)
    return AudioSource(stream_index=plan.source_stream["index"], channels=int(plan.source_stream.get("channels") or 2))
