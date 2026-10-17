"""Adapt local replacement partials to Vibe's text-insertion listener.

The copied input expects text deltas from a remote stream. Whisper can revise
earlier words, so the adapter replaces only the dictation span and preserves
the typed prefix, suffix and cursor. No copied input widget needs changing.
"""
from ovat.agentic.voice import VoiceManager
from ovat.tui.vibe.cli.voice_manager.voice_manager_port import TranscribeState


class PromptListener:
    def __init__(self, listener):
        self.listener = listener
        self.prefix = self.suffix = ""
        self.partial = False

    def on_transcribe_state_change(self, state):
        if state == TranscribeState.RECORDING:
            widget = getattr(self.listener, "input_widget", None)
            if widget is not None:
                # The input keeps its ! or / mode separately. replace_input
                # changes only the editable text, so keep that mode prefix
                # out of the replacement and the cursor calculation.
                text = widget.text
                offset = widget.get_cursor_offset()
                self.prefix, self.suffix = text[:offset], text[offset:]
            self.partial = False
        self.listener.on_transcribe_state_change(state)

    def on_transcribe_partial(self, text):
        replace = getattr(self.listener, "replace_input", None)
        if replace is not None:
            replace(self.prefix + text + self.suffix,
                    cursor_offset=len(self.prefix) + len(text))
            self.partial = True

    def on_transcribe_text(self, text):
        replace = getattr(self.listener, "replace_input", None)
        if replace is not None:
            replace(self.prefix + text + self.suffix,
                    cursor_offset=len(self.prefix) + len(text))
        else:
            self.listener.on_transcribe_text(text)

    def on_voice_mode_change(self, enabled):
        self.listener.on_voice_mode_change(enabled)

    def on_transcribe_error(self, message):
        self.listener.on_transcribe_error(message)

    def on_transcribe_notice(self, message):
        self.listener.on_transcribe_notice(message)


class PromptVoiceManager(VoiceManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._adapters = {}

    def add_listener(self, listener):
        if listener not in self._adapters:
            adapter = self._adapters[listener] = PromptListener(listener)
            super().add_listener(adapter)

    def remove_listener(self, listener):
        adapter = self._adapters.pop(listener, None)
        if adapter is not None:
            super().remove_listener(adapter)

    async def close(self):
        await super().close()
        self._adapters.clear()
