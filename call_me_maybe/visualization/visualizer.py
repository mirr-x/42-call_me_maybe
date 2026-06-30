"""Terminal visualization for the constrained decoding generation process.

Enable by passing --visualize on the CLI, or by setting
GenerationVisualizer(enabled=True) directly.
"""

from __future__ import annotations


# ANSI styling
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_GREEN = "\033[38;5;120m"
_RED = "\033[38;5;203m"
_AMBER = "\033[38;5;215m"
_BLUE = "\033[38;5;111m"
_GREY = "\033[38;5;245m"
_BG_PANEL = "\033[48;5;235m"

_CLEAR_LINE = "\033[2K"
_CURSOR_UP = "\033[1A"


def _truncate(text: str, width: int) -> str:
    """Truncate text to width, replacing control chars with visible glyphs."""
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    if len(text) > width:
        return text[: width - 1] + "…"
    return text.ljust(width)


class GenerationVisualizer:
    """Renders a live view of constrained decoding token-by-token.

    Each step shows: the model's raw best-guess token, whether it was
    accepted or rejected by the constraint engine / state machine, and
    the running output stream so far.
    """

    def __init__(self, enabled: bool = True, width: int = 90) -> None:
        """Initialize the visualizer.

        Args:
            enabled (bool): If False, all methods are no-ops.
            width (int): Panel width in characters.
        """
        self.enabled = enabled
        self.width = width
        self.step = 0
        self._output_stream = ""
        self._lines_printed = 0

    def start(self, prompt: str) -> None:
        """Print the visualizer header for a new generation run."""
        if not self.enabled:
            return
        self.step = 0
        self._output_stream = ""
        print()
        print(f"{_BOLD}{_BLUE}┌{'─' * (self.width)}┐{_RESET}")
        title = " CONSTRAINED DECODING — live trace "
        print(f"{_BOLD}{_BLUE}│{title.center(self.width)}│{_RESET}")
        print(f"{_BOLD}{_BLUE}├{'─' * (self.width)}┤{_RESET}")

        # wrap full prompt across multiple lines instead of truncating
        inner_width = self.width - 4
        for raw_line in prompt.split("\n"):
            if raw_line == "":
                print(f"{_BLUE}│{' ' * (self.width)}│{_RESET}")
            else:
                for i in range(0, len(raw_line), inner_width):
                    line = raw_line[i:i + inner_width]
                    content = f" {line}".ljust(self.width)
                    print(f"{_BLUE}│{_RESET}{content}{_BLUE}│{_RESET}")

        print(f"{_BOLD}{_BLUE}└{'─' * (self.width)}┘{_RESET}")
        print()

    def step_event(
            self,
            state_name: str,
            raw_best_token: str,
            allowed_count: int | None,
            final_token: str,
            was_masked: bool,
            retried: bool = False,
    ) -> None:
        """Render one generation step.

        Args:
            state_name (str): Current JSONState name.
            raw_best_token (str): The model's unconstrained top token.
            allowed_count (int | None): Size of the allowed token set, or
                None if unconstrained.
            final_token (str): The token actually emitted.
            was_masked (bool): True if raw_best_token != final_token.
            retried (bool): True if the state machine rejected a token
                and a retry was needed.
        """
        if not self.enabled:
            return

        self.step += 1
        self._output_stream += final_token

        status_color = _AMBER if retried else (_RED if was_masked else _GREEN)
        status_label = "RETRY" if retried else ("MASKED" if was_masked else "OK")

        state_col = f"{_GREY}{state_name.ljust(20)}{_RESET}"
        raw_col = f"{_DIM}raw:{_RESET} {_truncate(repr(raw_best_token), 14)}"

        if was_masked or retried:
            arrow = f"{status_color} ✗→ {_RESET}"
        else:
            arrow = f"{_GREEN} == {_RESET}"

        final_col = (
            f"{status_color}{_BOLD}{repr(final_token).ljust(10)}{_RESET}")
        constraint_col = (
            f"{_DIM}allowed={allowed_count}{_RESET}"
            if allowed_count is not None
            else f"{_DIM}unconstrained{_RESET}"
        )

        line = (
            f"{_DIM}[{self.step:>3} ]{_RESET} {state_col} "
            f"{raw_col}{arrow}{final_col} "
            f"{status_color}{status_label.ljust(6)}{_RESET} {constraint_col}"
        )
        print(line)

    def finish(self, final_json: str, success: bool) -> None:
        """Print the closing summary panel."""
        if not self.enabled:
            return

        print()
        color = _GREEN if success else _RED
        label = "COMPLETE" if success else "FAILED"
        print(f"{_BOLD}{color}┌{'─' * (self.width)}┐{_RESET}")
        title = f" {label} — {self.step} tokens generated "
        print(f"{_BOLD}{color}│{title.center(self.width)}│{_RESET}")
        print(f"{_BOLD}{color}├{'─' * (self.width)}┤{_RESET}")
        # out_line = f" {_truncate(final_json, self.width - 4)}"
        for i in range(0, len(final_json), self.width - 7):
            out_line = final_json[i:i + self.width - 7]
            print(
                f"{color}│{_RESET}{out_line.ljust(self.width)}{color}│{_RESET}"
            )
        print(f"{_BOLD}{color}└{'─' * (self.width)}┘{_RESET}")
        print()
