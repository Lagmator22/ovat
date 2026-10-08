# ovat/tools/fuzzy.py
"""Recover from a model that names a file slightly wrong.

Small models misspell paths: "meeting_note.wav" for "meeting-notes.wav", or a
bare "sample.wav" for "examples/audio-multimodal/sample.wav". Failing the tool
call costs a whole round trip, so a close-enough existing file is used instead.

Two rules make that safe rather than surprising:

  * A substitution is never silent. resolve_path returns a NOTE alongside the
    path, and the tools put it in front of their result, so the model -- and
    the transcript a human reads -- both see which file was actually used. The
    first version swapped files without a word: a model that asked for one
    recording was handed a different one and answered as if it were the first.
  * The search is bounded. The first version walked the ENTIRE tree under the
    working directory on every miss; started from a home directory, one tool
    call took 31.6 s. Where to look is _candidates() below.
"""
import difflib
import os

#: How alike two filenames must be before one stands in for the other.
#: 0.66 is "file.wav" vs "sample.wav", which must NOT match.
SIMILARITY = 0.70


def _candidates(requested: str):
    """Existing files worth comparing against `requested`, a path that is
    not there. Yield (or return a list of) file paths; resolve_path filters
    them by extension and scores them, so this only decides WHERE to look.
    """
    folder = os.path.dirname(requested)
    if folder and (os.path.isdir(folder) or os.path.isabs(folder)):
        # The model named a folder that exists (or an absolute one): trust it
        # and look nowhere else. A typo in the FILENAME is the case worth
        # rescuing; reaching into some other tree is how a different
        # recording gets swapped in.
        return _list_files(folder)
    if folder:
        # A relative folder that is not there. Measured on the AI PC: asked
        # for examples/audio-multimodal/sample.wav, Qwen3.5 sent
        # audio-multimodal/sample.wav on every engine and every arm, so the
        # file was "not found" while it sat one folder down. Look only in
        # folders whose path ENDS with the one named, within the same two
        # levels as a bare name: the named folder still decides where.
        tail = os.sep + os.path.normpath(folder)
        return [path for root, files in _walk_two_levels()
                if (os.sep + os.path.normpath(root)).endswith(tail)
                for path in files]
    # A bare name ("sample.wav") says nothing about where, so look a fixed
    # two levels under the working directory: enough for
    # examples/audio-multimodal/sample.wav, nowhere near a whole home folder.
    return [path for _root, files in _walk_two_levels() for path in files]


def _walk_two_levels():
    """(folder, [its file paths]) for the cwd and two levels below it."""
    for root, dirs, files in os.walk("."):
        depth = 0 if root == "." else root.count(os.sep)
        dirs[:] = [d for d in dirs
                   if not d.startswith(".") and d not in _SKIP and depth < 2]
        yield root, [os.path.join(root, name) for name in files]


#: Folders that hold thousands of files and never a user's input.
_SKIP = {"node_modules", "__pycache__", "venv", "site-packages"}


def _list_files(folder: str) -> list:
    try:
        return [os.path.join(folder, name) for name in os.listdir(folder)]
    except OSError:                 # the named folder does not exist either
        return []


def resolve_path(requested: str) -> tuple[str, str | None]:
    """(path to use, note for the model or None).

    An existing path comes back untouched with no note. Otherwise the most
    similar file from _candidates() with the same extension is used, and the
    note says so. No good match leaves the request as it was, so the tool
    reports its usual "could not find" error.
    """
    if os.path.isfile(requested):
        return requested, None
    basename = os.path.basename(requested).lower()
    ext = os.path.splitext(basename)[1]
    if not ext:
        return requested, None

    best, best_ratio = None, SIMILARITY
    # sorted: the same tree must always pick the same file.
    for candidate in sorted(_candidates(requested)):
        name = os.path.basename(candidate).lower()
        if not name.endswith(ext) or not os.path.isfile(candidate):
            continue
        ratio = difflib.SequenceMatcher(None, basename, name).ratio()
        if ratio > best_ratio:
            best, best_ratio = candidate, ratio
    if best is None:
        return requested, None
    best = os.path.normpath(best)
    return best, _note(requested, best)


def _note(requested: str, used: str) -> str:
    """The sentence that leads the tool result.

    Plain quotes, not repr(): repr doubles every backslash, so on Windows the
    model read 'docs\\\\jfk.wav' and could copy that doubled path into its
    next call, a path that exists nowhere.
    """
    return (f"Note: there is no file at '{requested}'; used the closest "
            f"match, '{used}'.")
