import textwrap


def start(text: str):
    """Start of a bigger operation"""
    print(f"{f' {text} ':>^80}")


def stop(text: str = "Done!!!"):
    """Stop of a bigger operation"""
    print(f"{f' {text} ':<^80}")
    print("")


def middle(text: str):
    print("")
    """Intermediate step in a bigger operation"""
    print(f"{f' {text} ':-^80}")


def plain(text: str):
    """Plain commentary"""
    print("\n".join(textwrap.wrap(str(text), 80, break_long_words=False)))
