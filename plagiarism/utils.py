def read_source_file(path):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if not src.endswith("\n"):
        src += "\n"
    return src
