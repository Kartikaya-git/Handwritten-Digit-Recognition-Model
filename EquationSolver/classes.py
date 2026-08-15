CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "add", "sub", "mul", "div"]

SYMBOL_DISPLAY = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}


def class_to_display(cls):
    return SYMBOL_DISPLAY.get(cls, cls)
