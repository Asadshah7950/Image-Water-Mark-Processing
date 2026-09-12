class ColorSpaceConverter:
    @staticmethod
    def rgb_to_ycbcr(r: int, g: int, b: int) -> tuple:
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
        return (
            max(0, min(255, int(round(y)))),
            max(0, min(255, int(round(cb)))),
            max(0, min(255, int(round(cr)))),
        )

    @staticmethod
    def ycbcr_to_rgb(y: int, cb: int, cr: int) -> tuple:
        r = y + 1.402 * (cr - 128)
        g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
        b = y + 1.772 * (cb - 128)
        return (
            max(0, min(255, int(round(r)))),
            max(0, min(255, int(round(g)))),
            max(0, min(255, int(round(b)))),
        )
