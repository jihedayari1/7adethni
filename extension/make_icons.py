#!/usr/bin/env python3
"""Draw the 7adethni logo with Pillow and export icon512/128/48/16.png (Chrome needs PNG)."""
import numpy as np
from PIL import Image, ImageDraw
S = 512

def grad(c1, c2):
    yy, xx = np.mgrid[0:S, 0:S]
    t = ((xx + yy) / (2 * S))[..., None]
    arr = (np.array(c1) * (1 - t) + np.array(c2) * t).astype("uint8")
    return Image.fromarray(arr, "RGB").convert("RGBA")

canvas = Image.new("RGBA", (S, S), (0, 0, 0, 0))

# rounded tile with red->terracotta diagonal gradient
tile = Image.new("L", (S, S), 0)
ImageDraw.Draw(tile).rounded_rectangle([16, 16, S - 16, S - 16], radius=118, fill=255)
canvas.paste(grad((231, 0, 19), (194, 80, 47)), (0, 0), tile)

d = ImageDraw.Draw(canvas)
# speech bubble (talk = 7adethni)
d.polygon([(150, 318), (150, 398), (222, 332)], fill=(255, 255, 255, 255))
d.rounded_rectangle([104, 112, 408, 326], radius=54, fill=(255, 255, 255, 255))

# the "7" filled with a red gradient (via mask)
seven = Image.new("L", (S, S), 0)
sd = ImageDraw.Draw(seven)
sd.rounded_rectangle([168, 156, 348, 204], radius=10, fill=255)
sd.polygon([(348, 156), (300, 156), (214, 296), (262, 296)], fill=255)
canvas.paste(grad((231, 0, 19), (255, 59, 78)), (0, 0), seven)

# AI sparkle (jasmine gold), upper-right
cx, cy, R, r = 420, 120, 40, 12
star = [(cx, cy - R), (cx + r, cy - r), (cx + R, cy), (cx + r, cy + r),
        (cx, cy + R), (cx - r, cy + r), (cx - R, cy), (cx - r, cy - r)]
ImageDraw.Draw(canvas).polygon(star, fill=(244, 211, 94, 255))

canvas.save("extension/icon512.png")
for s in (128, 48, 16):
    canvas.resize((s, s), Image.LANCZOS).save(f"extension/icon{s}.png")
print("exported icon512/128/48/16.png")

if __name__ == "__main__":
    pass
