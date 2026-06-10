"""在截图上点击获取真实坐标。用法: python pick_coord.py <图片路径>"""
import sys
from tkinter import Tk, Canvas, Label
from PIL import Image, ImageTk
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if not path or not path.exists():
    print("Usage: python pick_coord.py <image>")
    sys.exit(1)

pil = Image.open(path)
orig_w, orig_h = pil.size
print(f"原图: {orig_w}x{orig_h}")

# 缩放到适合屏幕 (最大1000x700)
scale = min(1000 / orig_w, 700 / orig_h, 1.0)
if scale < 1.0:
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    pil = pil.resize((new_w, new_h), Image.LANCZOS)
    print(f"缩放: {new_w}x{new_h} (比例: {scale:.3f})")

root = Tk()
root.title("点击获取坐标")
img = ImageTk.PhotoImage(pil)
canvas = Canvas(root, width=pil.width, height=pil.height)
canvas.pack()
canvas.create_image(0, 0, anchor="nw", image=img)
info = Label(root, text="点击目标位置获取坐标", font=("", 14), fg="blue")
info.pack()

def click(e):
    real_x = int(e.x / scale)
    real_y = int(e.y / scale)
    info.config(text=f"X={real_x}  Y={real_y}")
    print(f"X={real_x}  Y={real_y}")
    root.clipboard_clear()
    root.clipboard_append(f"{real_x},{real_y}")

canvas.bind("<Button-1>", click)
root.mainloop()
