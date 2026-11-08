"""
生成应用图标（build/icon.ico 和 build/icon.png）
"""
from PIL import Image, ImageDraw, ImageFont
import struct
import io


def create_png(size: int, output_path: str = None):
    """创建圆形渐变图标，带有 'J' 字母"""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 渐变背景：从 #89b4fa (蓝) 到 #b4a0fa (紫)
    for y in range(size):
        r = int(0x89 + (0xb4 - 0x89) * y / size)
        g = int(0xb4 + (0xa0 - 0xb4) * y / size)
        b = int(0xfa + (0xfa - 0xfa) * y / size)
        for x in range(size):
            dx, dy = x - size/2, y - size/2
            dist = (dx*dx + dy*dy) ** 0.5
            if dist <= size/2 - 2:
                draw.point((x, y), fill=(r, g, b, 255))

    # 绘制白色"J"字母
    font_size = int(size * 0.6)
    try:
        font = ImageFont.truetype("segoeui.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), "J", font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) / 2 - bbox[0]
    y = (size - th) / 2 - bbox[1] - size * 0.05
    draw.text((x, y), "J", fill="white", font=font)

    if output_path:
        img.save(output_path, "PNG")
    return img


def create_ico(output_path: str, sizes=[16, 32, 48, 64, 128, 256]):
    """创建多分辨率 ICO 文件"""
    images = [create_png(s) for s in sizes]

    # 保存为 ICO
    images[0].save(
        output_path,
        format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=images[1:],
    )
    print(f"图标已生成: {output_path}")


if __name__ == "__main__":
    from pathlib import Path
    build_dir = Path(__file__).resolve().parent
    build_dir.mkdir(parents=True, exist_ok=True)

    create_png(256, str(build_dir / "icon.png"))
    create_ico(str(build_dir / "icon.ico"))
    print("完成！")
