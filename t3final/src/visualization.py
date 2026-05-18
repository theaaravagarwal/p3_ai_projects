from __future__ import annotations

import io

import chess
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


UNICODE_PIECES = {
    "P": "P",
    "N": "N",
    "B": "B",
    "R": "R",
    "Q": "Q",
    "K": "K",
    "p": "p",
    "n": "n",
    "b": "b",
    "r": "r",
    "q": "q",
    "k": "k",
}


def render_board_image(fen: str) -> Image.Image:
    board = chess.Board(fen)
    size = 512
    square = size // 8
    light = (238, 238, 210)
    dark = (118, 150, 86)
    image = Image.new("RGB", (size, size), light)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
    except OSError:
        font = ImageFont.load_default()

    for rank in range(8):
        for file in range(8):
            x0 = file * square
            y0 = rank * square
            color = light if (rank + file) % 2 == 0 else dark
            draw.rectangle([x0, y0, x0 + square, y0 + square], fill=color)
            square_index = chess.square(file, 7 - rank)
            piece = board.piece_at(square_index)
            if piece:
                text = UNICODE_PIECES[piece.symbol()]
                fill = (245, 245, 245) if piece.color == chess.WHITE else (35, 35, 35)
                bbox = draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                draw.text((x0 + (square - tw) / 2, y0 + (square - th) / 2 - 3), text, fill=fill, font=font)
    return image


def evaluation_bar(cp: float) -> Image.Image:
    clipped = max(-1000.0, min(1000.0, float(cp)))
    white_fraction = (clipped + 1000.0) / 2000.0
    width, height = 420, 72
    image = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    margin = 18
    bar_w = width - 2 * margin
    split_x = margin + int(bar_w * white_fraction)
    draw.rectangle([margin, 22, margin + bar_w, 50], fill=(35, 35, 35))
    draw.rectangle([margin, 22, split_x, 50], fill=(245, 245, 245))
    draw.rectangle([margin, 22, margin + bar_w, 50], outline=(30, 30, 30), width=2)
    center = margin + bar_w // 2
    draw.line([center, 16, center, 56], fill=(180, 60, 60), width=2)
    draw.text((margin, 54), "Black", fill=(30, 30, 30))
    draw.text((width - margin - 38, 54), "White", fill=(30, 30, 30))
    return image


def save_confusion_matrix(cm, labels: list[str], path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
