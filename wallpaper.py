from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Image dimensions
width, height = 1920, 1080

# Create a new image with a gradient background
img = Image.new("RGB", (width, height))
pixels = img.load()

# Define start and end colors for a cosmic gradient (deep blue to purple)
start_color = np.array([10, 10, 50])   # Dark blue
end_color   = np.array([50, 10, 70])    # Purple-ish

for y in range(height):
    # Interpolation factor
    factor = y / height
    # Interpolated color for the row
    row_color = tuple((start_color * (1 - factor) + end_color * factor).astype(int))
    for x in range(width):
        pixels[x, y] = row_color

# Prepare to draw text overlays
draw = ImageDraw.Draw(img)

# Load a font (using a default one if specific fonts are not available)
try:
    font_large = ImageFont.truetype("arial.ttf", 64)
    font_medium = ImageFont.truetype("arial.ttf", 48)
    font_small = ImageFont.truetype("arial.ttf", 32)
except IOError:
    font_large = ImageFont.load_default()
    font_medium = ImageFont.load_default()
    font_small = ImageFont.load_default()

# Draw physics equations (semi-transparent white text)
equations = [
    "∇⋅E = ρ/ε₀",
    "∇×B = μ₀J + μ₀ε₀∂E/∂t",
    "iħ∂ψ/∂t = Hψ"
]
y_text = 100
for eq in equations:
    draw.text((100, y_text), eq, font=font_medium, fill=(255, 255, 255, 180))
    y_text += 80

# Draw a subtle Egyptian hieroglyph pattern (using a Unicode hieroglyph, repeated)
hieroglyph = "𓀀"
for i in range(5):
    for j in range(10):
        x = 50 + j * 100
        y = height - 200 + i * 40
        draw.text((x, y), hieroglyph, font=font_small, fill=(255, 215, 0, 180))  # Gold-like color

# Draw the motivational quote
quote = "Explore the universe within and beyond."
text_width, text_height = draw.textsize(quote, font=font_large)
quote_x = width - text_width - 50
quote_y = height - text_height - 50
draw.text((quote_x, quote_y), quote, font=font_large, fill=(255, 255, 255, 200))

# Optionally, add some particle-like effects (simple white dots)
import random
for _ in range(300):
    x = random.randint(0, width-1)
    y = random.randint(0, height-1)
    draw.point((x, y), fill=(255, 255, 255))

# Save the image
img.save("cosmic_equations_wallscreen.png")
img.show()