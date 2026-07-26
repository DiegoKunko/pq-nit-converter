#!/usr/bin/env python3
"""Public web app: convert images to JPEG with Rec. 2100 PQ profile."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from flask import Flask, render_template_string, request, send_file
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB limit

APP_DIR = Path(__file__).resolve().parent
PROFILE_PATH = APP_DIR / "Rec2100-PQ.icc"
PROFILE_BYTES: bytes | None = None


def get_profile() -> bytes:
    global PROFILE_BYTES
    if PROFILE_BYTES is None:
        if not PROFILE_PATH.exists():
            raise FileNotFoundError("ICC profile not found.")
        PROFILE_BYTES = PROFILE_PATH.read_bytes()
    return PROFILE_BYTES


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def pq_oetf(luminance_nits: np.ndarray) -> np.ndarray:
    """SMPTE ST 2084: absolute luminance (cd/m²) to PQ code values."""
    luminance = np.clip(luminance_nits / 10000.0, 0.0, 1.0)
    m1, m2 = 2610.0 / 16384.0, 2523.0 / 32.0
    c1, c2, c3 = 3424.0 / 4096.0, 2413.0 / 128.0, 2392.0 / 128.0
    lm1 = luminance**m1
    return ((c1 + c2 * lm1) / (1.0 + c3 * lm1)) ** m2


def convert_to_pq(image_bytes: bytes, peak_nits: int) -> bytes:
    """Convert image bytes to PQ JPEG with ICC profile, all in memory."""
    if not 500 <= peak_nits <= 1200:
        raise ValueError("El blanco HDR debe estar entre 500 y 1200 nits.")

    source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    srgb = np.asarray(source, dtype=np.float32) / 255.0
    linear_srgb = srgb_to_linear(srgb)

    # Linear sRGB/D65 → linear Rec.2020/D65
    matrix = np.array(
        [
            [0.6274040, 0.3292820, 0.0433136],
            [0.0690970, 0.9195400, 0.0113612],
            [0.0163916, 0.0880132, 0.8955950],
        ],
        dtype=np.float32,
    )
    rec2020 = np.einsum("...c,dc->...d", linear_srgb, matrix)
    pq = pq_oetf(np.clip(rec2020, 0.0, 1.0) * peak_nits)
    pixels = np.round(np.clip(pq, 0.0, 1.0) * 255.0).astype(np.uint8)

    output = io.BytesIO()
    Image.fromarray(pixels, "RGB").save(
        output,
        format="JPEG",
        quality=95,
        subsampling=0,
        icc_profile=get_profile(),
    )
    output.seek(0)
    return output.getvalue()


HTML = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PQ Nit Converter</title>
<style>
:root { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:#17171c; background:#f7f7f8; }
body { margin:0; min-height:100vh; display:grid; place-items:center; padding:24px; box-sizing:border-box; }
main { width:min(520px, 100%); background:#fff; padding:40px; border-radius:24px; box-shadow:0 16px 48px #00000014; }
h1 { margin:0 0 8px; font-size:28px; }
p { color:#575762; line-height:1.5; margin-bottom:0; }
label { display:block; font-weight:600; margin-top:24px; margin-bottom:8px; }
input, select, button { font:inherit; }
input, select { width:100%; box-sizing:border-box; padding:12px; border:1px solid #d9d9df; border-radius:10px; background:#fff; }
input[type="file"] { padding:10px; }
button { margin-top:28px; width:100%; padding:14px; border:0; border-radius:10px; background:#2b2f7e; color:#fff; font-weight:700; cursor:pointer; transition:background .15s; }
button:hover { background:#1e2260; }
button:disabled { background:#999; cursor:not-allowed; }
small { display:block; margin-top:20px; color:#6d6d77; line-height:1.45; }
.error { background:#fee; border:1px solid #c55; color:#900; padding:14px; border-radius:10px; margin-top:20px; }
.info { background:#e8f4e8; border:1px solid #5a5; color:#252; padding:14px; border-radius:10px; margin-top:20px; }
</style>
</head>
<body>
<main>
<h1>PQ Nit Converter</h1>
<p>Subí una imagen JPG o PNG, elegí el blanco HDR y descargá un JPEG con perfil Rec. 2100 PQ.</p>
{% if error %}
<div class="error">{{ error }}</div>
{% endif %}
<form method="post" enctype="multipart/form-data">
<label for="image">Imagen (máx 10 MB)</label>
<input id="image" name="image" type="file" accept="image/jpeg,image/png" required>
<label for="nits">Blanco HDR objetivo</label>
<select id="nits" name="nits">
{% for n in nits_options %}
<option value="{{ n }}"{{ ' selected' if n == 800 else '' }}>{{ n }} nits</option>
{% endfor %}
</select>
<button type="submit">Convertir y descargar</button>
</form>
<small>El JPEG resultante incluye el perfil ICC "Rec. ITU-R BT.2100 PQ". Para ver HDR, usá una app o plataforma que respete el perfil.</small>
</main>
</body>
</html>"""

NITS_OPTIONS = list(range(500, 1201, 50))


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            file = request.files.get("image")
            if not file or file.filename == "":
                raise ValueError("Seleccioná una imagen.")

            filename = file.filename.lower()
            if not filename.endswith((".jpg", ".jpeg", ".png")):
                raise ValueError("Solo se aceptan archivos JPG o PNG.")

            image_bytes = file.read()
            if len(image_bytes) == 0:
                raise ValueError("El archivo está vacío.")

            nits = int(request.form.get("nits", 800))
            if nits not in NITS_OPTIONS:
                raise ValueError("Valor de nits inválido.")

            result = convert_to_pq(image_bytes, nits)

            return send_file(
                io.BytesIO(result),
                mimetype="image/jpeg",
                as_attachment=True,
                download_name=f"imagen_PQ_{nits}nits.jpg",
            )
        except ValueError as e:
            return render_template_string(HTML, nits_options=NITS_OPTIONS, error=str(e))
        except Exception as e:
            return render_template_string(HTML, nits_options=NITS_OPTIONS, error=f"Error: {e}")

    return render_template_string(HTML, nits_options=NITS_OPTIONS, error=None)


@app.errorhandler(413)
def too_large(e):
    return render_template_string(HTML, nits_options=NITS_OPTIONS, error="El archivo supera el límite de 10 MB."), 413


if __name__ == "__main__":
    app.run(debug=True, port=5000)
