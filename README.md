# Texturizer

Texturizer is a minimal Pinokio web app for texturing an existing mesh with `Hunyuan3D-2GP`, which integrates `mmgp` for lower VRAM use.

The intended workflow is:

1. Upload a mesh, ideally a rigged `.glb`.
2. Upload a reference image, or provide a text prompt.
3. Generate a textured output.
4. If the mesh is a rigged GLB and the textured result keeps the same vertex layout, Texturizer merges the new UV/material data back into the original GLB so the rig stays intact.

## What it is good for

- Texturing an existing mesh with a reference image
- Texturing a rigged GLB while preserving joints when Hunyuan keeps the original vertex layout
- Using Hunyuan in a lower-VRAM mode through `mmgp` profile-based offloading

## What it is not good for

- Re-rigging or re-topologizing meshes
- Preserving a rig when Hunyuan changes the mesh vertex layout
- Guaranteeing prompt-only quality; prompt-only mode is supported through Hunyuan's optional text-to-image path and may download extra weights on first use

## How to use

1. Click `Install`.
2. Click `Start Web App`.
3. Upload a mesh, or click `Use bundled Geno demo mesh`.
4. Upload a reference image or enter a prompt.
5. Leave `Preserve original rig` enabled if you need a rig-safe output for another animation app.
6. Download the output `.glb`.

## Bundled example

The launcher now includes the AI4Animation biped default mesh as:

- `app/examples/geno.glb`

This is the main intended path when you want to texture a mesh and bring it back into AI4Animation without changing the rig contract.

## API

The app also exposes a local HTTP API.

### Health check

`GET /api/health`

### Texture an existing mesh

`POST /api/texture`

Multipart form fields:

- `mesh`: required mesh file
- `reference_image`: optional reference image file
- `prompt`: optional text prompt
- `preserve_rig`: optional boolean, defaults to `true`

If `preserve_rig=true` and the uploaded file is a rigged `.glb`, the server will only return a result when the textured output can be merged back without breaking the original rig.

## VRAM notes

- This launcher defaults to `mmgp` profile `4` (`LowRAM_LowVRAM`).
- The practical target is still an NVIDIA GPU.
- `6 GB` may be possible in aggressive `mmgp` configurations, but `8-12 GB` is the more realistic target for mesh + reference image texturing.
- Prompt-only mode may need more headroom because it also loads Hunyuan's text-to-image pipeline.

### JavaScript

```javascript
const form = new FormData();
form.append("mesh", meshFile);
form.append("reference_image", imageFile);
form.append("preserve_rig", "true");

const response = await fetch("http://127.0.0.1:7860/api/texture", {
  method: "POST",
  body: form
});

if (!response.ok) {
  throw new Error(await response.text());
}

const blob = await response.blob();
const url = URL.createObjectURL(blob);
```

### Python

```python
import requests

with open("character.glb", "rb") as mesh, open("reference.png", "rb") as image:
    response = requests.post(
        "http://127.0.0.1:7860/api/texture",
        files={
            "mesh": ("character.glb", mesh, "model/gltf-binary"),
            "reference_image": ("reference.png", image, "image/png"),
        },
        data={"preserve_rig": "true"},
        timeout=3600,
    )

response.raise_for_status()
with open("textured.glb", "wb") as f:
    f.write(response.content)
```

### Curl

```bash
curl -X POST http://127.0.0.1:7860/api/texture \
  -F "mesh=@character.glb" \
  -F "reference_image=@reference.png" \
  -F "preserve_rig=true" \
  --output textured.glb
```
