# Texturizer

Texturizer is a minimal Pinokio web app for turning one image into a textured character mesh and putting that mesh onto a usable rig. Its default workflow uses Hunyuan3D to generate new character geometry from the image, texture it, and transfer the selected rig onto the generated mesh.

The intended workflow is:

1. Upload a rigged `.glb`, or use the bundled Geno demo mesh.
2. Upload an image, or provide a text prompt for AI mode.
3. Use the default generated-character mode.
4. Generate a textured character output.
5. If the selected mesh is rigged, Texturizer transfers that skeleton onto the generated mesh. If no mesh is selected in the UI, it uses the bundled Geno rig.

## What it is good for

- Generating character geometry from a reference image, then transferring a rig onto it
- AI texturing an existing mesh with a reference image or prompt when the original geometry is already correct
- Texturing a rigged GLB while preserving joints when the output can be merged back into the original vertex layout
- Applying an uploaded image as the exact texture map when the mesh already has usable UVs
- Using Hunyuan in a lower-VRAM mode through `mmgp` profile-based offloading

## What it is not good for

- Producing production-quality skinning automatically in every pose; rig transfer is approximate and may need cleanup
- Preserving the exact original vertex layout in generated-character mode; that mode creates new geometry
- Guaranteeing prompt-only quality; prompt-only mode is supported through Hunyuan's optional text-to-image path and may download extra weights on first use

## How to use

1. Click `Install`.
2. Click `Start Web App`.
3. Upload a mesh, or click `Use bundled Geno demo mesh`.
4. Upload an image or enter a prompt.
5. Leave texture mode on `Generate character geometry + transfer rig` for the normal workflow.
6. Leave rig transfer enabled if you need the result attached to the selected/bundled skeleton.
7. Download the output `.glb`.

## Bundled example

The launcher now includes the AI4Animation biped default mesh as:

- `app/examples/geno.glb`

This is the main intended path when you want to generate a character from an image and bring it back into AI4Animation with the Geno skeleton contract.

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
- `texture_mode`: optional mode. Omit it, or use `character`, for the generated-character rig-transfer path. Use `ai` to only infer a texture on the existing mesh. Use `image` only when `reference_image` is already an exact UV texture map.

If `texture_mode=character` and `preserve_rig=true`, the server generates new geometry and transfers the uploaded rig onto it. If `texture_mode=ai`, `preserve_rig=true`, and the uploaded file is a rigged `.glb`, the server keeps the original rig and merges only the generated material/texture back into that GLB.

## VRAM notes

- This launcher defaults to `mmgp` profile `4` (`LowRAM_LowVRAM`).
- The practical target is still an NVIDIA GPU.
- `8-12 GB` is a realistic target for existing-mesh texturing. Generated-character mode can need more headroom because it loads shape generation and texture generation sequentially.
- Generated-character mode uses Hunyuan's fp16 safetensors shape weights; the `.ckpt` variants are not required.
- Prompt-only mode may need more headroom because it also loads Hunyuan's text-to-image pipeline.

### JavaScript

```javascript
const form = new FormData();
form.append("mesh", meshFile);
form.append("reference_image", imageFile);
form.append("preserve_rig", "true");
form.append("texture_mode", "character");

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
        data={"preserve_rig": "true", "texture_mode": "character"},
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
  -F "texture_mode=character" \
  --output textured.glb
```
