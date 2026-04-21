import argparse
import copy
import os
import shutil
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gradio as gr
import torch
import trimesh
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from mmgp import offload
from PIL import Image
from pygltflib import GLTF2

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.texgen import Hunyuan3DPaintPipeline

try:
    from hy3dgen.text2image import HunyuanDiTPipeline
except Exception:
    HunyuanDiTPipeline = None


torch.set_default_device("cpu")


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
EXAMPLE_MESH = BASE_DIR / "examples" / "geno.glb"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TextureResult:
    output_path: Path
    conditioning_image_path: Path
    status: str


def align4(value: int) -> int:
    return (value + 3) & ~3


def append_bytes(blob: bytearray, chunk: bytes) -> int:
    offset = align4(len(blob))
    if offset > len(blob):
        blob.extend(b"\x00" * (offset - len(blob)))
    blob.extend(chunk)
    return offset


def deep_copy_gltf(path: Path) -> GLTF2:
    return copy.deepcopy(GLTF2().load_binary(str(path)))


def has_rig(gltf: GLTF2) -> bool:
    if gltf.skins:
        return True
    return any(getattr(node, "skin", None) is not None for node in (gltf.nodes or []))


def create_geometry_only_glb(source_path: Path, target_path: Path) -> Path:
    gltf = deep_copy_gltf(source_path)
    gltf.skins = []
    gltf.animations = []
    for node in gltf.nodes or []:
        node.skin = None
    gltf.save_binary(str(target_path))
    return target_path


def accessor_count(gltf: GLTF2, accessor_index: int) -> int:
    return gltf.accessors[accessor_index].count


class RigSafeMerger:
    def __init__(self, original: GLTF2, textured: GLTF2):
        self.original = original
        self.textured = textured
        self.original_blob = bytearray(original.binary_blob() or b"")
        self.textured_blob = textured.binary_blob() or b""
        for attr in ("bufferViews", "accessors", "images", "textures", "materials", "samplers"):
            if getattr(self.original, attr) is None:
                setattr(self.original, attr, [])
        self.maps = {
            "bufferViews": {},
            "accessors": {},
            "images": {},
            "textures": {},
            "materials": {},
            "samplers": {},
        }

    def copy_buffer_view(self, index: Optional[int]) -> Optional[int]:
        if index is None:
            return None
        cached = self.maps["bufferViews"].get(index)
        if cached is not None:
            return cached
        src = copy.deepcopy(self.textured.bufferViews[index])
        start = src.byteOffset or 0
        end = start + src.byteLength
        new_offset = append_bytes(self.original_blob, self.textured_blob[start:end])
        src.buffer = 0
        src.byteOffset = new_offset
        new_index = len(self.original.bufferViews)
        self.original.bufferViews.append(src)
        self.maps["bufferViews"][index] = new_index
        return new_index

    def copy_accessor(self, index: Optional[int]) -> Optional[int]:
        if index is None:
            return None
        cached = self.maps["accessors"].get(index)
        if cached is not None:
            return cached
        src = copy.deepcopy(self.textured.accessors[index])
        src.bufferView = self.copy_buffer_view(src.bufferView)
        new_index = len(self.original.accessors)
        self.original.accessors.append(src)
        self.maps["accessors"][index] = new_index
        return new_index

    def copy_sampler(self, index: Optional[int]) -> Optional[int]:
        if index is None:
            return None
        cached = self.maps["samplers"].get(index)
        if cached is not None:
            return cached
        src = copy.deepcopy(self.textured.samplers[index])
        new_index = len(self.original.samplers)
        self.original.samplers.append(src)
        self.maps["samplers"][index] = new_index
        return new_index

    def copy_image(self, index: Optional[int]) -> Optional[int]:
        if index is None:
            return None
        cached = self.maps["images"].get(index)
        if cached is not None:
            return cached
        src = copy.deepcopy(self.textured.images[index])
        src.bufferView = self.copy_buffer_view(getattr(src, "bufferView", None))
        new_index = len(self.original.images)
        self.original.images.append(src)
        self.maps["images"][index] = new_index
        return new_index

    def copy_texture(self, index: Optional[int]) -> Optional[int]:
        if index is None:
            return None
        cached = self.maps["textures"].get(index)
        if cached is not None:
            return cached
        src = copy.deepcopy(self.textured.textures[index])
        src.source = self.copy_image(getattr(src, "source", None))
        src.sampler = self.copy_sampler(getattr(src, "sampler", None))
        new_index = len(self.original.textures)
        self.original.textures.append(src)
        self.maps["textures"][index] = new_index
        return new_index

    def remap_texture_info(self, info):
        if info is None:
            return None
        new_info = copy.deepcopy(info)
        new_info.index = self.copy_texture(info.index)
        return new_info

    def copy_material(self, index: Optional[int]) -> Optional[int]:
        if index is None:
            return None
        cached = self.maps["materials"].get(index)
        if cached is not None:
            return cached
        src = copy.deepcopy(self.textured.materials[index])
        if src.pbrMetallicRoughness is not None:
            src.pbrMetallicRoughness.baseColorTexture = self.remap_texture_info(src.pbrMetallicRoughness.baseColorTexture)
            src.pbrMetallicRoughness.metallicRoughnessTexture = self.remap_texture_info(src.pbrMetallicRoughness.metallicRoughnessTexture)
        src.normalTexture = self.remap_texture_info(src.normalTexture)
        src.occlusionTexture = self.remap_texture_info(src.occlusionTexture)
        src.emissiveTexture = self.remap_texture_info(src.emissiveTexture)
        new_index = len(self.original.materials)
        self.original.materials.append(src)
        self.maps["materials"][index] = new_index
        return new_index

    def save(self, destination: Path) -> Path:
        if self.original.buffers:
            self.original.buffers[0].byteLength = len(self.original_blob)
        self.original.set_binary_blob(bytes(self.original_blob))
        self.original.save_binary(str(destination))
        return destination


def merge_texture_into_rigged_glb(original_path: Path, textured_path: Path, output_path: Path) -> Path:
    original = GLTF2().load_binary(str(original_path))
    textured = GLTF2().load_binary(str(textured_path))

    if not original.meshes or not original.meshes[0].primitives:
        raise RuntimeError("Original GLB has no mesh primitives.")
    if not textured.meshes or not textured.meshes[0].primitives:
        raise RuntimeError("Textured GLB has no mesh primitives.")

    original_primitive = original.meshes[0].primitives[0]
    textured_primitive = textured.meshes[0].primitives[0]

    original_position = getattr(original_primitive.attributes, "POSITION", None)
    textured_position = getattr(textured_primitive.attributes, "POSITION", None)
    textured_uv = getattr(textured_primitive.attributes, "TEXCOORD_0", None)

    if original_position is None or textured_position is None or textured_uv is None:
        raise RuntimeError("Missing POSITION or TEXCOORD_0 accessor required for a rig-safe merge.")

    if accessor_count(original, original_position) != accessor_count(textured, textured_position):
        raise RuntimeError("Hunyuan changed the vertex layout, so a rig-safe merge is unsafe.")

    merger = RigSafeMerger(original, textured)
    original_primitive.attributes.TEXCOORD_0 = merger.copy_accessor(textured_uv)
    if textured_primitive.material is not None:
        original_primitive.material = merger.copy_material(textured_primitive.material)
    return merger.save(output_path)


class TextureService:
    def __init__(self, device: str = "cuda", low_vram: bool = True, profile: int = 4, verbose: int = 0):
        self.device = device
        self.low_vram = low_vram
        self.profile = profile
        self.verbose = verbose
        self.paint_pipeline = None
        self.prompt_pipeline = None
        self.background_remover = None
        self.offload_applied = False

    def apply_mmgp(self):
        pipe = {}
        if self.paint_pipeline is not None:
            pipe.update(offload.extract_models("texgen_worker", self.paint_pipeline))
            try:
                self.paint_pipeline.models["multiview_model"].pipeline.vae.use_slicing = True
            except Exception:
                pass
        if self.prompt_pipeline is not None:
            pipe.update(offload.extract_models("t2i_worker", self.prompt_pipeline))
        if not pipe:
            return
        kwargs = {}
        if self.paint_pipeline is not None and self.profile < 5:
            kwargs["pinnedMemory"] = "texgen_worker/model"
        if self.profile != 1 and self.profile != 3:
            kwargs["budgets"] = {"*": 2200}
        offload.default_verboseLevel = int(self.verbose)
        offload.profile(
            pipe,
            profile_no=int(self.profile),
            verboseLevel=int(self.verbose),
            **kwargs,
        )
        self.offload_applied = True

    def ensure_paint(self):
        if self.paint_pipeline is None:
            self.paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
            if self.low_vram:
                self.apply_mmgp()
        return self.paint_pipeline

    def ensure_prompt(self):
        if HunyuanDiTPipeline is None:
            raise RuntimeError("Prompt-only mode is unavailable because HunyuanDiT could not be imported.")
        if self.prompt_pipeline is None:
            self.prompt_pipeline = HunyuanDiTPipeline("Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled")
            if self.low_vram:
                self.apply_mmgp()
        return self.prompt_pipeline

    def ensure_remover(self):
        if self.background_remover is None:
            self.background_remover = BackgroundRemover()
        return self.background_remover

    def build_conditioning_image(self, image_path: Optional[Path], prompt: str, job_dir: Path) -> Path:
        if image_path is not None:
            reference_image = Image.open(image_path).convert("RGB")
            source_label = "reference image"
        else:
            prompt = (prompt or "").strip()
            if not prompt:
                raise RuntimeError("Upload a reference image or enter a prompt.")
            prompt_pipeline = self.ensure_prompt()
            reference_image = prompt_pipeline(prompt)
            source_label = "generated prompt image"

        remover = self.ensure_remover()
        conditioning = remover(reference_image)
        conditioning_path = job_dir / "conditioning.png"
        conditioning.save(conditioning_path)
        return conditioning_path

    def load_mesh_for_texturing(self, mesh_path: Path):
        file_type = mesh_path.suffix.lower().lstrip(".") or "glb"
        mesh = trimesh.load(str(mesh_path), file_type=file_type, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        return mesh

    def texture(self, mesh_path: Path, image_path: Optional[Path], prompt: str, preserve_rig: bool) -> TextureResult:
        if not mesh_path.exists():
            raise RuntimeError("Mesh file does not exist.")

        job_dir = OUTPUT_DIR / str(uuid.uuid4())
        job_dir.mkdir(parents=True, exist_ok=True)

        input_copy = job_dir / mesh_path.name
        shutil.copy2(mesh_path, input_copy)

        conditioning_path = self.build_conditioning_image(image_path, prompt, job_dir)
        rigged_glb = input_copy.suffix.lower() == ".glb" and has_rig(GLTF2().load_binary(str(input_copy)))
        geometry_input = input_copy
        if rigged_glb:
            geometry_input = create_geometry_only_glb(input_copy, job_dir / "geometry_only.glb")

        mesh = self.load_mesh_for_texturing(geometry_input)
        paint = self.ensure_paint()
        textured_mesh = paint(mesh, image=Image.open(conditioning_path).convert("RGBA"))
        preview_path = job_dir / "textured_preview.glb"
        textured_mesh.export(str(preview_path))

        if self.low_vram and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if rigged_glb and preserve_rig:
            final_path = job_dir / "textured_rigged.glb"
            merge_texture_into_rigged_glb(input_copy, preview_path, final_path)
            status = "Textured successfully. Rig preserved by merging UVs and materials back into the original GLB."
            return TextureResult(final_path, conditioning_path, status)

        if rigged_glb and not preserve_rig:
            status = "Textured successfully. Returned raw Hunyuan output without rig preservation."
        else:
            status = "Textured successfully. Input mesh was not a rigged GLB, so no rig-preservation merge was needed."
        return TextureResult(preview_path, conditioning_path, status)


service = None


def run_ui(mesh_path: Optional[str], image_path: Optional[str], prompt: str, preserve_rig: bool):
    if not mesh_path:
        raise gr.Error("Upload a mesh first.")
    if service is None:
        raise gr.Error("Service is not initialized.")
    try:
        result = service.texture(
            Path(mesh_path),
            Path(image_path) if image_path else None,
            prompt or "",
            preserve_rig,
        )
        return (
            str(result.conditioning_image_path),
            str(result.output_path),
            str(result.output_path),
            result.status,
        )
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(str(exc))


def load_bundled_mesh():
    if not EXAMPLE_MESH.exists():
        raise gr.Error("Bundled demo mesh is missing.")
    return str(EXAMPLE_MESH)


def build_ui():
    with gr.Blocks(title="Texturizer") as demo:
        gr.Markdown(
            """
            # Texturizer

            Upload an existing mesh and a reference image, or use a prompt.
            If the input is a rigged GLB and Hunyuan keeps the original vertex layout,
            the output is merged back into the original file so the rig stays intact.
            The bundled `geno.glb` example matches the AI4Animation biped default mesh.
            """
        )
        with gr.Row():
            with gr.Column():
                mesh_input = gr.File(label="Mesh", file_types=[".glb", ".gltf", ".obj"], type="filepath")
                bundled_mesh_button = gr.Button("Use bundled Geno demo mesh")
                image_input = gr.Image(label="Reference image", type="filepath")
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Optional. Used only when no reference image is uploaded.",
                    lines=3,
                )
                preserve_rig_input = gr.Checkbox(
                    label="Preserve original rig when possible",
                    value=True,
                )
                submit = gr.Button("Texturize", variant="primary")
            with gr.Column():
                conditioning_output = gr.Image(label="Conditioning image")
                model_output = gr.Model3D(label="Output preview")
                file_output = gr.File(label="Download output")
                status_output = gr.Textbox(label="Status", lines=4)

        bundled_mesh_button.click(
            fn=load_bundled_mesh,
            outputs=mesh_input,
        )

        submit.click(
            fn=run_ui,
            inputs=[mesh_input, image_input, prompt_input, preserve_rig_input],
            outputs=[conditioning_output, model_output, file_output, status_output],
        )

    return demo


api = FastAPI(title="Texturizer API")


@api.get("/api/health")
def health():
    return JSONResponse({"ok": True})


@api.post("/api/texture")
async def texture_endpoint(
    mesh: UploadFile = File(...),
    reference_image: Optional[UploadFile] = File(None),
    prompt: Optional[str] = Form(None),
    preserve_rig: bool = Form(True),
):
    if service is None:
        raise HTTPException(status_code=503, detail="Service is not initialized.")
    mesh_path = UPLOAD_DIR / f"{uuid.uuid4()}_{mesh.filename}"
    with mesh_path.open("wb") as handle:
        handle.write(await mesh.read())

    image_path = None
    if reference_image is not None:
        image_path = UPLOAD_DIR / f"{uuid.uuid4()}_{reference_image.filename}"
        with image_path.open("wb") as handle:
            handle.write(await reference_image.read())

    try:
        result = service.texture(mesh_path, image_path, prompt or "", preserve_rig)
        return FileResponse(result.output_path, filename=result.output_path.name, media_type="model/gltf-binary")
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=str(exc))


gradio_app = build_ui()
app = gr.mount_gradio_app(api, gradio_app, path="/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    parser.add_argument("--profile", type=int, default=4)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()
    service = TextureService(device="cuda", low_vram=True, profile=args.profile, verbose=args.verbose)
    print(f"http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
