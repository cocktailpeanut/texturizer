import argparse
import copy
import gc
import os
import shutil
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import torch
import trimesh
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from mmgp import offload
from PIL import Image, ImageOps
from pygltflib import Accessor, BufferView, GLTF2, PbrMetallicRoughness

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import (
    DegenerateFaceRemover,
    FaceReducer,
    FloaterRemover,
    Hunyuan3DDiTFlowMatchingPipeline,
)
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
QUADRUPED_EXAMPLE_MESH = BASE_DIR / "examples" / "dog.glb"
LOCAL_MODELS_DIR = BASE_DIR / "models"
TEXTURE_MODEL_REPO = "tencent/Hunyuan3D-2"
TEXTURE_MODEL_FOLDERS = ("hunyuan3d-delight-v2-0", "hunyuan3d-paint-v2-0")
SHAPE_MODEL_FOLDERS = ("hunyuan3d-dit-v2-0",)
SHAPE_MODEL_FILES = ("config.yaml", "model.fp16.safetensors")
PROMPT_MODEL_REPO = "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled"
CHARACTER_TEXTURE_MODE = "Generate character geometry + transfer rig"
AI_TEXTURE_MODE = "AI infer full texture from image"
IMAGE_TEXTURE_MODE = "Use image as exact UV texture map"
TEXTURE_MODE_CHOICES = [CHARACTER_TEXTURE_MODE, AI_TEXTURE_MODE, IMAGE_TEXTURE_MODE]
TEXTURE_MAX_FACES = 40000
CHARACTER_BASE_COLOR_FACTOR = [1.0, 1.0, 1.0, 1.0]
CHARACTER_METALLIC_FACTOR = 0.0
CHARACTER_ROUGHNESS_FACTOR = 0.75
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def log_step(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} [texturizer] {message}", flush=True)


def gpu_memory_gb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    total = float(torch.cuda.get_device_properties(0).total_memory)
    try:
        free, _ = torch.cuda.mem_get_info()
    except Exception:
        free = 0.0
    gib = 1024.0 ** 3
    return total / gib, float(free) / gib


def local_model_dir(repo_id: str) -> Path:
    return LOCAL_MODELS_DIR.joinpath(*repo_id.split("/"))


def download_snapshot_to_local(repo_id: str, allow_patterns: Optional[list[str]] = None) -> Path:
    import huggingface_hub

    destination = local_model_dir(repo_id)
    destination.mkdir(parents=True, exist_ok=True)
    huggingface_hub.snapshot_download(
        repo_id=repo_id,
        local_dir=str(destination),
        allow_patterns=allow_patterns,
        max_workers=4,
    )
    (destination / ".texturizer_snapshot_complete").write_text(repo_id, encoding="utf-8")
    return destination


def ensure_texture_models() -> str:
    os.environ["HY3DGEN_MODELS"] = str(LOCAL_MODELS_DIR)
    model_dir = local_model_dir(TEXTURE_MODEL_REPO)
    marker = model_dir / ".texturizer_snapshot_complete"
    missing_folders = [folder for folder in TEXTURE_MODEL_FOLDERS if not (model_dir / folder).is_dir()]
    if missing_folders or not marker.is_file():
        download_snapshot_to_local(
            TEXTURE_MODEL_REPO,
            allow_patterns=[f"{folder}/*" for folder in TEXTURE_MODEL_FOLDERS],
        )
    missing_folders = [folder for folder in TEXTURE_MODEL_FOLDERS if not (model_dir / folder).is_dir()]
    if missing_folders:
        raise RuntimeError(f"Missing Hunyuan3D texture model folders: {', '.join(missing_folders)}")
    return TEXTURE_MODEL_REPO


def ensure_shape_models() -> str:
    os.environ["HY3DGEN_MODELS"] = str(LOCAL_MODELS_DIR)
    model_dir = local_model_dir(TEXTURE_MODEL_REPO)
    missing_files = [
        f"{folder}/{filename}"
        for folder in SHAPE_MODEL_FOLDERS
        for filename in SHAPE_MODEL_FILES
        if not (model_dir / folder / filename).is_file()
    ]
    if missing_files:
        download_snapshot_to_local(
            TEXTURE_MODEL_REPO,
            allow_patterns=[
                f"{folder}/{filename}"
                for folder in SHAPE_MODEL_FOLDERS
                for filename in SHAPE_MODEL_FILES
            ],
        )
    missing_files = [
        f"{folder}/{filename}"
        for folder in SHAPE_MODEL_FOLDERS
        for filename in SHAPE_MODEL_FILES
        if not (model_dir / folder / filename).is_file()
    ]
    if missing_files:
        raise RuntimeError(f"Missing Hunyuan3D shape model files: {', '.join(missing_files)}")
    return TEXTURE_MODEL_REPO


@dataclass
class TextureResult:
    output_path: Path
    conditioning_image_path: Path
    status: str
    preview_path: Optional[Path] = None


def align4(value: int) -> int:
    return (value + 3) & ~3


def append_bytes(blob: bytearray, chunk: bytes) -> int:
    offset = align4(len(blob))
    if offset > len(blob):
        blob.extend(b"\x00" * (offset - len(blob)))
    blob.extend(chunk)
    return offset


COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
ACCESSOR_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
ARRAY_BUFFER = 34962


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


def mesh_has_usable_uv(mesh) -> bool:
    uv = getattr(getattr(mesh, "visual", None), "uv", None)
    if uv is None:
        return False
    uv = np.asarray(uv)
    if uv.ndim != 2 or uv.shape[1] != 2 or uv.shape[0] != len(mesh.vertices):
        return False
    return bool(np.isfinite(uv).all())


def read_accessor_array(gltf: GLTF2, accessor_index: int) -> np.ndarray:
    accessor = gltf.accessors[accessor_index]
    if accessor.bufferView is None:
        raise RuntimeError("Sparse or bufferless accessors are not supported.")
    dtype = np.dtype(COMPONENT_DTYPES[accessor.componentType])
    component_count = ACCESSOR_COMPONENTS[accessor.type]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    blob = gltf.binary_blob() or b""
    offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    row_bytes = dtype.itemsize * component_count
    stride = buffer_view.byteStride or row_bytes

    if stride == row_bytes:
        data = np.frombuffer(
            blob,
            dtype=dtype,
            count=accessor.count * component_count,
            offset=offset,
        )
        return data.reshape(accessor.count, component_count).copy()

    rows = np.empty((accessor.count, component_count), dtype=dtype)
    for index in range(accessor.count):
        start = offset + (index * stride)
        rows[index] = np.frombuffer(blob, dtype=dtype, count=component_count, offset=start)
    return rows


def compute_vertex_normals(positions: np.ndarray, indices: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        return None

    if indices is None:
        if positions.shape[0] % 3 != 0:
            return None
        faces = np.arange(positions.shape[0], dtype=np.int64).reshape((-1, 3))
    else:
        faces = np.asarray(indices, dtype=np.int64).reshape((-1, 3))

    if faces.shape[0] == 0:
        return None

    valid = (faces >= 0).all(axis=1) & (faces < positions.shape[0]).all(axis=1)
    faces = faces[valid]
    if faces.shape[0] == 0:
        return None

    p0 = positions[faces[:, 0]]
    p1 = positions[faces[:, 1]]
    p2 = positions[faces[:, 2]]
    face_normals = np.cross(p1 - p0, p2 - p0)
    face_lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    valid_faces = face_lengths[:, 0] > 1e-8
    faces = faces[valid_faces]
    face_normals = face_normals[valid_faces] / face_lengths[valid_faces]
    if faces.shape[0] == 0:
        return None

    normals = np.zeros_like(positions, dtype=np.float32)
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)

    vertex_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    valid_vertices = vertex_lengths[:, 0] > 1e-8
    normals[valid_vertices] /= vertex_lengths[valid_vertices]
    normals[~valid_vertices] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return normals.astype(np.float32)


def accessor_attribute_min_max(array: np.ndarray):
    if not np.issubdtype(array.dtype, np.floating):
        return None, None
    if array.ndim != 2 or array.shape[0] == 0:
        return None, None
    return array.min(axis=0).astype(float).tolist(), array.max(axis=0).astype(float).tolist()


def normalize_character_material(material) -> None:
    if material.pbrMetallicRoughness is None:
        material.pbrMetallicRoughness = PbrMetallicRoughness()
    pbr = material.pbrMetallicRoughness
    pbr.baseColorFactor = CHARACTER_BASE_COLOR_FACTOR.copy()
    pbr.metallicFactor = CHARACTER_METALLIC_FACTOR
    pbr.roughnessFactor = CHARACTER_ROUGHNESS_FACTOR


def normalize_character_materials(gltf: GLTF2) -> int:
    count = 0
    for material in gltf.materials or []:
        normalize_character_material(material)
        count += 1
    return count


def normalize_glb_character_materials(path: Path) -> Path:
    gltf = GLTF2().load_binary(str(path))
    count = normalize_character_materials(gltf)
    if count:
        gltf.save_binary(str(path))
        log_step(f"normalized character materials in {path.name}: {count} materials")
    return path


def normalize_texture_mode(texture_mode: Optional[str], image_path: Optional[Path]) -> str:
    value = (texture_mode or "auto").strip().lower()
    if value in {"auto", ""}:
        return "character"
    if value in {
        "character",
        "shape",
        "geometry",
        "full",
        "full character",
        "generated character",
        "generate character geometry",
        CHARACTER_TEXTURE_MODE.lower(),
    }:
        return "character"
    if value in {
        "image",
        "texture",
        "uv",
        "map",
        "use image as texture map",
        IMAGE_TEXTURE_MODE.lower(),
    }:
        return "image"
    if value in {
        "ai",
        "hunyuan",
        "generate",
        "reference",
        "ai retexture from reference/prompt",
        AI_TEXTURE_MODE.lower(),
    }:
        return "ai"
    raise RuntimeError(f"Unknown texture mode: {texture_mode}")


def open_rgba_image(image_path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(image_path)).convert("RGBA")


def open_rgb_image(image_path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")


def apply_image_as_existing_uv_texture(mesh, image_path: Path, output_path: Path) -> Path:
    if not mesh_has_usable_uv(mesh):
        raise RuntimeError("Image texture mode requires the mesh to have usable UVs.")

    texture_image = open_rgba_image(image_path)
    textured_mesh = mesh.copy()
    uv = np.asarray(mesh.visual.uv).copy()
    material = trimesh.visual.texture.SimpleMaterial(image=texture_image)
    textured_mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=uv,
        image=texture_image,
        material=material,
    )
    textured_mesh.export(str(output_path))
    normalize_glb_character_materials(output_path)
    return output_path


@torch.no_grad()
def paint_mesh_preserving_existing_uv(paint_pipeline, mesh, image):
    if isinstance(image, str):
        image_prompt = Image.open(image)
    else:
        image_prompt = image

    image_prompt = paint_pipeline.recenter_image(image_prompt)
    image_prompt = paint_pipeline.models["delight_model"](image_prompt)

    # Rig preservation depends on the original vertex/UV layout. The default
    # Hunyuan path calls xatlas here, which can duplicate vertices and break skinning.
    paint_pipeline.render.load_mesh(mesh)

    selected_camera_elevs = paint_pipeline.config.candidate_camera_elevs
    selected_camera_azims = paint_pipeline.config.candidate_camera_azims
    selected_view_weights = paint_pipeline.config.candidate_view_weights

    normal_maps = paint_pipeline.render_normal_multiview(
        selected_camera_elevs,
        selected_camera_azims,
        use_abs_coor=True,
    )
    position_maps = paint_pipeline.render_position_multiview(
        selected_camera_elevs,
        selected_camera_azims,
    )

    camera_info = [
        (((azim // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[elev]
        + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elev]
        for azim, elev in zip(selected_camera_azims, selected_camera_elevs)
    ]
    multiviews = paint_pipeline.models["multiview_model"](
        image_prompt,
        normal_maps + position_maps,
        camera_info,
    )

    for index in range(len(multiviews)):
        multiviews[index] = multiviews[index].resize(
            (paint_pipeline.config.render_size, paint_pipeline.config.render_size)
        )

    texture, mask = paint_pipeline.bake_from_multiview(
        multiviews,
        selected_camera_elevs,
        selected_camera_azims,
        selected_view_weights,
        method=paint_pipeline.config.merge_method,
    )

    mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
    texture = paint_pipeline.texture_inpaint(texture, mask_np)
    paint_pipeline.render.set_texture(texture)
    return paint_pipeline.render.save_mesh()


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

    def append_accessor(
        self,
        array: np.ndarray,
        component_type: int,
        accessor_type: str,
        target: Optional[int] = None,
        include_bounds: bool = False,
    ) -> int:
        dtype = np.dtype(COMPONENT_DTYPES[component_type])
        array = np.ascontiguousarray(array, dtype=dtype)
        component_count = ACCESSOR_COMPONENTS[accessor_type]
        if component_count == 1 and array.ndim == 1:
            count = array.shape[0]
        else:
            array = array.reshape((-1, component_count))
            count = array.shape[0]

        chunk = array.tobytes()
        byte_offset = append_bytes(self.original_blob, chunk)
        buffer_view = BufferView(
            buffer=0,
            byteOffset=byte_offset,
            byteLength=len(chunk),
            target=target,
        )
        buffer_view_index = len(self.original.bufferViews)
        self.original.bufferViews.append(buffer_view)

        accessor_min, accessor_max = accessor_attribute_min_max(array) if include_bounds else (None, None)
        accessor = Accessor(
            bufferView=buffer_view_index,
            byteOffset=0,
            componentType=component_type,
            count=count,
            type=accessor_type,
            min=accessor_min,
            max=accessor_max,
        )
        accessor_index = len(self.original.accessors)
        self.original.accessors.append(accessor)
        return accessor_index

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
        normalize_character_material(src)
        new_index = len(self.original.materials)
        self.original.materials.append(src)
        self.maps["materials"][index] = new_index
        return new_index

    def save(self, destination: Path) -> Path:
        normalize_character_materials(self.original)
        final_length = align4(len(self.original_blob))
        if final_length > len(self.original_blob):
            self.original_blob.extend(b"\x00" * (final_length - len(self.original_blob)))
        if self.original.buffers:
            self.original.buffers[0].byteLength = len(self.original_blob)
        self.original.set_binary_blob(bytes(self.original_blob))
        self.original.save_binary(str(destination))
        return destination


def merge_texture_into_rigged_glb(
    original_path: Path,
    textured_path: Path,
    output_path: Path,
    copy_uv: bool = True,
) -> Path:
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
    if copy_uv:
        original_primitive.attributes.TEXCOORD_0 = merger.copy_accessor(textured_uv)
    if textured_primitive.material is not None:
        original_primitive.material = merger.copy_material(textured_primitive.material)
    else:
        raise RuntimeError("Textured output has no material to merge into the rigged GLB.")
    return merger.save(output_path)


def load_mesh_geometry(mesh_path: Path):
    mesh = trimesh.load(str(mesh_path), file_type=mesh_path.suffix.lower().lstrip(".") or "glb", force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def mesh_face_count(mesh) -> int:
    faces = getattr(mesh, "faces", None)
    return int(len(faces)) if faces is not None else 0


def prepare_generated_mesh_for_texture(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    log_step(
        "generated mesh before texture postprocess: "
        f"{len(mesh.vertices)} vertices, {mesh_face_count(mesh)} faces"
    )

    try:
        mesh = FloaterRemover()(mesh)
        log_step(f"floater removal complete: {mesh_face_count(mesh)} faces")
    except Exception as exc:
        log_step(f"floater removal skipped: {exc}")

    try:
        mesh = DegenerateFaceRemover()(mesh)
        log_step(f"degenerate face cleanup complete: {mesh_face_count(mesh)} faces")
    except Exception as exc:
        log_step(f"degenerate face cleanup skipped: {exc}")

    face_count = mesh_face_count(mesh)
    if face_count > TEXTURE_MAX_FACES:
        log_step(f"reducing generated mesh from {face_count} to {TEXTURE_MAX_FACES} faces before UV unwrap")
        mesh = FaceReducer()(mesh, max_facenum=TEXTURE_MAX_FACES)
        log_step(f"face reduction complete: {mesh_face_count(mesh)} faces")
    else:
        log_step(f"face reduction not needed: {face_count} faces")

    return mesh


def align_mesh_to_template_bounds(mesh, template_path: Path):
    template_mesh = load_mesh_geometry(template_path)
    aligned = mesh.copy()
    if isinstance(aligned, trimesh.Scene):
        aligned = aligned.dump(concatenate=True)

    source_bounds = np.asarray(aligned.bounds, dtype=np.float32)
    target_bounds = np.asarray(template_mesh.bounds, dtype=np.float32)
    source_size = source_bounds[1] - source_bounds[0]
    target_size = target_bounds[1] - target_bounds[0]

    if not np.isfinite(source_size).all() or source_size[1] <= 1e-6:
        return aligned

    scale = float(target_size[1] / source_size[1]) if target_size[1] > 1e-6 else 1.0
    source_center = (source_bounds[0] + source_bounds[1]) * 0.5
    target_center = (target_bounds[0] + target_bounds[1]) * 0.5

    vertices = np.asarray(aligned.vertices, dtype=np.float32)
    vertices = (vertices - source_center) * scale
    vertices[:, 1] += target_bounds[0, 1] - vertices[:, 1].min()

    aligned_bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    aligned_center = (aligned_bounds[0] + aligned_bounds[1]) * 0.5
    vertices[:, 0] += target_center[0] - aligned_center[0]
    vertices[:, 2] += target_center[2] - aligned_center[2]
    aligned.vertices = vertices
    try:
        aligned.fix_normals()
    except Exception:
        pass
    return aligned


def transfer_rig_to_generated_glb(template_path: Path, generated_path: Path, output_path: Path) -> Path:
    template = GLTF2().load_binary(str(template_path))
    generated = GLTF2().load_binary(str(generated_path))

    if not has_rig(template):
        raise RuntimeError("Rig transfer requires a rigged GLB template.")
    if not template.meshes or not template.meshes[0].primitives:
        raise RuntimeError("Rigged template has no mesh primitives.")
    if not generated.meshes or not generated.meshes[0].primitives:
        raise RuntimeError("Generated GLB has no mesh primitives.")

    template_primitive = template.meshes[0].primitives[0]
    generated_primitive = generated.meshes[0].primitives[0]
    template_attributes = template_primitive.attributes
    generated_attributes = generated_primitive.attributes

    template_position_accessor = getattr(template_attributes, "POSITION", None)
    template_joints_accessor = getattr(template_attributes, "JOINTS_0", None)
    template_weights_accessor = getattr(template_attributes, "WEIGHTS_0", None)
    generated_position_accessor = getattr(generated_attributes, "POSITION", None)
    generated_uv_accessor = getattr(generated_attributes, "TEXCOORD_0", None)

    if template_position_accessor is None or template_joints_accessor is None or template_weights_accessor is None:
        raise RuntimeError("Rigged template is missing POSITION, JOINTS_0, or WEIGHTS_0 attributes.")
    if generated_position_accessor is None:
        raise RuntimeError("Generated GLB is missing POSITION geometry.")

    template_positions = read_accessor_array(template, template_position_accessor).astype(np.float32)
    template_joints = read_accessor_array(template, template_joints_accessor)
    template_weights = read_accessor_array(template, template_weights_accessor).astype(np.float32)
    generated_positions = read_accessor_array(generated, generated_position_accessor).astype(np.float32)
    generated_indices = (
        read_accessor_array(generated, generated_primitive.indices).reshape(-1)
        if generated_primitive.indices is not None
        else None
    )

    if generated_positions.shape[0] == 0:
        raise RuntimeError("Generated GLB has no vertices.")

    from scipy.spatial import cKDTree

    _, nearest_indices = cKDTree(template_positions).query(generated_positions, k=1)
    transferred_joints = template_joints[nearest_indices]
    transferred_weights = template_weights[nearest_indices]
    weight_sums = transferred_weights.sum(axis=1, keepdims=True)
    nonzero = weight_sums[:, 0] > 1e-8
    transferred_weights[nonzero] = transferred_weights[nonzero] / weight_sums[nonzero]
    if np.any(~nonzero):
        transferred_weights[~nonzero] = 0
        transferred_weights[~nonzero, 0] = 1

    joint_component_type = template.accessors[template_joints_accessor].componentType
    weight_component_type = template.accessors[template_weights_accessor].componentType

    merger = RigSafeMerger(template, generated)
    output_primitive = template.meshes[0].primitives[0]
    output_primitive.indices = merger.copy_accessor(generated_primitive.indices)
    output_primitive.mode = generated_primitive.mode
    output_primitive.material = merger.copy_material(generated_primitive.material)

    output_attributes = output_primitive.attributes
    output_attributes.POSITION = merger.copy_accessor(generated_position_accessor)
    generated_normal_accessor = getattr(generated_attributes, "NORMAL", None)
    if generated_normal_accessor is not None:
        output_attributes.NORMAL = merger.copy_accessor(generated_normal_accessor)
    else:
        generated_normals = compute_vertex_normals(generated_positions, generated_indices)
        output_attributes.NORMAL = (
            merger.append_accessor(generated_normals, 5126, "VEC3", target=ARRAY_BUFFER)
            if generated_normals is not None
            else None
        )
    output_attributes.TEXCOORD_0 = merger.copy_accessor(generated_uv_accessor) if generated_uv_accessor is not None else None
    output_attributes.JOINTS_0 = merger.append_accessor(
        transferred_joints,
        joint_component_type,
        "VEC4",
        target=ARRAY_BUFFER,
    )
    output_attributes.WEIGHTS_0 = merger.append_accessor(
        transferred_weights,
        weight_component_type,
        "VEC4",
        target=ARRAY_BUFFER,
    )
    template.meshes[0].primitives = [output_primitive]
    return merger.save(output_path)


class TextureService:
    def __init__(
        self,
        device: str = "cuda",
        low_vram: bool = True,
        profile: str = "auto",
        verbose: int = 0,
        mmgp_budget_mb: Optional[int] = None,
    ):
        self.device = device
        self.low_vram = low_vram
        self.profile = profile
        self.verbose = verbose
        self.mmgp_budget_mb = mmgp_budget_mb
        self.paint_pipeline = None
        self.shape_pipeline = None
        self.prompt_pipeline = None
        self.background_remover = None
        self.offload_applied = False

    def resolve_mmgp_profile(self) -> tuple[int, Optional[int]]:
        profile_text = str(self.profile).strip().lower()
        if profile_text != "auto":
            profile_no = int(profile_text)
            budget_mb = self.mmgp_budget_mb
            if budget_mb is None and profile_no not in (1, 3):
                budget_mb = 2200
            log_step(f"mmgp manual profile {profile_no}; budget={budget_mb or 'unlimited'} MB")
            return profile_no, budget_mb

        total_gb, free_gb = gpu_memory_gb()
        if total_gb >= 18:
            profile_no, budget_mb = 3, None
        elif total_gb >= 14:
            profile_no, budget_mb = 4, 6000
        elif total_gb >= 10:
            profile_no, budget_mb = 4, 3600
        else:
            profile_no, budget_mb = 5, 2200

        if self.mmgp_budget_mb is not None:
            budget_mb = self.mmgp_budget_mb

        log_step(
            "mmgp auto VRAM: "
            f"total={total_gb:.1f} GB free={free_gb:.1f} GB; "
            f"profile={profile_no}; budget={budget_mb or 'unlimited'} MB"
        )
        return profile_no, budget_mb

    def enable_texture_progress(self):
        if self.paint_pipeline is None:
            return
        for model in self.paint_pipeline.models.values():
            pipeline = getattr(model, "pipeline", None)
            if pipeline is not None and hasattr(pipeline, "set_progress_bar_config"):
                try:
                    pipeline.set_progress_bar_config(disable=False)
                except Exception:
                    pass

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
        profile_no, budget_mb = self.resolve_mmgp_profile()
        kwargs = {}
        if budget_mb is not None:
            kwargs["budgets"] = {"*": budget_mb}
        offload.default_verboseLevel = int(self.verbose)
        log_step(f"applying mmgp profile {profile_no} to models: {', '.join(pipe.keys())}")
        offload.profile(
            pipe,
            profile_no=profile_no,
            verboseLevel=int(self.verbose),
            **kwargs,
        )
        log_step("mmgp profile applied")
        self.offload_applied = True

    def ensure_paint(self):
        if self.paint_pipeline is None:
            log_step("loading Hunyuan texture pipeline")
            self.paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(ensure_texture_models())
            self.enable_texture_progress()
            log_step("Hunyuan texture pipeline loaded")
            if self.low_vram:
                self.apply_mmgp()
        return self.paint_pipeline

    def ensure_shape(self):
        if not torch.cuda.is_available():
            raise RuntimeError("Character geometry mode requires a CUDA GPU.")
        if self.shape_pipeline is None:
            log_step("loading Hunyuan shape pipeline")
            self.shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                ensure_shape_models(),
                subfolder="hunyuan3d-dit-v2-0",
                variant="fp16",
            )
            log_step("Hunyuan shape pipeline loaded")
        return self.shape_pipeline

    def release_shape(self):
        if self.shape_pipeline is None:
            return
        self.shape_pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_step("released Hunyuan shape pipeline from VRAM")

    def ensure_prompt(self):
        if HunyuanDiTPipeline is None:
            raise RuntimeError("Prompt-only mode is unavailable because HunyuanDiT could not be imported.")
        if self.prompt_pipeline is None:
            prompt_model_path = download_snapshot_to_local(PROMPT_MODEL_REPO)
            self.prompt_pipeline = HunyuanDiTPipeline(str(prompt_model_path))
            if self.low_vram:
                self.apply_mmgp()
        return self.prompt_pipeline

    def ensure_remover(self):
        if self.background_remover is None:
            self.background_remover = BackgroundRemover()
        return self.background_remover

    def build_conditioning_image(self, image_path: Optional[Path], prompt: str, job_dir: Path) -> Path:
        if image_path is not None:
            log_step("using uploaded reference image for conditioning")
            reference_image = open_rgb_image(image_path)
        else:
            prompt = (prompt or "").strip()
            if not prompt:
                raise RuntimeError("Upload a reference image or enter a prompt.")
            log_step("generating prompt image for conditioning")
            prompt_pipeline = self.ensure_prompt()
            reference_image = prompt_pipeline(prompt)

        remover = self.ensure_remover()
        log_step("removing background from conditioning image")
        conditioning = remover(reference_image)
        conditioning_path = job_dir / "conditioning.png"
        conditioning.save(conditioning_path)
        log_step(f"conditioning image saved: {conditioning_path.name}")
        return conditioning_path

    def load_mesh_for_texturing(self, mesh_path: Path):
        return load_mesh_geometry(mesh_path)

    def generate_textured_character_mesh(self, conditioning_path: Path, template_path: Path, job_dir: Path) -> Path:
        conditioning = Image.open(conditioning_path).convert("RGBA")
        shape = self.ensure_shape()
        log_step("generating new character mesh from conditioning image")
        generated = shape(
            image=conditioning,
            num_inference_steps=50,
            octree_resolution=384,
            num_chunks=8000,
            generator=torch.manual_seed(12345),
            output_type="trimesh",
        )
        generated_mesh = generated[0] if generated else None
        if generated_mesh is None:
            raise RuntimeError("Hunyuan shape generation did not return a mesh.")
        log_step("new character mesh generated")

        if self.low_vram:
            self.release_shape()

        generated_mesh = prepare_generated_mesh_for_texture(generated_mesh)
        untextured_path = job_dir / "generated_character_untextured.glb"
        generated_mesh.export(str(untextured_path))
        log_step(f"postprocessed untextured GLB exported: {untextured_path.name}")

        paint = self.ensure_paint()
        log_step("texturing generated character mesh")
        textured_mesh = paint(generated_mesh, image=conditioning)
        log_step("generated character mesh textured")
        textured_mesh = align_mesh_to_template_bounds(textured_mesh, template_path)
        log_step("aligned generated mesh to template bounds")
        preview_path = job_dir / "generated_character_preview.glb"
        textured_mesh.export(str(preview_path))
        normalize_glb_character_materials(preview_path)
        log_step(f"preview GLB exported: {preview_path.name}")
        return preview_path

    def texture(
        self,
        mesh_path: Path,
        image_path: Optional[Path],
        prompt: str,
        preserve_rig: bool,
        texture_mode: Optional[str] = None,
    ) -> TextureResult:
        if not mesh_path.exists():
            raise RuntimeError("Mesh file does not exist.")

        mode = normalize_texture_mode(texture_mode, image_path)
        job_dir = OUTPUT_DIR / str(uuid.uuid4())
        job_dir.mkdir(parents=True, exist_ok=True)
        log_step(f"started job {job_dir.name}; mode={mode}")

        input_copy = job_dir / mesh_path.name
        shutil.copy2(mesh_path, input_copy)

        rigged_glb = input_copy.suffix.lower() == ".glb" and has_rig(GLTF2().load_binary(str(input_copy)))
        geometry_input = input_copy
        if rigged_glb:
            geometry_input = create_geometry_only_glb(input_copy, job_dir / "geometry_only.glb")

        mesh = self.load_mesh_for_texturing(geometry_input)

        if mode == "character":
            conditioning_path = self.build_conditioning_image(image_path, prompt, job_dir)
            preview_path = self.generate_textured_character_mesh(conditioning_path, input_copy, job_dir)

            rig_template = None
            if preserve_rig and rigged_glb:
                rig_template = input_copy
            elif preserve_rig and EXAMPLE_MESH.exists():
                rig_template = EXAMPLE_MESH

            if rig_template is not None:
                final_path = job_dir / "generated_character_rigged.glb"
                transfer_rig_to_generated_glb(rig_template, preview_path, final_path)
                log_step(f"rigged final GLB exported: {final_path.name}")
                if rigged_glb:
                    status = "Generated new character geometry from the image, textured it, and transferred the original rig onto the generated mesh."
                else:
                    status = "Generated new character geometry from the image, textured it, and transferred the bundled Geno rig onto the generated mesh."
                return TextureResult(final_path, conditioning_path, status, preview_path)

            status = "Generated and textured new character geometry. No rig was transferred because rig transfer was disabled."
            return TextureResult(preview_path, conditioning_path, status, preview_path)

        if mode == "image":
            if image_path is None:
                raise RuntimeError("Image texture mode requires an uploaded image.")

            log_step("applying uploaded image as exact UV texture")
            texture_source_path = job_dir / "texture_source.png"
            open_rgba_image(image_path).save(texture_source_path)
            preview_path = job_dir / "textured_preview.glb"
            apply_image_as_existing_uv_texture(mesh, texture_source_path, preview_path)
            log_step(f"preview GLB exported: {preview_path.name}")

            if rigged_glb and preserve_rig:
                final_path = job_dir / "textured_rigged.glb"
                merge_texture_into_rigged_glb(input_copy, preview_path, final_path, copy_uv=False)
                log_step(f"rigged final GLB exported: {final_path.name}")
                status = "Applied uploaded image as the existing UV texture map. Rig preserved in the original GLB."
                return TextureResult(final_path, texture_source_path, status, preview_path)

            if rigged_glb and not preserve_rig:
                status = "Applied uploaded image as the existing UV texture map. Returned geometry-only output without rig preservation."
            else:
                status = "Applied uploaded image as the existing UV texture map."
            return TextureResult(preview_path, texture_source_path, status, preview_path)

        conditioning_path = self.build_conditioning_image(image_path, prompt, job_dir)
        paint = self.ensure_paint()
        if rigged_glb and preserve_rig:
            if not mesh_has_usable_uv(mesh):
                raise RuntimeError("Rig-preserve mode requires the rigged GLB to have usable UVs.")
            log_step("texturing existing mesh while preserving original UV layout")
            textured_mesh = paint_mesh_preserving_existing_uv(
                paint,
                mesh,
                Image.open(conditioning_path).convert("RGBA"),
            )
        else:
            log_step("texturing mesh with Hunyuan texture pipeline")
            textured_mesh = paint(mesh, image=Image.open(conditioning_path).convert("RGBA"))
        log_step("mesh texturing completed")
        preview_path = job_dir / "textured_preview.glb"
        textured_mesh.export(str(preview_path))
        normalize_glb_character_materials(preview_path)
        log_step(f"preview GLB exported: {preview_path.name}")

        if self.low_vram and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if rigged_glb and preserve_rig:
            final_path = job_dir / "textured_rigged.glb"
            merge_texture_into_rigged_glb(input_copy, preview_path, final_path, copy_uv=False)
            log_step(f"rigged final GLB exported: {final_path.name}")
            status = "Textured successfully. Hunyuan inferred the full texture and the original rigged GLB was preserved."
            return TextureResult(final_path, conditioning_path, status, preview_path)

        if rigged_glb and not preserve_rig:
            status = "Textured successfully. Returned raw Hunyuan output without rig preservation."
        else:
            status = "Textured successfully. Input mesh was not a rigged GLB, so no rig-preservation merge was needed."
        return TextureResult(preview_path, conditioning_path, status, preview_path)


service = None

TARGET_GENO = "Geno biped"
TARGET_DOG = "Dog quadruped"
TARGET_CUSTOM = "Custom GLB"


def run_ui(
    mesh_path: Optional[str],
    target_rig: str,
    image_path: Optional[str],
    prompt: str,
    texture_mode: str,
    preserve_rig: bool,
):
    mode = normalize_texture_mode(texture_mode, Path(image_path) if image_path else None)
    if target_rig == TARGET_GENO:
        mesh_path = str(EXAMPLE_MESH)
    elif target_rig == TARGET_DOG:
        mesh_path = str(QUADRUPED_EXAMPLE_MESH)
    elif not mesh_path and mode == "character" and EXAMPLE_MESH.exists():
        mesh_path = str(EXAMPLE_MESH)
    if not mesh_path:
        raise gr.Error("Choose a bundled rig template or upload a custom GLB.")
    if service is None:
        raise gr.Error("Service is not initialized.")
    try:
        result = service.texture(
            Path(mesh_path),
            Path(image_path) if image_path else None,
            prompt or "",
            preserve_rig,
            texture_mode,
        )
        return (
            gr.update(visible=True, value=str(result.conditioning_image_path)),
            gr.update(visible=True, value=str(result.output_path)),
            gr.update(visible=True, value=str(result.output_path)),
            gr.update(visible=True, value=result.status),
            gr.update(visible=False),
        )
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(str(exc))


def select_target_rig(target_rig: str):
    custom_visible = target_rig == TARGET_CUSTOM
    if target_rig == TARGET_GENO:
        mesh_path = str(EXAMPLE_MESH)
        status = "Selected: Geno biped rig. Output is intended for the biped locomotion demo."
    elif target_rig == TARGET_DOG:
        mesh_path = str(QUADRUPED_EXAMPLE_MESH)
        status = "Selected: Dog quadruped rig. Output is intended for the quadruped locomotion demo."
    else:
        mesh_path = None
        status = "Selected: Custom GLB. Upload a rigged mesh below."
    return (
        mesh_path,
        gr.update(visible=custom_visible),
        status,
    )


def load_bundled_mesh(mesh_path: Path):
    if not mesh_path.exists():
        raise gr.Error(f"Bundled demo mesh is missing: {mesh_path.name}")
    return str(mesh_path)


def load_bundled_geno_mesh():
    return (
        TARGET_GENO,
        str(EXAMPLE_MESH),
        gr.update(visible=False),
        "Selected: Geno biped rig. Output is intended for the biped locomotion demo.",
    )


def load_bundled_quadruped_mesh():
    return (
        TARGET_DOG,
        str(QUADRUPED_EXAMPLE_MESH),
        gr.update(visible=False),
        "Selected: Dog quadruped rig. Output is intended for the quadruped locomotion demo.",
    )


def custom_mesh_uploaded(mesh_path: Optional[str]):
    if mesh_path:
        return (
            TARGET_CUSTOM,
            mesh_path,
            gr.update(visible=True),
            f"Selected: Custom GLB ({Path(mesh_path).name}).",
        )
    return (
        TARGET_CUSTOM,
        None,
        gr.update(visible=True),
        "Selected: Custom GLB. Upload a rigged mesh below.",
    )


def clear_outputs():
    return (
        gr.update(visible=False, value=None),
        gr.update(visible=False, value=None),
        gr.update(visible=False, value=None),
        gr.update(visible=False, value=""),
        gr.update(visible=True),
    )


def build_ui():
    with gr.Blocks(title="Texturizer") as demo:
        gr.Markdown(
            """
            # Texturizer

            Pick a target rig, add a reference image or prompt, then generate a textured GLB.
            Bundled rigs preserve the AI4Animation skeleton contracts for biped and quadruped demos.
            """
        )
        with gr.Row():
            with gr.Column():
                mesh_input = gr.File(
                    label="Custom mesh",
                    file_types=[".glb", ".gltf", ".obj"],
                    type="filepath",
                    visible=False,
                )
                target_rig_input = gr.Radio(
                    choices=[TARGET_GENO, TARGET_DOG, TARGET_CUSTOM],
                    value=TARGET_GENO,
                    label="Target rig",
                )
                selected_mesh_state = gr.State(str(EXAMPLE_MESH))
                target_status = gr.Markdown("Selected: Geno biped rig. Output is intended for the biped locomotion demo.")
                with gr.Row():
                    with gr.Column():
                        gr.Model3D(
                            value=str(EXAMPLE_MESH) if EXAMPLE_MESH.exists() else None,
                            label="Geno biped preset",
                            height=180,
                        )
                        bundled_geno_button = gr.Button("Select Geno")
                    with gr.Column():
                        gr.Model3D(
                            value=str(QUADRUPED_EXAMPLE_MESH) if QUADRUPED_EXAMPLE_MESH.exists() else None,
                            label="Dog quadruped preset",
                            height=180,
                        )
                        bundled_quadruped_button = gr.Button("Select Dog")
                image_input = gr.Image(label="Reference image", type="filepath")
                texture_mode_input = gr.Radio(
                    choices=TEXTURE_MODE_CHOICES,
                    value=CHARACTER_TEXTURE_MODE,
                    label="Texture mode",
                )
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Used when no image is uploaded.",
                    lines=3,
                )
                preserve_rig_input = gr.Checkbox(
                    label="Transfer/preserve rig when possible",
                    value=True,
                )
                submit = gr.Button("Texturize", variant="primary")
            with gr.Column():
                empty_output = gr.Markdown("Output preview will appear here after texturizing.")
                conditioning_output = gr.Image(label="Source or conditioning image", visible=False)
                model_output = gr.Model3D(label="Final GLB preview", visible=False)
                file_output = gr.File(label="Download final GLB", visible=False)
                status_output = gr.Textbox(label="Status", lines=4, visible=False)

        target_rig_input.change(
            fn=select_target_rig,
            inputs=target_rig_input,
            outputs=[selected_mesh_state, mesh_input, target_status],
        ).then(
            fn=clear_outputs,
            outputs=[conditioning_output, model_output, file_output, status_output, empty_output],
        )

        bundled_geno_button.click(
            fn=load_bundled_geno_mesh,
            outputs=[target_rig_input, selected_mesh_state, mesh_input, target_status],
        ).then(
            fn=clear_outputs,
            outputs=[conditioning_output, model_output, file_output, status_output, empty_output],
        )

        bundled_quadruped_button.click(
            fn=load_bundled_quadruped_mesh,
            outputs=[target_rig_input, selected_mesh_state, mesh_input, target_status],
        ).then(
            fn=clear_outputs,
            outputs=[conditioning_output, model_output, file_output, status_output, empty_output],
        )

        mesh_input.change(
            fn=custom_mesh_uploaded,
            inputs=mesh_input,
            outputs=[target_rig_input, selected_mesh_state, mesh_input, target_status],
        ).then(
            fn=clear_outputs,
            outputs=[conditioning_output, model_output, file_output, status_output, empty_output],
        )

        submit.click(
            fn=run_ui,
            inputs=[selected_mesh_state, target_rig_input, image_input, prompt_input, texture_mode_input, preserve_rig_input],
            outputs=[conditioning_output, model_output, file_output, status_output, empty_output],
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
    texture_mode: Optional[str] = Form(None),
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
        result = service.texture(mesh_path, image_path, prompt or "", preserve_rig, texture_mode)
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
    parser.add_argument("--profile", type=str, default="auto")
    parser.add_argument("--mmgp-budget-mb", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()
    service = TextureService(
        device="cuda",
        low_vram=True,
        profile=args.profile,
        verbose=args.verbose,
        mmgp_budget_mb=args.mmgp_budget_mb,
    )
    print(f"http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
