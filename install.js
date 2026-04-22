module.exports = {
  run: [{
    when: "{{!exists('app/hunyuan3d')}}",
    method: "shell.run",
    params: {
      path: "app",
      message: "git clone --depth 1 https://github.com/deepbeepmeep/Hunyuan3D-2GP.git hunyuan3d"
    }
  }, {
    method: "script.start",
    params: {
      uri: "torch.js",
      params: {
        venv: "env",
        path: "app"
      }
    }
  }, {
    method: "shell.run",
    params: {
      build: true,
      env: {
        USE_NINJA: 0,
        DISTUTILS_USE_SDK: 1
      },
      venv: "../env",
      path: "app/hunyuan3d",
      message: [
        "uv pip install setuptools==65.5.0 wheel typing_extensions filelock fsspec jinja2 networkx sympy==1.14.0",
        "{{platform === 'win32' ? 'uv pip install --force-reinstall https://raw.githubusercontent.com/pinokiofactory/Hunyuan3d-2-lowvram/main/wheels/diso-0.1.4-cp310-cp310-win_amd64.whl' : null}}",
        "{{platform === 'linux' ? 'uv pip install --no-cache --force-reinstall --no-binary diso --no-build-isolation diso==0.1.4' : null}}",
        "uv pip install -r requirements.txt",
        "uv pip install -e ."
      ]
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "uv pip install -r requirements.txt"
      ]
    }
  }, {
    when: "{{platform === 'linux'}}",
    method: "shell.run",
    params: {
      build: true,
      venv: "../../../../env",
      env: {
        USE_NINJA: 0,
        DISTUTILS_USE_SDK: 1,
        NVCC_PREPEND_FLAGS: "-ccbin {{which('g++')}}"
      },
      path: "app/hunyuan3d/hy3dgen/texgen/custom_rasterizer",
      message: [
        "python setup.py install"
      ]
    }
  }, {
    when: "{{platform === 'win32'}}",
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "uv pip install --force-reinstall https://raw.githubusercontent.com/pinokiofactory/Hunyuan3d-2-lowvram/main/wheels/custom_rasterizer-0.1-cp310-cp310-win_amd64.whl"
      ]
    }
  }, {
    when: "{{platform === 'linux'}}",
    method: "shell.run",
    params: {
      build: true,
      venv: "../../../../env",
      env: {
        USE_NINJA: 0,
        DISTUTILS_USE_SDK: 1,
        NVCC_PREPEND_FLAGS: "-ccbin {{which('g++')}}"
      },
      path: "app/hunyuan3d/hy3dgen/texgen/differentiable_renderer",
      message: [
        "python setup.py install"
      ]
    }
  }, {
    when: "{{platform === 'win32'}}",
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "uv pip install --force-reinstall https://raw.githubusercontent.com/pinokiofactory/Hunyuan3d-2-lowvram/main/wheels/mesh_processor-0.0.0-cp310-cp310-win_amd64.whl"
      ]
    }
  }]
}
