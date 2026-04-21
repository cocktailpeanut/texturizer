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
      venv: "../env",
      path: "app/hunyuan3d",
      message: [
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
      build: true,
      venv: "../../../../env",
      env: {
        USE_NINJA: 0,
        DISTUTILS_USE_SDK: 1
      },
      path: "app/hunyuan3d/hy3dgen/texgen/custom_rasterizer",
      message: [
        "where link",
        "where cl",
        "set",
        "python setup.py install"
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
      build: true,
      venv: "../../../../env",
      env: {
        USE_NINJA: 0,
        DISTUTILS_USE_SDK: 1
      },
      path: "app/hunyuan3d/hy3dgen/texgen/differentiable_renderer",
      message: [
        "python setup.py install"
      ]
    }
  }]
}
