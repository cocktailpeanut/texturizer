module.exports = {
  run: [{
    method: "shell.run",
    params: {
      path: "app/hunyuan3d",
      message: "git pull"
    }
  }, {
    method: "fs.rm",
    params: {
      path: "app/env"
    }
  }, {
    method: "script.start",
    params: {
      uri: "install.js"
    }
  }]
}

