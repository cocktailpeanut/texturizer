module.exports = {
  daemon: true,
  run: [{
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      env: {
        PYTHONUNBUFFERED: "1"
      },
      message: [
        "python app.py --host 127.0.0.1 --port {{port}} --profile 4"
      ],
      on: [{
        event: "/(http:\\/\\/[0-9.:]+)/",
        done: true
      }]
    }
  }, {
    method: "local.set",
    params: {
      url: "{{input.event[1]}}"
    }
  }]
}
