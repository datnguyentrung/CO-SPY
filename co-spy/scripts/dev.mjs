import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { join } from 'node:path'

const appRoot = process.cwd()
const projectRoot = join(appRoot, '..')
const configuredPython = process.env.CO_SPY_PYTHON
const python = configuredPython || (process.platform === 'win32' ? 'python' : 'python3')
const isWindows = process.platform === 'win32'
const npm = isWindows ? 'npm.cmd' : 'npm'

if (configuredPython && !existsSync(configuredPython)) {
  console.error(`[DEV] CO_SPY_PYTHON does not exist: ${configuredPython}`)
  process.exit(1)
}

const children = [
  // Windows cannot directly spawn npm.cmd with shell disabled (EINVAL).
  spawn(
    isWindows ? 'cmd.exe' : npm,
    isWindows ? ['/d', '/s', '/c', npm, 'run', 'dev:frontend'] : ['run', 'dev:frontend'],
    { cwd: appRoot, stdio: 'inherit', shell: false },
  ),
  spawn(python, ['-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', '8000'], {
    cwd: appRoot,
    stdio: 'inherit',
    shell: false,
    env: { ...process.env, PYTHONPATH: projectRoot },
  }),
]

const stop = () => children.forEach((child) => child.kill())
process.on('SIGINT', stop)
process.on('SIGTERM', stop)
children.forEach((child) => child.on('exit', (code) => {
  if (code && code !== 0) stop()
}))
