import { spawn, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const build = spawnSync(process.execPath, [path.join(root, 'scripts/build-native.mjs')], { stdio: 'inherit' });
if (build.error || build.status) process.exit(build.status || 1);
const backend = path.join(root, 'backend/build/native/stockassist-server');
if (!existsSync(backend)) { console.error('Native backend build is missing.'); process.exit(1); }
const children = [
  spawn(backend, ['--port', '8000'], { cwd: path.join(root, 'backend'), stdio: 'inherit' }),
  spawn(process.platform === 'win32' ? 'npm.cmd' : 'npm', ['run', 'dev', '--', '--hostname', '127.0.0.1'], { cwd: path.join(root, 'frontend'), stdio: 'inherit' }),
];
let stopping = false;
function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  for (const child of children) child.kill('SIGTERM');
  process.exitCode = code;
}
for (const child of children) {
  child.on('error', error => { console.error(error.message); stop(1); });
  child.on('exit', code => { if (!stopping) stop(code || 0); });
}
process.on('SIGINT', () => stop());
process.on('SIGTERM', () => stop());
