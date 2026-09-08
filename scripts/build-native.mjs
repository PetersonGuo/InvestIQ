import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { existsSync } from 'node:fs';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const python = path.join(root, 'backend/.venv/bin/python');
const args = ['-S', path.join(root, 'backend/native'), '-B', path.join(root, 'backend/build/native'), '-DCMAKE_BUILD_TYPE=Release'];
if (existsSync('/opt/homebrew')) args.push('-DCMAKE_PREFIX_PATH=/opt/homebrew');
if (existsSync(python)) args.push(`-DPython3_EXECUTABLE=${python}`);
if (process.env.IBKR_CPP_SDK_DIR) args.push(`-DIBKR_CPP_SDK_DIR=${process.env.IBKR_CPP_SDK_DIR}`);
for (const command of [args, ['--build', path.join(root, 'backend/build/native'), '-j', '4']]) {
  const result = spawnSync('cmake', command, { stdio: 'inherit' });
  if (result.error) { console.error(result.error.message); process.exit(1); }
  if (result.status) process.exit(result.status);
}
