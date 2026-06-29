const { spawn } = require('child_process');

const proc = spawn('vllm-omni', [
  'serve',
  '/data1/moonshotai/Kimi-Audio-7B-Instruct',
  '--omni',
  '--port', '8091',
  '--deploy-config', 'vllm_omni/deploy/kimi_audio.yaml',
  '--trust-remote-code'
], {
  cwd: __dirname,
  stdio: 'inherit',
  env: { ...process.env, CUDA_VISIBLE_DEVICES: '6' }
});

proc.on('error', (err) => {
  console.error('Failed to start vllm-omni:', err);
  process.exit(1);
});

proc.on('close', (code) => {
  console.log(`vllm-omni process exited with code ${code}`);
  process.exit(code);
});
