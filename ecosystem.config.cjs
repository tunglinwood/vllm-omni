module.exports = {
  apps: [
    {
      name: 'kimi-audio-8091',
      cwd: '/root/learning/vllm_integration/vllm-omni',
      script: 'start_kimi_audio.cjs',
      interpreter: '/root/.nvm/versions/node/v24.13.0/bin/node',
      env: {
        PYTHONUNBUFFERED: '1',
        CUDA_VISIBLE_DEVICES: '6'
      },
      max_memory_restart: '80G',
      error_file: '/tmp/kimi_audio_error.log',
      out_file: '/tmp/kimi_audio_out.log',
      merge_logs: true,
      autorestart: true,
      watch: false
    }
  ]
};
