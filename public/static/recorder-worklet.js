// AudioWorklet that captures raw PCM (mono mix-down) while recording is on.
// Samples are posted to the main thread in blocks; no processing happens here.

class RecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.recording = false;
    this.block = new Float32Array(8192);
    this.fill = 0;
    this.port.onmessage = (e) => {
      if (e.data === 'start') {
        this.fill = 0;
        this.recording = true;
      } else if (e.data === 'stop') {
        this.recording = false;
        this.flush();
        this.port.postMessage({ type: 'stopped' });
      }
    };
  }

  flush() {
    if (this.fill > 0) {
      const out = this.block.slice(0, this.fill);
      this.port.postMessage({ type: 'data', samples: out }, [out.buffer]);
      this.fill = 0;
    }
  }

  process(inputs) {
    const input = inputs[0];
    if (!this.recording || !input || input.length === 0) return true;
    const channels = input.length;
    const frames = input[0].length;
    for (let i = 0; i < frames; i++) {
      let v = 0;
      for (let c = 0; c < channels; c++) v += input[c][i];
      this.block[this.fill++] = v / channels;
      if (this.fill === this.block.length) this.flush();
    }
    return true;
  }
}

registerProcessor('recorder', RecorderProcessor);
