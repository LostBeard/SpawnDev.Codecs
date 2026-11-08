// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Kernels whose double arithmetic is a port of reference C code (FFmpeg's FLAC LPC analysis) must round as IEEE-754
// binary64 to make the reference's decisions. WebGPU and WebGL have no f64: SpawnDev.ILGPU emulates it, by default with
// Dekker double-float (two f32, ~48 bits: close, not IEEE), or with Ozaki (strict IEEE-754). This loads such kernels
// with the accelerator in Ozaki mode and puts the mode back: ILGPU fixes a kernel's f64 mode when it compiles the
// kernel (the shader cache keys on the mode), so only the kernels loaded here pay for strict emulation.

using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGL;
using SpawnDev.ILGPU.WebGPU;

namespace SpawnDev.Codecs;

/// <summary>Loads kernels that need IEEE-754 double rounding on every backend (see the file header).</summary>
internal static class StrictFloat64
{
    /// <summary>Run <paramref name="load"/> (kernel loads) with strict f64 emulation on WebGPU / WebGL; other backends
    /// have native IEEE doubles and load as they are.</summary>
    public static T Load<T>(Accelerator accelerator, Func<T> load)
    {
        switch (accelerator)
        {
            case WebGPUAccelerator g when g.F64Mode != F64EmulationMode.Ozaki:
            {
                var mode = g.F64Mode;
                g.F64Mode = F64EmulationMode.Ozaki;
                try { return load(); }
                finally { g.F64Mode = mode; }
            }
            case WebGLAccelerator gl when gl.F64Mode != F64EmulationMode.Ozaki:
            {
                var mode = gl.F64Mode;
                gl.F64Mode = F64EmulationMode.Ozaki;
                try { return load(); }
                finally { gl.F64Mode = mode; }
            }
            default:
                return load();
        }
    }
}
