// Cross-backend tests for SilkDecodeCoreGpu - the SILK synthesis chain
// orchestrator. Runs CPU SilkDecodeCore.Decode as oracle, runs GPU
// SilkDecodeCoreGpu.Decode via SilkDecodeCoreGpuTestKernel, compares
// PCM output + LPC history + PrevGainQ16 bit-exactly.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>Build an NB (8 kHz, 4 subframes, lpcOrder=10) parameter set with simple
    /// known values. Voiced + non-voiced variants supplied via signalType arg.</summary>
    private static (SilkChannelDecoderState state, SilkDecodedParameters parameters)
        SilkDecodeCoreTest_BuildNbState(int signalType)
    {
        var state = new SilkChannelDecoderState();
        state.Configure(fsKHz: 8, nbSubfr: 4, lpcOrder: 10);
        state.Reset();

        var parameters = new SilkDecodedParameters();
        // Gains: simple ramp 0x10000, 0x12000, 0x14000, 0x16000.
        parameters.GainsQ16[0] = 0x10000;
        parameters.GainsQ16[1] = 0x12000;
        parameters.GainsQ16[2] = 0x14000;
        parameters.GainsQ16[3] = 0x16000;

        // PredCoefQ12: tiny LP filter (lo-pass at 8 kHz). Two halves identical.
        // a[0]=3500, a[1]=-1200, a[2..9]=0. Stable on a 1024Q12-norm domain.
        for (int half = 0; half < 2; half++)
        {
            parameters.PredCoefQ12[half * 16 + 0] = 3500;
            parameters.PredCoefQ12[half * 16 + 1] = -1200;
            for (int i = 2; i < 10; i++) parameters.PredCoefQ12[half * 16 + i] = 0;
            for (int i = 10; i < 16; i++) parameters.PredCoefQ12[half * 16 + i] = 0;
        }

        // Pitch lags: spread but well within max (PE_MAX_LAG_MS * fs_kHz = 18*8 = 144
        // for NB).
        parameters.PitchL[0] = 60;
        parameters.PitchL[1] = 65;
        parameters.PitchL[2] = 70;
        parameters.PitchL[3] = 75;

        // LTP coefs: small voiced tap pattern.
        for (int k = 0; k < 4; k++)
        {
            parameters.LtpCoefQ14[k * 5 + 0] = 1000;
            parameters.LtpCoefQ14[k * 5 + 1] = 2000;
            parameters.LtpCoefQ14[k * 5 + 2] = 4000;
            parameters.LtpCoefQ14[k * 5 + 3] = 2000;
            parameters.LtpCoefQ14[k * 5 + 4] = 1000;
        }
        parameters.LtpScaleQ14 = 15565; // index 0

        return (state, parameters);
    }

    private static short[] SilkDecodeCoreTest_BuildSparsePulses(int frameLength, int seed)
    {
        int alignedLen = (frameLength + 15) & ~15;
        var pulses = new short[alignedLen];
        // Simple deterministic sparse pulse pattern, ±1/±2 amplitudes.
        var rng = new Random(seed);
        for (int i = 0; i < frameLength; i++)
        {
            int r = rng.Next(0, 32);
            if (r == 0) pulses[i] = 2;
            else if (r == 1) pulses[i] = -2;
            else if (r < 6) pulses[i] = 1;
            else if (r < 11) pulses[i] = -1;
            // else 0
        }
        return pulses;
    }

    /// <summary>Drive SilkDecodeCoreGpu on the given accelerator + return the resulting
    /// xqOut PCM, the updated SLpcQ14Buf, and the updated PrevGainQ16.</summary>
    private static async Task<(short[] xq, int[] sLpcQ14Buf, int prevGainQ16)>
        SilkDecodeCoreTest_RunGpuAsync(
            Accelerator acc,
            SilkDecodedParameters parameters,
            SilkChannelDecoderState state,
            short[] pulses,
            int signalType, int quantOffsetType, int seed,
            bool nlsfInterpEnabled)
    {
        int frameLength = state.FrameLength;
        int subfrLength = state.SubfrLength;
        int ltpMemLength = state.LtpMemLength;
        int nbSubfr = state.NbSubfr;
        int lpcOrder = state.LpcOrder;

        const int MaxLpcOrder = 16;
        const int MaxFrameLength = 320;
        const int MaxSubfrLength = 80;
        const int MaxLtpMemLength = 320;

        using var dPredCoefQ12 = acc.Allocate1D<short>(2 * MaxLpcOrder);
        using var dGainsQ16 = acc.Allocate1D<int>(4);
        using var dPitchL = acc.Allocate1D<int>(4);
        using var dLtpCoefQ14 = acc.Allocate1D<short>(20);
        using var dPulses = acc.Allocate1D<short>(pulses.Length);

        using var dOutBuf = acc.Allocate1D<short>(MaxLtpMemLength + MaxFrameLength);
        using var dSLpcQ14Buf = acc.Allocate1D<int>(MaxLpcOrder);
        using var dExcQ14 = acc.Allocate1D<int>(MaxFrameLength);
        using var dPrevGain = acc.Allocate1D<int>(1);

        using var dSLpcScratch = acc.Allocate1D<int>(MaxLpcOrder + MaxSubfrLength);
        using var dSLtpQ15Scratch = acc.Allocate1D<int>(MaxLtpMemLength + MaxFrameLength);
        using var dSLtpScratch = acc.Allocate1D<short>(MaxLtpMemLength);
        using var dPresQ14Scratch = acc.Allocate1D<int>(MaxSubfrLength);
        using var dGainAdjScratch = acc.Allocate1D<int>(1);

        using var dXqOut = acc.Allocate1D<short>(MaxFrameLength);

        dPredCoefQ12.View.CopyFromCPU(parameters.PredCoefQ12);
        dGainsQ16.View.CopyFromCPU(parameters.GainsQ16);
        dPitchL.View.CopyFromCPU(parameters.PitchL);
        dLtpCoefQ14.View.CopyFromCPU(parameters.LtpCoefQ14);
        dPulses.View.CopyFromCPU(pulses);

        dOutBuf.View.CopyFromCPU(state.OutBuf);
        dSLpcQ14Buf.View.CopyFromCPU(state.SLpcQ14Buf);
        dPrevGain.View.CopyFromCPU(new int[] { state.PrevGainQ16 });

        var inputs = new SilkDecodeCoreInputs
        {
            PredCoefQ12 = dPredCoefQ12.View,
            GainsQ16 = dGainsQ16.View,
            PitchL = dPitchL.View,
            LtpCoefQ14 = dLtpCoefQ14.View,
            Pulses = dPulses.View,
            OutBufInOut = dOutBuf.View,
            SLpcQ14BufInOut = dSLpcQ14Buf.View,
            ExcQ14Out = dExcQ14.View,
            PrevGainQ16InOut = dPrevGain.View,
            SLpcScratch = dSLpcScratch.View,
            SLtpQ15Scratch = dSLtpQ15Scratch.View,
            SLtpScratch = dSLtpScratch.View,
            PresQ14Scratch = dPresQ14Scratch.View,
            GainAdjScratch = dGainAdjScratch.View,
            XqOut = dXqOut.View,
        };

        var scalars = new SilkDecodeCoreScalars
        {
            SignalType = signalType,
            QuantOffsetType = quantOffsetType,
            Seed = seed,
            LpcOrder = lpcOrder,
            NbSubfr = nbSubfr,
            SubfrLength = subfrLength,
            FrameLength = frameLength,
            LtpMemLength = ltpMemLength,
            LtpScaleQ14 = parameters.LtpScaleQ14,
            NlsfInterpEnabled = nlsfInterpEnabled ? 1 : 0,
        };

        using var kernel = new SilkDecodeCoreGpuTestKernel(acc);
        kernel.Run(inputs, scalars);
        await acc.SynchronizeAsync();

        var xq = await dXqOut.CopyToHostAsync();
        var sLpcOut = await dSLpcQ14Buf.CopyToHostAsync();
        var prevGainOut = await dPrevGain.CopyToHostAsync();

        var xqSlice = new short[frameLength];
        Array.Copy(xq, xqSlice, frameLength);
        return (xqSlice, sLpcOut, prevGainOut[0]);
    }

    /// <summary>
    /// 2026-05-04 SilkDecodeCoreGpu hits two ILGPU backend bugs documented at
    /// _DevComms/SpawnDev.ILGPU/tuvok-to-geordi-silkdecodecoregpu-cuda-webgpu-2026-05-04.md.
    /// CUDA: "too many resources requested for launch" (auto-grouped picks block
    /// size whose register × thread footprint exceeds SM register file).
    /// WebGPU: "Invalid BindGroupLayout" (15-view body struct exceeds default
    /// device storage-buffer binding count). Both are ILGPU codegen / launch
    /// config issues, not codec logic - bit-exact passes on CPU + OpenCL + Wasm.
    /// </summary>
    private static void SilkDecodeCoreTest_GateBackend(Accelerator acc)
    {
        if (acc.AcceleratorType == AcceleratorType.Cuda)
            throw new UnsupportedTestException(
                "SilkDecodeCoreGpu auto-grouped launch hits 'too many resources requested for launch' on CUDA. "
                + "ILGPU auto-grouper picks a block size whose per-thread register footprint exceeds SM limits "
                + "for this kernel's body-struct + LPC-synthesis-unroll combination. Verified 2026-05-04: "
                + "removing b0..b4 local caching in the LTP loop did NOT shrink register count enough; "
                + "the body struct's 15 ArrayView fields dominate. Tracked at "
                + "_DevComms/SpawnDev.ILGPU/tuvok-to-geordi-silkdecodecoregpu-cuda-webgpu-2026-05-04.md.");
        if (acc.AcceleratorType == AcceleratorType.WebGPU)
            throw new UnsupportedTestException(
                "SilkDecodeCoreGpu hits WebGPU 'Invalid BindGroupLayout' - 15 ArrayView fields in the body struct "
                + "exceed the default storage-buffer binding count after rc.5 same-element-type coalesce. "
                + "Tracked at _DevComms/SpawnDev.ILGPU/tuvok-to-geordi-silkdecodecoregpu-cuda-webgpu-2026-05-04.md.");
    }

    [TestMethod]
    public async Task SilkDecodeCoreGpu_Unvoiced_NB_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            SilkDecodeCoreTest_GateBackend(acc);
            // NB unvoiced: 8 kHz, 4 subframes, lpcOrder=10, frameLength=160.
            var (state, parameters) = SilkDecodeCoreTest_BuildNbState(signalType: 1);
            int frameLength = state.FrameLength;
            var pulses = SilkDecodeCoreTest_BuildSparsePulses(frameLength, seed: 1234);

            // CPU oracle.
            var cpuState = SilkDecodeCoreTest_CloneState(state);
            var cpuXq = new short[frameLength];
            SilkDecodeCore.Decode(
                cpuState, parameters, pulses.AsSpan(0, frameLength),
                signalType: 1, quantOffsetType: 0, seed: 5,
                nlsfInterpolationEnabled: false,
                cpuXq.AsSpan(0, frameLength));

            // GPU.
            var (gpuXq, gpuSLpc, gpuPrevGain) = await SilkDecodeCoreTest_RunGpuAsync(
                acc, parameters, state, pulses,
                signalType: 1, quantOffsetType: 0, seed: 5,
                nlsfInterpEnabled: false);

            for (int i = 0; i < frameLength; i++)
                if (cpuXq[i] != gpuXq[i])
                    throw new Exception($"PCM mismatch at sample {i}: cpu={cpuXq[i]} gpu={gpuXq[i]}");
            for (int i = 0; i < 16; i++)
                if (cpuState.SLpcQ14Buf[i] != gpuSLpc[i])
                    throw new Exception($"SLpcQ14Buf mismatch at {i}: cpu={cpuState.SLpcQ14Buf[i]} gpu={gpuSLpc[i]}");
            if (cpuState.PrevGainQ16 != gpuPrevGain)
                throw new Exception($"PrevGainQ16 mismatch: cpu={cpuState.PrevGainQ16} gpu={gpuPrevGain}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkDecodeCoreGpu_Voiced_NB_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            SilkDecodeCoreTest_GateBackend(acc);
            // NB voiced: same 8 kHz / 4 sub, but signalType=2 - exercises the
            // LTP rewhitening path (k==0) and 5-tap LTP loop.
            var (state, parameters) = SilkDecodeCoreTest_BuildNbState(signalType: 2);
            int frameLength = state.FrameLength;
            var pulses = SilkDecodeCoreTest_BuildSparsePulses(frameLength, seed: 5678);

            var cpuState = SilkDecodeCoreTest_CloneState(state);
            var cpuXq = new short[frameLength];
            SilkDecodeCore.Decode(
                cpuState, parameters, pulses.AsSpan(0, frameLength),
                signalType: 2, quantOffsetType: 1, seed: 7,
                nlsfInterpolationEnabled: false,
                cpuXq.AsSpan(0, frameLength));

            var (gpuXq, gpuSLpc, gpuPrevGain) = await SilkDecodeCoreTest_RunGpuAsync(
                acc, parameters, state, pulses,
                signalType: 2, quantOffsetType: 1, seed: 7,
                nlsfInterpEnabled: false);

            for (int i = 0; i < frameLength; i++)
                if (cpuXq[i] != gpuXq[i])
                    throw new Exception($"PCM mismatch at sample {i}: cpu={cpuXq[i]} gpu={gpuXq[i]}");
            for (int i = 0; i < 16; i++)
                if (cpuState.SLpcQ14Buf[i] != gpuSLpc[i])
                    throw new Exception($"SLpcQ14Buf mismatch at {i}: cpu={cpuState.SLpcQ14Buf[i]} gpu={gpuSLpc[i]}");
            if (cpuState.PrevGainQ16 != gpuPrevGain)
                throw new Exception($"PrevGainQ16 mismatch: cpu={cpuState.PrevGainQ16} gpu={gpuPrevGain}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    /// <summary>Make a deep copy of state so CPU and GPU runs don't interfere.</summary>
    private static SilkChannelDecoderState SilkDecodeCoreTest_CloneState(SilkChannelDecoderState src)
    {
        var copy = new SilkChannelDecoderState();
        copy.Configure(src.FsKHz, src.NbSubfr, src.LpcOrder);
        copy.Reset();
        copy.PrevGainQ16 = src.PrevGainQ16;
        Array.Copy(src.OutBuf, copy.OutBuf, src.OutBuf.Length);
        Array.Copy(src.SLpcQ14Buf, copy.SLpcQ14Buf, src.SLpcQ14Buf.Length);
        Array.Copy(src.ExcQ14, copy.ExcQ14, src.ExcQ14.Length);
        return copy;
    }

    /// <summary>WB (16 kHz, 4 subframes, lpcOrder=16) parameter set. Exercises
    /// the FULL 16-tap LPC synthesis filter unroll (vs NB's 10-tap path).</summary>
    private static (SilkChannelDecoderState state, SilkDecodedParameters parameters)
        SilkDecodeCoreTest_BuildWbState(int signalType)
    {
        var state = new SilkChannelDecoderState();
        state.Configure(fsKHz: 16, nbSubfr: 4, lpcOrder: 16);
        state.Reset();

        var parameters = new SilkDecodedParameters();
        parameters.GainsQ16[0] = 0x18000;
        parameters.GainsQ16[1] = 0x1A000;
        parameters.GainsQ16[2] = 0x1C000;
        parameters.GainsQ16[3] = 0x1E000;

        // 16-tap LP filter for both halves: low-pass, all 16 taps populated.
        for (int half = 0; half < 2; half++)
        {
            parameters.PredCoefQ12[half * 16 + 0] = 2800;
            parameters.PredCoefQ12[half * 16 + 1] = -900;
            parameters.PredCoefQ12[half * 16 + 2] = 200;
            parameters.PredCoefQ12[half * 16 + 3] = -50;
            parameters.PredCoefQ12[half * 16 + 4] = 30;
            for (int i = 5; i < 16; i++) parameters.PredCoefQ12[half * 16 + i] = (short)((i % 2 == 0) ? 10 : -10);
        }

        // Pitch lags within WB max (PE_MAX_LAG_MS * fs_kHz = 18 * 16 = 288).
        parameters.PitchL[0] = 100;
        parameters.PitchL[1] = 110;
        parameters.PitchL[2] = 120;
        parameters.PitchL[3] = 130;

        for (int k = 0; k < 4; k++)
        {
            parameters.LtpCoefQ14[k * 5 + 0] = 800;
            parameters.LtpCoefQ14[k * 5 + 1] = 1500;
            parameters.LtpCoefQ14[k * 5 + 2] = 3000;
            parameters.LtpCoefQ14[k * 5 + 3] = 1500;
            parameters.LtpCoefQ14[k * 5 + 4] = 800;
        }
        parameters.LtpScaleQ14 = 12288; // index 1

        return (state, parameters);
    }

    /// <summary>MB (12 kHz, 4 subframes, lpcOrder=10) parameter set. Exercises
    /// the 12 kHz sample rate path with subfrLength=60 (vs NB's 40, WB's 80).</summary>
    private static (SilkChannelDecoderState state, SilkDecodedParameters parameters)
        SilkDecodeCoreTest_BuildMbState(int signalType)
    {
        var state = new SilkChannelDecoderState();
        state.Configure(fsKHz: 12, nbSubfr: 4, lpcOrder: 10);
        state.Reset();

        var parameters = new SilkDecodedParameters();
        parameters.GainsQ16[0] = 0x14000;
        parameters.GainsQ16[1] = 0x16000;
        parameters.GainsQ16[2] = 0x18000;
        parameters.GainsQ16[3] = 0x1A000;

        for (int half = 0; half < 2; half++)
        {
            parameters.PredCoefQ12[half * 16 + 0] = 3200;
            parameters.PredCoefQ12[half * 16 + 1] = -1100;
            parameters.PredCoefQ12[half * 16 + 2] = 100;
            parameters.PredCoefQ12[half * 16 + 3] = -50;
            for (int i = 4; i < 10; i++) parameters.PredCoefQ12[half * 16 + i] = 0;
            for (int i = 10; i < 16; i++) parameters.PredCoefQ12[half * 16 + i] = 0;
        }

        // Pitch lags within MB max (PE_MAX_LAG_MS * fs_kHz = 18 * 12 = 216).
        parameters.PitchL[0] = 80;
        parameters.PitchL[1] = 90;
        parameters.PitchL[2] = 100;
        parameters.PitchL[3] = 110;

        for (int k = 0; k < 4; k++)
        {
            parameters.LtpCoefQ14[k * 5 + 0] = 900;
            parameters.LtpCoefQ14[k * 5 + 1] = 1800;
            parameters.LtpCoefQ14[k * 5 + 2] = 3500;
            parameters.LtpCoefQ14[k * 5 + 3] = 1800;
            parameters.LtpCoefQ14[k * 5 + 4] = 900;
        }
        parameters.LtpScaleQ14 = 8192; // index 2

        return (state, parameters);
    }

    [TestMethod]
    public async Task SilkDecodeCoreGpu_Voiced_MB_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            SilkDecodeCoreTest_GateBackend(acc);
            // MB voiced: 12 kHz × 4 subframes × 60 = 240 samples/frame.
            // Different subfrLength than NB (40) or WB (80), exercising the
            // synthesis loop's per-sample stride math at a third length.
            var (state, parameters) = SilkDecodeCoreTest_BuildMbState(signalType: 2);
            int frameLength = state.FrameLength;
            var pulses = SilkDecodeCoreTest_BuildSparsePulses(frameLength, seed: 31337);

            var cpuState = SilkDecodeCoreTest_CloneState(state);
            var cpuXq = new short[frameLength];
            SilkDecodeCore.Decode(
                cpuState, parameters, pulses.AsSpan(0, frameLength),
                signalType: 2, quantOffsetType: 0, seed: 21,
                nlsfInterpolationEnabled: false,
                cpuXq.AsSpan(0, frameLength));

            var (gpuXq, gpuSLpc, gpuPrevGain) = await SilkDecodeCoreTest_RunGpuAsync(
                acc, parameters, state, pulses,
                signalType: 2, quantOffsetType: 0, seed: 21,
                nlsfInterpEnabled: false);

            for (int i = 0; i < frameLength; i++)
                if (cpuXq[i] != gpuXq[i])
                    throw new Exception($"MB PCM mismatch at sample {i}: cpu={cpuXq[i]} gpu={gpuXq[i]}");
            for (int i = 0; i < 16; i++)
                if (cpuState.SLpcQ14Buf[i] != gpuSLpc[i])
                    throw new Exception($"MB SLpcQ14Buf mismatch at {i}: cpu={cpuState.SLpcQ14Buf[i]} gpu={gpuSLpc[i]}");
            if (cpuState.PrevGainQ16 != gpuPrevGain)
                throw new Exception($"MB PrevGainQ16 mismatch: cpu={cpuState.PrevGainQ16} gpu={gpuPrevGain}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkDecodeCoreGpu_Voiced_WB_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            SilkDecodeCoreTest_GateBackend(acc);
            // WB voiced: 16 kHz × 4 subframes × 80 samples = 320 samples/frame.
            // Exercises the lpcOrder=16 full-unroll LPC synthesis path.
            var (state, parameters) = SilkDecodeCoreTest_BuildWbState(signalType: 2);
            int frameLength = state.FrameLength;
            var pulses = SilkDecodeCoreTest_BuildSparsePulses(frameLength, seed: 9999);

            var cpuState = SilkDecodeCoreTest_CloneState(state);
            var cpuXq = new short[frameLength];
            SilkDecodeCore.Decode(
                cpuState, parameters, pulses.AsSpan(0, frameLength),
                signalType: 2, quantOffsetType: 1, seed: 11,
                nlsfInterpolationEnabled: false,
                cpuXq.AsSpan(0, frameLength));

            var (gpuXq, gpuSLpc, gpuPrevGain) = await SilkDecodeCoreTest_RunGpuAsync(
                acc, parameters, state, pulses,
                signalType: 2, quantOffsetType: 1, seed: 11,
                nlsfInterpEnabled: false);

            for (int i = 0; i < frameLength; i++)
                if (cpuXq[i] != gpuXq[i])
                    throw new Exception($"WB PCM mismatch at sample {i}: cpu={cpuXq[i]} gpu={gpuXq[i]}");
            for (int i = 0; i < 16; i++)
                if (cpuState.SLpcQ14Buf[i] != gpuSLpc[i])
                    throw new Exception($"WB SLpcQ14Buf mismatch at {i}: cpu={cpuState.SLpcQ14Buf[i]} gpu={gpuSLpc[i]}");
            if (cpuState.PrevGainQ16 != gpuPrevGain)
                throw new Exception($"WB PrevGainQ16 mismatch: cpu={cpuState.PrevGainQ16} gpu={gpuPrevGain}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkDecodeCoreGpu_Voiced_NB_NlsfInterpEnabled_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            SilkDecodeCoreTest_GateBackend(acc);
            // Voiced NB with NLSF interpolation enabled. This is the ONLY config
            // that takes the second-half rewhitening path inside the per-subframe
            // loop (k==2 rewhitens because nlsfInterpolationEnabled is true and
            // the k==0 || k==2&&interp predicate fires at k==2). Exercises the
            // OutBuf staging step (xqOut[0..2*subfrLength) -> OutBuf[ltpMemLength..])
            // which doesn't fire on any of the other configurations.
            var (state, parameters) = SilkDecodeCoreTest_BuildNbState(signalType: 2);
            int frameLength = state.FrameLength;
            var pulses = SilkDecodeCoreTest_BuildSparsePulses(frameLength, seed: 4242);

            var cpuState = SilkDecodeCoreTest_CloneState(state);
            var cpuXq = new short[frameLength];
            SilkDecodeCore.Decode(
                cpuState, parameters, pulses.AsSpan(0, frameLength),
                signalType: 2, quantOffsetType: 0, seed: 17,
                nlsfInterpolationEnabled: true,
                cpuXq.AsSpan(0, frameLength));

            var (gpuXq, gpuSLpc, gpuPrevGain) = await SilkDecodeCoreTest_RunGpuAsync(
                acc, parameters, state, pulses,
                signalType: 2, quantOffsetType: 0, seed: 17,
                nlsfInterpEnabled: true);

            for (int i = 0; i < frameLength; i++)
                if (cpuXq[i] != gpuXq[i])
                    throw new Exception($"NLSF-interp PCM mismatch at sample {i}: cpu={cpuXq[i]} gpu={gpuXq[i]}");
            for (int i = 0; i < 16; i++)
                if (cpuState.SLpcQ14Buf[i] != gpuSLpc[i])
                    throw new Exception($"NLSF-interp SLpcQ14Buf mismatch at {i}: cpu={cpuState.SLpcQ14Buf[i]} gpu={gpuSLpc[i]}");
            if (cpuState.PrevGainQ16 != gpuPrevGain)
                throw new Exception($"NLSF-interp PrevGainQ16 mismatch: cpu={cpuState.PrevGainQ16} gpu={gpuPrevGain}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkDecodeCoreGpu_VoicedNB_3FramesStateRolling_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            SilkDecodeCoreTest_GateBackend(acc);
            // Decode 3 consecutive voiced NB frames, threading the persistent
            // state buffers (SLpcQ14Buf, OutBuf, PrevGainQ16) across calls.
            // Catches state-rolling bugs that single-frame tests miss: e.g.
            // outBuf-shift bugs visible only on frame 2+, or SLpcQ14Buf
            // contamination between frames.
            var (state, parameters) = SilkDecodeCoreTest_BuildNbState(signalType: 2);
            int frameLength = state.FrameLength;
            var cpuState = SilkDecodeCoreTest_CloneState(state);

            for (int frame = 0; frame < 3; frame++)
            {
                var pulses = SilkDecodeCoreTest_BuildSparsePulses(frameLength, seed: 1000 + frame);

                // CPU oracle.
                var cpuXq = new short[frameLength];
                SilkDecodeCore.Decode(
                    cpuState, parameters, pulses.AsSpan(0, frameLength),
                    signalType: 2, quantOffsetType: 0, seed: (sbyte)(13 + frame),
                    nlsfInterpolationEnabled: false,
                    cpuXq.AsSpan(0, frameLength));
                // CPU SilkDecodeFrame normally shifts outBuf[frameLength..] -> outBuf[0..]
                // and writes xq into outBuf[mvLen..]. Mirror that here so the LTP buffer
                // is realistic on the next iteration.
                int mvLen = cpuState.LtpMemLength - frameLength;
                Array.Copy(cpuState.OutBuf, frameLength, cpuState.OutBuf, 0, mvLen);
                Array.Copy(cpuXq, 0, cpuState.OutBuf, mvLen, frameLength);

                // GPU: clone the GPU-side state to the SAME starting state CPU just used,
                // run, then mirror the same shift on the GPU clone for next iter.
                var (gpuXq, gpuSLpc, gpuPrevGain) = await SilkDecodeCoreTest_RunGpuAsync(
                    acc, parameters, state, pulses,
                    signalType: 2, quantOffsetType: 0, seed: 13 + frame,
                    nlsfInterpEnabled: false);

                for (int i = 0; i < frameLength; i++)
                    if (cpuXq[i] != gpuXq[i])
                        throw new Exception(
                            $"frame {frame} sample {i} mismatch: cpu={cpuXq[i]} gpu={gpuXq[i]}");
                for (int i = 0; i < 16; i++)
                    if (cpuState.SLpcQ14Buf[i] != gpuSLpc[i])
                        throw new Exception(
                            $"frame {frame} SLpcQ14Buf mismatch at {i}: cpu={cpuState.SLpcQ14Buf[i]} gpu={gpuSLpc[i]}");
                if (cpuState.PrevGainQ16 != gpuPrevGain)
                    throw new Exception(
                        $"frame {frame} PrevGainQ16 mismatch: cpu={cpuState.PrevGainQ16} gpu={gpuPrevGain}");

                // Update GPU-side state for next iteration: copy persistent state from
                // cpuState into the GPU's "state" object (since cpuState was shifted).
                Array.Copy(cpuState.OutBuf, state.OutBuf, cpuState.OutBuf.Length);
                Array.Copy(cpuState.SLpcQ14Buf, state.SLpcQ14Buf, cpuState.SLpcQ14Buf.Length);
                state.PrevGainQ16 = cpuState.PrevGainQ16;
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
