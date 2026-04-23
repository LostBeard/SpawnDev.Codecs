using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for the extended <see cref="SilkChannelDecoderState"/>: Configure sets
/// frame geometry correctly, Reset clears all scalar + buffer state, and the
/// buffers are sized for the worst-case decode configuration.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void StateConfigure_NbMb20Ms_SetsExpectedDimensions()
    {
        var state = new SilkChannelDecoderState();
        state.Configure(fsKHz: 8, nbSubfr: 4, lpcOrder: 10);

        Equal(8, state.FsKHz);
        Equal(4, state.NbSubfr);
        Equal(10, state.LpcOrder);
        Equal(40, state.SubfrLength);     // 5 * 8
        Equal(160, state.FrameLength);    // 4 * 40
        Equal(160, state.LtpMemLength);   // 20 * 8
    }

    [TestMethod]
    public void StateConfigure_Wb20Ms_SetsExpectedDimensions()
    {
        var state = new SilkChannelDecoderState();
        state.Configure(fsKHz: 16, nbSubfr: 4, lpcOrder: 16);

        Equal(16, state.FsKHz);
        Equal(4, state.NbSubfr);
        Equal(16, state.LpcOrder);
        Equal(80, state.SubfrLength);     // 5 * 16
        Equal(320, state.FrameLength);    // 4 * 80
        Equal(320, state.LtpMemLength);   // 20 * 16
    }

    [TestMethod]
    public void StateConfigure_Wb10Ms_HalfFrame()
    {
        var state = new SilkChannelDecoderState();
        state.Configure(fsKHz: 16, nbSubfr: 2, lpcOrder: 16);

        Equal(80, state.SubfrLength);     // 5 * 16
        Equal(160, state.FrameLength);    // 2 * 80
        Equal(320, state.LtpMemLength);   // LTP mem is independent of frame size
    }

    [TestMethod]
    public void StateConfigure_Mb20Ms_SetsExpectedDimensions()
    {
        var state = new SilkChannelDecoderState();
        state.Configure(fsKHz: 12, nbSubfr: 4, lpcOrder: 10);

        Equal(60, state.SubfrLength);     // 5 * 12
        Equal(240, state.FrameLength);    // 4 * 60
        Equal(240, state.LtpMemLength);   // 20 * 12
    }

    [TestMethod]
    public void StateConfigure_InvalidFsKHz_Throws()
    {
        var state = new SilkChannelDecoderState();
        Throws<ArgumentException>(() => state.Configure(fsKHz: 24, nbSubfr: 4, lpcOrder: 16));
    }

    [TestMethod]
    public void StateConfigure_InvalidNbSubfr_Throws()
    {
        var state = new SilkChannelDecoderState();
        Throws<ArgumentException>(() => state.Configure(fsKHz: 16, nbSubfr: 3, lpcOrder: 16));
    }

    [TestMethod]
    public void StateConfigure_InvalidLpcOrder_Throws()
    {
        var state = new SilkChannelDecoderState();
        Throws<ArgumentException>(() => state.Configure(fsKHz: 16, nbSubfr: 4, lpcOrder: 12));
    }

    [TestMethod]
    public void StateReset_ClearsAllScalarsAndBuffers()
    {
        var state = new SilkChannelDecoderState();
        state.Configure(fsKHz: 16, nbSubfr: 4, lpcOrder: 16);

        // Dirty everything.
        state.LastGainIndex = 42;
        state.PrevLagIndex = 100;
        state.PrevSignalTypeWasVoiced = true;
        state.PrevNlsfQ15[0] = 1234;
        state.PrevNlsfQ15[15] = 9999;
        state.OutBuf[0] = 7;
        state.OutBuf[500] = -7;
        state.SLtpQ15[50] = 12345;
        state.SLpcQ14Buf[3] = 54321;
        state.ExcQ14[100] = -55555;
        state.PrevGainQ16 = 555555;
        state.LossCnt = 3;
        state.PrevSignalType = SilkConstants.TYPE_VOICED;
        state.FirstFrameAfterReset = false;
        state.LagPrev = 75;

        state.Reset();

        Equal((sbyte)0, state.LastGainIndex);
        Equal((short)0, state.PrevLagIndex);
        True(!state.PrevSignalTypeWasVoiced, "prevVoiced should be false after Reset");
        for (int i = 0; i < state.PrevNlsfQ15.Length; i++) Equal((short)0, state.PrevNlsfQ15[i], $"prevNlsf[{i}]");
        for (int i = 0; i < state.OutBuf.Length; i++) Equal((short)0, state.OutBuf[i], $"outBuf[{i}]");
        for (int i = 0; i < state.SLtpQ15.Length; i++) Equal(0, state.SLtpQ15[i], $"sLtp[{i}]");
        for (int i = 0; i < state.SLpcQ14Buf.Length; i++) Equal(0, state.SLpcQ14Buf[i], $"sLpc[{i}]");
        for (int i = 0; i < state.ExcQ14.Length; i++) Equal(0, state.ExcQ14[i], $"exc[{i}]");
        Equal(65536, state.PrevGainQ16);
        Equal(0, state.LossCnt);
        Equal(SilkConstants.TYPE_NO_VOICE_ACTIVITY, state.PrevSignalType);
        True(state.FirstFrameAfterReset, "FirstFrameAfterReset should be true after Reset");
        Equal(100, state.LagPrev);
    }

    [TestMethod]
    public void StateBuffers_SizedForWorstCaseWb20Ms()
    {
        var state = new SilkChannelDecoderState();
        // OutBuf and SLtpQ15 need to hold LTP_MEM_LENGTH + FRAME_LENGTH at worst case WB 20ms.
        int worstCase = SilkConstants.MAX_LTP_MEM_LENGTH + SilkConstants.MAX_FRAME_LENGTH;
        Equal(worstCase, state.OutBuf.Length);
        Equal(worstCase, state.SLtpQ15.Length);
        Equal(SilkConstants.MAX_LPC_ORDER, state.SLpcQ14Buf.Length);
        Equal(SilkConstants.MAX_FRAME_LENGTH, state.ExcQ14.Length);
    }
}
