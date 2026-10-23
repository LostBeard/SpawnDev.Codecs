// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VorbisPacketDecodeKernel - GPU kernel for Vorbis v2 audio packet bit-stream decode (Vorbis I sec 4.3):
//
//   1. VorbisAudioPacketHeaderGpu.Parse    (header bits)
//   2. VorbisFloor1DecoderGpu.Decode       (per channel)
//   3. residue decode                      (per submap; types 0, 1 and 2, sec 8.6)
//
// Replaces the CPU `DecodeSpectrumOnCpu` path inside VorbisAudioDecoderGpu - the v2 step in
// Plans/PLAN-Vorbis-Decoder-V2-GPU-BitStream-Decode.md.
//
// Bit-stream decoding is sequential (one GPU thread per packet), so the decode is RESUMABLE: every loop
// position (phase, channel, submap, pass, partition, member, codeword index inside the partition, the
// type-2 de-interleave cursor) and the bit reader live in a small device int region (StateInts), and each
// launch of Step does at most StepBudget work units before it stores them and returns. A unit is one loop
// iteration plus one per bit read plus one per decoded vector element, so no launch grows with the packet
// (a packet's work grows with its bits, channels and block size: up to 255 channels x 4096 coefficients x
// 8 passes). A launch after the end does nothing. The host launches the steps in batches and reads back
// one done flag per batch (the work is content-dependent).
//
// The setup and codebook tables of the stream live in ONE int pool and one double pool (VorbisPacketDecodeSetup:
// the pool starts with the offset of every table), so the kernel binds 8 buffers whatever the setup (WebGPU
// allows at most 16 storage buffers per stage).
//
// Caller responsibility (host = pure coordinator):
//   - Build + upload the pools ONCE per stream (metadata struct setup carve-out per CARDINAL rule).
//   - Per packet: zero the outputs and the state, launch Step until the state says done.
//   - Read back the small header + floorOk flags (the decision bits the post-spectrum chain needs).

using ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// The setup header + codebook set of a stream flattened into one int pool and one double pool for
/// <see cref="VorbisPacketDecodeKernel"/>. Ints [0, <see cref="TableCount"/>) hold the start of every table
/// (T* ids); the tables follow. Doubles: [0, codebooks) MinValues, [codebooks, 2 * codebooks) DeltaValues.
/// </summary>
public sealed class VorbisPacketDecodeSetup
{
    /// <summary>Table ids (index of the table's start offset in the int pool).</summary>
    public const int
        TFloorScalars = 0, TFloorPartitionClassList = 1, TFloorPartitionClassListOffsets = 2,
        TFloorClassDimensions = 3, TFloorClassDimensionsOffsets = 4, TFloorClassSubclasses = 5,
        TFloorClassSubclassesOffsets = 6, TFloorClassMasterbooks = 7, TFloorClassMasterbooksOffsets = 8,
        TFloorClassSubclassBooks = 9, TFloorClassSubclassBooksOffsets = 10,
        TResidueScalars = 11, TResidueBooks = 12, TResidueBooksOffsets = 13,
        TMappingScalars = 14, TMappingMux = 15, TMappingMuxOffsets = 16, TMappingFloors = 17,
        TMappingResidues = 18, TMappingSubmapOffsets = 19, TModeBlockFlags = 20, TModeMappings = 21,
        TAllChildren = 22, TAllLeafToEntry = 23, TChildrenOffsets = 24, TLeafOffsets = 25, TMaxDepths = 26,
        TCodebookParams = 27, TAllMultiplicands = 28, TMultOffsets = 29, TMultLengths = 30,
        TCodebookDimensions = 31, TCodebookEntries = 32, TCodebookLookupTypes = 33, TCodebookQuantvals = 34,
        TCodebookSequenceP = 35, TableCount = 36;

    /// <summary>The int pool.</summary>
    public required int[] Ints { get; init; }
    /// <summary>The double pool (MinValues then DeltaValues).</summary>
    public required double[] Doubles { get; init; }
    /// <summary>Codebook count (offset of DeltaValues in <see cref="Doubles"/>).</summary>
    public required int CodebookCount { get; init; }

    /// <summary>
    /// Flattens <paramref name="setup"/>. CodebookParams holds, per codebook, [children start, leaf start, max
    /// depth] as ABSOLUTE int-pool offsets, so VorbisFloor1DecoderGpu walks the pool directly.
    /// </summary>
    public static VorbisPacketDecodeSetup Build(VorbisSetupHeader setup)
    {
        ArgumentNullException.ThrowIfNull(setup);
        var s = VorbisSetupHeaderGpu.Build(setup);
        var cb = VorbisHuffmanCodebookSetGpu.Build(setup.Codebooks);
        var modeFlags = new int[s.ModeBlockFlags.Length];
        for (int i = 0; i < modeFlags.Length; i++) modeFlags[i] = s.ModeBlockFlags[i];

        var tables = new int[TableCount][];
        tables[TFloorScalars] = s.FloorScalars;
        tables[TFloorPartitionClassList] = s.FloorPartitionClassList;
        tables[TFloorPartitionClassListOffsets] = s.FloorPartitionClassListOffsets;
        tables[TFloorClassDimensions] = s.FloorClassDimensions;
        tables[TFloorClassDimensionsOffsets] = s.FloorClassDimensionsOffsets;
        tables[TFloorClassSubclasses] = s.FloorClassSubclasses;
        tables[TFloorClassSubclassesOffsets] = s.FloorClassSubclassesOffsets;
        tables[TFloorClassMasterbooks] = s.FloorClassMasterbooks;
        tables[TFloorClassMasterbooksOffsets] = s.FloorClassMasterbooksOffsets;
        tables[TFloorClassSubclassBooks] = s.FloorClassSubclassBooks;
        tables[TFloorClassSubclassBooksOffsets] = s.FloorClassSubclassBooksOffsets;
        tables[TResidueScalars] = s.ResidueScalars;
        tables[TResidueBooks] = s.ResidueBooks;
        tables[TResidueBooksOffsets] = s.ResidueBooksOffsets;
        tables[TMappingScalars] = s.MappingScalars;
        tables[TMappingMux] = s.MappingMux;
        tables[TMappingMuxOffsets] = s.MappingMuxOffsets;
        tables[TMappingFloors] = s.MappingFloors;
        tables[TMappingResidues] = s.MappingResidues;
        tables[TMappingSubmapOffsets] = s.MappingSubmapOffsets;
        tables[TModeBlockFlags] = modeFlags;
        tables[TModeMappings] = s.ModeMappings;
        tables[TAllChildren] = cb.AllChildren;
        tables[TAllLeafToEntry] = cb.AllLeafToEntry;
        tables[TChildrenOffsets] = cb.ChildrenOffsets;
        tables[TLeafOffsets] = cb.LeafOffsets;
        tables[TMaxDepths] = cb.MaxDepths;
        tables[TCodebookParams] = new int[cb.MaxDepths.Length * 3]; // filled below (needs the pool offsets)
        tables[TAllMultiplicands] = cb.AllMultiplicands;
        tables[TMultOffsets] = cb.MultOffsets;
        tables[TMultLengths] = cb.MultLengths;
        tables[TCodebookDimensions] = cb.CodebookDimensions;
        tables[TCodebookEntries] = cb.CodebookEntries;
        tables[TCodebookLookupTypes] = cb.CodebookLookupTypes;
        tables[TCodebookQuantvals] = cb.CodebookQuantvals;
        tables[TCodebookSequenceP] = cb.CodebookSequenceP;

        var starts = new int[TableCount];
        int at = TableCount;
        for (int t = 0; t < TableCount; t++)
        {
            starts[t] = at;
            at += tables[t].Length;
        }
        var cp = tables[TCodebookParams];
        for (int i = 0; i < cb.MaxDepths.Length; i++)
        {
            cp[i * 3 + 0] = starts[TAllChildren] + cb.ChildrenOffsets[i];
            cp[i * 3 + 1] = starts[TAllLeafToEntry] + cb.LeafOffsets[i];
            cp[i * 3 + 2] = cb.MaxDepths[i];
        }
        var ints = new int[Math.Max(1, at)];
        for (int t = 0; t < TableCount; t++)
        {
            ints[t] = starts[t];
            tables[t].CopyTo(ints, starts[t]);
        }
        int books = cb.CodebookMinValues.Length;
        var doubles = new double[Math.Max(1, 2 * books)];
        cb.CodebookMinValues.CopyTo(doubles, 0);
        cb.CodebookDeltaValues.CopyTo(doubles, books);
        return new VorbisPacketDecodeSetup { Ints = ints, Doubles = doubles, CodebookCount = books };
    }
}

/// <summary>Scalars of a <see cref="VorbisPacketDecodeKernel.Step"/> launch.</summary>
public struct VorbisPacketDecodeScalars
{
    /// <summary>Precomputed VorbisMath.Ilog(modes - 1); 0 if modes == 1.</summary>
    public int ModeBits;
    /// <summary>Audio channel count (= ident.AudioChannels).</summary>
    public int Channels;
    /// <summary>Short-block size from ident.</summary>
    public int BlockSize0;
    /// <summary>Long-block size from ident.</summary>
    public int BlockSize1;
    /// <summary>Maximum XList length across all floors (sized for the posterior section).</summary>
    public int MaxXListLength;
    /// <summary>Codebook count (offset of DeltaValues in the double pool).</summary>
    public int CodebookCount;
}

/// <summary>
/// Resumable GPU decode of one Vorbis audio packet's bit-stream payload (header + per-channel floor +
/// per-submap residue). Single-thread launches; see the file header.
/// </summary>
public static class VorbisPacketDecodeKernel
{
    /// <summary>
    /// Layout offsets within the combined int-output buffer (allIntOut).
    /// Section 0 [0..5)         : packet header (ModeNumber, BlockSize, IsLongBlock, PreviousWindowLong, NextWindowLong)
    /// Section 1 [5..5+ch)      : floorOk per channel
    /// Section 2 [5+ch..5+2ch)  : floorIndex per channel
    /// Section 3 [5+2ch..5+2ch+ch*maxXList) : floor posteriors per channel (channel-major)
    /// Section 4 [last 1 int]   : err flag (last position = 5+2ch + ch*maxXList)
    /// </summary>
    public const int PacketHeaderOffset = 0;
    /// <summary>Length of the packet header section in ints.</summary>
    public const int PacketHeaderLength = 5;

    /// <summary>Work units of one <see cref="Step"/> launch (a unit is about one dependent load).</summary>
    public const int StepBudget = 8192;

    /// <summary>Ints of the resumable state region (zero it before the first Step of a packet).</summary>
    public const int StateInts = 16;

    /// <summary>State slot holding the phase; <see cref="PhaseDone"/> once the packet is decoded.</summary>
    public const int SPhase = 0;
    private const int SBytePos = 1, SBitPos = 2, SCh = 3, SSub = 4, SPass = 5, SPc = 6, SSubPhase = 7,
        SMember = 8, SIdx = 9, SDeintCh = 10, SDeintI = 11;

    /// <summary>Phases.</summary>
    public const int PhaseStart = 0, PhaseFloors = 1, PhaseSubmap = 2, PhaseResidue = 3, PhaseDeinterleave = 4,
        PhaseDone = 5;

    /// <summary>Compute the required <c>allIntOut</c> length for the given configuration.</summary>
    public static long ComputeAllIntOutLength(int channels, int maxXListLength)
        => PacketHeaderLength + (long)channels * 2 + (long)channels * maxXListLength + 1;

    /// <summary>Compute the offset of the err flag within <c>allIntOut</c>.</summary>
    public static long ErrOutOffset(int channels, int maxXListLength)
        => PacketHeaderLength + (long)channels * 2 + (long)channels * maxXListLength;

    /// <summary>
    /// Ints of the <c>intScratch</c> a stream needs: classifications (one per partition, at most halfBlock per
    /// channel for types 0/1 and channels * halfBlock for type 2) plus the doNotDecode flags (channels ints, at the end).
    /// </summary>
    public static long ComputeIntScratchLength(int channels, int blockSize1)
        => (long)channels * (blockSize1 / 2) + channels;

    /// <summary>
    /// One bounded launch of the packet decode (see the file header). Residues are written channel-major with a
    /// row stride of the packet's halfBlock; the caller pre-zeroes allIntOut, residuesFlatOut and state.
    /// </summary>
    /// <param name="_">Dispatch index (unused; single thread).</param>
    /// <param name="packet">Encoded audio packet bytes.</param>
    /// <param name="setupInts">The int pool of <see cref="VorbisPacketDecodeSetup"/>.</param>
    /// <param name="setupDoubles">The double pool of <see cref="VorbisPacketDecodeSetup"/>.</param>
    /// <param name="allIntOut">Combined int output buffer; layout per <see cref="PacketHeaderOffset"/> + sections.</param>
    /// <param name="residuesFlatOut">[channels * blockSize1 / 2] residue floats, channel-major.</param>
    /// <param name="intScratch">Classifications + doNotDecode flags (<see cref="ComputeIntScratchLength"/>).</param>
    /// <param name="entryVecScratch">[max codebook dimensions] residue decode entry vector scratch.</param>
    /// <param name="state">The resumable state (<see cref="StateInts"/> ints).</param>
    /// <param name="sc">Stream scalars.</param>
    public static void Step(
        Index1D _,
        ArrayView<byte> packet,
        ArrayView<int> setupInts,
        ArrayView<double> setupDoubles,
        ArrayView<int> allIntOut,
        ArrayView<float> residuesFlatOut,
        ArrayView<int> intScratch,
        ArrayView<float> entryVecScratch,
        ArrayView<int> state,
        VorbisPacketDecodeScalars sc)
    {
        int channels = sc.Channels;
        long floorOkBase = PacketHeaderLength;
        long floorIndexBase = floorOkBase + channels;
        long posteriorsBase = floorIndexBase + channels;
        long errBase = posteriorsBase + (long)channels * sc.MaxXListLength;
        long doNotDecodeBase = intScratch.Length - channels;

        int phase = state[SPhase];
        var r = VorbisBitReaderGpu.Init((int)packet.Length);
        r.BytePos = state[SBytePos];
        r.BitPos = state[SBitPos];
        int work = 0;

        // ---- Header (first launch only) ----
        if (phase == PhaseStart)
        {
            var header = VorbisAudioPacketHeaderGpu.Parse(
                ref r, packet, sc.ModeBits,
                setupInts, setupInts[VorbisPacketDecodeSetup.TModeBlockFlags],
                sc.BlockSize0, sc.BlockSize1);
            allIntOut[PacketHeaderOffset + 0] = header.ModeNumber;
            allIntOut[PacketHeaderOffset + 1] = header.BlockSize;
            allIntOut[PacketHeaderOffset + 2] = header.IsLongBlock;
            allIntOut[PacketHeaderOffset + 3] = header.PreviousWindowLong;
            allIntOut[PacketHeaderOffset + 4] = header.NextWindowLong;
            state[SCh] = 0;
            phase = PhaseFloors;
            work += 8;
        }

        int modeNumber = allIntOut[PacketHeaderOffset + 0];
        int halfBlock = allIntOut[PacketHeaderOffset + 1] / 2;
        int mappingIdx = setupInts[setupInts[VorbisPacketDecodeSetup.TModeMappings] + modeNumber];
        int muxBase = setupInts[VorbisPacketDecodeSetup.TMappingMux]
            + setupInts[setupInts[VorbisPacketDecodeSetup.TMappingMuxOffsets] + mappingIdx];
        int submapBase = setupInts[setupInts[VorbisPacketDecodeSetup.TMappingSubmapOffsets] + mappingIdx];
        int submapCount = setupInts[setupInts[VorbisPacketDecodeSetup.TMappingScalars] + mappingIdx];

        // ---- Per-channel floor decode ----
        if (phase == PhaseFloors)
        {
            int ch = state[SCh];
            // At least one floor per launch: every launch makes progress whatever the budget.
            bool go = ch < channels;
            while (go)
            {
                int bitsBefore = r.BytePos * 8 + r.BitPos;
                int submap = setupInts[muxBase + ch];
                int floorIdx = setupInts[setupInts[VorbisPacketDecodeSetup.TMappingFloors] + submapBase + submap];
                allIntOut[floorIndexBase + ch] = floorIdx;
                int fs = setupInts[VorbisPacketDecodeSetup.TFloorScalars] + floorIdx * 5;
                int xListLen = setupInts[fs + 3];
                int scbBase = setupInts[VorbisPacketDecodeSetup.TFloorClassSubclassBooksOffsets];
                int yLen = VorbisFloor1DecoderGpu.Decode(
                    ref r, packet,
                    setupInts[fs + 0], setupInts[fs + 1], xListLen,
                    setupInts, Table(setupInts, VorbisPacketDecodeSetup.TFloorPartitionClassList, VorbisPacketDecodeSetup.TFloorPartitionClassListOffsets, floorIdx),
                    setupInts, Table(setupInts, VorbisPacketDecodeSetup.TFloorClassDimensions, VorbisPacketDecodeSetup.TFloorClassDimensionsOffsets, floorIdx),
                    setupInts, Table(setupInts, VorbisPacketDecodeSetup.TFloorClassSubclasses, VorbisPacketDecodeSetup.TFloorClassSubclassesOffsets, floorIdx),
                    setupInts, Table(setupInts, VorbisPacketDecodeSetup.TFloorClassMasterbooks, VorbisPacketDecodeSetup.TFloorClassMasterbooksOffsets, floorIdx),
                    setupInts, Table(setupInts, VorbisPacketDecodeSetup.TFloorClassSubclassBooks, VorbisPacketDecodeSetup.TFloorClassSubclassBooksOffsets, floorIdx),
                    setupInts, scbBase,
                    setupInts, setupInts,
                    setupInts, setupInts[VorbisPacketDecodeSetup.TCodebookParams],
                    allIntOut, posteriorsBase + (long)ch * sc.MaxXListLength);
                allIntOut[floorOkBase + ch] = yLen > 0 ? 1 : 0;
                work += 1 + xListLen + (r.BytePos * 8 + r.BitPos - bitsBefore);
                ch++;
                go = (ch < channels) & (work < StepBudget);
            }
            state[SCh] = ch;
            if (ch >= channels)
            {
                phase = PhaseSubmap;
                state[SSub] = 0;
            }
        }

        // ---- Per-submap residue decode ----
        bool more = (phase != PhaseDone) & (phase != PhaseFloors) & (work < StepBudget);
        while (more)
        {
            int s = state[SSub];
            int residueIdx = 0;
            int members = 0;
            int firstSubmapChannel = -1;
            if (s < submapCount)
            {
                residueIdx = setupInts[setupInts[VorbisPacketDecodeSetup.TMappingResidues] + submapBase + s];
                for (int ch = 0; ch < channels; ch++)
                {
                    if (setupInts[muxBase + ch] == s)
                    {
                        if (phase == PhaseSubmap)
                            intScratch[doNotDecodeBase + members] = allIntOut[floorOkBase + ch] != 0 ? 0 : 1;
                        if (firstSubmapChannel < 0) firstSubmapChannel = ch;
                        members++;
                    }
                }
                work += 1 + channels;
            }
            int rs = setupInts[VorbisPacketDecodeSetup.TResidueScalars] + residueIdx * 6;
            int residueType = setupInts[rs + 0];
            int partitionSize = setupInts[rs + 3];
            int classbook = setupInts[rs + 5];
            // Rows of this submap start at its first channel (channels of a submap are contiguous; v1 mapping).
            long outBase = (long)(firstSubmapChannel < 0 ? 0 : firstSubmapChannel) * halfBlock;
            // Type 2 decodes one interleaved vector of members * halfBlock as a single virtual channel.
            int vlen = residueType == 2 ? members * halfBlock : halfBlock;
            int begin = setupInts[rs + 1];
            int end = setupInts[rs + 2];
            int actualBegin = begin < vlen ? begin : vlen;
            int actualEnd = end < vlen ? end : vlen;
            int nToRead = actualEnd - actualBegin;
            int partitionsToRead = nToRead > 0 ? nToRead / partitionSize : 0;

            if (phase == PhaseSubmap)
            {
                if (s >= submapCount)
                {
                    phase = PhaseDone;
                }
                else
                {
                    // Skip the submap when it has no members, every member is do-not-decode, or nothing to read.
                    int anyDecode = 0;
                    for (int m = 0; m < members; m++)
                        if (intScratch[doNotDecodeBase + m] == 0) anyDecode = 1;
                    bool skip = (members == 0) | (anyDecode == 0) | (nToRead <= 0);
                    if (skip)
                    {
                        state[SSub] = s + 1;
                    }
                    else
                    {
                        state[SPass] = 0;
                        state[SPc] = 0;
                        state[SSubPhase] = 0;
                        state[SMember] = 0;
                        state[SIdx] = 0;
                        phase = PhaseResidue;
                    }
                }
            }
            else if (phase == PhaseResidue)
            {
                int result = ResidueSteps(
                    ref r, packet, setupInts, setupDoubles, sc.CodebookCount,
                    residueType, residueIdx, partitionSize, classbook, actualBegin, partitionsToRead,
                    residueType == 2 ? 1 : members,
                    residuesFlatOut, outBase, halfBlock,
                    intScratch, doNotDecodeBase, entryVecScratch,
                    state, ref work);
                // result: 0 = budget spent, 1 = finished, 2 = end of packet (stops this submap, no de-interleave).
                if (result != 0)
                {
                    bool deinterleave = (result == 1) & (residueType == 2) & (members > 1);
                    if (deinterleave)
                    {
                        state[SDeintCh] = members - 1;
                        state[SDeintI] = halfBlock - 1;
                        phase = PhaseDeinterleave;
                    }
                    else
                    {
                        state[SSub] = s + 1;
                        phase = PhaseSubmap;
                    }
                }
            }
            else
            {
                // Type 2 de-interleave, in place, from the end: interleaved [i * members + ch] -> row ch.
                int dch = state[SDeintCh];
                int di = state[SDeintI];
                bool go = dch >= 0; // at least one element per launch
                while (go)
                {
                    long src = outBase + (long)di * members + dch;
                    long dst = outBase + (long)dch * halfBlock + di;
                    bool add = (intScratch[doNotDecodeBase + dch] == 0) & (src != dst);
                    if (add) residuesFlatOut[dst] += residuesFlatOut[src];
                    work++;
                    di--;
                    if (di < 0)
                    {
                        di = halfBlock - 1;
                        dch--;
                    }
                    go = (dch >= 0) & (work < StepBudget);
                }
                state[SDeintCh] = dch;
                state[SDeintI] = di;
                if (dch < 0)
                {
                    state[SSub] = s + 1;
                    phase = PhaseSubmap;
                }
            }
            more = (phase != PhaseDone) & (work < StepBudget);
        }

        if (phase == PhaseDone) allIntOut[errBase] = 0;
        state[SPhase] = phase;
        state[SBytePos] = r.BytePos;
        state[SBitPos] = r.BitPos;
    }

    /// <summary>Start of the per-floor slice of table <paramref name="table"/> (offsets in <paramref name="offsets"/>).</summary>
    private static int Table(ArrayView<int> pool, int table, int offsets, int index)
        => pool[table] + pool[pool[offsets] + index];

    /// <summary>
    /// Advances the residue decode of one submap (Vorbis I sec 8.6.2, the 8-pass partition loop) until the budget
    /// is spent. Flattened order, identical to the nested loops: for pass, for partition pc: (pass 0 and pc a
    /// multiple of the classwords per codeword: one classword per decoding member), then per member the
    /// partition's codewords. Returns 0 (budget spent), 1 (all passes done) or 2 (end of packet / bad codeword).
    /// </summary>
    private static int ResidueSteps(
        ref VorbisBitReaderGpuState r, ArrayView<byte> packet,
        ArrayView<int> pool, ArrayView<double> doubles, int codebookCount,
        int residueType, int residueIdx, int partitionSize, int classbook, int actualBegin, int partitionsToRead,
        int members,
        ArrayView<float> outFlat, long outBase, int n,
        ArrayView<int> intScratch, long doNotDecodeBase, ArrayView<float> entryVec,
        ArrayView<int> state, ref int work)
    {
        int classifications = pool[pool[VorbisPacketDecodeSetup.TResidueScalars] + residueIdx * 6 + 4];
        int booksBase = pool[VorbisPacketDecodeSetup.TResidueBooks]
            + pool[pool[VorbisPacketDecodeSetup.TResidueBooksOffsets] + residueIdx];
        int cpc = pool[pool[VorbisPacketDecodeSetup.TCodebookDimensions] + classbook];
        // A 0-dimension classbook never advances the partition counter in the nested-loop form (it spins); the
        // flattened form treats it as 1 so the modulo below stays defined.
        if (cpc < 1) cpc = 1;
        bool type2 = residueType == 2;
        // Type 2 decodes its virtual channel as type 1 and ignores the do-not-decode flags (any member decoding
        // means decode all).
        int partType = type2 ? 1 : residueType;

        int pass = state[SPass];
        int pc = state[SPc];
        int sub = state[SSubPhase];
        int m = state[SMember];
        int idx = state[SIdx];
        int result = 0;
        // At least one iteration per call: the submap bookkeeping before it may already have spent the budget
        // (budget below channels + 1), and a launch must always make progress.
        bool go = true;
        while (go)
        {
            if (pass >= 8)
            {
                result = 1;
            }
            else if (pc >= partitionsToRead)
            {
                pass++;
                pc = 0;
                sub = 0;
                m = 0;
                idx = 0;
                work++;
            }
            else if (sub == 0)
            {
                // Classwords: pass 0, at the first partition of each classword group.
                bool classwordAt = (pass == 0) & (pc % cpc == 0);
                if (!classwordAt)
                {
                    sub = 1;
                    m = 0;
                }
                else if (m >= members)
                {
                    sub = 1;
                    m = 0;
                }
                else if ((!type2) & (intScratch[doNotDecodeBase + m] != 0))
                {
                    m++;
                    work++;
                }
                else
                {
                    int bitsBefore = r.BytePos * 8 + r.BitPos;
                    int temp = DecodeEntry(ref r, packet, pool, classbook);
                    work += 1 + (r.BytePos * 8 + r.BitPos - bitsBefore);
                    if (temp < 0)
                    {
                        result = 2;
                    }
                    else
                    {
                        for (int i = cpc - 1; i >= 0; i--)
                        {
                            int pi = pc + i;
                            if (pi < partitionsToRead)
                                intScratch[(long)m * partitionsToRead + pi] = temp % classifications;
                            temp /= classifications;
                        }
                        work += cpc;
                        m++;
                    }
                }
            }
            else if (m >= members)
            {
                pc++;
                sub = 0;
                m = 0;
                idx = 0;
                work++;
            }
            else if ((!type2) & (intScratch[doNotDecodeBase + m] != 0))
            {
                m++;
                work++;
            }
            else
            {
                int vqclass = intScratch[(long)m * partitionsToRead + pc];
                int bookIndex = pool[booksBase + vqclass * 8 + pass];
                if (bookIndex < 0)
                {
                    m++;
                    work++;
                }
                else
                {
                    int dims = pool[pool[VorbisPacketDecodeSetup.TCodebookDimensions] + bookIndex];
                    // Type 0: codewords i < partitionSize / dims, element d at i + d * step; type 1: contiguous.
                    int step = partitionSize / dims;
                    int limit = partType == 0 ? step : partitionSize;
                    if (idx >= limit)
                    {
                        idx = 0;
                        m++;
                        work++;
                    }
                    else
                    {
                        int bitsBefore = r.BytePos * 8 + r.BitPos;
                        int entry = DecodeEntry(ref r, packet, pool, bookIndex);
                        work += 1 + dims + (r.BytePos * 8 + r.BitPos - bitsBefore);
                        if (entry < 0)
                        {
                            result = 2;
                        }
                        else
                        {
                            VorbisCodebookVectorLookupGpu.LookupVector(
                                pool,
                                pool[VorbisPacketDecodeSetup.TAllMultiplicands] + pool[pool[VorbisPacketDecodeSetup.TMultOffsets] + bookIndex],
                                pool[pool[VorbisPacketDecodeSetup.TMultLengths] + bookIndex],
                                entry,
                                pool[pool[VorbisPacketDecodeSetup.TCodebookEntries] + bookIndex],
                                dims,
                                pool[pool[VorbisPacketDecodeSetup.TCodebookLookupTypes] + bookIndex],
                                pool[pool[VorbisPacketDecodeSetup.TCodebookQuantvals] + bookIndex],
                                doubles[bookIndex], doubles[codebookCount + bookIndex],
                                pool[pool[VorbisPacketDecodeSetup.TCodebookSequenceP] + bookIndex],
                                entryVec, 0);
                            long partBase = outBase + (long)m * n + actualBegin + (long)pc * partitionSize;
                            if (partType == 0)
                            {
                                for (int d = 0; d < dims; d++)
                                    outFlat[partBase + idx + (long)d * step] += entryVec[d];
                                idx++;
                            }
                            else
                            {
                                for (int d = 0; d < dims; d++)
                                    outFlat[partBase + idx + d] += entryVec[d];
                                idx += dims;
                            }
                        }
                    }
                }
            }
            go = (result == 0) & (work < StepBudget);
        }
        state[SPass] = pass;
        state[SPc] = pc;
        state[SSubPhase] = sub;
        state[SMember] = m;
        state[SIdx] = idx;
        return result;
    }

    /// <summary>Huffman-decodes one entry of codebook <paramref name="book"/> (negative: end of packet / bad code).</summary>
    private static int DecodeEntry(ref VorbisBitReaderGpuState r, ArrayView<byte> packet, ArrayView<int> pool, int book)
        => VorbisHuffmanDecoderGpu.TryDecode(
            ref r, packet,
            pool, pool[VorbisPacketDecodeSetup.TAllChildren] + pool[pool[VorbisPacketDecodeSetup.TChildrenOffsets] + book],
            pool, pool[VorbisPacketDecodeSetup.TAllLeafToEntry] + pool[pool[VorbisPacketDecodeSetup.TLeafOffsets] + book],
            pool[pool[VorbisPacketDecodeSetup.TMaxDepths] + book]);
}
