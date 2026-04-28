// Inspect BBB AV1 SequenceHeader bytes - what does libaom emit?

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.IO;
using System.Linq;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;

var bytes = File.ReadAllBytes("SpawnDev.Codecs.Demo.Shared/TestData/bbb_180_2s.ivf");
var first = IvfReader.EnumerateFrames(bytes).First();

foreach (var obu in Av1ObuParser.EnumerateObus(first.Data))
{
    if (obu.Type == Av1ObuType.SequenceHeader)
    {
        var payload = first.Data.Slice(obu.PayloadOffset, obu.PayloadLength).ToArray();
        Console.WriteLine($"BBB SH OBU header: type={obu.Type}, hasExt={obu.HasExtension}, hasSize={obu.HasSizeField}, payloadLen={payload.Length}");
        Console.WriteLine($"BBB SH payload bytes: {string.Join(" ", payload.Select(b => b.ToString("X2")))}");
        Console.WriteLine($"Bits: {string.Join("", payload.Select(b => Convert.ToString(b, 2).PadLeft(8, '0')))}");

        // Decode the SH manually to list every field
        DecodeShBitFields(payload);
        break;
    }
}

static void DecodeShBitFields(byte[] data)
{
    var br = new Av1BitReader(data);

    int seqProfile = (int)br.ReadBits(3);
    bool stillPic = br.ReadFlag();
    bool reducedStill = br.ReadFlag();
    Console.WriteLine($"  seq_profile = {seqProfile}");
    Console.WriteLine($"  still_picture = {stillPic}");
    Console.WriteLine($"  reduced_still_picture_header = {reducedStill}");

    bool timing = false, decModel = false, displayDelay = false;
    if (reducedStill)
    {
        Console.WriteLine($"  seq_level_idx[0] = {br.ReadBits(5)}");
    }
    else
    {
        timing = br.ReadFlag();
        Console.WriteLine($"  timing_info_present = {timing}");
        if (timing)
        {
            Console.WriteLine($"    num_units_in_display_tick = {br.ReadBits(32)}");
            Console.WriteLine($"    time_scale = {br.ReadBits(32)}");
            bool eq = br.ReadFlag();
            Console.WriteLine($"    equal_pic_interval = {eq}");
            if (eq) Console.WriteLine($"      num_ticks_per_picture_minus_1 (UVLC, skipping)");
            decModel = br.ReadFlag();
            Console.WriteLine($"    decoder_model_info_present = {decModel}");
            if (decModel)
            {
                br.ReadBits(5); br.ReadBits(32); br.ReadBits(5); br.ReadBits(5);
            }
        }
        displayDelay = br.ReadFlag();
        Console.WriteLine($"  initial_display_delay_present = {displayDelay}");
        int opCnt = (int)br.ReadBits(5);
        Console.WriteLine($"  operating_points_cnt_minus_1 = {opCnt}");
        for (int i = 0; i <= opCnt; i++)
        {
            Console.WriteLine($"    op[{i}].operating_point_idc = {br.ReadBits(12)}");
            int level = (int)br.ReadBits(5);
            Console.WriteLine($"    op[{i}].seq_level_idx = {level}");
            if (level > 7) Console.WriteLine($"    op[{i}].seq_tier = {br.ReadBits(1)}");
            if (decModel)
            {
                bool present = br.ReadFlag();
                if (present) br.ReadBits(20);
            }
            if (displayDelay)
            {
                bool d = br.ReadFlag();
                if (d) br.ReadBits(4);
            }
        }
    }

    int wBits = (int)br.ReadBits(4) + 1;
    int hBits = (int)br.ReadBits(4) + 1;
    int wMinus1 = (int)br.ReadBits(wBits);
    int hMinus1 = (int)br.ReadBits(hBits);
    Console.WriteLine($"  frame_width_bits = {wBits}, max_frame_width = {wMinus1 + 1}");
    Console.WriteLine($"  frame_height_bits = {hBits}, max_frame_height = {hMinus1 + 1}");

    if (!reducedStill)
    {
        bool fid = br.ReadFlag();
        Console.WriteLine($"  frame_id_numbers_present = {fid}");
        if (fid)
        {
            Console.WriteLine($"    delta_frame_id_length_minus_2 = {br.ReadBits(4)}");
            Console.WriteLine($"    frame_id_length_minus_7 = {br.ReadBits(3)}");
        }
    }

    Console.WriteLine($"  use_128x128_superblock = {br.ReadFlag()}");
    Console.WriteLine($"  enable_filter_intra = {br.ReadFlag()}");
    Console.WriteLine($"  enable_intra_edge_filter = {br.ReadFlag()}");

    if (!reducedStill)
    {
        Console.WriteLine($"  enable_interintra_compound = {br.ReadFlag()}");
        Console.WriteLine($"  enable_masked_compound = {br.ReadFlag()}");
        Console.WriteLine($"  enable_warped_motion = {br.ReadFlag()}");
        Console.WriteLine($"  enable_dual_filter = {br.ReadFlag()}");
        bool oh = br.ReadFlag();
        Console.WriteLine($"  enable_order_hint = {oh}");
        if (oh)
        {
            Console.WriteLine($"    enable_jnt_comp = {br.ReadFlag()}");
            Console.WriteLine($"    enable_ref_frame_mvs = {br.ReadFlag()}");
        }
        bool sccChoose = br.ReadFlag();
        Console.WriteLine($"  seq_choose_screen_content_tools = {sccChoose}");
        int sccForce = sccChoose ? 2 : (int)br.ReadBits(1);
        if (!sccChoose) Console.WriteLine($"    seq_force_screen_content_tools = {sccForce}");
        if (sccForce > 0)
        {
            bool imChoose = br.ReadFlag();
            Console.WriteLine($"    seq_choose_integer_mv = {imChoose}");
            if (!imChoose) Console.WriteLine($"      seq_force_integer_mv = {br.ReadBits(1)}");
        }
        if (oh) Console.WriteLine($"  order_hint_bits_minus_1 = {br.ReadBits(3)}");
    }

    Console.WriteLine($"  enable_superres = {br.ReadFlag()}");
    Console.WriteLine($"  enable_cdef = {br.ReadFlag()}");
    Console.WriteLine($"  enable_restoration = {br.ReadFlag()}");

    bool hbd = br.ReadFlag();
    Console.WriteLine($"  high_bitdepth = {hbd}");
    int bitDepth = 8;
    int seqProfileSnap = ((data[0] >> 5) & 7);
    if (seqProfileSnap == 2 && hbd) { bool tw = br.ReadFlag(); bitDepth = tw ? 12 : 10; Console.WriteLine($"    twelve_bit = {tw}"); }
    else if (hbd) bitDepth = 10;
    Console.WriteLine($"  bit_depth = {bitDepth}");

    bool mono = false;
    if (seqProfileSnap != 1) { mono = br.ReadFlag(); Console.WriteLine($"  monochrome = {mono}"); }

    bool colorDescPresent = br.ReadFlag();
    Console.WriteLine($"  color_description_present = {colorDescPresent}");
    if (colorDescPresent)
    {
        Console.WriteLine($"    color_primaries = {br.ReadBits(8)}");
        Console.WriteLine($"    transfer_chars  = {br.ReadBits(8)}");
        Console.WriteLine($"    matrix_coefs    = {br.ReadBits(8)}");
    }
    if (mono)
    {
        Console.WriteLine($"  color_range = {br.ReadFlag()}");
    }
    else
    {
        Console.WriteLine($"  color_range = {br.ReadFlag()}");
        // Profile 0 -> subsampling fixed (1,1); Profile 1 -> (0,0); Profile 2 -> conditional
        // chroma_sample_position only if subX=1 && subY=1.
        if (seqProfileSnap == 0 || (seqProfileSnap == 2 && bitDepth != 12))
        {
            // subX=1, subY=1 for profile 0; profile 2 with bit_depth != 12 has subX=1, subY=0; only profile 0 reaches here
        }
        if (seqProfileSnap == 0)
        {
            Console.WriteLine($"  chroma_sample_position = {br.ReadBits(2)}");
        }
        Console.WriteLine($"  separate_uv_deltas = {br.ReadFlag()}");
    }
    Console.WriteLine($"  film_grain_params_present = {br.ReadFlag()}");

    Console.WriteLine($"  Bits remaining at this point: {br.BitsRemaining}");
}
