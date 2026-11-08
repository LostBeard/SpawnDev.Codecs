/*
 * SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
 *
 * Reference decoder for the MP3 decoder's fixtures (scripts/audio/make_mp3_decoder_fixtures.py): minimp3
 * (github.com/lieff/minimp3, CC0 / public domain) with float output. Frames are decoded by minimp3's mp3dec_decode_frame;
 * the container steps use minimp3_ex's own helpers (mp3dec_skip_id3: ID3v2 / ID3v1 / APEv2; mp3d_find_frame +
 * mp3dec_check_vbrtag: the Xing / Info frame is skipped and gives delay + 529 and padding - 529). The trim is the
 * convention FFmpeg follows (and our decoder): delay + 529 samples dropped at the start, padding - 529 (when positive)
 * at the end of the frames decoded. minimp3_ex's mp3dec_load_buf instead cuts the output to the Info frame's frame count
 * minus delay and padding; the two agree unless that count is wrong (FFmpeg's muxer writes a short count for some
 * encoders, e.g. MediaFoundation's mp3_mf, and mp3dec_load_buf then drops most of the audio).
 * minimp3.h and minimp3_ex.h are NOT in this repository: download them and point /I at their folder.
 *
 * Build (x64 Native Tools prompt, or after vcvars64.bat):
 *   cl /nologo /O2 /I <folder with minimp3.h + minimp3_ex.h> minimp3_ref.c /Fe:minimp3_ref.exe
 * x64 builds take minimp3's SSE paths (MINIMP3_ONLY_SIMD); our decoder follows those paths' arithmetic.
 *
 * Usage: minimp3_ref in.mp3 out.f32
 *   writes the interleaved float32 samples (little-endian) and prints
 *   "channels hz samples_per_channel load_buf_samples_per_channel" (the last: minimp3_ex's mp3dec_load_buf count).
 */
#define MINIMP3_IMPLEMENTATION
#define MINIMP3_FLOAT_OUTPUT
#include "minimp3_ex.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    FILE *f;
    long size;
    unsigned char *file;
    const uint8_t *buf;
    size_t buf_size;
    mp3dec_t dec;
    mp3dec_frame_info_t info;
    mp3dec_file_info_t ex;
    float *pcm;
    size_t n = 0, cap, skip, total, ex_samples = 0;
    int channels = 0, hz = 0, free_format_bytes = 0, frame_size = 0, i, ret, delay = 0, padding = 0;
    uint32_t tag_frames = 0;
    if (argc != 3)
    {
        fprintf(stderr, "usage: minimp3_ref in.mp3 out.f32\n");
        return 2;
    }
    f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    file = (unsigned char *)malloc(size > 0 ? size : 1);
    if (fread(file, 1, size, f) != (size_t)size) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);

    /* minimp3_ex's own count, for the report */
    if (!mp3dec_load_buf(&dec, file, (size_t)size, &ex, NULL, NULL) && ex.channels)
    {
        ex_samples = ex.samples / ex.channels;
        free(ex.buffer);
    }

    buf = file;
    buf_size = (size_t)size;
    mp3dec_skip_id3(&buf, &buf_size);
    i = mp3d_find_frame(buf, (int)buf_size, &free_format_bytes, &frame_size);
    if (!frame_size) { fprintf(stderr, "no frames\n"); return 1; }
    buf += i;
    buf_size -= i;
    ret = mp3dec_check_vbrtag(buf, frame_size, &tag_frames, &delay, &padding);
    if (ret)
    {
        buf += frame_size;
        buf_size -= frame_size;
    }
    if (ret <= 0) delay = padding = 0;

    mp3dec_init(&dec);
    cap = 1 << 20;
    pcm = (float *)malloc(cap * sizeof(float));
    for (;;)
    {
        int samples;
        if (cap - n < MINIMP3_MAX_SAMPLES_PER_FRAME)
        {
            cap *= 2;
            pcm = (float *)realloc(pcm, cap * sizeof(float));
        }
        samples = mp3dec_decode_frame(&dec, buf, (int)buf_size, pcm + n, &info);
        if (!info.frame_bytes) break;
        buf += info.frame_bytes;
        buf_size -= info.frame_bytes;
        if (samples)
        {
            if (!channels) { channels = info.channels; hz = info.hz; }
            if (info.channels != channels) { fprintf(stderr, "channel count changes\n"); return 1; }
            n += (size_t)samples * channels;
        }
    }
    if (!channels) { fprintf(stderr, "nothing decoded\n"); return 1; }
    skip = (size_t)delay * channels;
    total = n > skip ? n - skip : 0;
    if (padding > 0) total = total > (size_t)padding * channels ? total - (size_t)padding * channels : 0;

    f = fopen(argv[2], "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    if (total) fwrite(pcm + skip, sizeof(float), total, f);
    fclose(f);
    printf("%d %d %zu %zu\n", channels, hz, total / channels, ex_samples);
    free(pcm);
    free(file);
    return 0;
}
