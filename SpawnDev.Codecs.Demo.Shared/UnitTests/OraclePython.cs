// The Python the desktop oracle tests run (Pillow / libwebp, libFLAC through soundfile): ONE lookup for every oracle.
// Two variable names grew up for it (SPAWNDEV_IMG_PYTHON for the image / FLAC oracles, CODECS_PILLOW_PYTHON for the WebP
// encoder tests); a run that set only one of them silently skipped the other half of the oracles (2026-11-15: 2 of the 7
// WebP oracle tests). Either name now serves every oracle.

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

internal static class OraclePython
{
    /// <summary>The variables read, in order; the first one naming an existing file wins.</summary>
    public const string Variables = "SPAWNDEV_IMG_PYTHON or CODECS_PILLOW_PYTHON";

    /// <summary>The oracle Python, or null when neither variable names an existing file (or in the browser).</summary>
    public static string? Find()
    {
        if (OperatingSystem.IsBrowser()) return null;
        foreach (string name in new[] { "SPAWNDEV_IMG_PYTHON", "CODECS_PILLOW_PYTHON" })
        {
            string? python = Environment.GetEnvironmentVariable(name);
            if (!string.IsNullOrEmpty(python) && File.Exists(python)) return python;
        }
        return null;
    }
}
