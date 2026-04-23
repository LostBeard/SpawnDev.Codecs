namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Small throw-on-fail assertion helpers. SpawnDev.UnitTesting treats thrown exceptions
/// as test failures, so these just wrap common checks to produce readable failure
/// messages without needing a full xunit-style Assert class.
/// </summary>
internal static class TestHelpers
{
    public static void Equal<T>(T expected, T actual, string? context = null)
    {
        if (!EqualityComparer<T>.Default.Equals(expected, actual))
            throw new Exception($"{Prefix(context)}Expected '{expected}' but got '{actual}'");
    }

    public static void EqualBytes(byte[] expected, byte[] actual, string? context = null)
    {
        if (expected.Length != actual.Length)
            throw new Exception($"{Prefix(context)}Length mismatch: expected {expected.Length}, got {actual.Length}");
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] != actual[i])
                throw new Exception($"{Prefix(context)}Byte {i}: expected 0x{expected[i]:X2}, got 0x{actual[i]:X2}");
        }
    }

    public static void EqualInts(int[] expected, int[] actual, string? context = null)
    {
        if (expected.Length != actual.Length)
            throw new Exception($"{Prefix(context)}Length mismatch: expected {expected.Length}, got {actual.Length}");
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] != actual[i])
                throw new Exception($"{Prefix(context)}Index {i}: expected {expected[i]}, got {actual[i]}");
        }
    }

    public static void True(bool condition, string? message = null)
    {
        if (!condition) throw new Exception(message ?? "Expected true but was false");
    }

    public static void False(bool condition, string? message = null)
    {
        if (condition) throw new Exception(message ?? "Expected false but was true");
    }

    public static void Throws<TException>(Action action, string? context = null) where TException : Exception
    {
        try { action(); }
        catch (TException) { return; }
        catch (Exception ex)
        {
            throw new Exception($"{Prefix(context)}Expected {typeof(TException).Name} but got {ex.GetType().Name}: {ex.Message}");
        }
        throw new Exception($"{Prefix(context)}Expected {typeof(TException).Name} but no exception was thrown");
    }

    public static void InRange(int value, int min, int max)
    {
        if (value < min || value > max)
            throw new Exception($"Expected {value} to be in [{min}, {max}]");
    }

    public static void InRange(uint value, uint min, uint max)
    {
        if (value < min || value > max)
            throw new Exception($"Expected {value} to be in [{min}, {max}]");
    }

    public static void Contains(string substring, string str, string? context = null)
    {
        if (!str.Contains(substring))
            throw new Exception($"{Prefix(context)}Expected string to contain '{substring}' but was '{str}'");
    }

    public static void NotNull<T>(T? value, string? message = null) where T : class
    {
        if (value is null) throw new Exception(message ?? "Expected non-null value");
    }

    private static string Prefix(string? ctx) => string.IsNullOrEmpty(ctx) ? "" : $"[{ctx}] ";
}
