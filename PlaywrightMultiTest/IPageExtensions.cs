using Microsoft.Playwright;

namespace PlaywrightMultiTest
{
    public static class IPageExtensions
    {
        public static async Task WaitForConditionAsync(this IPage page, Func<Task<bool>> condition, int timeoutMs = 660_000)
        {
            // Longer than the longest in-page [TestMethod(Timeout = 600000)]: the page must report its own timeout (with the
            // test's name and state) before this generic wait gives up; equal windows race.
            var start = DateTime.Now;
            while ((DateTime.Now - start).TotalMilliseconds < timeoutMs)
            {
                if (await condition()) return;
                await Task.Delay(100); // Polling interval
            }
            throw new TimeoutException("Condition was not met within the timeout.");
        }
    }
}
