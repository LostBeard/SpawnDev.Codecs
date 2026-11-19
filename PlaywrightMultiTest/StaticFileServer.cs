using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System.Net;

namespace PlaywrightMultiTest
{
    public class StaticFileServer
    {
        WebApplication? app;
        Task? runningTask;
        string WWWRoot;
        string RequestPath;
        string Url;
        string devcertPath;
        public StaticFileServer(string wwwroot, string url, string requestPath = "")
        {
            if (string.IsNullOrEmpty(wwwroot))
            {
                throw new ArgumentNullException(nameof(wwwroot));
            }
            if (!Directory.Exists(wwwroot))
            {
                throw new DirectoryNotFoundException(wwwroot);
            }
            WWWRoot = Path.GetFullPath(wwwroot);
            RequestPath = requestPath;
            Url = url;
            devcertPath = Path.GetFullPath("assets/testcert.pfx");
            if (!File.Exists(devcertPath))
                throw new Exception("testcert.pfx not found. Cannot create static server");
        }
        public bool Running => runningTask?.IsCompleted == false;
        // completes when the server listens, or faults when it cannot (e.g. the port is taken)
        TaskCompletionSource? listening;

        /// <summary>
        /// Starts the server and returns once it LISTENS; throws when it cannot. It used to return at once and lose a bind
        /// failure inside the server task: with the fixed port taken by another Codecs tree's PMT, this run's browser then
        /// loaded the OTHER tree's published app and tested that build (2026-11-19).
        /// </summary>
        // servers listening in this process, by URL: Init runs once per browser lane in one PMT process, and every run builds a
        // server for the same URL and folder; the later ones reuse the first (before 2026-11-19 their bind failed silently)
        static readonly Dictionary<string, StaticFileServer> listeningInProcess = new();

        public void Start()
        {
            lock (listeningInProcess)
            {
                if (listeningInProcess.TryGetValue(Url, out var other) && !ReferenceEquals(other, this) && other.Running)
                {
                    if (!string.Equals(other.WWWRoot, WWWRoot, StringComparison.OrdinalIgnoreCase))
                        throw new InvalidOperationException($"{Url} is already served by this process from {other.WWWRoot}, not {WWWRoot}.");
                    return;   // the same app is already served at this URL
                }
            }
            if (runningTask == null)
            {
                listening = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
                runningTask = StartAsync();
            }
            try
            {
                listening!.Task.GetAwaiter().GetResult();
                lock (listeningInProcess) listeningInProcess[Url] = this;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"The static server could not listen on {Url} (another PlaywrightMultiTest of this " +
                    $"project, from another working tree, may be running and holding the port): {ex.Message}", ex);
            }
        }
        private async Task StartAsync()
        {
            try
            {
                var builder = WebApplication.CreateBuilder();
                var port = new Uri(Url).Port;

                // This wipes out Console, Debug, and any other default providers
                builder.Logging.ClearProviders();

                // Configure static file serving
                builder.WebHost.UseKestrel();
                builder.WebHost.ConfigureKestrel(serverOptions =>
                {
                    serverOptions.Listen(IPAddress.Loopback, port, listenOptions =>
                    {
                        listenOptions.UseHttps(devcertPath, "unittests");
                    });
                });
                // Use the current directory as the web root
                builder.Environment.WebRootPath = WWWRoot;
                builder.WebHost.UseUrls(Url);

                app = builder.Build();

                // (optional) add headers that enables: window.crossOriginIsolated == true
                app.Use(async (context, next) =>
                {
                    context.Response.Headers["Cross-Origin-Embedder-Policy"] = "credentialless";
                    context.Response.Headers["Cross-Origin-Opener-Policy"] = "same-origin";
                    await next();
                });

                // enable 404 fallback to default root
                app.UseStatusCodePagesWithReExecute(string.IsNullOrEmpty(RequestPath) ? "/" : RequestPath);

                // enable index.html fallback
                app.UseDefaultFiles(new DefaultFilesOptions
                {
                    FileProvider = new PhysicalFileProvider(WWWRoot),
                    RequestPath = RequestPath
                });
                // enable unknown file types (required)
                app.UseFileServer(new FileServerOptions
                {
                    FileProvider = new PhysicalFileProvider(WWWRoot),
                    RequestPath = RequestPath,
                    EnableDirectoryBrowsing = false, // Optional: allows browsing directory listings
                    StaticFileOptions = {
                        ServeUnknownFileTypes = true, // Crucial: serves all file types, even those without known MIME types
                        DefaultContentType = "application/octet-stream" // Optional: default MIME type for unknown files
                    }
                });
                // start hosting: report listening (or the bind failure) to Start, then serve until stopped
                await app.StartAsync();
                listening?.TrySetResult();
                await app.WaitForShutdownAsync();
            }
            catch (Exception ex)
            {
                listening?.TrySetException(ex);
                throw;
            }
            finally
            {
                lock (listeningInProcess)
                {
                    if (listeningInProcess.TryGetValue(Url, out var me) && ReferenceEquals(me, this)) listeningInProcess.Remove(Url);
                }
                app = null;
                runningTask = null;
            }
        }
        public async Task Stop()
        {
            if (app == null || runningTask == null) return;
            try
            {
                await app.StopAsync();
            }
            catch { }
            await app.DisposeAsync();
            if (runningTask != null)
            {
                try
                {
                    await runningTask;
                }
                catch { }
            }
        }
    }
}
