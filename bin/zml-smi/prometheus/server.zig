const std = @import("std");
const smi_info = @import("zml-smi/info");
const DeviceInfo = smi_info.device_info.DeviceInfo;
const HostInfo = smi_info.host_info.HostInfo;
const exposition = @import("exposition.zig");

// Serveur d'exposition Prometheus des metriques des appareils et de l'hote sur
// un ecoute TCP.
pub const Server = struct {
    io: std.Io,
    devices: []const *DeviceInfo,
    host: *const HostInfo,
    tcp: std.Io.net.Server,
    connection_group: std.Io.Group = .init,

    // Initialise le serveur Prometheus: parse `listen`, ouvre un ecoute TCP
    // reutilisable et construit la structure `Server`.
    pub fn init(io: std.Io, listen: []const u8, devices: []const *DeviceInfo, host: *const HostInfo) !Server {
        const address = try std.Io.net.IpAddress.parseLiteral(listen);
        var tcp = try address.listen(io, .{ .reuse_address = true });
        errdefer tcp.deinit(io);

        return .{
            .io = io,
            .devices = devices,
            .host = host,
            .tcp = tcp,
        };
    }

    // Point d'entree du serveur Prometheus: l'initialise, logue et abandonne en
    // cas d'echec, puis le fait servir jusqu'a l'arret (deinit automatique).
    pub fn run(io: std.Io, listen: []const u8, devices: []const *DeviceInfo, host_info: *const HostInfo) void {
        var self = init(io, listen, devices, host_info) catch |err| {
            std.log.err("prometheus server failed to start: {s}", .{@errorName(err)});
            return;
        };
        defer self.deinit();
        self.serve();
    }

    // Libere le serveur: attend la fin des connexions puis ferme le socket
    // d'ecoute TCP.
    pub fn deinit(self: *Server) void {
        self.connection_group.await(self.io) catch {};
        self.tcp.deinit(self.io);
    }

    // Boucle d'acceptation des connexions; chaque client est traite en
    // concurrence via `onConnection`.
    pub fn serve(self: *Server) void {
        while (true) {
            const stream = self.tcp.accept(self.io) catch break;
            self.connection_group.concurrent(self.io, Server.onConnection, .{ self, stream }) catch break;
        }
    }

    // Gestionnaire d'une connexion acceptee: ferme le flux en sortie et delegue
    // le traitement HTTP a handleConn en ignorant ses erreurs.
    fn onConnection(self: *Server, stream: std.Io.net.Stream) std.Io.Cancelable!void {
        defer stream.close(self.io);
        self.handleConn(stream) catch {};
    }

    // Boucle serveur HTTP d'une connexion: refuse les methodes autres que GET
    // (405), sinon repond 200 en text/plain avec l'exposition Prometheus.
    fn handleConn(self: *Server, stream: std.Io.net.Stream) !void {
        var read_buf: [8192]u8 = undefined;
        var reader = stream.reader(self.io, &read_buf);

        var write_buf: [4096]u8 = undefined;
        var writer = stream.writer(self.io, &write_buf);

        var http_server: std.http.Server = .init(&reader.interface, &writer.interface);

        while (true) {
            var request = http_server.receiveHead() catch |err| switch (err) {
                error.HttpConnectionClosing => return,
                else => return err,
            };

            if (request.head.method != .GET) {
                try request.respond("Method Not Allowed\n", .{ .status = .method_not_allowed });
                continue;
            }

            var body_buf: [4096]u8 = undefined;
            var body_writer = try request.respondStreaming(&body_buf, .{
                .respond_options = .{
                    .status = .ok,
                    .extra_headers = &.{
                        .{ .name = "content-type", .value = "text/plain" },
                    },
                },
            });

            try exposition.write(&body_writer.writer, self.devices, self.host);
            try body_writer.end();
        }
    }
};
